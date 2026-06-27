import XCTest
@testable import KrillHarness
import KrillTooling

/// Scripted generator (local to this file): returns canned outputs in order.
private actor EventScriptGen: HarnessGenerator {
    nonisolated let toolFormat: ToolCalling.ToolFormat = .hermes
    private let responses: [String]
    private var idx = 0
    init(_ responses: [String]) { self.responses = responses }
    func complete(messages: [[String: String]]) async -> String {
        defer { idx += 1 }
        return idx < responses.count ? responses[idx] : "done"
    }
}

private struct EchoTool: Tool {
    let name: String
    let isReadOnly: Bool
    let description = "echo"
    let parametersJSON = #"{"type":"object","properties":{"x":{"type":"string"}}}"#
    let reply: String
    func run(argumentsJSON: String) async -> ToolResult { ToolResult(content: reply) }
}

/// Collects events from a run in order. `@unchecked Sendable` because the loop
/// calls the sink synchronously from a single task (no concurrent access).
private final class EventSink: @unchecked Sendable {
    private let lock = NSLock()
    private(set) var events: [AgentEvent] = []
    var sink: @Sendable (AgentEvent) -> Void {
        { [self] e in lock.lock(); events.append(e); lock.unlock() }
    }
}

private func hermesCall(_ name: String, _ argsJSON: String) -> String {
    "<tool_call>{\"name\": \"\(name)\", \"arguments\": \(argsJSON)}</tool_call>"
}

final class AgentEventTests: XCTestCase {

    func testNoToolRunEmitsOnlyFinal() async {
        // A terminal turn emits ONLY finalAnswer - no assistantTurn - so a
        // consumer never double-prints the answer.
        let sink = EventSink()
        let gen = EventScriptGen(["The answer is 4."])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([EchoTool(name: "bash", isReadOnly: false, reply: "ok")]))
        _ = await loop.run(user: "2+2?", onEvent: sink.sink)

        XCTAssertEqual(sink.events, [.finalAnswer("The answer is 4.")])
    }

    func testSingleToolRunEmitsFullSequence() async {
        let sink = EventSink()
        let gen = EventScriptGen([
            "Checking.\n" + hermesCall("read_file", #"{"x":"a"}"#),
            "All done.",
        ])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([EchoTool(name: "read_file", isReadOnly: true, reply: "FILE")]))
        _ = await loop.run(user: "go", onEvent: sink.sink)

        XCTAssertEqual(sink.events.count, 4)
        XCTAssertEqual(sink.events[0], .assistantTurn(text: "Checking."))
        XCTAssertEqual(sink.events[1], .toolStarted(name: "read_file", argumentsJSON: #"{"x":"a"}"#))
        guard case .toolFinished(let inv) = sink.events[2] else { return XCTFail("expected toolFinished") }
        XCTAssertEqual(inv.name, "read_file")
        XCTAssertEqual(inv.result.content, "FILE")
        XCTAssertEqual(sink.events[3], .finalAnswer("All done."))
    }

    func testDeniedToolEmitsFinishedButNotStarted() async {
        // Plan mode denies a mutating tool: a toolFinished (error) is emitted,
        // but no toolStarted (it never ran).
        let sink = EventSink()
        let gen = EventScriptGen([hermesCall("edit_file", #"{"x":"a"}"#), "here is the plan"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([EchoTool(name: "edit_file", isReadOnly: false, reply: "EDITED")]),
            permission: PermissionPolicy(mode: .plan))
        _ = await loop.run(user: "go", onEvent: sink.sink)

        XCTAssertFalse(
            sink.events.contains { if case .toolStarted = $0 { return true } else { return false } },
            "a denied tool must not emit toolStarted")
        guard case .toolFinished(let inv) = sink.events[1] else { return XCTFail("expected toolFinished") }
        XCTAssertTrue(inv.result.isError)
        XCTAssertTrue(inv.result.content.contains("Permission denied"))
    }

    func testIterationLimitEmitsLimitEventAndNoFinal() async {
        let sink = EventSink()
        // Always calls a tool (varying args so the runaway dedupe does not trip
        // first): the cap stops it, so no finalAnswer is emitted.
        let always = (0 ..< 10).map { hermesCall("read_file", "{\"x\":\"\($0)\"}") }
        let loop = AgentLoop(
            generator: EventScriptGen(always),
            tools: ToolRegistry([EchoTool(name: "read_file", isReadOnly: true, reply: "ok")]),
            maxIterations: 2)
        _ = await loop.run(user: "go", onEvent: sink.sink)

        XCTAssertEqual(sink.events.last, .iterationLimitReached)
        XCTAssertFalse(
            sink.events.contains { if case .finalAnswer = $0 { return true } else { return false } },
            "no final answer when the loop hits the iteration cap")
    }

    func testCancelledTaskStopsLoopAndEmitsCancelled() async {
        let sink = EventSink()
        // A generator that would otherwise loop forever calling a tool.
        let always = Array(repeating: hermesCall("read_file", #"{"x":"a"}"#), count: 100)
        let loop = AgentLoop(
            generator: EventScriptGen(always),
            tools: ToolRegistry([EchoTool(name: "read_file", isReadOnly: true, reply: "ok")]),
            maxIterations: 100)
        let task = Task { await loop.run(user: "go", onEvent: sink.sink) }
        task.cancel()
        let t = await task.value

        XCTAssertTrue(t.wasCancelled, "a cancelled Task must stop the loop")
        XCTAssertFalse(t.hitIterationLimit, "cancellation is not the iteration cap")
        XCTAssertEqual(sink.events.last, .cancelled)
    }

    func testRunWithoutObserverReturnsSameTranscript() async {
        // The event seam is additive: omitting onEvent yields the original
        // batch result unchanged.
        let gen = EventScriptGen([
            "Checking.\n" + hermesCall("read_file", #"{"x":"a"}"#),
            "All done.",
        ])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([EchoTool(name: "read_file", isReadOnly: true, reply: "FILE")]))
        let t = await loop.run(user: "go")

        XCTAssertEqual(t.finalText, "All done.")
        XCTAssertFalse(t.hitIterationLimit)
        XCTAssertEqual(t.steps.count, 2)
        XCTAssertEqual(t.steps[0].toolCalls.first?.result.content, "FILE")
    }
}
