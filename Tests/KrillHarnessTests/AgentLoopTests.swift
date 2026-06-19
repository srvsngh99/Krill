import XCTest
@testable import KrillHarness
import KrillTooling

/// Deterministic generator: returns scripted outputs in order, records the last
/// message history it was handed (to assert tool observations were fed back).
private actor MockGenerator: HarnessGenerator {
    nonisolated let toolFormat: ToolCalling.ToolFormat
    private let responses: [String]
    private var idx = 0
    private(set) var lastMessages: [[String: String]] = []
    private(set) var callCount = 0

    init(toolFormat: ToolCalling.ToolFormat = .hermes, responses: [String]) {
        self.toolFormat = toolFormat
        self.responses = responses
    }

    func complete(messages: [[String: String]]) async -> String {
        lastMessages = messages
        callCount += 1
        defer { idx += 1 }
        return idx < responses.count ? responses[idx] : "Done."
    }

    func recordedCallCount() async -> Int { callCount }
    func recordedMessages() async -> [[String: String]] { lastMessages }
}

/// A tool that echoes back a fixed observation regardless of input.
private struct StubTool: Tool {
    let name: String
    let description = "test stub"
    let parametersJSON = #"{"type":"object","properties":{"command":{"type":"string"}}}"#
    let reply: String
    let asError: Bool
    init(name: String = "bash", reply: String = "ok", asError: Bool = false) {
        self.name = name
        self.reply = reply
        self.asError = asError
    }
    func run(argumentsJSON: String) async -> ToolResult {
        ToolResult(content: reply, isError: asError)
    }
}

private func hermesCall(_ name: String, _ argsJSON: String) -> String {
    "<tool_call>{\"name\": \"\(name)\", \"arguments\": \(argsJSON)}</tool_call>"
}

/// Generator that records whether the grammar-constrained repair pass fired and
/// returns a canned constrained reply.
private actor RepairMockGenerator: HarnessGenerator {
    nonisolated let toolFormat: ToolCalling.ToolFormat = .hermes
    private let freeResponses: [String]
    private let constrainedReply: String
    private var idx = 0
    private(set) var constrainedCalls = 0

    init(freeResponses: [String], constrainedReply: String) {
        self.freeResponses = freeResponses
        self.constrainedReply = constrainedReply
    }
    func complete(messages: [[String: String]]) async -> String {
        defer { idx += 1 }
        return idx < freeResponses.count ? freeResponses[idx] : "done"
    }
    func completeConstrained(messages: [[String: String]], jsonSchema: String) async -> String {
        constrainedCalls += 1
        return constrainedReply
    }
    func recordedConstrainedCalls() async -> Int { constrainedCalls }
}

/// Tool whose schema makes `command` required, so `{}` fails the schema check
/// (the empty-args case). Echoes the args it received so tests can assert which
/// args actually ran.
private struct RequiredArgTool: Tool {
    let name = "bash"
    let description = "test stub with a required argument"
    let parametersJSON = #"{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}"#
    func run(argumentsJSON: String) async -> ToolResult { ToolResult(content: "ARGS=\(argumentsJSON)") }
}

final class AgentLoopTests: XCTestCase {

    func testNoToolCallReturnsFinalAnswerDirectly() async {
        let gen = MockGenerator(responses: ["The answer is 4."])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([StubTool()]))
        let t = await loop.run(user: "What is 2+2?")

        XCTAssertEqual(t.finalText, "The answer is 4.")
        XCTAssertFalse(t.hitIterationLimit)
        XCTAssertEqual(t.steps.count, 1)
        XCTAssertTrue(t.steps[0].toolCalls.isEmpty)
        let calls = await gen.recordedCallCount()
        XCTAssertEqual(calls, 1, "should generate once when no tool is called")
    }

    func testSingleToolCallThenFinalAnswer() async {
        let gen = MockGenerator(responses: [
            "Let me check.\n" + hermesCall("bash", #"{"command":"echo hi"}"#),
            "The file says hi.",
        ])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([StubTool(reply: "hi")]))
        let t = await loop.run(user: "run echo hi")

        XCTAssertEqual(t.finalText, "The file says hi.")
        XCTAssertFalse(t.hitIterationLimit)
        XCTAssertEqual(t.steps.count, 2)
        XCTAssertEqual(t.steps[0].toolCalls.count, 1)
        let inv = t.steps[0].toolCalls[0]
        XCTAssertEqual(inv.name, "bash")
        XCTAssertEqual(inv.argumentsJSON, #"{"command":"echo hi"}"#)
        XCTAssertEqual(inv.result.content, "hi")
        XCTAssertFalse(inv.result.isError)
        XCTAssertTrue(t.steps[1].toolCalls.isEmpty)
    }

    func testToolObservationIsFedBackToTheModel() async {
        let gen = MockGenerator(responses: [
            hermesCall("bash", #"{"command":"echo hi"}"#),
            "done",
        ])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([StubTool(reply: "OBSERVED_42")]))
        _ = await loop.run(user: "go")

        // On the 2nd generation the history must contain the tool observation.
        let msgs = await gen.recordedMessages()
        XCTAssertTrue(
            msgs.contains { ($0["content"] ?? "").contains("OBSERVED_42") },
            "the tool result must be fed back into the next generation's messages")
    }

    func testUnknownToolYieldsErrorObservationAndContinues() async {
        let gen = MockGenerator(responses: [
            hermesCall("nonexistent", #"{}"#),
            "recovered",
        ])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([StubTool()]))
        let t = await loop.run(user: "go")

        XCTAssertEqual(t.finalText, "recovered")
        let inv = t.steps[0].toolCalls[0]
        XCTAssertEqual(inv.name, "nonexistent")
        XCTAssertTrue(inv.result.isError)
        XCTAssertTrue(inv.result.content.contains("unknown tool"))
    }

    func testIterationCapStopsAnInfiniteToolLoop() async {
        // Model that ALWAYS calls a tool: the cap must stop it.
        let always = Array(repeating: hermesCall("bash", #"{"command":"loop"}"#), count: 20)
        let gen = MockGenerator(responses: always)
        let loop = AgentLoop(
            generator: gen, tools: ToolRegistry([StubTool()]), maxIterations: 3)
        let t = await loop.run(user: "go")

        XCTAssertTrue(t.hitIterationLimit)
        XCTAssertEqual(t.steps.count, 3, "should stop exactly at the cap")
        XCTAssertEqual(t.finalText, "")
    }

    func testEmptyArgsAreRepairedViaConstrainedPass() async {
        // The model emits an empty-args call (the small-model failure); the
        // schema-constrained second pass must repair it and the tool runs with
        // the repaired args.
        let gen = RepairMockGenerator(
            freeResponses: [hermesCall("bash", "{}"), "all done"],
            constrainedReply: #"{"command":"echo hi"}"#)
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([RequiredArgTool()]))
        let t = await loop.run(user: "go")

        let inv = t.steps[0].toolCalls[0]
        XCTAssertEqual(inv.argumentsJSON, #"{"command":"echo hi"}"#, "empty args must be repaired")
        XCTAssertTrue(inv.result.content.contains("echo hi"), "tool must run with repaired args")
        let calls = await gen.recordedConstrainedCalls()
        XCTAssertEqual(calls, 1, "the constrained repair pass should fire exactly once")
    }

    func testValidArgsSkipTheConstrainedPass() async {
        // Capable model: args already satisfy the schema, so no repair (the gate
        // is selective / fail-open).
        let gen = RepairMockGenerator(
            freeResponses: [hermesCall("bash", #"{"command":"ls"}"#), "done"],
            constrainedReply: #"{"command":"SHOULD_NOT_BE_USED"}"#)
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([RequiredArgTool()]))
        let t = await loop.run(user: "go")

        XCTAssertEqual(t.steps[0].toolCalls[0].argumentsJSON, #"{"command":"ls"}"#)
        let calls = await gen.recordedConstrainedCalls()
        XCTAssertEqual(calls, 0, "valid args must not trigger a repair pass")
    }

    func testFailOpenKeepsOriginalArgsWhenRepairAlsoFails() async {
        // The constrained pass returns args that STILL fail the schema (and are
        // structurally distinct from the original, so the assertion proves the
        // ORIGINAL was kept, not coincidence); the loop must fall back to the
        // original args (fail-open), not run garbage.
        let gen = RepairMockGenerator(
            freeResponses: [hermesCall("bash", "{}"), "done"],
            constrainedReply: #"{"wrong_field":"x"}"#)
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([RequiredArgTool()]))
        let t = await loop.run(user: "go")

        XCTAssertEqual(t.steps[0].toolCalls[0].argumentsJSON, "{}",
                       "fail-open: keep the original args when repair also fails the schema")
        let calls = await gen.recordedConstrainedCalls()
        XCTAssertEqual(calls, 1, "repair should have been attempted exactly once")
    }

    func testConstrainDisabledSkipsRepair() async {
        let gen = RepairMockGenerator(
            freeResponses: [hermesCall("bash", "{}"), "done"],
            constrainedReply: #"{"command":"x"}"#)
        let loop = AgentLoop(
            generator: gen, tools: ToolRegistry([RequiredArgTool()]), constrainToolArgs: false)
        let t = await loop.run(user: "go")

        XCTAssertEqual(t.steps[0].toolCalls[0].argumentsJSON, "{}", "repair must be off when disabled")
        let calls = await gen.recordedConstrainedCalls()
        XCTAssertEqual(calls, 0)
    }

    func testReasoningIsStrippedFromFinalAnswer() async {
        let gen = MockGenerator(responses: [
            "<think>2+2 is basic arithmetic, the result is 4.</think>The answer is 4.",
        ])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([StubTool()]))
        let t = await loop.run(user: "What is 2+2?")

        XCTAssertEqual(t.finalText, "The answer is 4.", "reasoning must not leak into the final answer")
        XCTAssertEqual(t.steps[0].assistantText, "The answer is 4.")
    }

    func testReasoningIsStrippedFromToolCallStepButRawIsFedBack() async {
        let gen = MockGenerator(responses: [
            "<think>I should run the command first.</think>Checking.\n"
                + hermesCall("bash", #"{"command":"echo hi"}"#),
            "<think>The output looks right.</think>Done.",
        ])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([StubTool(reply: "hi")]))
        let t = await loop.run(user: "go")

        // Display text is stripped of reasoning on both the tool step and the final.
        XCTAssertEqual(t.steps[0].assistantText, "Checking.")
        XCTAssertEqual(t.finalText, "Done.")
        // But the raw assistant turn (with reasoning) is preserved in history so
        // the model keeps its own context.
        XCTAssertTrue(
            t.messages.contains {
                $0["role"] == "assistant" && ($0["content"] ?? "").contains("<think>")
            },
            "raw model output (including reasoning) must remain in the message history")
    }

    func testRegistrySpecsPreserveOrderAndDedupe() {
        let reg = ToolRegistry([
            StubTool(name: "read"),
            StubTool(name: "bash"),
            StubTool(name: "read"),  // duplicate ignored
        ])
        let specs = reg.specs()
        XCTAssertEqual(specs.map(\.name), ["read", "bash"])
        XCTAssertNotNil(reg.tool(named: "bash"))
        XCTAssertNil(reg.tool(named: "missing"))
    }
}
