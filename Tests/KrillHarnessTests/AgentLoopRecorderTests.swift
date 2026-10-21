import XCTest
import KrillObs
@testable import KrillHarness
import KrillTooling

private actor ScriptedGenerator: HarnessGenerator {
    nonisolated let toolFormat: ToolCalling.ToolFormat = .hermes
    private let responses: [String]
    private var idx = 0
    init(_ responses: [String]) { self.responses = responses }
    func complete(messages: [[String: String]]) async -> String {
        defer { idx += 1 }
        return idx < responses.count ? responses[idx] : "Done."
    }
}

private struct EchoTool: Tool {
    let name: String
    let description = "stub"
    let parametersJSON = #"{"type":"object","properties":{"command":{"type":"string"}}}"#
    let isReadOnly: Bool
    let isFileEdit = false
    func run(argumentsJSON: String) async -> ToolResult { ToolResult(content: "result-of-\(name)", isError: false) }
}

private func call(_ name: String, _ args: String) -> String {
    "<tool_call>{\"name\": \"\(name)\", \"arguments\": \(args)}</tool_call>"
}

/// The agent loop is the one place every surface (`krill code`, the TUI, hosted /ui sessions) goes
/// through, so recording there covers all of them.
final class AgentLoopRecorderTests: XCTestCase {
    var tmp: URL!
    var settings: ObsSettings!

    override func setUp() {
        tmp = FileManager.default.temporaryDirectory.appendingPathComponent("krill-loop-\(UUID().uuidString)")
        settings = ObsSettings(eventLog: false, sessionLog: .full, root: tmp)
    }
    override func tearDown() { try? FileManager.default.removeItem(at: tmp) }

    private func events(_ r: SessionRecorder) -> [[String: Any]] {
        r.flush()
        return ((try? String(contentsOf: r.fileURL, encoding: .utf8)) ?? "").split(separator: "\n").compactMap {
            (try? JSONSerialization.jsonObject(with: Data($0.utf8))) as? [String: Any]
        }
    }

    func testToolRunThenFinalAnswerIsRecordedInOrder() async {
        let rec = SessionRecorder.start(surface: "code", project: "/p", model: "m", provider: "local", settings: settings)!
        let loop = AgentLoop(
            generator: ScriptedGenerator([call("bash", #"{"command":"ls"}"#), "All done."]),
            tools: ToolRegistry([EchoTool(name: "bash", isReadOnly: false)]),
            permission: PermissionPolicy(mode: .acceptAll), recorder: rec)
        let t = await loop.run(user: "list files")
        XCTAssertEqual(t.finalText, "All done.")
        let evs = events(rec)
        let names = evs.compactMap { $0["event"] as? String }
        XCTAssertEqual(names, ["session_start", "user", "assistant", "permission", "tool_call", "tool_result", "assistant", "stop"])
        XCTAssertEqual(evs[1]["text"] as? String, "list files")
        XCTAssertEqual(evs[3]["decision"] as? String, "allow")
        XCTAssertEqual(evs[4]["tool"] as? String, "bash")
        XCTAssertEqual((evs[4]["args"] as? String)?.contains("ls"), true)
        XCTAssertEqual(evs[5]["result"] as? String, "result-of-bash")
        XCTAssertEqual(evs[5]["is_error"] as? Bool, false)
        XCTAssertEqual(evs[7]["reason"] as? String, "final_answer")
    }

    func testPlanModeDenialIsRecordedAsADeniedPermissionWithReason() async {
        let rec = SessionRecorder.start(surface: "code", project: nil, model: nil, provider: nil, settings: settings)!
        let loop = AgentLoop(
            generator: ScriptedGenerator([call("bash", #"{"command":"rm -rf /"}"#), "Understood."]),
            tools: ToolRegistry([EchoTool(name: "bash", isReadOnly: false)]),
            permission: PermissionPolicy(mode: .plan), recorder: rec)
        _ = await loop.run(user: "clean up")
        let perm = events(rec).first { $0["event"] as? String == "permission" }
        XCTAssertEqual(perm?["decision"] as? String, "deny")
        XCTAssertNotNil(perm?["reason"] as? String)
        let result = events(rec).first { $0["event"] as? String == "tool_result" }
        XCTAssertEqual(result?["is_error"] as? Bool, true)
    }

    func testUnknownToolIsRecorded() async {
        let rec = SessionRecorder.start(surface: "code", project: nil, model: nil, provider: nil, settings: settings)!
        let loop = AgentLoop(
            generator: ScriptedGenerator([call("teleport", "{}"), "ok"]),
            tools: ToolRegistry([EchoTool(name: "bash", isReadOnly: false)]),
            constrainToolArgs: false, recorder: rec)
        _ = await loop.run(user: "x")
        let perm = events(rec).first { $0["event"] as? String == "permission" }
        XCTAssertEqual(perm?["decision"] as? String, "unknown_tool")
    }

    func testIterationLimitStopReason() async {
        let rec = SessionRecorder.start(surface: "code", project: nil, model: nil, provider: nil, settings: settings)!
        let many = (0..<5).map { call("bash", #"{"command":"echo \#($0)"}"#) }
        let loop = AgentLoop(
            generator: ScriptedGenerator(many), tools: ToolRegistry([EchoTool(name: "bash", isReadOnly: false)]),
            maxIterations: 2, permission: PermissionPolicy(mode: .acceptAll), recorder: rec)
        let t = await loop.run(user: "loop")
        XCTAssertTrue(t.hitIterationLimit)
        XCTAssertEqual(events(rec).last?["reason"] as? String, "iteration_limit")
    }

    func testNoRecorderChangesNothing() async {
        let loop = AgentLoop(generator: ScriptedGenerator(["hi"]), tools: ToolRegistry([]))
        let t = await loop.run(user: "x")
        XCTAssertEqual(t.finalText, "hi")
        XCTAssertFalse(FileManager.default.fileExists(atPath: tmp.path))
    }
}
