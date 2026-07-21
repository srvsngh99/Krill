import XCTest
@testable import KrillHarness
import KrillTooling

// MARK: - Test doubles

/// Scripted generator: returns canned outputs in order, then "done".
private actor ScriptGen: HarnessGenerator {
    nonisolated let toolFormat: ToolCalling.ToolFormat = .hermes
    private let responses: [String]
    private var idx = 0
    init(_ responses: [String]) { self.responses = responses }
    func complete(messages: [[String: String]]) async -> String {
        defer { idx += 1 }
        return idx < responses.count ? responses[idx] : "done"
    }
}

/// Records whether `run` was actually invoked, so a denied tool can be proven
/// to have never executed (not merely overridden in the transcript).
private actor RunFlag {
    private(set) var ran = false
    func mark() { ran = true }
}

private struct FlagTool: Tool {
    let name: String
    let isReadOnly: Bool
    var isFileEdit: Bool = false
    let description = "flag tool"
    let parametersJSON = #"{"type":"object","properties":{"x":{"type":"string"}}}"#
    let flag: RunFlag
    func run(argumentsJSON: String) async -> ToolResult {
        await flag.mark()
        return ToolResult(content: "RAN")
    }
}

private struct ApproveGate: PermissionGate {
    let approveAll: Bool
    func approve(toolName: String, argumentsJSON: String) async -> Bool { approveAll }
}

private actor RecordingGate: PermissionGate {
    private let answer: Bool
    private(set) var seenArgs: [String] = []
    init(answer: Bool) { self.answer = answer }
    func approve(toolName: String, argumentsJSON: String) async -> Bool {
        seenArgs.append(argumentsJSON)
        return answer
    }
}

private func hermesCall(_ name: String, _ argsJSON: String) -> String {
    "<tool_call>{\"name\": \"\(name)\", \"arguments\": \(argsJSON)}</tool_call>"
}

final class PermissionPolicyTests: XCTestCase {

    func testDenyListBeatsEverything() {
        let p = PermissionPolicy(mode: .acceptAll, allow: ["bash"], deny: ["bash"])
        // Deny wins over the allow list and over read-only status.
        if case .deny = p.decision(toolName: "bash", isReadOnly: false) {} else {
            XCTFail("deny list must win")
        }
        if case .deny = p.decision(toolName: "bash", isReadOnly: true) {} else {
            XCTFail("deny list must win even for a read-only tool")
        }
    }

    func testAllowListBeatsMode() {
        for mode in PermissionMode.allCases {
            let p = PermissionPolicy(mode: mode, allow: ["write_file"])
            XCTAssertEqual(
                p.decision(toolName: "write_file", isReadOnly: false), .allow,
                "allow list must override mode \(mode.rawValue)")
        }
    }

    func testReadOnlyAlwaysAllowedRegardlessOfMode() {
        for mode in PermissionMode.allCases {
            let p = PermissionPolicy(mode: mode)
            XCTAssertEqual(
                p.decision(toolName: "read_file", isReadOnly: true), .allow,
                "read-only tools must run in every mode (\(mode.rawValue))")
        }
    }

    func testAcceptAllAllowsMutating() {
        let p = PermissionPolicy(mode: .acceptAll)
        XCTAssertEqual(p.decision(toolName: "bash", isReadOnly: false), .allow)
    }

    func testAskDefersMutatingToGate() {
        let p = PermissionPolicy(mode: .ask)
        XCTAssertEqual(p.decision(toolName: "bash", isReadOnly: false), .ask)
    }

    func testPlanDeniesMutating() {
        let p = PermissionPolicy(mode: .plan)
        if case .deny = p.decision(toolName: "edit_file", isReadOnly: false) {} else {
            XCTFail("plan mode must deny mutating tools")
        }
    }

    func testAcceptEditsAllowsFileEditButAsksForCommand() {
        let p = PermissionPolicy(mode: .acceptEdits)
        // A file edit auto-applies...
        XCTAssertEqual(
            p.decision(toolName: "edit_file", isReadOnly: false, isFileEdit: true), .allow)
        // ...while a command (not a file edit) still defers to the gate.
        XCTAssertEqual(
            p.decision(toolName: "bash", isReadOnly: false, isFileEdit: false), .ask)
        // Read-only tools always run.
        XCTAssertEqual(
            p.decision(toolName: "read_file", isReadOnly: true), .allow)
    }
}

final class PermissionPostureTests: XCTestCase {

    func testParseAcceptsSynonyms() {
        XCTAssertEqual(PermissionMode.parse("auto"), .acceptAll)
        XCTAssertEqual(PermissionMode.parse("accept-all"), .acceptAll)
        XCTAssertEqual(PermissionMode.parse("accept-edits"), .acceptEdits)
        XCTAssertEqual(PermissionMode.parse("edits"), .acceptEdits)
        XCTAssertEqual(PermissionMode.parse("ask"), .ask)
        XCTAssertEqual(PermissionMode.parse("PLAN"), .plan)
        XCTAssertNil(PermissionMode.parse("nonsense"))
    }

    func testConfiguredPermissionDefaultFailsClosed() {
        XCTAssertEqual(PermissionMode.configuredDefault(nil), .plan)
        XCTAssertEqual(PermissionMode.configuredDefault(""), .plan)
        XCTAssertEqual(PermissionMode.configuredDefault("nonsense"), .plan)
        XCTAssertEqual(PermissionMode.configuredDefault("ask"), .ask)
        XCTAssertEqual(PermissionMode.configuredDefault("accept-edits"), .acceptEdits)
        XCTAssertEqual(PermissionMode.configuredDefault("auto"), .acceptAll)
    }

    func testShiftTabCycleOrderWraps() {
        XCTAssertEqual(PermissionMode.plan.next, .ask)
        XCTAssertEqual(PermissionMode.ask.next, .acceptEdits)
        XCTAssertEqual(PermissionMode.acceptEdits.next, .acceptAll)
        XCTAssertEqual(PermissionMode.acceptAll.next, .plan)
    }

    func testLabelsAreStable() {
        XCTAssertEqual(PermissionMode.acceptAll.label, "auto")
        XCTAssertEqual(PermissionMode.acceptEdits.label, "accept-edits")
        XCTAssertEqual(PermissionMode.ask.label, "ask")
        XCTAssertEqual(PermissionMode.plan.label, "plan")
    }
}

final class AgentLoopPermissionTests: XCTestCase {

    func testPlanModeDeniesMutatingToolAndNeverRunsIt() async {
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("edit_file", #"{"x":"y"}"#), "here is the plan"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "edit_file", isReadOnly: false, flag: flag)]),
            permission: PermissionPolicy(mode: .plan))
        let t = await loop.run(user: "change the file")

        let inv = t.steps[0].toolCalls[0]
        XCTAssertTrue(inv.result.isError)
        XCTAssertTrue(inv.result.content.contains("Permission denied"))
        let ran = await flag.ran
        XCTAssertFalse(ran, "a denied tool must never execute")
        // The loop keeps going and the model produces its plan.
        XCTAssertEqual(t.finalText, "here is the plan")
    }

    func testPlanModeAllowsReadOnlyTool() async {
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("read_file", #"{"x":"y"}"#), "done reading"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "read_file", isReadOnly: true, flag: flag)]),
            permission: PermissionPolicy(mode: .plan))
        let t = await loop.run(user: "read the file")

        let ran = await flag.ran
        XCTAssertTrue(ran, "read-only tools must run in plan mode")
        XCTAssertEqual(t.steps[0].toolCalls[0].result.content, "RAN")
    }

    func testAskModeRunsToolWhenGateApproves() async {
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("bash", #"{"x":"ls"}"#), "done"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "bash", isReadOnly: false, flag: flag)]),
            permission: PermissionPolicy(mode: .ask),
            gate: ApproveGate(approveAll: true))
        _ = await loop.run(user: "go")

        let ran = await flag.ran
        XCTAssertTrue(ran, "an approved tool must run")
    }

    func testAskModeDeniesToolWhenGateDeclines() async {
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("bash", #"{"x":"rm -rf"}"#), "ok, skipped"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "bash", isReadOnly: false, flag: flag)]),
            permission: PermissionPolicy(mode: .ask),
            gate: ApproveGate(approveAll: false))
        let t = await loop.run(user: "go")

        let ran = await flag.ran
        XCTAssertFalse(ran, "a declined tool must not run")
        XCTAssertTrue(t.steps[0].toolCalls[0].result.content.contains("Permission denied"))
    }

    func testAskModeWithNoGateDeniesByDefault() async {
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("bash", #"{"x":"ls"}"#), "done"])
        // mode .ask but gate == nil: fail-safe deny, never silently run.
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "bash", isReadOnly: false, flag: flag)]),
            permission: PermissionPolicy(mode: .ask))
        _ = await loop.run(user: "go")

        let ran = await flag.ran
        XCTAssertFalse(ran, "ask mode with no approver must deny, not run")
    }

    func testGateSeesTheArgsThatWillRun() async {
        // The gate must be shown the args the tool will actually run with, so
        // the user approves the real call (not an empty placeholder).
        let gate = RecordingGate(answer: true)
        let gen = ScriptGen([hermesCall("write_file", #"{"x":"hello"}"#), "done"])
        _ = await AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "write_file", isReadOnly: false, flag: RunFlag())]),
            permission: PermissionPolicy(mode: .ask),
            gate: gate).run(user: "go")

        let seen = await gate.seenArgs
        XCTAssertTrue(seen.contains(#"{"x":"hello"}"#), "gate must see the args that will run")
    }

    func testAcceptEditsRunsEditWithoutGateButAsksForBash() async {
        // A file edit auto-applies (no gate consulted); the bash command defers
        // to the gate, which declines here, so bash never runs.
        let editFlag = RunFlag(), bashFlag = RunFlag()
        let gen = ScriptGen([
            hermesCall("write_file", #"{"x":"a"}"#),
            hermesCall("bash", #"{"x":"rm -rf"}"#),
            "done",
        ])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([
                FlagTool(name: "write_file", isReadOnly: false, isFileEdit: true, flag: editFlag),
                FlagTool(name: "bash", isReadOnly: false, flag: bashFlag),
            ]),
            permission: PermissionPolicy(mode: .acceptEdits),
            gate: ApproveGate(approveAll: false))
        _ = await loop.run(user: "go")

        let edited = await editFlag.ran
        let bashed = await bashFlag.ran
        XCTAssertTrue(edited, "accept-edits must auto-apply a file edit")
        XCTAssertFalse(bashed, "accept-edits must still gate (and here deny) a command")
    }

    func testDenyListBlocksToolEvenInAcceptAll() async {
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("bash", #"{"x":"ls"}"#), "done"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "bash", isReadOnly: false, flag: flag)]),
            permission: PermissionPolicy(mode: .acceptAll, deny: ["bash"]))
        let t = await loop.run(user: "go")

        let ran = await flag.ran
        XCTAssertFalse(ran, "a deny-listed tool must not run even in accept-all")
        XCTAssertTrue(t.steps[0].toolCalls[0].result.content.contains("deny list"))
    }

    func testDefaultPolicyRunsToolsUnchanged() async {
        // No permission args: the default PermissionPolicy() is accept-all, so
        // mutating tools run exactly as before this PR.
        let flag = RunFlag()
        let gen = ScriptGen([hermesCall("bash", #"{"x":"ls"}"#), "done"])
        let loop = AgentLoop(
            generator: gen,
            tools: ToolRegistry([FlagTool(name: "bash", isReadOnly: false, flag: flag)]))
        _ = await loop.run(user: "go")

        let ran = await flag.ran
        XCTAssertTrue(ran, "default policy must preserve the autonomous run")
    }
}
