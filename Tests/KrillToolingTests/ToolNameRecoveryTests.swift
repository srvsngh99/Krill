import XCTest
@testable import KrillTooling

/// Tool-name canonicalization against the offered set: casing, mangled
/// prefixes, and (the gemma-4-12b-agentic slip observed in QA) CamelCase /
/// kebab-case variants of a snake_case tool.
final class ToolNameRecoveryTests: XCTestCase {

    private let known = [
        "write_file", "edit_file", "read_file", "bash", "ask_user", "request_execute",
    ]

    private func canon(_ name: String) -> String {
        let calls = [ToolCalling.ParsedToolCall(name: name, argumentsJSON: "{}")]
        return ToolCalling.canonicalizeNames(calls, known: known)[0].name
    }

    func testExactNameUntouched() {
        XCTAssertEqual(canon("write_file"), "write_file")
    }

    func testCaseSlipRecovered() {
        XCTAssertEqual(canon("Write_File"), "write_file")
    }

    func testCamelCaseSlipRecovered() {
        // gemma-4-12b-agentic emitted `WriteFile` for `write_file`.
        XCTAssertEqual(canon("WriteFile"), "write_file")
    }

    func testKebabCaseSlipRecovered() {
        XCTAssertEqual(canon("write-file"), "write_file")
    }

    func testHallucinatedNameNotRecovered() {
        // Never resolve to a tool that was not offered.
        XCTAssertEqual(canon("delete_everything"), "delete_everything")
    }

    func testAmbiguousSquashMatchNotRecovered() {
        // Two offered tools that squash identically: the separator-
        // insensitive match is ambiguous, so the name must stay
        // unresolved rather than silently dispatching either one.
        let ambiguous = ["write_file", "wri_tefile", "bash"]
        let calls = [ToolCalling.ParsedToolCall(name: "Write-File", argumentsJSON: "{}")]
        XCTAssertEqual(
            ToolCalling.canonicalizeNames(calls, known: ambiguous)[0].name,
            "Write-File")
    }

    // MARK: - Claude-Code-style vocabulary

    // `gemma-4-12b-agentic` emits `Read` for `read_file`. That is a different
    // WORD, not a casing or separator slip, so none of the rules above can
    // bridge it and the agent died on its first tool call with
    // "unknown tool 'Read'".
    func testClaudeStyleReadAliased() {
        XCTAssertEqual(canon("Read"), "read_file")
    }

    func testClaudeStyleEditAliased() {
        XCTAssertEqual(canon("Edit"), "edit_file")
    }

    func testClaudeStyleWriteAliased() {
        XCTAssertEqual(canon("Write"), "write_file")
    }

    func testShellAliasesToBash() {
        XCTAssertEqual(canon("Shell"), "bash")
    }

    func testQuestionAliasesToAskUser() {
        for alias in ["question", "askuserquestion", "ask"] {
            XCTAssertEqual(canon(alias), "ask_user")
        }
    }

    func testPlanExitAliasesToRequestExecute() {
        for alias in ["exitplanmode", "plan_exit", "exitplan", "approve_plan", "start_implementing"] {
            XCTAssertEqual(canon(alias), "request_execute")
        }
    }

    func testAliasNeverInventsAnUnofferedTool() {
        // `read_file` is NOT offered here, so `Read` must stay unresolved
        // rather than resolving to a tool the caller never exposed.
        let calls = [ToolCalling.ParsedToolCall(name: "Read", argumentsJSON: "{}")]
        XCTAssertEqual(
            ToolCalling.canonicalizeNames(calls, known: ["bash"])[0].name,
            "Read")
    }
}
