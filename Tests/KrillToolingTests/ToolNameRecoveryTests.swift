import XCTest
@testable import KrillTooling

/// Tool-name canonicalization against the offered set: casing, mangled
/// prefixes, and (the gemma-4-12b-agentic slip observed in QA) CamelCase /
/// kebab-case variants of a snake_case tool.
final class ToolNameRecoveryTests: XCTestCase {

    private let known = ["write_file", "edit_file", "read_file", "bash"]

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
}
