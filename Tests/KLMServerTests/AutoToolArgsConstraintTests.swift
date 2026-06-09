import XCTest
import KLMGrammar
@testable import KLMServer

/// Auto tool-call argument constraining (two-pass). Verifies the gate that
/// decides whether a re-generation is needed, the bare-args parser, and that
/// a tool's parameter schema compiles in the grammar (so pass 2 can constrain
/// decoding to it).
final class AutoToolArgsConstraintTests: XCTestCase {

    // codex's shell tool shape: an object with a required `cmd` array.
    private let shell = ServerToolSpec(
        name: "shell", description: "Run a shell command",
        parametersJSON: #"{"type":"object","properties":{"cmd":{"type":"array","items":{"type":"string"}}},"required":["cmd"]}"#)
    private let noReq = ServerToolSpec(
        name: "ping", description: "No required args",
        parametersJSON: #"{"type":"object","properties":{"host":{"type":"string"}}}"#)

    // MARK: - The gate (argsSatisfySchema)

    func testEmptyArgsMissingRequiredFails() {
        // This is the exact codex failure: shell called with {} (no cmd).
        XCTAssertFalse(ToolCalling.argsSatisfySchema(
            argumentsJSON: "{}", parametersJSON: shell.parametersJSON))
    }

    func testArgsWithRequiredKeyPasses() {
        XCTAssertTrue(ToolCalling.argsSatisfySchema(
            argumentsJSON: #"{"cmd":["python3","test_stats.py"]}"#,
            parametersJSON: shell.parametersJSON))
    }

    func testPartialRequiredFails() {
        let twoReq = #"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a","b"]}"#
        XCTAssertFalse(ToolCalling.argsSatisfySchema(
            argumentsJSON: #"{"a":"x"}"#, parametersJSON: twoReq))
    }

    func testNoRequiredKeysAlwaysPasses() {
        // A tool with no required args: empty {} is valid, no re-generation.
        XCTAssertTrue(ToolCalling.argsSatisfySchema(
            argumentsJSON: "{}", parametersJSON: noReq.parametersJSON))
    }

    func testMalformedArgsFails() {
        XCTAssertFalse(ToolCalling.argsSatisfySchema(
            argumentsJSON: "not json", parametersJSON: shell.parametersJSON))
        // A non-object (array) is not a valid arguments object.
        XCTAssertFalse(ToolCalling.argsSatisfySchema(
            argumentsJSON: "[1,2]", parametersJSON: shell.parametersJSON))
    }

    // MARK: - Pass-2 plumbing

    func testToolSchemaCompilesForPass2() {
        // Pass 2 constrains decoding to the tool's own parameter schema.
        XCTAssertNotNil(SchemaGrammar.compile(shell.parametersJSON),
            "the tool parameter schema must compile so pass 2 can constrain args")
    }

    func testParseArgsObjectBareAndFenced() {
        XCTAssertEqual(
            parseDict(ToolCalling.parseArgsObject(#"{"cmd":["ls"]}"#))["cmd"] as? [String], ["ls"])
        // Tolerates a code fence + surrounding whitespace.
        let fenced = "\n```json\n{\"cmd\": [\"ls\", \"-la\"]}\n```\n"
        XCTAssertEqual(
            parseDict(ToolCalling.parseArgsObject(fenced))["cmd"] as? [String], ["ls", "-la"])
        XCTAssertNil(ToolCalling.parseArgsObject("no object here"))
    }

    func testArgsRegenPromptNamesToolAndSchema() {
        let p = ToolCalling.argsRegenPrompt(tool: shell)
        XCTAssertTrue(p.contains("shell"))
        XCTAssertTrue(p.contains("cmd"))
    }

    private func parseDict(_ s: String?) -> [String: Any] {
        guard let s, let d = s.data(using: .utf8) else { return [:] }
        return (try? JSONSerialization.jsonObject(with: d)) as? [String: Any] ?? [:]
    }
}
