import XCTest
@testable import KLMServer

/// Unit tests for the Stage D CFG format wiring at the server boundary:
/// `ResponseFormat` parsing (OpenAI + Ollama), `StructuredOutput.engineFormat`,
/// `systemPrompt`, and `coerce` behavior for `.cfg`. MLX-free (no engine /
/// model load), so they cover the user-facing entry point for CFG-constrained
/// decoding directly.
final class CFGFormatWiringTests: XCTestCase {

    private let sampleGrammar = "start: item*\nitem: \"(\" item* \")\""

    // MARK: - OpenAI response_format parsing

    func testOpenAICfgType() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "cfg", "cfg": sampleGrammar])
        XCTAssertEqual(f, .cfg(sampleGrammar))
    }

    func testOpenAILarkTypeAlias() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "lark", "lark": sampleGrammar])
        XCTAssertEqual(f, .cfg(sampleGrammar))
    }

    func testOpenAICfgUnderGrammarKey() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "lark", "grammar": sampleGrammar])
        XCTAssertEqual(f, .cfg(sampleGrammar))
    }

    func testOpenAINestedCfgGrammar() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "lark", "lark": ["grammar": sampleGrammar]])
        XCTAssertEqual(f, .cfg(sampleGrammar))
    }

    func testOpenAICfgTypeMissingGrammarIsNil() {
        XCTAssertNil(ServerParsing.parseOpenAIResponseFormat(["type": "cfg"]))
    }

    func testOpenAIGrammarTypeStillAliasesToRegex() {
        // `type:"grammar"` remains the Stage C regex alias, unchanged.
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "grammar", "grammar": "yes|no"])
        XCTAssertEqual(f, .regex("yes|no"))
    }

    // MARK: - Ollama format parsing

    func testOllamaBareCfgObject() {
        let f = ServerParsing.parseOllamaFormat(["cfg": sampleGrammar])
        XCTAssertEqual(f, .cfg(sampleGrammar))
    }

    func testOllamaBareLarkObject() {
        let f = ServerParsing.parseOllamaFormat(["lark": sampleGrammar])
        XCTAssertEqual(f, .cfg(sampleGrammar))
    }

    func testOllamaSchemaWithCfgPropertyIsNotShadowed() {
        // A real JSON schema declaring a property named "cfg" must parse as a
        // schema, not a CFG request (more than one key).
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["cfg": ["type": "string"]],
        ]
        let f = ServerParsing.parseOllamaFormat(schema)
        if case .schema = f { /* ok */ } else {
            XCTFail("expected .schema, got \(String(describing: f))")
        }
    }

    func testOllamaCfgObjectWithExtraKeyIsSchema() {
        let f = ServerParsing.parseOllamaFormat(["cfg": "start: \"x\"", "extra": "y"])
        if case .schema = f { /* ok */ } else {
            XCTFail("expected .schema, got \(String(describing: f))")
        }
    }

    // MARK: - coerce: CFG output must not be JSON-extracted

    func testCoerceCfgReturnsVerbatim() {
        // A balanced-brace output must survive intact (JSON extraction would
        // corrupt it).
        let out = "{ a { b } }"
        XCTAssertEqual(StructuredOutput.coerce(out, format: .cfg(sampleGrammar)), out)
    }

    func testCoerceCfgPlainVerbatim() {
        XCTAssertEqual(StructuredOutput.coerce("(()())", format: .cfg(sampleGrammar)), "(()())")
    }

    // MARK: - engineFormat mapping

    func testEngineFormatMapsCfg() {
        let ef = StructuredOutput.engineFormat(for: .cfg(sampleGrammar))
        if case .cfg(let g) = ef { XCTAssertEqual(g, sampleGrammar) }
        else { XCTFail("expected .cfg, got \(String(describing: ef))") }
    }

    // MARK: - systemPrompt

    func testSystemPromptForCfgMentionsGrammar() {
        let p = StructuredOutput.systemPrompt(for: .cfg(sampleGrammar))
        XCTAssertTrue(p.contains(sampleGrammar))
        XCTAssertTrue(p.lowercased().contains("context-free grammar"))
    }
}
