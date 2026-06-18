import XCTest
@testable import KrillServer

/// Unit tests for the Stage C regex format wiring at the server boundary:
/// `ResponseFormat` parsing (OpenAI + Ollama) and `StructuredOutput.coerce`
/// behavior for `.regex`. These are MLX-free (no engine / model load), so they
/// cover the user-facing entry point for regex-constrained decoding directly.
final class RegexFormatWiringTests: XCTestCase {

    // MARK: - OpenAI response_format parsing

    func testOpenAIRegexType() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "regex", "regex": #"\d{3}-\d{4}"#])
        XCTAssertEqual(f, .regex(#"\d{3}-\d{4}"#))
    }

    func testOpenAIGrammarTypeAliasesToRegex() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "grammar", "grammar": "yes|no"])
        XCTAssertEqual(f, .regex("yes|no"))
    }

    func testOpenAINestedRegexPattern() {
        let f = ServerParsing.parseOpenAIResponseFormat(
            ["type": "regex", "regex": ["pattern": "[a-z]+"]])
        XCTAssertEqual(f, .regex("[a-z]+"))
    }

    func testOpenAIRegexTypeMissingPatternIsNil() {
        // type:regex with no pattern is not a usable constraint.
        XCTAssertNil(ServerParsing.parseOpenAIResponseFormat(["type": "regex"]))
    }

    func testOpenAIJsonStillParses() {
        XCTAssertEqual(
            ServerParsing.parseOpenAIResponseFormat(["type": "json_object"]), .json)
    }

    func testOpenAIJsonSchemaStillParses() {
        let f = ServerParsing.parseOpenAIResponseFormat([
            "type": "json_schema",
            "json_schema": ["schema": ["type": "object"]],
        ])
        if case .schema = f { /* ok */ } else {
            XCTFail("expected .schema, got \(String(describing: f))")
        }
    }

    // MARK: - Ollama format parsing

    func testOllamaBareRegexObject() {
        let f = ServerParsing.parseOllamaFormat(["regex": #"[A-Z]{2}\d{4}"#])
        XCTAssertEqual(f, .regex(#"[A-Z]{2}\d{4}"#))
    }

    func testOllamaJsonStringStillParses() {
        XCTAssertEqual(ServerParsing.parseOllamaFormat("json"), .json)
    }

    func testOllamaSchemaWithRegexPropertyIsNotShadowed() {
        // A real JSON schema that happens to declare a property named "regex"
        // must parse as a schema, NOT as a regex request (more than one key).
        let schema: [String: Any] = [
            "type": "object",
            "properties": ["regex": ["type": "string"]],
        ]
        let f = ServerParsing.parseOllamaFormat(schema)
        if case .schema = f { /* ok */ } else {
            XCTFail("expected .schema, got \(String(describing: f))")
        }
    }

    func testOllamaRegexObjectWithExtraKeyIsSchema() {
        // {"regex": "...", "flags": "..."} is not the bare single-key shape, so
        // it is treated as a schema rather than a regex request.
        let f = ServerParsing.parseOllamaFormat(["regex": "abc", "flags": "i"])
        if case .schema = f { /* ok */ } else {
            XCTFail("expected .schema, got \(String(describing: f))")
        }
    }

    // MARK: - coerce: regex output must not be JSON-extracted

    func testCoerceRegexReturnsVerbatim() {
        // A matching string that happens to contain braces must survive intact
        // (the JSON-extraction path would corrupt it).
        let out = "set { x = 1 }"
        XCTAssertEqual(StructuredOutput.coerce(out, format: .regex(".*")), out)
    }

    func testCoerceRegexPlainStringVerbatim() {
        XCTAssertEqual(StructuredOutput.coerce("415-555-1234", format: .regex(#"\d{3}-\d{3}-\d{4}"#)),
                       "415-555-1234")
    }

    func testCoerceJsonStillExtracts() {
        // Sanity: the JSON path still strips surrounding prose.
        let out = StructuredOutput.coerce("Here: {\"a\":1} done", format: .json)
        XCTAssertEqual(out, "{\"a\":1}")
    }

    func testCoerceNilFormatVerbatim() {
        XCTAssertEqual(StructuredOutput.coerce("anything {", format: nil), "anything {")
    }

    // MARK: - engineFormat mapping

    func testEngineFormatMapsRegex() {
        let ef = StructuredOutput.engineFormat(for: .regex("x+"))
        if case .regex(let p) = ef { XCTAssertEqual(p, "x+") }
        else { XCTFail("expected .regex, got \(String(describing: ef))") }
    }

    func testSystemPromptForRegexMentionsPattern() {
        let p = StructuredOutput.systemPrompt(for: .regex(#"\d{4}"#))
        XCTAssertTrue(p.contains(#"\d{4}"#))
        XCTAssertTrue(p.lowercased().contains("regular expression"))
    }
}
