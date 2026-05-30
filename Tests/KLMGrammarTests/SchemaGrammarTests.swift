import XCTest
@testable import KLMGrammar

/// Unit tests for the Stage B JSON-Schema grammar (compiler + automaton),
/// no MLX. Each test compiles a schema, feeds candidate output strings
/// through the automaton, and asserts accept-and-complete or reject.
final class SchemaGrammarTests: XCTestCase {

    /// Compile `schema`, run `s` through it; returns the final state or nil if
    /// any character was rejected.
    private func run(_ schema: String, _ s: String) -> SchemaGrammar.State? {
        guard let g = SchemaGrammar.compile(schema) else {
            XCTFail("schema did not compile: \(schema)")
            return nil
        }
        return g.advance(g.initialState, piece: s)
    }

    private func assertAccepts(_ schema: String, _ s: String,
                               file: StaticString = #filePath, line: UInt = #line) {
        guard let g = SchemaGrammar.compile(schema) else {
            return XCTFail("schema did not compile", file: file, line: line)
        }
        guard let st = g.advance(g.initialState, piece: s) else {
            return XCTFail("rejected valid-for-schema output: \(s)", file: file, line: line)
        }
        XCTAssertTrue(g.isComplete(st), "expected complete: \(s)", file: file, line: line)
    }

    private func assertRejects(_ schema: String, _ s: String,
                               file: StaticString = #filePath, line: UInt = #line) {
        guard let g = SchemaGrammar.compile(schema) else {
            return XCTFail("schema did not compile", file: file, line: line)
        }
        // Rejected = either some char refused, or the full string never reaches
        // a complete state (e.g. a required key never supplied).
        if let st = g.advance(g.initialState, piece: s) {
            XCTAssertFalse(g.isComplete(st),
                           "expected rejection/incompleteness: \(s)", file: file, line: line)
        }
    }

    // MARK: - Scalar types

    func testStringType() {
        let s = #"{"type":"string"}"#
        assertAccepts(s, #""hello""#)
        assertRejects(s, "42")
        assertRejects(s, "true")
    }

    func testIntegerRejectsFraction() {
        let s = #"{"type":"integer"}"#
        assertAccepts(s, "42")
        assertAccepts(s, "-7")
        assertRejects(s, "4.5")     // integer must not have a fractional part
        assertRejects(s, #""4""#)
    }

    func testNumberAcceptsFraction() {
        let s = #"{"type":"number"}"#
        assertAccepts(s, "3.14")
        assertAccepts(s, "-2.5e10")
        assertAccepts(s, "42")
    }

    func testBooleanAndNull() {
        assertAccepts(#"{"type":"boolean"}"#, "true")
        assertAccepts(#"{"type":"boolean"}"#, "false")
        assertRejects(#"{"type":"boolean"}"#, "null")
        assertAccepts(#"{"type":"null"}"#, "null")
        assertRejects(#"{"type":"null"}"#, "true")
    }

    // MARK: - enum / const

    func testEnum() {
        let s = #"{"enum":["red","green","blue"]}"#
        assertAccepts(s, #""red""#)
        assertAccepts(s, #""blue""#)
        assertRejects(s, #""purple""#)
        assertRejects(s, #""gre""#)        // prefix of green, not complete
    }

    func testConst() {
        let s = #"{"const":42}"#
        assertAccepts(s, "42")
        assertRejects(s, "43")
    }

    func testEnumMixedScalars() {
        let s = #"{"enum":[1,2,true,null]}"#
        assertAccepts(s, "1")
        assertAccepts(s, "true")
        assertAccepts(s, "null")
        assertRejects(s, "3")
    }

    // MARK: - Objects

    func testObjectRequiredKeys() {
        let s = #"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"]}"#
        assertAccepts(s, #"{"name":"Ada","age":36}"#)
        assertAccepts(s, #"{"age":36,"name":"Ada"}"#)   // order-independent
        assertRejects(s, #"{"name":"Ada"}"#)             // missing required 'age'
        assertRejects(s, "{}")                            // both required missing
    }

    func testObjectTypedValues() {
        let s = #"{"type":"object","properties":{"age":{"type":"integer"}},"required":["age"]}"#
        assertAccepts(s, #"{"age":36}"#)
        assertRejects(s, #"{"age":"old"}"#)   // value must be integer, not string
        assertRejects(s, #"{"age":3.5}"#)     // integer, not number
    }

    func testAdditionalPropertiesFalseRejectsUndeclaredKey() {
        let s = #"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}"#
        assertAccepts(s, #"{"name":"x"}"#)
        assertRejects(s, #"{"name":"x","extra":1}"#)  // 'extra' not declared
    }

    func testAdditionalPropertiesTrueAllowsExtra() {
        let s = #"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#
        // additionalProperties defaults to allowed.
        assertAccepts(s, #"{"name":"x","extra":[1,2,3]}"#)
    }

    func testAdditionalPropertiesSchemaConstrainsExtraValues() {
        let s = #"{"type":"object","properties":{},"additionalProperties":{"type":"integer"}}"#
        assertAccepts(s, #"{"a":1,"b":2}"#)
        assertRejects(s, #"{"a":"str"}"#)   // extra value must be integer
    }

    // MARK: - Arrays

    func testArrayOfTyped() {
        let s = #"{"type":"array","items":{"type":"integer"}}"#
        assertAccepts(s, "[1,2,3]")
        assertAccepts(s, "[]")
        assertRejects(s, #"[1,"two",3]"#)   // element must be integer
    }

    func testNestedObjectArray() {
        let s = #"{"type":"object","properties":{"tags":{"type":"array","items":{"type":"string"}}},"required":["tags"]}"#
        assertAccepts(s, #"{"tags":["a","b"]}"#)
        assertAccepts(s, #"{"tags":[]}"#)
        assertRejects(s, #"{"tags":[1]}"#)         // items must be strings
        assertRejects(s, #"{"tags":"a"}"#)          // tags must be an array
    }

    func testDeepNesting() {
        let s = #"{"type":"object","properties":{"user":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}},"required":["user"]}"#
        assertAccepts(s, #"{"user":{"id":7}}"#)
        assertRejects(s, #"{"user":{}}"#)           // inner required 'id' missing
        assertRejects(s, #"{"user":{"id":"x"}}"#)   // inner id must be integer
    }

    // MARK: - Whitespace tolerance

    func testWhitespaceInObject() {
        let s = #"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#
        assertAccepts(s, "{ \"a\" : 1 }")
    }

    // MARK: - Unsupported keywords relax to "any value"

    func testAnyOfRelaxesToAnyValidJSON() {
        // anyOf cannot be made deterministic; the compiler relaxes it to an
        // unconstrained value, so any well-formed JSON is accepted.
        let s = #"{"anyOf":[{"type":"string"},{"type":"integer"}]}"#
        assertAccepts(s, #""hello""#)
        assertAccepts(s, "42")
        assertAccepts(s, "[1,2]")    // relaxed: even a non-listed shape is ok
        assertRejects(s, "not json") // still must be valid JSON
    }

    func testEmptySchemaIsAnyJSON() {
        assertAccepts("{}", #"{"anything":[1,true,null]}"#)
        assertAccepts("{}", "42")
        assertRejects("{}", "}{")
    }

    // MARK: - Type inference when 'type' omitted

    func testInferObjectFromProperties() {
        let s = #"{"properties":{"a":{"type":"integer"}},"required":["a"]}"#
        assertAccepts(s, #"{"a":1}"#)
        assertRejects(s, #"{"a":"x"}"#)
    }
}
