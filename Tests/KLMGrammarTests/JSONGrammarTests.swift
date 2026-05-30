import XCTest
@testable import KLMGrammar

/// Unit tests for the character-level JSON automaton (no MLX). Each test
/// feeds a string through `JSONGrammar.advance` from the initial state and
/// asserts acceptance/rejection and completeness.
final class JSONGrammarTests: XCTestCase {

    /// Run `s` through the automaton; returns the final state or nil if any
    /// character was rejected.
    private func run(_ s: String) -> JSONGrammar.State? {
        JSONGrammar.advance(JSONGrammar.initialState, piece: s)
    }

    /// A string that should be accepted AND be a complete JSON value.
    private func assertComplete(_ s: String, file: StaticString = #filePath, line: UInt = #line) {
        guard let st = run(s) else {
            return XCTFail("rejected valid JSON: \(s)", file: file, line: line)
        }
        XCTAssertTrue(JSONGrammar.isComplete(st),
                      "expected complete: \(s)", file: file, line: line)
    }

    /// A string the automaton must reject outright.
    private func assertRejected(_ s: String, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertNil(run(s), "expected rejection: \(s)", file: file, line: line)
    }

    /// A valid prefix that is accepted but NOT yet a complete value.
    private func assertIncomplete(_ s: String, file: StaticString = #filePath, line: UInt = #line) {
        guard let st = run(s) else {
            return XCTFail("rejected valid prefix: \(s)", file: file, line: line)
        }
        XCTAssertFalse(JSONGrammar.isComplete(st),
                       "expected incomplete: \(s)", file: file, line: line)
    }

    // MARK: - Primitive values

    func testLiterals() {
        assertComplete("true")
        assertComplete("false")
        assertComplete("null")
        assertRejected("tru e")
        assertRejected("nulls")  // 'null' completes, then 's' is trailing garbage
    }

    func testLiteralIncompletePrefix() {
        assertIncomplete("tru")
        assertIncomplete("fal")
        assertIncomplete("nul")
    }

    func testNumbers() {
        for n in ["0", "-0", "42", "-42", "3.14", "-3.14", "1e10", "1E10",
                  "1e+10", "1e-10", "0.5", "123.456e-7"] {
            assertComplete(n)
        }
    }

    func testInvalidNumbers() {
        assertRejected("01")     // leading zero
        assertRejected("1..2")
        assertRejected("1ee2")
        assertRejected(".5")     // must start with a digit or '-'
        assertRejected("+5")
    }

    func testNumberIncomplete() {
        assertIncomplete("-")
        assertIncomplete("1.")
        assertIncomplete("1e")
        assertIncomplete("1e+")
    }

    func testStrings() {
        assertComplete("\"\"")
        assertComplete("\"hello\"")
        assertComplete("\"with \\\"escaped\\\" quotes\"")
        assertComplete("\"unicode \\u00e9\"")
        assertComplete("\"tab\\tnewline\\n\"")
    }

    func testStringIncompleteAndInvalid() {
        assertIncomplete("\"unterminated")
        assertRejected("\"bad \\x escape\"")
        assertRejected("\"\\u00zz\"")           // non-hex in \u
        // A raw newline (control char) inside a string is invalid.
        assertRejected("\"line\nbreak\"")
    }

    // MARK: - Containers

    func testObjects() {
        assertComplete("{}")
        assertComplete("{\"a\":1}")
        assertComplete("{\"a\":1,\"b\":2}")
        assertComplete("{\"nested\":{\"x\":[1,2,3]}}")
        assertComplete("{ \"a\" : 1 , \"b\" : true }")  // whitespace tolerant
    }

    func testInvalidObjects() {
        assertRejected("{,}")
        assertRejected("{\"a\":1,}")      // trailing comma
        assertRejected("{\"a\"}")          // missing colon+value
        assertRejected("{\"a\":}")         // missing value
        assertRejected("{1:2}")            // non-string key
        assertRejected("{\"a\":1 \"b\":2}") // missing comma
    }

    func testArrays() {
        assertComplete("[]")
        assertComplete("[1]")
        assertComplete("[1,2,3]")
        assertComplete("[\"a\",true,null,3.14]")
        assertComplete("[[1],[2,3],[]]")
        assertComplete("[ 1 , 2 ]")
    }

    func testInvalidArrays() {
        assertRejected("[,]")
        assertRejected("[1,]")             // trailing comma
        assertRejected("[1 2]")            // missing comma
        assertRejected("[1,,2]")
    }

    func testObjectIncomplete() {
        assertIncomplete("{")
        assertIncomplete("{\"a\"")
        assertIncomplete("{\"a\":")
        assertIncomplete("{\"a\":1,")
        assertIncomplete("[1,")
        assertIncomplete("[")
    }

    // MARK: - Trailing content

    func testTrailingWhitespaceAllowed() {
        assertComplete("{}  ")
        assertComplete("42\n")
    }

    func testTrailingGarbageRejected() {
        assertRejected("{}x")
        assertRejected("truefalse")
        assertRejected("[]{}")
    }

    // MARK: - Step-by-step state tracking

    func testIncrementalAdvance() {
        // EOS gating proxy: isComplete must be false until the value closes.
        var st = JSONGrammar.initialState
        XCTAssertFalse(JSONGrammar.isComplete(st))  // empty: nothing emitted
        for (i, ch) in Array("{\"k\":1}").enumerated() {
            guard let next = JSONGrammar.step(st, ch) else {
                return XCTFail("rejected char \(ch) at \(i)")
            }
            st = next
        }
        XCTAssertTrue(JSONGrammar.isComplete(st))
    }
}
