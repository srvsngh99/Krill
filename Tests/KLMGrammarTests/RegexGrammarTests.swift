import XCTest
@testable import KLMGrammar

/// Unit tests for the Stage C regex grammar (compiler + NFA automaton),
/// no MLX. Each test compiles a pattern, feeds candidate strings through the
/// automaton, and asserts full-match acceptance or rejection. The pattern is
/// matched as a FULL match, so trailing junk must be rejected.
final class RegexGrammarTests: XCTestCase {

    private func accepts(_ pattern: String, _ s: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        guard let g = RegexGrammar.compile(pattern) else {
            return XCTFail("pattern did not compile: \(pattern)", file: file, line: line)
        }
        guard let st = g.advance(g.initialState, piece: s) else {
            return XCTFail("rejected matching string: /\(pattern)/ vs \(s)", file: file, line: line)
        }
        XCTAssertTrue(g.isComplete(st),
                      "expected full match: /\(pattern)/ vs \(s)", file: file, line: line)
    }

    private func rejects(_ pattern: String, _ s: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        guard let g = RegexGrammar.compile(pattern) else {
            return XCTFail("pattern did not compile: \(pattern)", file: file, line: line)
        }
        if let st = g.advance(g.initialState, piece: s) {
            XCTAssertFalse(g.isComplete(st),
                           "expected non-match: /\(pattern)/ vs \(s)", file: file, line: line)
        }
        // advance == nil is also a valid rejection.
    }

    /// A valid prefix that is accepted but not yet a complete match.
    private func incomplete(_ pattern: String, _ s: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        guard let g = RegexGrammar.compile(pattern) else {
            return XCTFail("pattern did not compile: \(pattern)", file: file, line: line)
        }
        guard let st = g.advance(g.initialState, piece: s) else {
            return XCTFail("rejected valid prefix: /\(pattern)/ vs \(s)", file: file, line: line)
        }
        XCTAssertFalse(g.isComplete(st),
                       "expected incomplete: /\(pattern)/ vs \(s)", file: file, line: line)
    }

    // MARK: - Literals

    func testLiteral() {
        accepts("abc", "abc")
        rejects("abc", "abd")
        rejects("abc", "ab")     // incomplete is a non-match at EOS
        rejects("abc", "abcd")   // full match: no trailing
        incomplete("abc", "ab")
    }

    func testEmptyPattern() {
        accepts("", "")
        rejects("", "x")
    }

    // MARK: - Dot and escapes

    func testDot() {
        accepts(".", "x")
        accepts("a.c", "axc")
        rejects("a.c", "a\nc")   // '.' excludes newline
        rejects(".", "")          // dot requires one char
    }

    func testEscapedMetacharacters() {
        accepts(#"a\.b"#, "a.b")
        rejects(#"a\.b"#, "axb")
        accepts(#"\(\)"#, "()")
        accepts(#"\d\+"#, "5+")
    }

    func testClassEscapes() {
        accepts(#"\d"#, "7")
        rejects(#"\d"#, "a")
        accepts(#"\w"#, "_")
        accepts(#"\w"#, "Q")
        rejects(#"\w"#, "-")
        accepts(#"\s"#, " ")
        rejects(#"\s"#, "x")
        accepts(#"\D"#, "a")
        rejects(#"\D"#, "3")
    }

    // MARK: - Character classes

    func testCharClass() {
        accepts("[abc]", "b")
        rejects("[abc]", "d")
        accepts("[a-z]", "m")
        rejects("[a-z]", "M")
        accepts("[A-Za-z0-9]", "Q")
        accepts("[A-Za-z0-9]", "7")
    }

    func testNegatedClass() {
        accepts("[^0-9]", "a")
        rejects("[^0-9]", "5")
    }

    func testClassWithEscapes() {
        accepts(#"[\d.]"#, "3")
        accepts(#"[\d.]"#, ".")
        rejects(#"[\d.]"#, "a")
    }

    // MARK: - Alternation

    func testAlternation() {
        accepts("yes|no", "yes")
        accepts("yes|no", "no")
        rejects("yes|no", "maybe")
        rejects("yes|no", "ye")
    }

    func testAlternationWithGroup() {
        accepts("(cat|dog)s", "cats")
        accepts("(cat|dog)s", "dogs")
        rejects("(cat|dog)s", "cat")
        rejects("(cat|dog)s", "fishs")
    }

    // MARK: - Quantifiers

    func testStar() {
        accepts("a*", "")
        accepts("a*", "aaaa")
        rejects("a*", "aab")
        accepts("ab*c", "ac")
        accepts("ab*c", "abbbc")
    }

    func testPlus() {
        rejects("a+", "")
        accepts("a+", "a")
        accepts("a+", "aaa")
    }

    func testOptional() {
        accepts("ab?c", "ac")
        accepts("ab?c", "abc")
        rejects("ab?c", "abbc")
    }

    func testCounted() {
        accepts(#"\d{3}"#, "123")
        rejects(#"\d{3}"#, "12")
        rejects(#"\d{3}"#, "1234")
        accepts("a{2,4}", "aa")
        accepts("a{2,4}", "aaaa")
        rejects("a{2,4}", "a")
        rejects("a{2,4}", "aaaaa")
        accepts("a{2,}", "aa")
        accepts("a{2,}", "aaaaaa")
        rejects("a{2,}", "a")
    }

    // MARK: - Realistic patterns

    func testPhoneNumber() {
        let p = #"\d{3}-\d{3}-\d{4}"#
        accepts(p, "415-555-1234")
        rejects(p, "415-555-123")    // last group too short
        rejects(p, "abc-555-1234")   // letters
        incomplete(p, "415-")
        incomplete(p, "415-555-")
    }

    func testISODate() {
        let p = #"\d{4}-\d{2}-\d{2}"#
        accepts(p, "2026-05-30")
        rejects(p, "2026-5-30")
        incomplete(p, "2026-05")
    }

    func testYesNoMaybe() {
        let p = "(yes|no|maybe)"
        accepts(p, "yes")
        accepts(p, "maybe")
        rejects(p, "nope")
    }

    // MARK: - Compile failures fall back (return nil)

    func testInvalidPatternsReturnNil() {
        XCTAssertNil(RegexGrammar.compile("("))      // unbalanced group
        XCTAssertNil(RegexGrammar.compile("a)"))     // stray close
        XCTAssertNil(RegexGrammar.compile("[a"))     // unterminated class
        XCTAssertNil(RegexGrammar.compile("*a"))     // quantifier with no atom
        XCTAssertNil(RegexGrammar.compile(#"\q"#))   // unknown escape
        XCTAssertNil(RegexGrammar.compile("a{2,1}")) // hi < lo
    }

    func testCountedOnGroupUnsupported() {
        // Counted repetition on a group is unsupported -> compile returns nil
        // (caller falls back to unconstrained).
        XCTAssertNil(RegexGrammar.compile("(ab){2}"))
    }

    /// Patterns with epsilon-heavy / nested-optional structure must compile
    /// and run without indexing an NFA node out of range (the recognizer is
    /// fed arbitrary user patterns). These exercise the epsilon-closure on
    /// split nodes built by `?`, `*`, alternation, and empty branches.
    func testEpsilonHeavyPatternsDoNotCrash() {
        accepts("(a?)b", "ab")
        accepts("(a?)b", "b")
        accepts("a?|b", "a")
        accepts("a?|b", "b")
        accepts("a?|b", "")        // empty matches the optional-a branch
        accepts("(a|)b", "ab")
        accepts("(a|)b", "b")
        accepts("(ab)?c", "abc")
        accepts("(ab)?c", "c")
        accepts("x?y?z?", "")
        accepts("x?y?z?", "xyz")
        accepts("x?y?z?", "xz")
        accepts("a*", "")
    }
}
