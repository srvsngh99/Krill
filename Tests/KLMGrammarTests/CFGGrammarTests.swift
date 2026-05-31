import XCTest
@testable import KLMGrammar

/// Unit tests for the Stage D context-free grammar (compiler + Earley
/// recognizer), no MLX. Each test compiles a grammar, feeds candidate strings
/// through the automaton character-by-character, and asserts full-parse
/// acceptance or rejection. The start symbol must span the whole input, so
/// trailing junk and incomplete prefixes are non-matches at EOS.
///
/// The signature capability is unbounded balanced nesting - the thing a regular
/// grammar (Stage C) provably cannot do.
final class CFGGrammarTests: XCTestCase {

    private func accepts(_ grammar: String, _ s: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        guard let g = CFGGrammar.compile(grammar) else {
            return XCTFail("grammar did not compile", file: file, line: line)
        }
        guard let st = g.advance(g.initialState, piece: s) else {
            return XCTFail("rejected matching string: \(s)", file: file, line: line)
        }
        XCTAssertTrue(g.isComplete(st), "expected full parse: \(s)", file: file, line: line)
    }

    private func rejects(_ grammar: String, _ s: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        guard let g = CFGGrammar.compile(grammar) else {
            return XCTFail("grammar did not compile", file: file, line: line)
        }
        if let st = g.advance(g.initialState, piece: s) {
            XCTAssertFalse(g.isComplete(st), "expected non-match: \(s)", file: file, line: line)
        }
        // advance == nil is also a valid rejection.
    }

    /// A valid prefix that is accepted but not yet a complete parse.
    private func incomplete(_ grammar: String, _ s: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        guard let g = CFGGrammar.compile(grammar) else {
            return XCTFail("grammar did not compile", file: file, line: line)
        }
        guard let st = g.advance(g.initialState, piece: s) else {
            return XCTFail("rejected valid prefix: \(s)", file: file, line: line)
        }
        XCTAssertFalse(g.isComplete(st), "expected incomplete: \(s)", file: file, line: line)
    }

    // MARK: - Basics

    func testLiteralRule() {
        let g = #"start: "abc""#
        accepts(g, "abc")
        rejects(g, "abd")
        rejects(g, "ab")      // incomplete at EOS
        rejects(g, "abcd")    // trailing junk
        incomplete(g, "ab")
    }

    func testAlternation() {
        let g = #"start: "cat" | "dog" | "fish""#
        accepts(g, "cat")
        accepts(g, "dog")
        accepts(g, "fish")
        rejects(g, "bird")
        incomplete(g, "ca")
        incomplete(g, "fi")
    }

    func testSequenceOfRefs() {
        let g = """
        start: greeting name
        greeting: "hi "
        name: "bob" | "alice"
        """
        accepts(g, "hi bob")
        accepts(g, "hi alice")
        rejects(g, "hi carol")
        incomplete(g, "hi ")
    }

    func testCharClassAndQuantifier() {
        // A number is one or more digits - expressed grammatically.
        let g = "start: [0-9]+"
        accepts(g, "0")
        accepts(g, "42")
        accepts(g, "1000000")
        rejects(g, "")
        rejects(g, "12a")
        incomplete(g, "")
    }

    func testOptionalAndStar() {
        let g = #"start: "a"? "b"*"#
        accepts(g, "")          // both empty
        accepts(g, "a")
        accepts(g, "b")
        accepts(g, "ab")
        accepts(g, "abbbb")
        accepts(g, "bbb")
        rejects(g, "aa")        // at most one 'a'
        rejects(g, "ba")        // 'a' cannot follow 'b'
    }

    // MARK: - The signature test: unbounded balanced nesting

    func testBalancedParens() {
        // start: item*   item: "(" item* ")"
        let g = """
        start: item*
        item: "(" item* ")"
        """
        accepts(g, "")
        accepts(g, "()")
        accepts(g, "()()")
        accepts(g, "(())")
        accepts(g, "((()))")
        accepts(g, "(()())")
        accepts(g, "(()(()))()")

        rejects(g, "(")         // unbalanced - incomplete
        rejects(g, "(()")       // unbalanced - incomplete
        rejects(g, "())")       // too many closes
        rejects(g, ")(")        // close before open

        incomplete(g, "(")
        incomplete(g, "((")
        incomplete(g, "(()")
    }

    func testDeepNesting() {
        let g = """
        start: item
        item: "(" item ")" | "x"
        """
        accepts(g, "x")
        accepts(g, "(x)")
        accepts(g, "((((((((((x))))))))))")
        rejects(g, "((((((((((x)))))))))")   // one missing close
        incomplete(g, "((((((((((x")
    }

    func testNestedArithmetic() {
        // Classic expression grammar: precedence + recursion + parentheses.
        let g = """
        start: expr
        expr: term (("+" | "-") term)*
        term: factor (("*" | "/") factor)*
        factor: [0-9]+ | "(" expr ")"
        """
        accepts(g, "1")
        accepts(g, "1+2")
        accepts(g, "1+2*3")
        accepts(g, "(1+2)*3")
        accepts(g, "((1+2)*(3-4))/5")
        accepts(g, "12*34+56")

        rejects(g, "1+")        // dangling operator
        rejects(g, "(1+2")      // unbalanced
        rejects(g, "1++2")      // double operator
        rejects(g, "*3")        // leading operator
        incomplete(g, "1+")
        incomplete(g, "(1+2)*")
    }

    // MARK: - Left recursion, ambiguity, nullable

    func testLeftRecursion() {
        // a: a "x" | "x"  - left-recursive; Earley handles it.
        let g = #"a: a "x" | "x""#   // no `start`; first rule is the start
        accepts(g, "x")
        accepts(g, "xx")
        accepts(g, "xxxxx")
        rejects(g, "")
        rejects(g, "xy")
    }

    func testAmbiguousGrammar() {
        // a: a a | "x"  - highly ambiguous, but acceptance is unaffected.
        let g = #"a: a a | "x""#
        accepts(g, "x")
        accepts(g, "xx")
        accepts(g, "xxx")
        rejects(g, "")
    }

    func testNullableRule() {
        // opt is nullable; start may be empty or "a".
        let g = """
        start: opt "a"
        opt: "z"?
        """
        accepts(g, "a")
        accepts(g, "za")
        rejects(g, "")
        rejects(g, "z")
        incomplete(g, "z")
    }

    func testStartRuleSelection() {
        // `start` is chosen even when it is not the first rule defined.
        let g = """
        helper: "nope"
        start: "yes"
        """
        accepts(g, "yes")
        rejects(g, "nope")
    }

    // MARK: - Completeness boundaries

    func testNoPrematureCompletion() {
        let g = #"start: "ab" "cd""#
        XCTAssertNotNil(CFGGrammar.compile(g))
        incomplete(g, "ab")
        incomplete(g, "abc")
        accepts(g, "abcd")
    }

    func testComments() {
        let g = """
        // a greeting grammar
        start: "hi"   // trailing comment
        """
        accepts(g, "hi")
        rejects(g, "bye")
    }

    // MARK: - Compile-failure fallbacks (must return nil → engine fails open)

    func testCompileFailures() {
        XCTAssertNil(CFGGrammar.compile(""))                     // empty grammar
        XCTAssertNil(CFGGrammar.compile("   \n  // only comment\n"))
        XCTAssertNil(CFGGrammar.compile("start: missing"))       // undefined nonterminal
        XCTAssertNil(CFGGrammar.compile("start: (\"a\""))        // unbalanced group
        XCTAssertNil(CFGGrammar.compile("start \"a\""))          // no colon
        XCTAssertNil(CFGGrammar.compile("start: \"a\"\nstart: \"b\""))  // duplicate rule
        XCTAssertNil(CFGGrammar.compile("start: [a"))            // unterminated class
        XCTAssertNil(CFGGrammar.compile("start: \"unterminated"))// unterminated string
        XCTAssertNil(CFGGrammar.compile("start: *"))             // bare quantifier
    }

    func testUndefinedReferenceInDeepBody() {
        XCTAssertNil(CFGGrammar.compile("""
        start: a b
        a: "x"
        b: "y" c
        """))   // `c` is undefined
    }

    func testOversizedGrammarRejected() {
        // A rule with more alternatives than the production cap must fail to
        // compile (fail open), bounding per-token recognizer work. A grammar
        // just under the cap still compiles.
        func grammar(alternatives n: Int) -> String {
            "start: " + (0 ..< n).map { "\"a\($0)\"" }.joined(separator: " | ")
        }
        XCTAssertNotNil(CFGGrammar.compile(grammar(alternatives: 100)))
        XCTAssertNil(CFGGrammar.compile(grammar(alternatives: CFGGrammar.maxProductions + 5)))
    }
}
