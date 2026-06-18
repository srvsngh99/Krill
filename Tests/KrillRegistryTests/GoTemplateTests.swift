import XCTest
@testable import KrillRegistry

/// Tests for the Go `text/template` engine that renders Ollama Modelfile
/// `TEMPLATE` overrides. Covers the operand/pipeline/control-flow surface
/// real Modelfiles use, whitespace trimming, the builtin functions, the
/// `OllamaTemplateContext` bridge, and the parse/eval failure modes the
/// engine signals so the caller can fall back to the built-in template.
final class GoTemplateTests: XCTestCase {

    private func render(_ src: String, _ v: GoValue) throws -> String {
        try GoTemplate.render(src, v)
    }

    // MARK: - Literals + field access

    func testPlainTextPassesThrough() throws {
        XCTAssertEqual(try render("hello world", .null), "hello world")
    }

    func testDotString() throws {
        XCTAssertEqual(try render("{{ . }}", .string("hi")), "hi")
    }

    func testFieldAccess() throws {
        let ctx = GoValue.dict(["System": .string("be nice"), "Prompt": .string("hi")])
        XCTAssertEqual(try render("{{ .System }}|{{ .Prompt }}", ctx), "be nice|hi")
    }

    func testNestedFieldAccessAndMissingKeyIsEmpty() throws {
        let ctx = GoValue.dict(["A": .dict(["B": .string("x")])])
        XCTAssertEqual(try render("{{ .A.B }}", ctx), "x")
        XCTAssertEqual(try render("{{ .A.Missing }}", ctx), "")
        XCTAssertEqual(try render("{{ .Nope }}", ctx), "")
    }

    // MARK: - if / else

    func testIfTruthy() throws {
        let ctx = GoValue.dict(["System": .string("S")])
        XCTAssertEqual(try render("{{ if .System }}[{{ .System }}]{{ end }}", ctx), "[S]")
    }

    func testIfElseFalsey() throws {
        let ctx = GoValue.dict(["System": .string("")])
        XCTAssertEqual(try render("{{ if .System }}yes{{ else }}no{{ end }}", ctx), "no")
    }

    func testElseIfChain() throws {
        func r(_ role: String) throws -> String {
            try render(
                "{{ if eq .Role \"user\" }}U{{ else if eq .Role \"assistant\" }}A{{ else }}?{{ end }}",
                .dict(["Role": .string(role)]))
        }
        XCTAssertEqual(try r("user"), "U")
        XCTAssertEqual(try r("assistant"), "A")
        XCTAssertEqual(try r("system"), "?")
    }

    // MARK: - range

    func testRangeValueOnly() throws {
        let ctx = GoValue.dict(["Messages": .list([
            .dict(["Content": .string("a")]),
            .dict(["Content": .string("b")]),
        ])])
        XCTAssertEqual(
            try render("{{ range .Messages }}{{ .Content }};{{ end }}", ctx),
            "a;b;")
    }

    func testRangeIndexAndValueVars() throws {
        let ctx = GoValue.dict(["Messages": .list([
            .dict(["Content": .string("a")]),
            .dict(["Content": .string("b")]),
        ])])
        XCTAssertEqual(
            try render("{{ range $i, $m := .Messages }}{{ $i }}:{{ $m.Content }} {{ end }}", ctx),
            "0:a 1:b ")
    }

    func testRangeElseOnEmpty() throws {
        let ctx = GoValue.dict(["Messages": .list([])])
        XCTAssertEqual(
            try render("{{ range .Messages }}x{{ else }}empty{{ end }}", ctx),
            "empty")
    }

    func testRootDotInsideRange() throws {
        // `$` (or a captured var) reaches the root while `.` is the item.
        let ctx = GoValue.dict([
            "Sep": .string("-"),
            "Items": .list([.string("a"), .string("b")]),
        ])
        XCTAssertEqual(
            try render("{{ range .Items }}{{ . }}{{ $.Sep }}{{ end }}", ctx),
            "a-b-")
    }

    // MARK: - with

    func testWithRebindsDot() throws {
        let ctx = GoValue.dict(["Inner": .dict(["X": .string("v")])])
        XCTAssertEqual(try render("{{ with .Inner }}{{ .X }}{{ end }}", ctx), "v")
    }

    func testWithElse() throws {
        let ctx = GoValue.dict(["Inner": .null])
        XCTAssertEqual(try render("{{ with .Inner }}{{ .X }}{{ else }}none{{ end }}", ctx), "none")
    }

    // MARK: - pipelines + builtins

    func testPipeIntoPrintf() throws {
        let ctx = GoValue.dict(["Name": .string("ada")])
        XCTAssertEqual(try render("{{ printf \"<%s>\" .Name }}", ctx), "<ada>")
        XCTAssertEqual(try render("{{ .Name | printf \"<%s>\" }}", ctx), "<ada>")
    }

    func testComparisonBuiltins() throws {
        XCTAssertEqual(try render("{{ if lt 1 2 }}y{{ end }}", .null), "y")
        XCTAssertEqual(try render("{{ if ge 2 2 }}y{{ end }}", .null), "y")
        XCTAssertEqual(try render("{{ if eq 3 3 }}y{{ end }}", .null), "y")
        XCTAssertEqual(try render("{{ if ne \"a\" \"b\" }}y{{ end }}", .null), "y")
    }

    func testAndOrNot() throws {
        XCTAssertEqual(try render("{{ if and true true }}y{{ end }}", .null), "y")
        XCTAssertEqual(try render("{{ if or false true }}y{{ end }}", .null), "y")
        XCTAssertEqual(try render("{{ if not false }}y{{ end }}", .null), "y")
    }

    func testLenIndexSlice() throws {
        let ctx = GoValue.dict(["L": .list([.string("a"), .string("b"), .string("c")])])
        XCTAssertEqual(try render("{{ len .L }}", ctx), "3")
        XCTAssertEqual(try render("{{ index .L 1 }}", ctx), "b")
        XCTAssertEqual(try render("{{ range slice .L 1 3 }}{{ . }}{{ end }}", ctx), "bc")
    }

    func testDefaultBuiltin() throws {
        XCTAssertEqual(try render("{{ default \"fallback\" .Missing }}", .dict([:])), "fallback")
        XCTAssertEqual(try render("{{ default \"fallback\" .V }}", .dict(["V": .string("set")])), "set")
    }

    func testParenthesizedSubPipeline() throws {
        XCTAssertEqual(try render("{{ if eq (len .L) 2 }}two{{ end }}",
            .dict(["L": .list([.int(1), .int(2)])])), "two")
    }

    // MARK: - whitespace trimming

    func testTrimMarkers() throws {
        // `{{-` eats preceding ws; `-}}` eats following ws.
        XCTAssertEqual(try render("a  {{- \"x\" }}", .null), "ax")
        XCTAssertEqual(try render("{{ \"x\" -}}  b", .null), "xb")
        XCTAssertEqual(try render("a\n{{- if true -}}\n  b{{- end }}", .null), "ab")
    }

    // MARK: - realistic Ollama ChatML TEMPLATE

    func testChatMLStyleTemplate() throws {
        let tmpl = """
        {{- if .System }}<|im_start|>system
        {{ .System }}<|im_end|>
        {{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}
        {{ .Content }}<|im_end|>
        {{ end }}<|im_start|>assistant
        """
        let ctx = GoValue.dict([
            "System": .string("You are helpful."),
            "Messages": .list([
                .dict(["Role": .string("user"), "Content": .string("Hi")]),
                .dict(["Role": .string("assistant"), "Content": .string("Hello!")]),
            ]),
        ])
        let out = try render(tmpl, ctx)
        XCTAssertTrue(out.contains("<|im_start|>system\nYou are helpful.<|im_end|>"), out)
        XCTAssertTrue(out.contains("<|im_start|>user\nHi<|im_end|>"), out)
        XCTAssertTrue(out.contains("<|im_start|>assistant\nHello!<|im_end|>"), out)
        XCTAssertTrue(out.hasSuffix("<|im_start|>assistant"), out)
    }

    // MARK: - OllamaTemplateContext bridge

    func testContextBuildMapsMessages() throws {
        let ctx = OllamaTemplateContext.build(messages: [
            ["role": "system", "content": "S"],
            ["role": "user", "content": "Q1"],
            ["role": "assistant", "content": "A1"],
            ["role": "user", "content": "Q2"],
        ])
        guard case .dict(let d) = ctx else { return XCTFail("expected dict") }
        XCTAssertEqual(d["System"], .string("S"))
        XCTAssertEqual(d["Prompt"], .string("Q2"))  // last user turn
        guard case .list(let msgs)? = d["Messages"] else { return XCTFail("expected Messages list") }
        XCTAssertEqual(msgs.count, 3)  // system pulled out of Messages
        XCTAssertEqual(msgs.first, .dict(["Role": .string("user"), "Content": .string("Q1")]))
    }

    func testContextThroughTemplateEndToEnd() throws {
        let ctx = OllamaTemplateContext.build(messages: [
            ["role": "user", "content": "ping"],
        ])
        XCTAssertEqual(try render("user: {{ .Prompt }}", ctx), "user: ping")
    }

    // MARK: - failure modes (caller falls back)

    func testUnknownFunctionThrows() {
        XCTAssertThrowsError(try render("{{ frobnicate .X }}", .dict(["X": .string("y")]))) { err in
            guard case GoTemplateError.eval = err else {
                return XCTFail("expected eval error, got \(err)")
            }
        }
    }

    func testUnclosedActionThrows() {
        XCTAssertThrowsError(try render("{{ .Prompt ", .null)) { err in
            guard case GoTemplateError.parse = err else {
                return XCTFail("expected parse error, got \(err)")
            }
        }
    }

    func testMissingEndThrows() {
        XCTAssertThrowsError(try render("{{ if .X }}y", .dict(["X": .bool(true)]))) { err in
            guard case GoTemplateError.parse = err else {
                return XCTFail("expected parse error, got \(err)")
            }
        }
    }

    // MARK: - recursion-depth guard (DoS on user-supplied Modelfiles)

    /// Deeply nested blocks must throw a catchable parse error rather than
    /// overflowing the stack (an uncatchable SIGSEGV would crash the
    /// server past the engine's catch-and-fallback contract). The
    /// template is user-supplied via a Modelfile, so this is a DoS guard.
    func testDeeplyNestedBlocksThrowNotCrash() {
        let n = 5000
        let src = String(repeating: "{{ if true }}", count: n)
            + "x" + String(repeating: "{{ end }}", count: n)
        XCTAssertThrowsError(try render(src, .null)) { err in
            guard case GoTemplateError.parse = err else {
                return XCTFail("expected parse error, got \(err)")
            }
        }
    }

    /// Deeply nested parenthesized sub-pipelines must likewise throw, not
    /// overflow the stack.
    func testDeeplyNestedParensThrowNotCrash() {
        let n = 5000
        let src = "{{ " + String(repeating: "(", count: n) + "len .L"
            + String(repeating: ")", count: n) + " }}"
        XCTAssertThrowsError(try render(src, .dict(["L": .list([.int(1)])]))) { err in
            guard case GoTemplateError.parse = err else {
                return XCTFail("expected parse error, got \(err)")
            }
        }
    }

    /// A long `else if` chain self-recurses through `parseIfBody`; it
    /// must hit the depth cap and throw, not pile native frames into a
    /// stack overflow.
    func testDeepElseIfChainThrowsNotCrash() {
        let n = 5000
        var src = "{{ if eq .R \"r0\" }}0"
        for i in 1..<n { src += "{{ else if eq .R \"r\(i)\" }}\(i)" }
        src += "{{ else }}none{{ end }}"
        XCTAssertThrowsError(try render(src, .dict(["R": .string("rNope")]))) { err in
            guard case GoTemplateError.parse = err else {
                return XCTFail("expected parse error, got \(err)")
            }
        }
    }

    /// A normal template nesting a handful of levels must still render
    /// (the cap is far above realistic nesting).
    func testModerateNestingStillRenders() throws {
        let src = "{{ if .A }}{{ if .B }}{{ range .L }}{{ . }}{{ end }}{{ end }}{{ end }}"
        let ctx = GoValue.dict([
            "A": .bool(true), "B": .bool(true),
            "L": .list([.string("x"), .string("y")]),
        ])
        XCTAssertEqual(try render(src, ctx), "xy")
    }
}
