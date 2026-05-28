import Foundation

// MARK: - Go text/template engine (Ollama Modelfile TEMPLATE)

/// A from-scratch evaluator for the Go `text/template` language, enough
/// to render Ollama `Modelfile` `TEMPLATE` directives. Ollama prompts
/// are Go templates -- `{{ .System }}`, `{{ range .Messages }}`,
/// `{{ if .Tools }}`, `{{- -}}` whitespace trimming -- whereas KrillLM's
/// built-in chat formatting goes through HF Jinja templates. When a
/// created model carries a `TEMPLATE` override we must render with the
/// Go-template semantics the author wrote against, so this engine is the
/// missing renderer the `ModelOverrides.template` field round-tripped
/// through `/api/show` without ever applying.
///
/// Supported surface (the practical TEMPLATE language):
///   - text + `{{ ... }}` actions, with `{{-` / `-}}` whitespace trims
///   - pipelines: `cmd arg arg | fn | fn`
///   - control flow: `if` / `else if` / `else` / `end`,
///     `range` (with `$i, $v :=`) / `else` / `end`, `with` / `else` / `end`
///   - variables: `$`, `$name`, `$name := pipeline`, `$name = pipeline`
///   - operands: `.`, field chains `.A.B.C`, `$var`, `(sub pipeline)`,
///     string / raw-string / number / `true` / `false` / `nil` literals
///   - builtins: `and or not eq ne lt le gt ge len index slice
///     print printf println default json` (the set Ollama TEMPLATEs use)
///
/// Unsupported constructs (`define` / `block` / `template` association,
/// custom functions) throw `GoTemplateError`; the caller falls back to
/// the model's built-in chat template so an exotic Modelfile never hard
/// fails a request.
public enum GoTemplateError: Error, CustomStringConvertible, Equatable {
    case parse(String)
    case eval(String)

    public var description: String {
        switch self {
        case .parse(let m): return "Go template parse error: \(m)"
        case .eval(let m): return "Go template eval error: \(m)"
        }
    }
}

// MARK: - Value model

/// A dynamically-typed value flowing through template evaluation. The
/// render context (`.System`, `.Messages`, etc.) is built from these.
public indirect enum GoValue: Equatable {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case list([GoValue])
    case dict([String: GoValue])
    case null

    /// Go's notion of "truthy" for `if` / `with`: false, 0, "", empty
    /// list/dict, and nil are falsey; everything else is true.
    var isTruthy: Bool {
        switch self {
        case .bool(let b): return b
        case .int(let i): return i != 0
        case .double(let d): return d != 0
        case .string(let s): return !s.isEmpty
        case .list(let l): return !l.isEmpty
        case .dict(let d): return !d.isEmpty
        case .null: return false
        }
    }

    /// Text rendering of a value when it lands directly in output (the
    /// `{{ .Prompt }}` case). Mirrors Go's default `%v`-ish printing.
    var asOutput: String {
        switch self {
        case .string(let s): return s
        case .int(let i): return String(i)
        case .double(let d): return trimmedDouble(d)
        case .bool(let b): return b ? "true" : "false"
        case .null: return ""
        case .list, .dict:
            // Go prints maps/slices with a structured form; templates
            // almost never print a composite directly, so a JSON-ish
            // dump is a reasonable, non-crashing rendering.
            return (try? jsonString()) ?? ""
        }
    }

    func jsonString() throws -> String {
        let obj = toFoundation()
        let data = try JSONSerialization.data(
            withJSONObject: obj, options: [.sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    private func toFoundation() -> Any {
        switch self {
        case .string(let s): return s
        case .int(let i): return i
        case .double(let d): return d
        case .bool(let b): return b
        case .null: return NSNull()
        case .list(let l): return l.map { $0.toFoundation() }
        case .dict(let d): return d.mapValues { $0.toFoundation() }
        }
    }
}

private func trimmedDouble(_ d: Double) -> String {
    if d == d.rounded() && abs(d) < 1e15 { return String(Int(d)) }
    return String(d)
}

// MARK: - AST

private indirect enum Node {
    case text(String)
    case action(Pipeline)
    case ifNode(Pipeline, [Node], [Node])           // cond, then, else
    case rangeNode(RangeSpec, [Node], [Node])        // spec, body, else
    case withNode(Pipeline, [Node], [Node])
}

private struct RangeSpec {
    var indexVar: String?    // $i (optional)
    var valueVar: String?    // $v (optional)
    var pipeline: Pipeline
}

/// A pipeline is one or more commands joined by `|`; each command's
/// output becomes the trailing argument of the next.
private struct Pipeline {
    var assignVar: String?   // `$x := ...` / `$x = ...`
    var declare: Bool        // := vs =
    var commands: [Command]
}

private struct Command {
    var args: [Operand]      // args[0] is the function/operand head
}

private indirect enum Operand {
    case dot                       // .
    case field([String])           // .A.B.C  (relative to dot)
    case variable(String, [String])// $x  or  $x.A.B  (`$` is root dot)
    case stringLit(String)
    case intLit(Int)
    case doubleLit(Double)
    case boolLit(Bool)
    case nilLit
    case identifier(String)        // a builtin/function name
    case pipeline(Pipeline)        // ( ... )
}

// MARK: - Public entry

public struct GoTemplate {
    private let nodes: [Node]

    /// Parse a Go-template source string. Throws `GoTemplateError.parse`
    /// on malformed or unsupported syntax.
    public init(_ source: String) throws {
        var lexer = Lexer(source)
        var tokens = try lexer.tokenize()
        GoTemplate.applyTrims(&tokens)
        var parser = Parser(tokens)
        self.nodes = try parser.parseTemplate(stopAt: [])
        if !parser.atEnd {
            throw GoTemplateError.parse("unexpected '\(parser.peekKeyword() ?? "?")' (stray end/else?)")
        }
    }

    /// Resolve `{{-` / `-}}` whitespace trims on the token stream, the
    /// way Go's lexer does: a left-trim removes trailing whitespace from
    /// the immediately preceding text token, a right-trim removes leading
    /// whitespace from the immediately following text token. Doing this
    /// before parsing makes trims work uniformly across control-flow
    /// tokens (`{{- if -}}`, `{{- end }}`), where parse-time trimming
    /// could not reach the text inside or after a block.
    private static func applyTrims(_ tokens: inout [Lexer.Token]) {
        for idx in tokens.indices {
            guard case .action(_, let tl, let tr) = tokens[idx] else { continue }
            if tl, idx > 0, case .text(let t) = tokens[idx - 1] {
                tokens[idx - 1] = .text(
                    String(t.reversed().drop(while: { $0.isWhitespace }).reversed()))
            }
            if tr, idx + 1 < tokens.count, case .text(let t) = tokens[idx + 1] {
                tokens[idx + 1] = .text(String(t.drop(while: { $0.isWhitespace })))
            }
        }
    }

    /// Render with the given root context. Throws `GoTemplateError.eval`
    /// on a runtime error (unknown function, bad field on a non-dict,
    /// arity mismatch).
    public func render(_ root: GoValue) throws -> String {
        var ev = Evaluator(root: root)
        return try ev.run(nodes)
    }

    /// Convenience: parse + render in one call.
    public static func render(_ source: String, _ root: GoValue) throws -> String {
        try GoTemplate(source).render(root)
    }
}

// MARK: - Ollama chat context

/// Builds the render context an Ollama chat `TEMPLATE` expects from a
/// KrillLM message list. Ollama templates address `.Messages` (a list
/// of `{Role, Content}`), `.System` (the system text), and `.Prompt`
/// (the latest user turn, for legacy single-turn templates); `.Response`
/// is empty at prompt-build time. This is the bridge from KrillLM's
/// `[[String: String]]` chat messages to the `GoValue` tree the engine
/// evaluates against.
public enum OllamaTemplateContext {
    public static func build(messages: [[String: String]]) -> GoValue {
        var msgList: [GoValue] = []
        var systemText = ""
        var lastUserPrompt = ""
        for m in messages {
            let role = m["role"] ?? ""
            let content = m["content"] ?? ""
            if role == "system" {
                // Concatenate multiple system turns the way chat
                // front-ends collapse them.
                systemText += (systemText.isEmpty ? "" : "\n\n") + content
            } else {
                msgList.append(.dict([
                    "Role": .string(role),
                    "Content": .string(content),
                ]))
                if role == "user" { lastUserPrompt = content }
            }
        }
        return .dict([
            "Messages": .list(msgList),
            "System": .string(systemText),
            "Prompt": .string(lastUserPrompt),
            "Response": .string(""),
        ])
    }
}

// MARK: - Lexer

/// Splits source into text runs and `{{ ... }}` action bodies, honoring
/// `{{-` / `-}}` trim markers by emitting trim flags the parser applies
/// to the adjacent text.
private struct Lexer {
    enum Token: Equatable {
        case text(String)
        case action(String, trimLeft: Bool, trimRight: Bool)
    }

    let chars: [Character]
    var i = 0

    init(_ s: String) { chars = Array(s) }

    mutating func tokenize() throws -> [Token] {
        var out: [Token] = []
        var textBuf = ""
        while i < chars.count {
            if chars[i] == "{" && i + 1 < chars.count && chars[i + 1] == "{" {
                // flush pending text
                if !textBuf.isEmpty { out.append(.text(textBuf)); textBuf = "" }
                i += 2
                var trimLeft = false
                if i < chars.count && chars[i] == "-"
                    && i + 1 < chars.count && chars[i + 1].isWhitespace {
                    trimLeft = true; i += 1
                }
                // read until "}}"
                var body = ""
                var trimRight = false
                var closed = false
                while i < chars.count {
                    if chars[i] == "}" && i + 1 < chars.count && chars[i + 1] == "}" {
                        i += 2; closed = true; break
                    }
                    // "-}}" : a trailing "-" right before "}}" preceded by ws
                    if chars[i] == "-" && i + 1 < chars.count && chars[i + 1] == "}"
                        && i + 2 < chars.count && chars[i + 2] == "}"
                        && (body.last?.isWhitespace ?? true) {
                        trimRight = true; i += 3; closed = true; break
                    }
                    body.append(chars[i]); i += 1
                }
                guard closed else {
                    throw GoTemplateError.parse("unclosed action '{{'")
                }
                out.append(.action(
                    body.trimmingCharacters(in: .whitespaces),
                    trimLeft: trimLeft, trimRight: trimRight))
            } else {
                textBuf.append(chars[i]); i += 1
            }
        }
        if !textBuf.isEmpty { out.append(.text(textBuf)) }
        return out
    }
}

// MARK: - Parser

private struct Parser {
    let tokens: [Lexer.Token]
    var pos = 0

    /// Recursion-depth guard. The parser is recursive descent (block
    /// bodies and parenthesized sub-pipelines recurse), and the template
    /// is user-supplied (a Modelfile), so adversarially deep nesting like
    /// `{{ if }}`xN or `(((...)))` would overflow the stack. A stack
    /// overflow is an uncatchable trap -- it would crash the server
    /// instead of falling back to the built-in template -- so we cap the
    /// depth and throw a catchable `GoTemplateError.parse` instead. Real
    /// TEMPLATEs nest only a few levels deep; the cap is far above that.
    var depth = 0
    static let maxDepth = 200

    init(_ tokens: [Lexer.Token]) { self.tokens = tokens }

    var atEnd: Bool { pos >= tokens.count }

    private mutating func enter() throws {
        depth += 1
        if depth > Parser.maxDepth {
            throw GoTemplateError.parse("template nesting too deep (>\(Parser.maxDepth))")
        }
    }
    private mutating func leave() { depth -= 1 }

    /// Peek the leading keyword of the next action token (if/else/range/
    /// with/end), used to terminate block parsing.
    func peekKeyword() -> String? {
        guard pos < tokens.count, case .action(let body, _, _) = tokens[pos] else {
            return nil
        }
        return body.split(separator: " ", maxSplits: 1).first.map(String.init)
    }

    /// Parse nodes until one of `stopAt` keywords is the next action.
    /// Leaves the terminating action UNconsumed.
    mutating func parseTemplate(stopAt: Set<String>) throws -> [Node] {
        try enter(); defer { leave() }
        var nodes: [Node] = []
        while pos < tokens.count {
            switch tokens[pos] {
            case .text(let t):
                // A trim may have reduced a text token to "" -- skip empties
                // so they don't litter the output.
                if !t.isEmpty { nodes.append(.text(t)) }
                pos += 1
            case .action(let body, _, _):
                let kw = body.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
                if stopAt.contains(kw) { return nodes }
                if let node = try parseAction(body) { nodes.append(node) }
            }
        }
        return nodes
    }

    private mutating func parseAction(_ body: String) throws -> Node? {
        let kw = body.split(separator: " ", maxSplits: 1).first.map(String.init) ?? ""
        switch kw {
        case "if":
            return try parseIf(body)
        case "range":
            return try parseRange(body)
        case "with":
            return try parseWith(body)
        case "end", "else":
            throw GoTemplateError.parse("unexpected '\(kw)'")
        default:
            // a bare pipeline action: {{ .Prompt }}, {{ $x := ... }}, etc.
            // Consume this action token. (The control-flow cases above
            // advance `pos` themselves; the plain-action case must too,
            // or `parseTemplate`'s loop never advances and spins.)
            pos += 1
            let pipe = try parsePipeline(body)
            return .action(pipe)
        }
    }

    private mutating func parseIf(_ body: String) throws -> Node {
        pos += 1  // consume the `if` action token
        return try parseIfBody(body)
    }

    /// Parse an if-block whose leading token has ALREADY been consumed.
    /// `body` is the action text starting with `if` (the initial `if` or
    /// an `else if`). Splitting consume-vs-parse this way lets `else if`
    /// recurse without double-advancing `pos` past the first node of its
    /// then-branch.
    private mutating func parseIfBody(_ body: String) throws -> Node {
        // `else if` self-recurses here; guard the depth too, or a long
        // else-if chain piles native frames while the counter stays flat
        // (parsePipeline's bump resets via defer before the recursion).
        try enter(); defer { leave() }
        let cond = try parsePipeline(String(body.dropFirst(2)))  // drop "if"
        let thenNodes = try parseTemplate(stopAt: ["else", "end"])
        var elseNodes: [Node] = []
        if case .action(let b, _, _)? = current, b.hasPrefix("else") {
            let rest = b.dropFirst(4).trimmingCharacters(in: .whitespaces)
            pos += 1  // consume the `else` / `else if` token
            if rest.hasPrefix("if") {
                // `else if ...`: the token is already consumed, so parse
                // its body directly (do NOT call parseIf, which would
                // consume another token).
                elseNodes = [try parseIfBody(rest)]
                return .ifNode(cond, thenNodes, elseNodes)
            } else {
                elseNodes = try parseTemplate(stopAt: ["end"])
            }
        }
        try expectEnd()
        return .ifNode(cond, thenNodes, elseNodes)
    }

    private mutating func parseWith(_ body: String) throws -> Node {
        pos += 1
        let pipe = try parsePipeline(String(body.dropFirst(4)))  // drop "with"
        let bodyNodes = try parseTemplate(stopAt: ["else", "end"])
        var elseNodes: [Node] = []
        if case .action(let b, _, _)? = current, b.hasPrefix("else") {
            pos += 1
            elseNodes = try parseTemplate(stopAt: ["end"])
        }
        try expectEnd()
        return .withNode(pipe, bodyNodes, elseNodes)
    }

    private mutating func parseRange(_ body: String) throws -> Node {
        pos += 1
        var rest = String(body.dropFirst(5)).trimmingCharacters(in: .whitespaces) // drop "range"
        var indexVar: String?
        var valueVar: String?
        // Optional `$i, $v :=` or `$v :=` variable spec.
        if rest.hasPrefix("$"), let r = rest.range(of: ":=") {
            let spec = String(rest[..<r.lowerBound]).trimmingCharacters(in: .whitespaces)
            let vars = spec.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespaces)
            }
            if vars.count == 1 {
                valueVar = String(vars[0].dropFirst())
            } else if vars.count == 2 {
                indexVar = String(vars[0].dropFirst())
                valueVar = String(vars[1].dropFirst())
            }
            rest = String(rest[r.upperBound...]).trimmingCharacters(in: .whitespaces)
        }
        let pipe = try parsePipeline(rest, alreadyStripped: true)
        let bodyNodes = try parseTemplate(stopAt: ["else", "end"])
        var elseNodes: [Node] = []
        if case .action(let b, _, _)? = current, b.hasPrefix("else") {
            pos += 1
            elseNodes = try parseTemplate(stopAt: ["end"])
        }
        try expectEnd()
        return .rangeNode(
            RangeSpec(indexVar: indexVar, valueVar: valueVar, pipeline: pipe),
            bodyNodes, elseNodes)
    }

    private var current: Lexer.Token? { pos < tokens.count ? tokens[pos] : nil }

    private mutating func expectEnd() throws {
        guard case .action(let b, _, _)? = current,
              b.split(separator: " ").first.map(String.init) == "end" else {
            throw GoTemplateError.parse("expected 'end'")
        }
        pos += 1
    }

    // MARK: pipeline / command / operand parsing

    /// Parse a pipeline body string (the text inside `{{ }}` minus any
    /// leading control keyword). `alreadyStripped` means the keyword was
    /// already removed (range case).
    private mutating func parsePipeline(_ raw: String, alreadyStripped: Bool = false) throws -> Pipeline {
        try enter(); defer { leave() }
        var s = raw.trimmingCharacters(in: .whitespaces)
        var assignVar: String?
        var declare = false
        // assignment: `$x := pipe` or `$x = pipe`
        if s.hasPrefix("$") {
            if let r = s.range(of: ":=") {
                assignVar = String(s[s.index(after: s.startIndex)..<r.lowerBound])
                    .trimmingCharacters(in: .whitespaces)
                declare = true
                s = String(s[r.upperBound...]).trimmingCharacters(in: .whitespaces)
            } else if let r = rangeOfBareEquals(s) {
                assignVar = String(s[s.index(after: s.startIndex)..<r.lowerBound])
                    .trimmingCharacters(in: .whitespaces)
                s = String(s[r.upperBound...]).trimmingCharacters(in: .whitespaces)
            }
        }
        let segments = try splitTopLevel(s, on: "|")
        var commands: [Command] = []
        for seg in segments {
            let cmd = try parseCommand(seg)
            if !cmd.args.isEmpty { commands.append(cmd) }
        }
        if commands.isEmpty && assignVar == nil {
            throw GoTemplateError.parse("empty pipeline")
        }
        return Pipeline(assignVar: assignVar, declare: declare, commands: commands)
    }

    /// Find a bare `=` (assignment) that is not part of `==`/`:=`/`!=`.
    private func rangeOfBareEquals(_ s: String) -> Range<String.Index>? {
        let arr = Array(s)
        var idx = 0
        while idx < arr.count {
            if arr[idx] == "=" {
                let prev = idx > 0 ? arr[idx - 1] : " "
                let next = idx + 1 < arr.count ? arr[idx + 1] : " "
                if prev != ":" && prev != "=" && prev != "!"
                    && prev != "<" && prev != ">" && next != "=" {
                    let start = s.index(s.startIndex, offsetBy: idx)
                    return start..<s.index(after: start)
                }
            }
            idx += 1
        }
        return nil
    }

    private mutating func parseCommand(_ raw: String) throws -> Command {
        let operands = try tokenizeOperands(raw.trimmingCharacters(in: .whitespaces))
        return Command(args: operands)
    }

    /// Split a string on a top-level separator char, ignoring occurrences
    /// inside parentheses or quotes.
    private func splitTopLevel(_ s: String, on sep: Character) throws -> [String] {
        var out: [String] = []
        var depth = 0
        var inStr: Character? = nil
        var buf = ""
        let arr = Array(s)
        var k = 0
        while k < arr.count {
            let c = arr[k]
            if let q = inStr {
                buf.append(c)
                if c == q && arr[k - 1] != "\\" { inStr = nil }
            } else if c == "\"" || c == "`" {
                inStr = c; buf.append(c)
            } else if c == "(" {
                depth += 1; buf.append(c)
            } else if c == ")" {
                depth -= 1; buf.append(c)
            } else if c == sep && depth == 0 {
                out.append(buf); buf = ""
            } else {
                buf.append(c)
            }
            k += 1
        }
        out.append(buf)
        return out
    }

    /// Split a command into operand tokens (whitespace-separated at top
    /// level, respecting quotes and parens), then classify each.
    private mutating func tokenizeOperands(_ s: String) throws -> [Operand] {
        var rawTokens: [String] = []
        var depth = 0
        var inStr: Character? = nil
        var buf = ""
        let arr = Array(s)
        var k = 0
        func flush() { if !buf.isEmpty { rawTokens.append(buf); buf = "" } }
        while k < arr.count {
            let c = arr[k]
            if let q = inStr {
                buf.append(c)
                if c == q && arr[k - 1] != "\\" { inStr = nil }
            } else if c == "\"" || c == "`" {
                inStr = c; buf.append(c)
            } else if c == "(" {
                if depth == 0 && !buf.isEmpty { flush() }
                depth += 1; buf.append(c)
            } else if c == ")" {
                depth -= 1; buf.append(c)
                if depth == 0 { flush() }
            } else if c.isWhitespace && depth == 0 {
                flush()
            } else {
                buf.append(c)
            }
            k += 1
        }
        flush()
        return try rawTokens.map { try classifyOperand($0) }
    }

    private mutating func classifyOperand(_ tok: String) throws -> Operand {
        if tok.hasPrefix("(") && tok.hasSuffix(")") {
            let inner = String(tok.dropFirst().dropLast())
            return .pipeline(try parsePipeline(inner))
        }
        if tok == "." { return .dot }
        if tok.hasPrefix(".") {
            let fields = tok.dropFirst().split(separator: ".").map(String.init)
            return .field(fields)
        }
        if tok.hasPrefix("$") {
            let body = String(tok.dropFirst())
            if body.isEmpty { return .variable("", []) }      // `$` = root
            if body.hasPrefix(".") {
                // `$.Field.Path` = root dot followed by a field chain.
                let fields = body.dropFirst().split(separator: ".").map(String.init)
                return .variable("", fields)
            }
            // `$name` or `$name.Field` = a declared variable + field chain.
            let parts = body.split(separator: ".").map(String.init)
            return .variable(parts[0], Array(parts.dropFirst()))
        }
        if tok.hasPrefix("\"") && tok.hasSuffix("\"") && tok.count >= 2 {
            return .stringLit(unescape(String(tok.dropFirst().dropLast())))
        }
        if tok.hasPrefix("`") && tok.hasSuffix("`") && tok.count >= 2 {
            return .stringLit(String(tok.dropFirst().dropLast()))  // raw
        }
        if tok == "true" { return .boolLit(true) }
        if tok == "false" { return .boolLit(false) }
        if tok == "nil" { return .nilLit }
        if let i = Int(tok) { return .intLit(i) }
        if let d = Double(tok) { return .doubleLit(d) }
        return .identifier(tok)
    }

    private func unescape(_ s: String) -> String {
        var out = ""
        var it = s.makeIterator()
        while let c = it.next() {
            if c == "\\", let n = it.next() {
                switch n {
                case "n": out.append("\n")
                case "t": out.append("\t")
                case "r": out.append("\r")
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                default: out.append(n)
                }
            } else {
                out.append(c)
            }
        }
        return out
    }
}

// MARK: - Evaluator

private struct Evaluator {
    let root: GoValue
    var scopes: [[String: GoValue]] = [[:]]

    /// Render-time recursion guard, mirroring the parser's. A valid but
    /// deeply nested AST (the parser caps nesting, but a paren-pipeline
    /// chain or a hand-built tree could still recurse) must not overflow
    /// the stack and crash the process past the catch-and-fallback. Cap
    /// and throw a catchable `GoTemplateError.eval` instead.
    var depth = 0
    static let maxDepth = 200

    init(root: GoValue) { self.root = root }

    mutating func run(_ nodes: [Node]) throws -> String {
        var out = ""
        for node in nodes {
            try render(node, dot: root, into: &out)
        }
        return out
    }

    private mutating func render(_ node: Node, dot: GoValue, into out: inout String) throws {
        depth += 1
        defer { depth -= 1 }
        if depth > Evaluator.maxDepth {
            throw GoTemplateError.eval("template render nesting too deep (>\(Evaluator.maxDepth))")
        }
        switch node {
        case .text(let t):
            out += t
        case .action(let pipe):
            let v = try evalPipeline(pipe, dot: dot)
            // An assignment-only pipeline produces no output.
            if pipe.assignVar != nil && pipe.commands.isEmpty { return }
            out += v.asOutput
        case .ifNode(let cond, let thenN, let elseN):
            let c = try evalPipeline(cond, dot: dot)
            let branch = c.isTruthy ? thenN : elseN
            for n in branch { try render(n, dot: dot, into: &out) }
        case .withNode(let pipe, let bodyN, let elseN):
            let v = try evalPipeline(pipe, dot: dot)
            if v.isTruthy {
                for n in bodyN { try render(n, dot: v, into: &out) }
            } else {
                for n in elseN { try render(n, dot: dot, into: &out) }
            }
        case .rangeNode(let spec, let bodyN, let elseN):
            let v = try evalPipeline(spec.pipeline, dot: dot)
            let items = rangeItems(v)
            if items.isEmpty {
                for n in elseN { try render(n, dot: dot, into: &out) }
            } else {
                scopes.append([:])
                defer { scopes.removeLast() }
                for (idx, item) in items {
                    if let iv = spec.indexVar { scopes[scopes.count - 1][iv] = idx }
                    if let vv = spec.valueVar { scopes[scopes.count - 1][vv] = item }
                    for n in bodyN { try render(n, dot: item, into: &out) }
                }
            }
        }
    }

    private func rangeItems(_ v: GoValue) -> [(GoValue, GoValue)] {
        switch v {
        case .list(let l): return l.enumerated().map { (.int($0.offset), $0.element) }
        case .dict(let d):
            return d.sorted { $0.key < $1.key }.map { (.string($0.key), $0.value) }
        default: return []
        }
    }

    // MARK: pipeline / command evaluation

    private mutating func evalPipeline(_ pipe: Pipeline, dot: GoValue) throws -> GoValue {
        depth += 1
        defer { depth -= 1 }
        if depth > Evaluator.maxDepth {
            throw GoTemplateError.eval("template pipeline nesting too deep (>\(Evaluator.maxDepth))")
        }
        var carry: GoValue? = nil
        for cmd in pipe.commands {
            carry = try evalCommand(cmd, dot: dot, piped: carry)
        }
        let result = carry ?? .null
        if let v = pipe.assignVar {
            scopes[scopes.count - 1][v] = result
        }
        return result
    }

    private mutating func evalCommand(
        _ cmd: Command, dot: GoValue, piped: GoValue?
    ) throws -> GoValue {
        guard let head = cmd.args.first else { return .null }
        // A function/identifier head means a function call; otherwise the
        // command is a single operand (with possible piped trailing arg).
        if case .identifier(let name) = head {
            var args = try cmd.args.dropFirst().map { try evalOperand($0, dot: dot) }
            if let p = piped { args.append(p) }   // pipe feeds last arg
            return try callFunction(name, args)
        }
        // Non-function head: evaluate it; ignore piped (Go would error on
        // extra args to a non-function, but templates that pipe into a
        // field are malformed -- we just return the operand).
        return try evalOperand(head, dot: dot)
    }

    private mutating func evalOperand(_ op: Operand, dot: GoValue) throws -> GoValue {
        switch op {
        case .dot: return dot
        case .field(let fields): return try lookupFields(fields, in: dot)
        case .variable(let name, let fields):
            let base = name.isEmpty ? root : (lookupVar(name) ?? .null)
            return try lookupFields(fields, in: base)
        case .stringLit(let s): return .string(s)
        case .intLit(let i): return .int(i)
        case .doubleLit(let d): return .double(d)
        case .boolLit(let b): return .bool(b)
        case .nilLit: return .null
        case .identifier(let name):
            // A bare identifier as an operand is a zero-arg function call
            // (e.g. a custom value); only builtins with no args qualify.
            return try callFunction(name, [])
        case .pipeline(let p): return try evalPipeline(p, dot: dot)
        }
    }

    private func lookupVar(_ name: String) -> GoValue? {
        for scope in scopes.reversed() {
            if let v = scope[name] { return v }
        }
        return nil
    }

    private func lookupFields(_ fields: [String], in base: GoValue) throws -> GoValue {
        var cur = base
        for f in fields {
            guard case .dict(let d) = cur else {
                // Field access on a non-dict (or absent key) yields nil,
                // matching Go's behaviour for a missing map key rather
                // than throwing -- templates routinely guard with `if`.
                return .null
            }
            cur = d[f] ?? .null
        }
        return cur
    }

    // MARK: builtins

    private func callFunction(_ name: String, _ args: [GoValue]) throws -> GoValue {
        switch name {
        case "and":
            // returns first falsey arg, else last
            var last: GoValue = .bool(true)
            for a in args { if !a.isTruthy { return a }; last = a }
            return last
        case "or":
            var last: GoValue = .bool(false)
            for a in args { if a.isTruthy { return a }; last = a }
            return last
        case "not":
            try arity(name, args, 1)
            return .bool(!args[0].isTruthy)
        case "eq":
            try minArity(name, args, 2)
            return .bool(args.dropFirst().allSatisfy { goEqual($0, args[0]) })
        case "ne":
            try arity(name, args, 2)
            return .bool(!goEqual(args[0], args[1]))
        case "lt", "le", "gt", "ge":
            try arity(name, args, 2)
            return .bool(try compare(name, args[0], args[1]))
        case "len":
            try arity(name, args, 1)
            return .int(goLen(args[0]))
        case "index":
            try minArity(name, args, 2)
            return try indexInto(args[0], Array(args.dropFirst()))
        case "slice":
            return try sliceValue(args)
        case "print":
            return .string(args.map { $0.asOutput }.joined())
        case "println":
            return .string(args.map { $0.asOutput }.joined(separator: " ") + "\n")
        case "printf":
            return .string(try goPrintf(args))
        case "json":
            try arity(name, args, 1)
            return .string(try args[0].jsonString())
        case "default":
            // default <fallback> <value> : value if truthy else fallback
            try arity(name, args, 2)
            return args[1].isTruthy ? args[1] : args[0]
        default:
            throw GoTemplateError.eval("unknown function '\(name)'")
        }
    }

    private func arity(_ n: String, _ a: [GoValue], _ k: Int) throws {
        guard a.count == k else {
            throw GoTemplateError.eval("\(n): expected \(k) args, got \(a.count)")
        }
    }
    private func minArity(_ n: String, _ a: [GoValue], _ k: Int) throws {
        guard a.count >= k else {
            throw GoTemplateError.eval("\(n): expected >=\(k) args, got \(a.count)")
        }
    }

    private func compare(_ op: String, _ a: GoValue, _ b: GoValue) throws -> Bool {
        // numeric or string comparison
        if let x = numeric(a), let y = numeric(b) {
            switch op { case "lt": return x < y; case "le": return x <= y
                        case "gt": return x > y; default: return x >= y }
        }
        if case .string(let x) = a, case .string(let y) = b {
            switch op { case "lt": return x < y; case "le": return x <= y
                        case "gt": return x > y; default: return x >= y }
        }
        throw GoTemplateError.eval("\(op): incomparable operands")
    }

    private func numeric(_ v: GoValue) -> Double? {
        switch v { case .int(let i): return Double(i); case .double(let d): return d
                   default: return nil }
    }

    private func goEqual(_ a: GoValue, _ b: GoValue) -> Bool {
        if let x = numeric(a), let y = numeric(b) { return x == y }
        return a == b
    }

    private func goLen(_ v: GoValue) -> Int {
        switch v {
        case .string(let s): return s.count
        case .list(let l): return l.count
        case .dict(let d): return d.count
        default: return 0
        }
    }

    private func indexInto(_ base: GoValue, _ keys: [GoValue]) throws -> GoValue {
        var cur = base
        for k in keys {
            switch cur {
            case .list(let l):
                guard case .int(let i) = k, i >= 0, i < l.count else { return .null }
                cur = l[i]
            case .dict(let d):
                guard case .string(let s) = k else { return .null }
                cur = d[s] ?? .null
            default:
                return .null
            }
        }
        return cur
    }

    private func sliceValue(_ args: [GoValue]) throws -> GoValue {
        guard let first = args.first else {
            throw GoTemplateError.eval("slice: missing operand")
        }
        let lo = args.count > 1 ? (intOf(args[1]) ?? 0) : 0
        switch first {
        case .list(let l):
            let hi = args.count > 2 ? (intOf(args[2]) ?? l.count) : l.count
            let a = max(0, min(lo, l.count)); let b = max(a, min(hi, l.count))
            return .list(Array(l[a..<b]))
        case .string(let s):
            let chars = Array(s)
            let hi = args.count > 2 ? (intOf(args[2]) ?? chars.count) : chars.count
            let a = max(0, min(lo, chars.count)); let b = max(a, min(hi, chars.count))
            return .string(String(chars[a..<b]))
        default:
            throw GoTemplateError.eval("slice: operand is not a list or string")
        }
    }

    private func intOf(_ v: GoValue) -> Int? {
        if case .int(let i) = v { return i }
        if case .double(let d) = v { return Int(d) }
        return nil
    }

    /// A pragmatic `printf`: supports %s %d %v %q %f %% (the verbs Ollama
    /// TEMPLATEs use). Unknown verbs pass through literally.
    private func goPrintf(_ args: [GoValue]) throws -> String {
        guard case .string(let fmt)? = args.first else {
            throw GoTemplateError.eval("printf: first arg must be a format string")
        }
        var out = ""
        var argi = 1
        let f = Array(fmt)
        var k = 0
        while k < f.count {
            if f[k] == "%" && k + 1 < f.count {
                let verb = f[k + 1]
                k += 2
                if verb == "%" { out += "%"; continue }
                let a = argi < args.count ? args[argi] : .null
                argi += 1
                switch verb {
                case "s", "v": out += a.asOutput
                case "d": out += String(intOf(a) ?? 0)
                case "f": out += String(numeric(a) ?? 0)
                case "q": out += "\"" + a.asOutput + "\""
                default: out += "%" + String(verb)
                }
            } else {
                out.append(f[k]); k += 1
            }
        }
        return out
    }
}
