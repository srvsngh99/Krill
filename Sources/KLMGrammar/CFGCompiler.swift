import Foundation

/// Parses the regex-flavored, named-rule grammar dialect documented below and
/// lowers it to the pure-BNF productions a `CFGGrammar` Earley recognizer runs
/// on. Total: any error (syntax, an undefined nonterminal reference, an empty
/// grammar, or an oversized one) returns `nil`, and the caller falls back to
/// the unconstrained path.
///
/// Dialect (a superset of the Stage C regex dialect, plus named recursion):
/// - A grammar is a sequence of rules `name: body`. `name` is an identifier
///   `[A-Za-z_][A-Za-z0-9_]*`. Comments run from `//` to end of line.
/// - The start symbol is the rule named `start` if present, else the first
///   rule defined.
/// - A `body` is an alternation (`|`) of sequences; each sequence is a
///   whitespace-separated list of items; each item is an atom with an optional
///   postfix quantifier `?` `*` `+`.
/// - An atom is one of:
///   - a nonterminal reference: a bare identifier naming another rule
///     (this is what enables recursion / unbounded nesting);
///   - a string literal `"..."` (escapes `\n \t \r \" \\`; other `\x` → `x`),
///     matched as the literal text;
///   - a character class `[...]` / `[^...]` with ranges and class escapes,
///     matching exactly ONE character;
///   - `.` (any char except `\n`) or an escape `\d \D \w \W \s \S` / escaped
///     metachar, each matching exactly ONE character;
///   - a group `( ... )` (a parenthesized sub-alternation).
///
/// Every terminal matches a single character (a `RegexGrammar.Matcher`); the
/// compiler desugars string literals, groups, and the `? * +` quantifiers into
/// anonymous nonterminals so the recognizer only ever sees pure BNF.
public extension CFGGrammar {

    /// Upper bound on the number of nonterminals (named + anonymous) a grammar
    /// may expand to, and on a single string literal's length — to keep a
    /// request-supplied grammar's compiled size bounded.
    static let maxNonterminals = 50_000
    static let maxLiteralLength = 4_096

    static func compile(_ text: String) -> CFGGrammar? {
        var parser = Parser(text)
        guard let rules = parser.parseGrammar(), parser.atEnd, !rules.isEmpty else {
            return nil
        }

        // Name → nonterminal index (named rules occupy 0 ..< count, in order).
        var ruleIndex: [String: Int] = [:]
        for (i, rule) in rules.enumerated() {
            if ruleIndex[rule.name] != nil { return nil }   // duplicate rule name
            ruleIndex[rule.name] = i
        }
        // Start symbol: `start` if declared, else the first rule.
        let startName = ruleIndex["start"] != nil ? "start" : rules[0].name
        guard let realStart = ruleIndex[startName] else { return nil }

        let lowerer = Lowerer(ruleIndex: ruleIndex, namedCount: rules.count)
        do {
            for rule in rules {
                guard let nt = ruleIndex[rule.name] else { return nil }
                lowerer.prods[nt] = try lowerer.lowerExpr(rule.body)
            }
            // Augmented start S' → realStart.
            let sprime = try lowerer.addNonterminal()
            lowerer.prods[sprime] = [[.nonterminal(realStart)]]
            return CFGGrammar(prods: lowerer.prods, startNT: sprime)
        } catch {
            return nil
        }
    }

    // MARK: - Grammar AST

    fileprivate enum Quant { case one, opt, star, plus }

    fileprivate indirect enum Atom {
        case ref(String)
        case lit(String)
        case matcher(RegexGrammar.Matcher)         // one char
        case classAlt([RegexGrammar.Matcher])      // one char matching any alternative
        case group([[Quantified]])                 // a parenthesized Expr
    }

    fileprivate struct Quantified { let atom: Atom; let quant: Quant }
    fileprivate typealias Seq = [Quantified]
    fileprivate typealias Expr = [Seq]
    fileprivate struct Rule { let name: String; let body: Expr }

    // MARK: - EBNF → BNF lowering

    fileprivate struct CFGCompileError: Error {}

    fileprivate final class Lowerer {
        var prods: [[[Symbol]]]
        let ruleIndex: [String: Int]

        init(ruleIndex: [String: Int], namedCount: Int) {
            self.ruleIndex = ruleIndex
            self.prods = Array(repeating: [], count: namedCount)
        }

        func addNonterminal() throws -> Int {
            guard prods.count < CFGGrammar.maxNonterminals else { throw CFGCompileError() }
            prods.append([])
            return prods.count - 1
        }

        func lowerExpr(_ e: Expr) throws -> [[Symbol]] {
            try e.map { try lowerSeq($0) }
        }

        func lowerSeq(_ s: Seq) throws -> [Symbol] {
            var out: [Symbol] = []
            for q in s { out.append(contentsOf: try lowerQuantified(q)) }
            return out
        }

        func lowerQuantified(_ q: Quantified) throws -> [Symbol] {
            switch q.quant {
            case .one:
                return try lowerAtom(q.atom)
            case .opt:
                let u = try unit(q.atom); let n = try addNonterminal()
                prods[n] = [[u], []]                      // Q → u | ε
                return [.nonterminal(n)]
            case .star:
                let u = try unit(q.atom); let n = try addNonterminal()
                prods[n] = [[], [.nonterminal(n), u]]     // R → ε | R u
                return [.nonterminal(n)]
            case .plus:
                let u = try unit(q.atom); let n = try addNonterminal()
                prods[n] = [[u], [.nonterminal(n), u]]    // P → u | P u
                return [.nonterminal(n)]
            }
        }

        func lowerAtom(_ a: Atom) throws -> [Symbol] {
            switch a {
            case .ref(let name):
                guard let idx = ruleIndex[name] else { throw CFGCompileError() }
                return [.nonterminal(idx)]
            case .lit(let s):
                guard s.count <= CFGGrammar.maxLiteralLength else { throw CFGCompileError() }
                return s.map { .terminal(.literal($0)) }
            case .matcher(let m):
                return [.terminal(m)]
            case .classAlt(let ms):
                let n = try addNonterminal()
                prods[n] = ms.map { [.terminal($0)] }     // one char ∈ alternatives
                return [.nonterminal(n)]
            case .group(let e):
                let n = try addNonterminal()
                prods[n] = try lowerExpr(e)
                return [.nonterminal(n)]
            }
        }

        /// Collapse an atom to a single symbol so a quantifier can wrap it.
        /// A multi-symbol atom (a multi-char string literal) is boxed in an
        /// anonymous nonterminal with one production.
        func unit(_ a: Atom) throws -> Symbol {
            let syms = try lowerAtom(a)
            if syms.count == 1 { return syms[0] }
            let n = try addNonterminal()
            prods[n] = [syms]
            return .nonterminal(n)
        }
    }

    // MARK: - Recursive-descent parser

    fileprivate struct Parser {
        let chars: [Character]
        var pos = 0

        init(_ s: String) { chars = Array(s) }

        var atEnd: Bool { pos >= chars.count }
        func peek() -> Character? { pos < chars.count ? chars[pos] : nil }
        func peekAhead(_ n: Int) -> Character? {
            let i = pos + n
            return i < chars.count ? chars[i] : nil
        }
        mutating func eat(_ c: Character) -> Bool {
            if peek() == c { pos += 1; return true }
            return false
        }

        /// Skip whitespace and `// …` line comments.
        mutating func skipTrivia() {
            while let c = peek() {
                if c == " " || c == "\t" || c == "\n" || c == "\r" { pos += 1; continue }
                if c == "/", peekAhead(1) == "/" {
                    pos += 2
                    while let d = peek(), d != "\n" { pos += 1 }
                    continue
                }
                break
            }
        }

        mutating func parseIdentifier() -> String? {
            guard let c = peek(), c == "_" || c.isLetter else { return nil }
            var s = ""
            while let c = peek(), c == "_" || c.isLetter || c.isNumber {
                s.append(c); pos += 1
            }
            return s
        }

        /// Lookahead: are we positioned at the start of a `name:` rule header?
        /// Used to end a rule body when a nonterminal reference would otherwise
        /// swallow the next rule's name. Never consumes input.
        mutating func atRuleHeader() -> Bool {
            let save = pos
            defer { pos = save }
            guard parseIdentifier() != nil else { return false }
            skipTrivia()
            return peek() == ":"
        }

        // grammar := (rule)*
        mutating func parseGrammar() -> [Rule]? {
            var rules: [Rule] = []
            while true {
                skipTrivia()
                if peek() == nil { break }
                guard let name = parseIdentifier() else { return nil }
                skipTrivia()
                guard eat(":") else { return nil }
                guard let body = parseExpr() else { return nil }
                rules.append(Rule(name: name, body: body))
            }
            return rules
        }

        // expr := seq ('|' seq)*
        mutating func parseExpr() -> Expr? {
            guard let first = parseSeq() else { return nil }
            var alts: Expr = [first]
            while true {
                skipTrivia()
                if peek() == "|" {
                    pos += 1
                    guard let s = parseSeq() else { return nil }
                    alts.append(s)
                } else {
                    break
                }
            }
            return alts
        }

        // seq := quantified*   (stops at '|', ')', EOF, or the next rule header)
        mutating func parseSeq() -> Seq? {
            var items: Seq = []
            while true {
                skipTrivia()
                guard let c = peek() else { break }
                if c == "|" || c == ")" { break }
                if atRuleHeader() { break }
                guard let q = parseQuantified() else { return nil }
                items.append(q)
            }
            return items    // may be empty ⇒ an epsilon alternative
        }

        // quantified := atom ('?' | '*' | '+')?
        mutating func parseQuantified() -> Quantified? {
            guard let atom = parseAtom() else { return nil }
            switch peek() {
            case "?": pos += 1; return Quantified(atom: atom, quant: .opt)
            case "*": pos += 1; return Quantified(atom: atom, quant: .star)
            case "+": pos += 1; return Quantified(atom: atom, quant: .plus)
            default: return Quantified(atom: atom, quant: .one)
            }
        }

        // atom := string | class | '.' | escape | group | ref
        mutating func parseAtom() -> Atom? {
            guard let c = peek() else { return nil }
            switch c {
            case "\"":
                guard let s = parseStringLiteral() else { return nil }
                return .lit(s)
            case "[":
                return parseClass()
            case ".":
                pos += 1
                return .matcher(.any)
            case "\\":
                return parseEscape()
            case "(":
                pos += 1
                guard let e = parseExpr() else { return nil }
                skipTrivia()
                guard eat(")") else { return nil }
                return .group(e)
            default:
                if c == "_" || c.isLetter {
                    guard let id = parseIdentifier() else { return nil }
                    return .ref(id)
                }
                return nil   // a bare quantifier, ':' outside a header, etc.
            }
        }

        mutating func parseStringLiteral() -> String? {
            guard eat("\"") else { return nil }
            var s = ""
            while let c = peek() {
                pos += 1
                if c == "\"" { return s }
                if c == "\\" {
                    guard let e = peek() else { return nil }
                    pos += 1
                    switch e {
                    case "n": s.append("\n")
                    case "t": s.append("\t")
                    case "r": s.append("\r")
                    default: s.append(e)   // \" \\ and permissively any \x → x
                    }
                } else {
                    s.append(c)
                }
            }
            return nil   // unterminated literal
        }

        // MARK: classes and escapes (single-character matchers)

        mutating func parseClass() -> Atom? {
            guard eat("[") else { return nil }
            let negated = eat("^")
            var chars = Set<Character>()
            var ranges: [ClosedRange<Character>] = []
            var classEscapes: [RegexGrammar.Matcher] = []
            var first = true
            while let c = peek(), c != "]" || first {
                first = false
                if c == "\\" {
                    pos += 1
                    guard let e = peek() else { return nil }
                    pos += 1
                    if let m = Self.classEscape(e) {
                        classEscapes.append(m)
                    } else if let lit = Self.literalEscape(e) {
                        if peek() == "-", let after = peekAhead(1), after != "]" {
                            pos += 1
                            guard let hi = readClassChar() else { return nil }
                            guard lit <= hi else { return nil }
                            ranges.append(lit ... hi)
                        } else {
                            chars.insert(lit)
                        }
                    } else {
                        return nil
                    }
                    continue
                }
                pos += 1
                if peek() == "-", let after = peekAhead(1), after != "]" {
                    pos += 1
                    guard let hi = readClassChar() else { return nil }
                    guard c <= hi else { return nil }
                    ranges.append(c ... hi)
                } else {
                    chars.insert(c)
                }
            }
            guard eat("]") else { return nil }
            if chars.isEmpty && ranges.isEmpty && classEscapes.isEmpty { return nil }

            if classEscapes.isEmpty {
                return .matcher(.set(chars: chars, ranges: ranges, negated: negated))
            }
            // Embedded \d etc.: fold into a one-char alternation. Negation with
            // class escapes is ambiguous; reject it (mirrors RegexCompiler).
            if negated { return nil }
            var ms: [RegexGrammar.Matcher] = []
            if !chars.isEmpty || !ranges.isEmpty {
                ms.append(.set(chars: chars, ranges: ranges, negated: false))
            }
            ms.append(contentsOf: classEscapes)
            return .classAlt(ms)
        }

        /// Read one class character (literal or simple escape) for a range bound.
        mutating func readClassChar() -> Character? {
            guard let c = peek() else { return nil }
            if c == "\\" {
                pos += 1
                guard let e = peek() else { return nil }
                pos += 1
                return Self.literalEscape(e)
            }
            pos += 1
            return c
        }

        mutating func parseEscape() -> Atom? {
            guard eat("\\"), let e = peek() else { return nil }
            pos += 1
            if let m = Self.classEscape(e) { return .matcher(m) }
            if let lit = Self.literalEscape(e) { return .matcher(.literal(lit)) }
            return nil
        }

        static func classEscape(_ e: Character) -> RegexGrammar.Matcher? {
            switch e {
            case "d": return .digit
            case "D": return .notDigit
            case "w": return .word
            case "W": return .notWord
            case "s": return .space
            case "S": return .notSpace
            default: return nil
            }
        }

        static func literalEscape(_ e: Character) -> Character? {
            switch e {
            case "n": return "\n"
            case "t": return "\t"
            case "r": return "\r"
            case ".", "*", "+", "?", "(", ")", "[", "]", "{", "}",
                 "|", "\\", "/", "-", "^", "$", "\"", " ":
                return e
            default:
                return nil
            }
        }
    }
}
