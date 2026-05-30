import Foundation

/// Parses the bounded regex dialect documented on `RegexGrammar` and builds a
/// Thompson NFA. Total: any parse error (unbalanced groups, bad class, an
/// out-of-bounds counted repetition, an unsupported construct) returns `nil`,
/// and the caller falls back to the unconstrained path.
public extension RegexGrammar {

    /// Upper bound on the repetition count expanded for `{n}` / `{n,m}` /
    /// `x{n,}` style quantifiers, to keep the NFA size bounded.
    static let maxRepeat = 1000

    static func compile(_ pattern: String) -> RegexGrammar? {
        var parser = Parser(pattern)
        guard let frag = parser.parseAlternation(), parser.atEnd else { return nil }
        // Terminate the top fragment into a dedicated accept node.
        var b = parser.builder
        let accept = b.addAccept()
        b.patch(frag.outs, to: accept)
        return RegexGrammar(nodes: b.nodes, start: frag.start, accept: accept)
    }

    // MARK: - NFA builder

    /// Dangling outgoing edges of a fragment, recorded as (node, isEpsilonSlot)
    /// so they can be patched to a follow-on node. For a char node the slot is
    /// its `out`; for a split node the slot is one of its `eps` entries.
    fileprivate enum Dangle {
        case charOut(Int)            // node's `out`
        case eps(node: Int, index: Int)
    }

    fileprivate struct Fragment {
        var start: Int
        var outs: [Dangle]
        /// The matcher when this fragment is a SINGLE bare character atom
        /// (literal, `.`, class, or class-escape) — the only shape that counted
        /// repetition `{n,m}` can clone. `nil` for groups, alternations, and
        /// already-quantified fragments, so `x{2}` on a group is rejected.
        var singleChar: RegexGrammar.Matcher? = nil
    }

    fileprivate struct Builder {
        var nodes: [RegexGrammar.Node] = []

        mutating func addChar(_ m: RegexGrammar.Matcher) -> Int {
            nodes.append(.init(matcher: m, out: -1, eps: []))
            return nodes.count - 1
        }
        mutating func addSplit(_ a: Int? = nil, _ b: Int? = nil) -> Int {
            var eps: [Int] = []
            if let a { eps.append(a) }
            if let b { eps.append(b) }
            nodes.append(.init(matcher: nil, out: -1, eps: eps))
            return nodes.count - 1
        }
        mutating func addAccept() -> Int {
            nodes.append(.init(matcher: nil, out: -1, eps: []))
            return nodes.count - 1
        }
        mutating func patch(_ dangles: [Dangle], to target: Int) {
            for d in dangles {
                switch d {
                case .charOut(let n): nodes[n].out = target
                case .eps(let n, let i):
                    while nodes[n].eps.count <= i { nodes[n].eps.append(-1) }
                    nodes[n].eps[i] = target
                }
            }
        }
    }

    // MARK: - Recursive-descent parser

    fileprivate struct Parser {
        let chars: [Character]
        var pos = 0
        var builder = Builder()

        init(_ s: String) { chars = Array(s) }

        var atEnd: Bool { pos >= chars.count }
        func peek() -> Character? { pos < chars.count ? chars[pos] : nil }
        mutating func next() -> Character? {
            guard pos < chars.count else { return nil }
            defer { pos += 1 }
            return chars[pos]
        }
        mutating func eat(_ c: Character) -> Bool {
            if peek() == c { pos += 1; return true }
            return false
        }

        // alternation := concat ('|' concat)*
        mutating func parseAlternation() -> Fragment? {
            guard var left = parseConcat() else { return nil }
            while eat("|") {
                guard let right = parseConcat() else { return nil }
                let split = builder.addSplit(left.start, right.start)
                left = Fragment(start: split, outs: left.outs + right.outs)
            }
            return left
        }

        // concat := repeat*
        mutating func parseConcat() -> Fragment? {
            var frag: Fragment? = nil
            while let c = peek(), c != "|", c != ")" {
                guard let piece = parseRepeat() else { return nil }
                if var f = frag {
                    builder.patch(f.outs, to: piece.start)
                    f.outs = piece.outs
                    frag = f
                } else {
                    frag = piece
                }
            }
            // An empty concat (e.g. `a|`, `()`) is a pass-through epsilon node.
            if frag == nil {
                let s = builder.addSplit()
                return Fragment(start: s, outs: [.eps(node: s, index: 0)])
            }
            return frag
        }

        // repeat := atom ('*' | '+' | '?' | '{n}' | '{n,}' | '{n,m}')?
        mutating func parseRepeat() -> Fragment? {
            guard let atom = parseAtom() else { return nil }
            guard let q = peek() else { return atom }
            switch q {
            case "*": pos += 1; return star(atom, greedyOptional: true)
            case "+": pos += 1; return plus(atom)
            case "?": pos += 1; return optional(atom)
            case "{": return parseCounted(atom)
            default: return atom
            }
        }

        // atom := group | class | escape | dot | literal
        mutating func parseAtom() -> Fragment? {
            guard let c = peek() else { return nil }
            switch c {
            case "(":
                pos += 1
                guard let inner = parseAlternation(), eat(")") else { return nil }
                // A group is NOT a single bare char, even if it wraps one, so
                // `(ab){2}` and `(a|b){2}` are rejected by parseCounted.
                return inner
            case "[":
                return parseClass()
            case "\\":
                guard let m = parseEscape() else { return nil }
                let n = builder.addChar(m)
                return Fragment(start: n, outs: [.charOut(n)], singleChar: m)
            case ".":
                pos += 1
                let n = builder.addChar(.any)
                return Fragment(start: n, outs: [.charOut(n)], singleChar: .any)
            case ")", "|", "*", "+", "?", "{", "}", "]":
                // A bare quantifier / close with nothing to bind to is invalid.
                return nil
            default:
                pos += 1
                let n = builder.addChar(.literal(c))
                return Fragment(start: n, outs: [.charOut(n)], singleChar: .literal(c))
            }
        }

        // MARK: quantifier constructors

        mutating func star(_ f: Fragment, greedyOptional: Bool) -> Fragment {
            let split = builder.addSplit(f.start, nil)   // eps[0]=body, eps[1]=skip
            builder.patch(f.outs, to: split)             // loop back
            return Fragment(start: split, outs: [.eps(node: split, index: 1)])
        }
        mutating func plus(_ f: Fragment) -> Fragment {
            let split = builder.addSplit(f.start, nil)
            builder.patch(f.outs, to: split)
            return Fragment(start: f.start, outs: [.eps(node: split, index: 1)])
        }
        mutating func optional(_ f: Fragment) -> Fragment {
            let split = builder.addSplit(f.start, nil)
            return Fragment(start: split, outs: f.outs + [.eps(node: split, index: 1)])
        }

        // {n} {n,} {n,m} — counted repetition. Each copy needs a distinct NFA
        // fragment, so we clone the preceding atom's single-character matcher.
        // Supported only for a single-character atom (one node, one char edge)
        // — the overwhelmingly common case, e.g. `\d{3}`, `[a-z]{2,4}`. Counted
        // repetition on a group is unsupported (returns nil → fallback).
        mutating func parseCounted(_ atom: Fragment) -> Fragment? {
            // Only a single bare character atom can be cloned for `{n,m}`.
            guard let matcher = atom.singleChar else { return nil }
            pos += 1  // consume '{'
            guard let lo = parseInt(), lo >= 0, lo <= Self.maxRepeatLimit else { return nil }
            var hi: Int? = lo
            if eat(",") { hi = parseInt() }    // nil ⇒ unbounded {n,}
            guard eat("}") else { return nil }
            if let h = hi, (h < lo || h > Self.maxRepeatLimit) { return nil }

            func makeChar() -> Fragment {
                let n = builder.addChar(matcher)
                return Fragment(start: n, outs: [.charOut(n)])
            }
            func chain(_ frags: [Fragment]) -> Fragment {
                var acc = frags[0]
                for f in frags.dropFirst() {
                    builder.patch(acc.outs, to: f.start)
                    acc = Fragment(start: acc.start, outs: f.outs)
                }
                return acc
            }

            // `lo` required copies (reuse the already-built `atom` as the first).
            var parts: [Fragment] = []
            if lo >= 1 {
                parts.append(atom)
                for _ in 1 ..< lo { parts.append(makeChar()) }
            }
            if let hi {
                // Plus (hi - lo) optional copies.
                for _ in 0 ..< (hi - lo) { parts.append(optional(makeChar())) }
                if parts.isEmpty {           // {0} ⇒ empty match
                    let s = builder.addSplit()
                    return Fragment(start: s, outs: [.eps(node: s, index: 0)])
                }
                return chain(parts)
            } else {
                // {n,} ⇒ the required copies then a star of one more copy. When
                // lo == 0 this is just a plain star over the atom.
                parts.append(star(makeChar(), greedyOptional: true))
                return chain(parts)
            }
        }

        static let maxRepeatLimit = RegexGrammar.maxRepeat

        mutating func parseInt() -> Int? {
            var digits = ""
            while let c = peek(), c.isNumber { digits.append(c); pos += 1 }
            return digits.isEmpty ? nil : Int(digits)
        }

        // MARK: classes and escapes

        mutating func parseClass() -> Fragment? {
            guard eat("[") else { return nil }
            let negated = eat("^")
            var chars = Set<Character>()
            var ranges: [ClosedRange<Character>] = []
            var classMatchers: [RegexGrammar.Matcher] = []  // \d \w \s inside []
            var first = true
            while let c = peek(), c != "]" || first {
                first = false
                if c == "\\" {
                    pos += 1
                    guard let esc = peek() else { return nil }
                    pos += 1
                    if let m = Self.classEscape(esc) {
                        classMatchers.append(m)
                    } else if let lit = Self.literalEscape(esc) {
                        // Possible range start: lit '-' x
                        if peek() == "-", pos + 1 < chars.count, chars[pos + 1] != "]" {
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
                // Literal char, possibly a range `a-z`.
                pos += 1
                if peek() == "-", pos + 1 < self.chars.count, self.chars[pos + 1] != "]" {
                    pos += 1  // consume '-'
                    guard let hi = readClassChar() else { return nil }
                    guard c <= hi else { return nil }
                    ranges.append(c ... hi)
                } else {
                    chars.insert(c)
                }
            }
            guard eat("]") else { return nil }
            if chars.isEmpty && ranges.isEmpty && classMatchers.isEmpty { return nil }

            // If there are embedded class escapes (\d etc.), fold everything
            // into a custom matcher via a split over alternatives.
            if classMatchers.isEmpty {
                let m = RegexGrammar.Matcher.set(chars: chars, ranges: ranges, negated: negated)
                let n = builder.addChar(m)
                return Fragment(start: n, outs: [.charOut(n)], singleChar: m)
            }
            // Build an alternation node: set-part | each class matcher. Negation
            // with embedded class escapes is uncommon and ambiguous; reject it.
            if negated { return nil }
            var alts: [Fragment] = []
            if !chars.isEmpty || !ranges.isEmpty {
                let n = builder.addChar(.set(chars: chars, ranges: ranges, negated: false))
                alts.append(Fragment(start: n, outs: [.charOut(n)]))
            }
            for m in classMatchers {
                let n = builder.addChar(m)
                alts.append(Fragment(start: n, outs: [.charOut(n)]))
            }
            guard var acc = alts.first else { return nil }
            for f in alts.dropFirst() {
                let split = builder.addSplit(acc.start, f.start)
                acc = Fragment(start: split, outs: acc.outs + f.outs)
            }
            return acc
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

        mutating func parseEscape() -> RegexGrammar.Matcher? {
            guard eat("\\"), let e = next() else { return nil }
            if let m = Self.classEscape(e) { return m }
            if let lit = Self.literalEscape(e) { return .literal(lit) }
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
                 "|", "\\", "/", "-", "^", "$":
                return e
            default:
                return nil
            }
        }
    }
}
