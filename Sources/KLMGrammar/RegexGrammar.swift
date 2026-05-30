import Foundation

/// Stage C: a regular-grammar (regex) constraint automaton (parity plan §8).
///
/// `RegexGrammar.compile(_:)` parses a bounded regex dialect into a Thompson
/// NFA, and the automaton conforms to `GrammarAutomaton` by subset
/// construction (a `State` is the set of NFA states reachable for the text so
/// far). Because the state is a compact `Set<Int>` that recurs across
/// generations, the shared `GrammarTokenMask` caching is effective and each
/// step is cheap — no per-prefix blow-up.
///
/// The pattern is matched as a FULL match (implicitly anchored at both ends):
/// the output must be exactly a string in the language, so EOS is allowed only
/// when an accepting NFA state is in the current set, and any character that
/// cannot extend toward an accepting string is forbidden.
///
/// Supported dialect:
/// - literals (any non-metacharacter), `.` (any char except newline)
/// - escapes: `\d \D \w \W \s \S` and escaped metachars `\. \* \+ \? \( \)
///   \[ \] \{ \} \| \\ \/ \- \^ \$` plus `\n \t \r`
/// - character classes `[...]`, negated `[^...]`, ranges `a-z`, class escapes
/// - groups `( ... )`, alternation `|`
/// - quantifiers `*` `+` `?` and counted `{n}` `{n,}` `{n,m}` (m bounded)
///
/// Deliberately unsupported (compile returns nil → caller falls back to the
/// unconstrained path): backreferences, lookaround, named groups, anchors
/// `^`/`$` as literals-in-pattern (the whole pattern is already anchored),
/// non-greedy `*?` (greediness is irrelevant to a recognizer), and unicode
/// property classes. Counted repetition is expanded, so `{n,m}` with a large
/// `m` is rejected to bound NFA size.
public struct RegexGrammar: GrammarAutomaton {

    // MARK: NFA

    /// A single-character matcher on an NFA edge.
    enum Matcher: Hashable, Sendable {
        case literal(Character)
        case any                      // '.' — any char except '\n'
        case set(chars: Set<Character>, ranges: [ClosedRange<Character>], negated: Bool)
        case digit, notDigit          // \d \D
        case word, notWord            // \w \W
        case space, notSpace          // \s \S

        func matches(_ c: Character) -> Bool {
            switch self {
            case .literal(let l): return c == l
            case .any: return c != "\n"
            case .set(let chars, let ranges, let negated):
                let hit = chars.contains(c) || ranges.contains { $0.contains(c) }
                return negated ? !hit : hit
            case .digit: return c.isNumber && c.isASCII
            case .notDigit: return !(c.isNumber && c.isASCII)
            case .word: return c == "_" || ((c.isLetter || c.isNumber) && c.isASCII)
            case .notWord: return !(c == "_" || ((c.isLetter || c.isNumber) && c.isASCII))
            case .space: return c == " " || c == "\t" || c == "\n" || c == "\r"
            case .notSpace: return !(c == " " || c == "\t" || c == "\n" || c == "\r")
            }
        }
    }

    /// One NFA node. Either a character edge to `out`, or up to two epsilon
    /// edges (`eps`). Accept nodes have no outgoing edges.
    struct Node: Sendable {
        var matcher: Matcher?     // non-nil ⇒ a single character edge to `out`
        var out: Int              // target for the character edge
        var eps: [Int]            // epsilon targets
    }

    let nodes: [Node]
    let start: Int
    let accept: Int

    // MARK: State (subset construction)

    public struct State: Hashable, Sendable {
        /// NFA nodes reachable for the text consumed so far (epsilon-closed).
        /// Empty ⇒ dead (no string can extend the prefix); `step` never
        /// returns a dead state (it returns nil instead).
        var set: Set<Int>
    }

    public var initialState: State {
        State(set: epsilonClosure([start]))
    }

    public func isComplete(_ s: State) -> Bool {
        s.set.contains(accept)
    }

    public func step(_ s: State, _ c: Character) -> State? {
        var next = Set<Int>()
        for n in s.set {
            let node = nodes[n]
            if let m = node.matcher, m.matches(c), node.out >= 0, node.out < nodes.count {
                next.insert(node.out)
            }
        }
        if next.isEmpty { return nil }
        let closed = epsilonClosure(Array(next))
        return closed.isEmpty ? nil : State(set: closed)
    }

    /// Epsilon-closure of `seeds`. Out-of-range targets are skipped
    /// defensively: the Thompson builder patches every dangling edge before
    /// `compile` returns, so a sentinel `-1` should never be reachable here,
    /// but since patterns come from request bodies we never index a node out
    /// of range rather than trust that invariant blindly.
    private func epsilonClosure(_ seeds: [Int]) -> Set<Int> {
        var result = Set<Int>()
        var stack = seeds.filter { $0 >= 0 && $0 < nodes.count }
        while let n = stack.popLast() {
            guard result.insert(n).inserted else { continue }
            for e in nodes[n].eps where e >= 0 && e < nodes.count { stack.append(e) }
        }
        return result
    }
}
