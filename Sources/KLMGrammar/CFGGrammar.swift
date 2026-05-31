import Foundation

/// Stage D: a context-free-grammar constraint automaton (parity plan §8).
///
/// `CFGGrammar.compile(_:)` parses a regex-flavored grammar with named rules
/// (see `CFGCompiler.swift`) into a set of BNF productions, and the automaton
/// conforms to `GrammarAutomaton` by running a **character-level Earley
/// recognizer**: a `State` is the Earley chart (one column of dotted items per
/// character consumed so far), and `step` advances it by one character,
/// returning `nil` when no derivation can extend the prefix. Earley is the
/// standard correct choice for arbitrary CFGs — it handles left-recursion,
/// ambiguity, and nullable rules — and, unlike a regular grammar (Stage C), it
/// constrains **unbounded balanced nesting** (`(((…)))`, recursive expressions).
///
/// The grammar is matched as a FULL parse (the start symbol must span the whole
/// output), so EOS is allowed only when the start symbol is complete, and any
/// character that cannot extend toward a complete parse is forbidden.
///
/// PERFORMANCE CAVEAT (why this is opt-in): an Earley chart is near-unique per
/// prefix, so the shared `GrammarTokenMask` per-state cache mostly misses and
/// each newly-reached state triggers a full-vocab rescan. This is correct but
/// markedly slower than the JSON / schema / regex masks (whose states recur).
/// Use it for short, structurally-constrained outputs.
///
/// Every terminal matches exactly one character (a `RegexGrammar.Matcher`,
/// reused from Stage C); multi-character structure is expressed grammatically.
/// The grammar is desugared to pure BNF at compile time, so the automaton here
/// only ever sees productions that are arrays of `.nonterminal` / `.terminal`
/// symbols.
public struct CFGGrammar: GrammarAutomaton {

    // MARK: Compiled grammar

    /// One symbol in a production body.
    enum Symbol: Hashable, Sendable {
        case nonterminal(Int)            // index into `prods`
        case terminal(RegexGrammar.Matcher)   // matches exactly one character
    }

    /// `prods[nt]` is the list of productions for nonterminal `nt`; each
    /// production is its ordered symbol body (empty ⇒ an epsilon production).
    let prods: [[[Symbol]]]
    /// The augmented start nonterminal `S'`, whose single production is
    /// `[.nonterminal(realStart)]`. A complete parse is `S' → realStart •` at
    /// origin 0.
    let startNT: Int
    /// `nullable[nt]` ⇒ nonterminal `nt` can derive the empty string. Used to
    /// advance past a nullable symbol during prediction (the Aycock–Horspool
    /// fix), which is what makes same-column completion of nullable rules
    /// correct.
    let nullable: [Bool]

    init(prods: [[[Symbol]]], startNT: Int) {
        self.prods = prods
        self.startNT = startNT
        self.nullable = Self.computeNullable(prods)
    }

    /// Fixpoint over the productions: a nonterminal is nullable if any of its
    /// productions has every symbol nullable (terminals are never nullable, as
    /// each consumes exactly one character; an empty production is nullable).
    private static func computeNullable(_ prods: [[[Symbol]]]) -> [Bool] {
        var nullable = [Bool](repeating: false, count: prods.count)
        var changed = true
        while changed {
            changed = false
            for nt in prods.indices where !nullable[nt] {
                for body in prods[nt] {
                    let allNull = body.allSatisfy {
                        if case .nonterminal(let b) = $0, b >= 0, b < nullable.count {
                            return nullable[b]
                        }
                        return false   // terminal ⇒ not nullable
                    }
                    if allNull { nullable[nt] = true; changed = true; break }
                }
            }
        }
        return nullable
    }

    // MARK: State (Earley chart)

    /// A dotted Earley item: production `prods[nt][prod]` with the dot before
    /// symbol index `dot`, begun in column `origin`.
    struct Item: Hashable, Sendable {
        var nt: Int
        var prod: Int
        var dot: Int
        var origin: Int
    }

    public struct State: Hashable, Sendable {
        /// One set of items per character consumed (column 0 = before any
        /// character). The current column is the last.
        var columns: [Set<Item>]
    }

    public var initialState: State {
        var columns: [Set<Item>] = [[Item(nt: startNT, prod: 0, dot: 0, origin: 0)]]
        closeColumn(&columns, 0)
        return State(columns: columns)
    }

    public func isComplete(_ s: State) -> Bool {
        guard let last = s.columns.last else { return false }
        // S' → realStart • at origin 0 (the augmented start production has one
        // symbol, so dot == 1 means the whole start symbol has been parsed).
        return last.contains(Item(nt: startNT, prod: 0, dot: 1, origin: 0))
    }

    public func step(_ s: State, _ c: Character) -> State? {
        let k = s.columns.count - 1
        guard k >= 0 else { return nil }

        // Scan: advance every item whose dot is before a terminal matching `c`
        // into the seeds of the next column.
        var newCol = Set<Item>()
        for item in s.columns[k] {
            let body = bodyOf(item)
            if item.dot < body.count,
               case .terminal(let m) = body[item.dot], m.matches(c) {
                newCol.insert(Item(nt: item.nt, prod: item.prod,
                                   dot: item.dot + 1, origin: item.origin))
            }
        }
        if newCol.isEmpty { return nil }   // dead: no derivation extends the prefix

        var columns = s.columns
        columns.append(newCol)
        closeColumn(&columns, k + 1)
        return State(columns: columns)
    }

    // MARK: - Earley closure (predict + complete to a fixpoint)

    /// Run prediction and completion on column `k` until no new item is added.
    /// Both operations only ever insert into column `k`; completion reads the
    /// (earlier, immutable) origin columns. Nullable nonterminals are advanced
    /// during prediction so a rule referencing a nullable symbol is handled
    /// regardless of item discovery order within the column.
    private func closeColumn(_ columns: inout [Set<Item>], _ k: Int) {
        guard k >= 0, k < columns.count else { return }
        var work = Array(columns[k])
        var i = 0
        while i < work.count {
            let item = work[i]; i += 1
            let body = bodyOf(item)

            if item.dot < body.count {
                guard case .nonterminal(let b) = body[item.dot], b >= 0, b < prods.count else {
                    continue   // terminal: consumed by `step`, not here
                }
                // Predict: add B → • γ at column k for every production of B.
                for p in prods[b].indices {
                    let predicted = Item(nt: b, prod: p, dot: 0, origin: k)
                    if columns[k].insert(predicted).inserted { work.append(predicted) }
                }
                // Aycock–Horspool: if B is nullable, also advance past it now.
                if b < nullable.count, nullable[b] {
                    let advanced = Item(nt: item.nt, prod: item.prod,
                                        dot: item.dot + 1, origin: item.origin)
                    if columns[k].insert(advanced).inserted { work.append(advanced) }
                }
            } else {
                // Complete: item.nt finished; advance every waiting item in the
                // origin column whose dot is before item.nt.
                guard item.origin >= 0, item.origin < columns.count else { continue }
                for waiting in columns[item.origin] {
                    let wbody = bodyOf(waiting)
                    if waiting.dot < wbody.count,
                       case .nonterminal(let b) = wbody[waiting.dot], b == item.nt {
                        let advanced = Item(nt: waiting.nt, prod: waiting.prod,
                                            dot: waiting.dot + 1, origin: waiting.origin)
                        if columns[k].insert(advanced).inserted { work.append(advanced) }
                    }
                }
            }
        }
    }

    /// The symbol body for an item's production, or an empty body if the item's
    /// indices are somehow out of range (defensive: productions come from a
    /// request-supplied grammar, though the compiler builds them well-formed).
    private func bodyOf(_ item: Item) -> [Symbol] {
        guard item.nt >= 0, item.nt < prods.count,
              item.prod >= 0, item.prod < prods[item.nt].count else { return [] }
        return prods[item.nt][item.prod]
    }
}
