import Foundation

/// Stage A automaton: any JSON value. Adapts the character-level
/// `JSONGrammar` static functions to the `GrammarAutomaton` protocol so the
/// generic `GrammarTokenMask` masking layer can drive it.
public struct JSONValueAutomaton: GrammarAutomaton {
    public init() {}
    public var initialState: JSONGrammar.State { JSONGrammar.initialState }
    public func step(_ s: JSONGrammar.State, _ c: Character) -> JSONGrammar.State? {
        JSONGrammar.step(s, c)
    }
    public func isComplete(_ s: JSONGrammar.State) -> Bool {
        JSONGrammar.isComplete(s)
    }
}

/// The Stage A `format:"json"` mask: a token-level mask over the any-JSON-value
/// grammar. Now a thin specialization of the shared `GrammarTokenMask`.
public typealias JSONTokenMask = GrammarTokenMask<JSONValueAutomaton>

public extension GrammarTokenMask where A == JSONValueAutomaton {
    /// Build the any-JSON-value mask from a token-piece table and stop set.
    convenience init(pieces: [String], stopIds: Set<Int>) {
        self.init(automaton: JSONValueAutomaton(), pieces: pieces, stopIds: stopIds)
    }
}
