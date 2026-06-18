import Foundation
import MLX

/// An incremental grammar runtime over characters: given a `State` and the
/// next character, `step` returns the new state or `nil` if the character
/// cannot extend the output into a valid prefix of the target language.
/// `JSONGrammar` (Stage A, any JSON value) and `SchemaGrammar` (Stage B,
/// schema-constrained) both conform, so the token-level mask layer below is
/// written once and reused.
public protocol GrammarAutomaton: Sendable {
    associatedtype State: Hashable & Sendable
    var initialState: State { get }
    func step(_ s: State, _ c: Character) -> State?
    /// Whether the output so far is a complete value — i.e. whether EOS may
    /// be emitted here.
    func isComplete(_ s: State) -> Bool
}

public extension GrammarAutomaton {
    /// Advance over every character of `piece`; `nil` if any char is
    /// rejected. An empty piece is a no-op.
    func advance(_ s: State, piece: String) -> State? {
        var cur = s
        for ch in piece {
            guard let next = step(cur, ch) else { return nil }
            cur = next
        }
        return cur
    }
}

/// Type-erased token-level logit mask the engine holds without knowing which
/// concrete automaton drives it. The per-generation cursor is a
/// `GrammarLogitSession`, so the shared (per-model, thread-safe) mask object
/// can serve many concurrent generations.
public protocol GrammarLogitMask: AnyObject, Sendable {
    /// Number of real token pieces the grammar reasons over (the tokenizer
    /// vocab). Used for piece indexing / advance bounds.
    var vocabSize: Int { get }
    /// Width of the emitted additive mask vector. Equals the model's logits
    /// width (`lm_head` output), which can be PADDED beyond `vocabSize` (e.g.
    /// Gemma 4: 262144 logits vs 261707 tokenizer pieces). The engine validates
    /// this against the runtime logits width; padding slots are blocked.
    var maskWidth: Int { get }
    var stopIdSet: Set<Int> { get }
    func makeSession() -> any GrammarLogitSession
}

/// Per-generation cursor over a `GrammarLogitMask`. Holds the current grammar
/// state; the engine asks for the current step's mask, then advances it by the
/// accepted token.
public protocol GrammarLogitSession: AnyObject {
    /// Additive `[vocab]` logit mask for the cursor's current state.
    func currentMask() -> MLXArray
    /// Advance by an accepted token. Returns `false` if the token's piece does
    /// not extend the grammar (the caller should then stop masking for the
    /// rest of the generation).
    func advance(token: Int) -> Bool
}

/// Large negative bias for forbidden logits, matching the `-1e9` convention
/// the sampler's top-k/top-p/min-p filters use.
private let grammarBlockedBias: Float = -1e9

/// Tokenizer-aware logit mask driven by a `GrammarAutomaton`. Generic over the
/// automaton so Stage A (`JSONTokenMask = GrammarTokenMask<JSONValueAutomaton>`)
/// and Stage B (`GrammarTokenMask<SchemaGrammar>`) share all the masking,
/// caching, and fail-open logic.
///
/// `mask(for:)` returns an additive `[vocab]` mask (0 for tokens that keep the
/// output a valid prefix, a large negative bias otherwise), cached per grammar
/// state. Building a mask for a new state scans the full vocab once; the
/// constrained path's absolute speed is secondary (the unconstrained path is
/// untouched).
public final class GrammarTokenMask<A: GrammarAutomaton>: GrammarLogitMask, @unchecked Sendable {
    public let automaton: A
    /// Decoded string piece for each token id, indexed by id.
    private let pieces: [String]
    /// Stop tokens (EOS / end-of-turn) - allowed only at a complete value.
    private let stopIds: [Int]
    public let vocabSize: Int
    /// Emitted mask width = the model's (possibly padded) logits width. When it
    /// exceeds `vocabSize`, the tail `[vocabSize, maskWidth)` are unused token
    /// slots and are always blocked.
    public let maskWidth: Int
    public let stopIdSet: Set<Int>

    private var cache: [A.State: MLXArray] = [:]
    private let lock = NSLock()
    /// One-shot guard for the all-blocked fail-open notice.
    private var didWarnAllBlocked = false

    /// - Parameter outputWidth: the model's logits width. Pass the loaded
    ///   model's `vocabSize` when it is padded beyond the tokenizer (Gemma 4).
    ///   Defaults to `pieces.count` (no padding). Values below `pieces.count`
    ///   are clamped up (the mask must cover every real token).
    public init(automaton: A, pieces: [String], stopIds: Set<Int>, outputWidth: Int? = nil) {
        self.automaton = automaton
        self.pieces = pieces
        self.vocabSize = pieces.count
        self.maskWidth = max(outputWidth ?? pieces.count, pieces.count)
        self.stopIds = Array(stopIds)
        self.stopIdSet = stopIds
    }

    public func makeSession() -> any GrammarLogitSession {
        GrammarTokenSession(mask: self, state: automaton.initialState)
    }

    /// Advance the grammar state by an accepted token. Returns `nil` if the
    /// token's piece is not a valid continuation.
    public func advance(_ state: A.State, token: Int) -> A.State? {
        guard token >= 0 && token < vocabSize else { return nil }
        return automaton.advance(state, piece: pieces[token])
    }

    /// Additive `[vocab]` float32 logit mask for `state`. Cached per state.
    public func mask(for state: A.State) -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        if let cached = cache[state] { return cached }

        let complete = automaton.isComplete(state)
        // Width = logits width; entries in [vocabSize, maskWidth) are padding
        // token slots and stay blocked (their initial value).
        var bias = [Float](repeating: grammarBlockedBias, count: maskWidth)
        var allowedCount = 0
        for id in 0 ..< vocabSize {
            let piece = pieces[id]
            // Empty pieces (typically special tokens that decode to "") make
            // no textual progress; forbid them. EOS is handled via stopIds.
            if piece.isEmpty { continue }
            if automaton.advance(state, piece: piece) != nil {
                bias[id] = 0
                allowedCount += 1
            }
        }
        // Stop tokens may fire only when the value is complete.
        for id in stopIds where id >= 0 && id < vocabSize {
            if complete {
                if bias[id] != 0 { allowedCount += 1 }
                bias[id] = 0
            } else {
                bias[id] = grammarBlockedBias
            }
        }

        // Fail open: if NO token is allowed for this state (e.g. a
        // lossy-detokenized vocab where no single piece faithfully extends the
        // prefix, or a stop set missing the model's real EOS at a non-complete
        // state), an all-blocked mask would force argMax onto a forbidden token
        // and wedge the decode. Return an all-allowed mask for this step and
        // let the unconstrained sampler proceed; post-extraction coerce remains
        // the validity net. Logged once so the condition is observable.
        if allowedCount == 0 {
            if !didWarnAllBlocked {
                didWarnAllBlocked = true
                FileHandle.standardError.write(Data((
                    "[Krill] grammar mask: no token extends the current state; "
                    + "failing open for this step (output validity then relies on "
                    + "post-extraction).\n").utf8))
            }
            // Fail open over the REAL tokens only; padding slots stay blocked.
            var openBias = [Float](repeating: 0, count: maskWidth)
            for id in vocabSize ..< maskWidth { openBias[id] = grammarBlockedBias }
            let open = MLXArray(openBias)
            cache[state] = open
            return open
        }

        let arr = MLXArray(bias)
        cache[state] = arr
        return arr
    }
}

/// Per-generation cursor for `GrammarTokenMask`.
private final class GrammarTokenSession<A: GrammarAutomaton>: GrammarLogitSession {
    private let mask: GrammarTokenMask<A>
    private var state: A.State

    init(mask: GrammarTokenMask<A>, state: A.State) {
        self.mask = mask
        self.state = state
    }

    func currentMask() -> MLXArray { mask.mask(for: state) }

    func advance(token: Int) -> Bool {
        guard let next = mask.advance(state, token: token) else { return false }
        state = next
        return true
    }
}
