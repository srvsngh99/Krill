import Foundation
import MLX

/// Tokenizer-aware logit mask for JSON-constrained decoding (parity plan
/// §8, Stage A). Wraps the character-level `JSONGrammar` automaton with a
/// token-piece table so the sampler can, per step, forbid every vocab
/// token that would break the output's validity as a JSON-value prefix.
///
/// `mask(for:)` returns an ADDITIVE `[vocab]` logit mask (0 for allowed
/// tokens, a large negative bias for disallowed ones) that the sampler
/// adds to the raw logits before any temperature / top-p / top-k / greedy
/// step — so a forbidden token can never win. Masks are cached per
/// `JSONGrammar.State`; JSON has a small, repeating set of states, so the
/// per-step cost amortizes to a dictionary lookup after each state is
/// first seen. (Building a mask for a new state scans the full vocab
/// once; that one-time cost is acceptable for the constrained path, whose
/// absolute speed is secondary — the unconstrained path is untouched.)
public final class JSONTokenMask: @unchecked Sendable {
    /// Decoded string piece for each token id, indexed by id.
    private let pieces: [String]
    /// Stop tokens (EOS / end-of-turn) — allowed only at a complete value.
    private let stopIds: [Int]
    public let vocabSize: Int

    /// Large negative bias applied to forbidden logits, matching the
    /// `-1e9` convention used by the sampler's top-k/top-p/min-p filters.
    private static let blocked: Float = -1e9

    private var cache: [JSONGrammar.State: MLXArray] = [:]
    private let lock = NSLock()
    /// One-shot guard for the all-blocked fail-open notice.
    private var didWarnAllBlocked = false

    /// The stop-token set this mask was built with, exposed so the engine can
    /// detect when a later request's stop set differs and rebuild the mask
    /// rather than silently reuse a stale one.
    public let stopIdSet: Set<Int>

    public init(pieces: [String], stopIds: Set<Int>) {
        self.pieces = pieces
        self.vocabSize = pieces.count
        self.stopIds = Array(stopIds)
        self.stopIdSet = stopIds
    }

    /// Advance the grammar state by an accepted token. Returns `nil` if the
    /// token's piece is not a valid continuation (should not happen for a
    /// token the mask allowed; callers keep the prior state defensively).
    public func advance(_ state: JSONGrammar.State, token: Int) -> JSONGrammar.State? {
        guard token >= 0 && token < vocabSize else { return nil }
        return JSONGrammar.advance(state, piece: pieces[token])
    }

    /// Additive `[vocab]` float32 logit mask for `state`: `0` for tokens
    /// that keep the output a valid JSON-value prefix, a large negative
    /// bias otherwise. Cached per state.
    public func mask(for state: JSONGrammar.State) -> MLXArray {
        lock.lock(); defer { lock.unlock() }
        if let cached = cache[state] { return cached }

        let complete = JSONGrammar.isComplete(state)
        var bias = [Float](repeating: Self.blocked, count: vocabSize)
        var allowedCount = 0
        for id in 0 ..< vocabSize {
            let piece = pieces[id]
            // Empty pieces (typically special tokens that decode to "")
            // are forbidden: emitting one makes no textual progress and
            // could stall the decode. EOS is handled via `stopIds` below.
            if piece.isEmpty { continue }
            if JSONGrammar.advance(state, piece: piece) != nil {
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
                bias[id] = Self.blocked
            }
        }

        // Fail open: if NO token is allowed for this state (e.g. a
        // lossy-detokenized vocab where no single piece faithfully extends
        // the prefix, or a stop set that excludes the model's real EOS at a
        // non-complete state), an all-blocked mask would force argMax onto a
        // forbidden token and wedge the decode. Returning an all-allowed mask
        // hands control back to the unconstrained sampler for this step — the
        // post-extraction `coerce` remains the safety net for validity. This
        // is logged once so the condition is observable.
        if allowedCount == 0 {
            if !didWarnAllBlocked {
                didWarnAllBlocked = true
                FileHandle.standardError.write(Data((
                    "[KrillLM] JSON grammar mask: no token extends the current "
                    + "state; failing open for this step (output validity then "
                    + "relies on post-extraction).\n").utf8))
            }
            let open = MLXArray([Float](repeating: 0, count: vocabSize))
            cache[state] = open
            return open
        }

        let arr = MLXArray(bias)
        cache[state] = arr
        return arr
    }
}
