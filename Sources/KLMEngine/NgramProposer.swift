import Foundation

/// Prompt-lookup (n-gram) draft source for speculative decoding.
///
/// Instead of a separate draft *model* (which pays a per-token forward cost,
/// `alpha ~ 1/8` of the target), the proposal is copied straight from the
/// context: match the last `n` tokens of the running sequence against an earlier
/// identical `n`-gram and propose the tokens that followed it. Draft cost is a
/// host-side array scan (`alpha -> 0`), and on repetitive workloads (code, RAG,
/// JSON/structured output, summarization that quotes its source) the match runs
/// are long and exact, so a single verify forward commits many tokens.
///
/// The proposer is greedy-agnostic and holds no GPU state: every proposed token
/// is *verified* against the target model's own greedy argmax before acceptance
/// (`SpeculativeDecoder.ngramStep`), so a wrong proposal is simply rejected and
/// replaced — output stays byte-identical to plain greedy decode regardless of
/// proposal quality. See `docs/VERIFY_PROFILE.md` for the measured `beta` that
/// justifies the `maxDraft` default.
public final class NgramProposer: @unchecked Sendable {

    public struct Config: Sendable {
        /// Longest suffix length to try matching (tried high→low; longest match wins).
        public var maxN: Int
        /// Shortest suffix length to fall back to.
        public var minN: Int
        /// Cap on proposed tokens per round. Bounds the wasted-forward cost of a
        /// fully-rejected round (see `docs/VERIFY_PROFILE.md`).
        public var maxDraft: Int
        /// Only search the last `searchWindow` tokens for an earlier occurrence
        /// (0 = whole history). Bounds scan cost and biases toward recent,
        /// more-relevant repetition.
        public var searchWindow: Int

        public init(maxN: Int = 3, minN: Int = 1, maxDraft: Int = 16, searchWindow: Int = 2048) {
            self.maxN = max(1, maxN)
            self.minN = max(1, min(minN, self.maxN))
            self.maxDraft = max(1, maxDraft)
            self.searchWindow = max(0, searchWindow)
        }
    }

    private let config: Config
    private let eosIds: Set<Int>

    /// The running token sequence: prompt + every accepted token. This is exactly
    /// the context the target model has attended to, so any match is valid.
    public private(set) var history: [Int] = []

    /// Acceptance-adaptive proposal cap. A rejected k-token round still pays a
    /// k-wide verify forward (~k× the weight stream is amortized, but a wrong
    /// large match wastes that forward to commit a single token), so when
    /// acceptance collapses we shrink toward 1 to hold the >=1.0x floor; when
    /// matches pay off we grow back toward `maxDraft`. Starts optimistic.
    public private(set) var effectiveCap: Int

    public init(config: Config = .init(), eosIds: Set<Int> = []) {
        self.config = config
        self.eosIds = eosIds
        self.effectiveCap = config.maxDraft
    }

    /// Begin a new generation, seeding history with the FULL prompt tokens (not a
    /// prefix-cache-trimmed slice — matches must see the whole attended context).
    public func reset(prompt: [Int]) {
        history = prompt
        effectiveCap = config.maxDraft
    }

    /// Update the adaptive cap from a round's outcome: `acceptedDraft` of the
    /// `proposed` draft tokens were correct. Grow by one on a fully-accepted
    /// round (the match is paying off, reach further); otherwise clamp to just
    /// past where it broke, so the next wrong match wastes a small forward. With
    /// repeated zero-acceptance rounds the cap decays to 1 (≈ plain decode cost).
    public func recordOutcome(acceptedDraft: Int, proposed: Int) {
        if proposed > 0 && acceptedDraft >= proposed {
            effectiveCap = min(config.maxDraft, effectiveCap + 1)
        } else {
            effectiveCap = max(1, min(effectiveCap, acceptedDraft + 1))
        }
    }

    /// Record tokens that were actually accepted (emitted) into history.
    public func append(_ tokens: [Int]) {
        history.append(contentsOf: tokens)
    }

    /// Propose up to `maxDraft` continuation tokens by matching the current
    /// suffix against the most recent earlier occurrence. Returns `[]` when no
    /// match is found (the caller then does a plain single-token decode step).
    public func propose() -> [Int] {
        let L = history.count
        guard L >= config.minN else { return [] }

        let searchStart = config.searchWindow > 0 ? max(0, L - config.searchWindow) : 0
        let cap = max(1, min(config.maxDraft, effectiveCap))

        // Try the longest suffix first for precision, falling back to shorter n.
        var n = min(config.maxN, L)
        while n >= config.minN {
            if let start = lastOccurrence(suffixLen: n, searchStart: searchStart, L: L) {
                let from = start + n
                let to = min(from + cap, L)
                if from < to {
                    return truncateAtEOS(Array(history[from ..< to]))
                }
            }
            n -= 1
        }
        return []
    }

    /// Rightmost start index `p` in `[searchStart, L-n-1]` such that
    /// `history[p..<p+n] == history[L-n..<L]` (the live suffix). Scans backward
    /// and short-circuits on the suffix's last token, so a no-match suffix is
    /// rejected in one comparison per candidate position.
    private func lastOccurrence(suffixLen n: Int, searchStart: Int, L: Int) -> Int? {
        let suffixStart = L - n
        let lastTok = history[L - 1]
        // p+n-1 is the candidate's last token; it must equal the suffix's last
        // token. Candidate window must end strictly before the suffix's last
        // token, i.e. p+n-1 <= L-2  =>  p <= L-n-1.
        var p = L - n - 1
        while p >= searchStart {
            if history[p + n - 1] == lastTok {
                var match = true
                var i = 0
                while i < n - 1 {                       // last token already matched
                    if history[p + i] != history[suffixStart + i] { match = false; break }
                    i += 1
                }
                if match { return p }
            }
            p -= 1
        }
        return nil
    }

    /// Stop a proposal at the first EOS id — never speculate past end-of-sequence.
    private func truncateAtEOS(_ tokens: [Int]) -> [Int] {
        guard !eosIds.isEmpty else { return tokens }
        if let idx = tokens.firstIndex(where: { eosIds.contains($0) }) {
            return Array(tokens[0 ... idx])             // keep the EOS itself
        }
        return tokens
    }
}
