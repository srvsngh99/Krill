import Foundation
import MLX
import KLMCore
import KLMCache
import KLMSampler

/// Speculative decoding scheduler.
///
/// Uses a small draft model to propose K tokens, then verifies all K in a single
/// batched forward pass of the target model. Accepts tokens up to the first
/// rejection, yielding 1.5-3x decode speedup on draftable workloads when the
/// draft model is well-matched (shared tokenizer, smaller architecture).
///
/// Adaptive K: rolls acceptance rate over last 16 verifications. If acceptance
/// rate < 0.4, drops K to 2. If > 0.8, raises K to 6.
///
/// Sampling: this scheduler does greedy verification only. The caller MUST
/// gate the spec path to greedy decoding (temperature == 0, top-p >= 1,
/// top-k <= 0, no penalties / mirostat). Non-greedy sampling would require
/// Leviathan-style rejection sampling, which is not implemented.
///
/// Cache contract: the caller MUST prefill `draftCaches` with the same
/// prompt history that the target model was prefilled with, before calling
/// `step(lastToken:targetCaches:draftCaches:)`. The decoder does not own
/// or warm the draft cache. See `InferenceEngine` for the prefill path.
public final class SpeculativeDecoder: @unchecked Sendable {
    private let targetModel: LoadedModel?
    private let draftModel: LoadedModel?
    private let sampler: Sampler

    private var adaptiveK: Int
    private let initialK: Int
    private let minK: Int = 2
    private let maxK: Int = 6
    private var acceptanceHistory: [Double] = []
    private let historyWindow: Int = 16

    /// Total tokens accepted via speculation (for stats).
    public private(set) var totalAccepted: Int = 0

    /// Total verification rounds (for stats).
    public private(set) var totalRounds: Int = 0

    public var acceptanceRate: Double {
        guard !acceptanceHistory.isEmpty else { return 0 }
        return acceptanceHistory.reduce(0, +) / Double(acceptanceHistory.count)
    }

    /// The current adaptive K (proposals per round).
    public var currentK: Int { adaptiveK }

    /// The draft model (used by the engine to prefill draft caches).
    public var draft: LoadedModel? { draftModel }

    public init(
        targetModel: LoadedModel,
        draftModel: LoadedModel,
        initialK: Int = 4,
        temperature: Float = 0.0
    ) {
        self.targetModel = targetModel
        self.draftModel = draftModel
        self.adaptiveK = initialK
        self.initialK = initialK
        self.sampler = Sampler(params: SamplingParams(temperature: temperature))
    }

    internal convenience init(initialKForTesting initialK: Int) {
        self.init(targetModel: nil, draftModel: nil, initialK: initialK)
    }

    /// Build a draft-model-free (n-gram / prompt-lookup) speculative decoder. The
    /// draft is supplied by a `NgramProposer` passed to `ngramStep`; there is no
    /// draft model and no draft KV cache. `K` is proposer-driven, so the adaptive
    /// `K` machinery is inert here (`ngramStep` never calls `adaptK`).
    public static func ngram(targetModel: LoadedModel, temperature: Float = 0.0)
        -> SpeculativeDecoder
    {
        SpeculativeDecoder(
            targetModel: targetModel, draftModel: nil, initialK: 0, temperature: temperature)
    }

    private init(
        targetModel: LoadedModel?,
        draftModel: LoadedModel?,
        initialK: Int,
        temperature: Float = 0.0
    ) {
        self.targetModel = targetModel
        self.draftModel = draftModel
        self.adaptiveK = initialK
        self.initialK = initialK
        self.sampler = Sampler(params: SamplingParams(temperature: temperature))
    }

    internal var currentKForTesting: Int {
        adaptiveK
    }

    @discardableResult
    internal func recordVerificationForTesting(acceptedTokenCount: Int, proposedTokenCount: Int) -> Double {
        recordVerification(acceptedTokenCount: acceptedTokenCount, proposedTokenCount: proposedTokenCount)
    }

    /// Run one speculative decode step.
    ///
    /// - Parameters:
    ///   - lastToken: The last accepted token
    ///   - targetCaches: KV caches for the target model
    ///   - draftCaches: KV caches for the draft model
    /// - Returns: Array of accepted tokens (1 to K+1 tokens)
    public func step(
        lastToken: Int,
        targetCaches: [RestorableKVCache],
        draftCaches: [KVCache]
    ) -> [Int] {
        guard let targetModel, let draftModel else {
            preconditionFailure("SpeculativeDecoder.step requires target and draft models")
        }

        let k = adaptiveK

        // Draft: generate K tokens greedily.
        var draftTokens: [Int] = []
        var currentToken = lastToken

        for _ in 0 ..< k {
            let input = MLXArray([Int32(currentToken)]).reshaped(1, 1)
            let logits = draftModel.forward(input, draftCaches)
            MLX.eval(logits)
            let nextToken = sampler.sample(logits)
            draftTokens.append(nextToken)
            currentToken = nextToken
        }

        // Target: verify all K tokens in one batched forward pass
        // Build input: [lastToken] + draftTokens[0..<k-1] (k tokens total)
        var verifyTokens = [Int32(lastToken)]
        for t in draftTokens.dropLast() {
            verifyTokens.append(Int32(t))
        }
        let verifyInput = MLXArray(verifyTokens).reshaped(1, verifyTokens.count)

        // Record sequence lengths before verification so we can roll back on
        // rejection. Target and draft caches advance independently and may
        // start at different lengths (e.g. asymmetric prefill), so we
        // capture both. After the draft loop above, draft sequenceLength
        // has already grown by `k`; capture the post-draft length so we can
        // restore the draft cache to match the accepted prefix.
        let previousLength = targetCaches.first?.sequenceLength ?? 0
        let postDraftLength = draftCaches.first?.sequenceLength ?? 0

        let targetLogits = targetModel.forward(verifyInput, targetCaches)
        MLX.eval(targetLogits)

        // Accept/reject: compare target vs draft at each position.
        //
        // The spec path is greedy-gated (see the class doc), so target
        // verification is an argmax. Compute the argmax for ALL K
        // positions in one batched op plus one eval, rather than slicing
        // each position and calling `sampler.sample` K times - that cost
        // K-1 extra GPU synchronizations per round, pure overhead on a
        // path whose entire purpose is to beat plain decode. The result
        // is identical to the per-position greedy sample.
        let verifiedTokens = argMax(targetLogits, axis: -1)
            .asType(.int32)
        MLX.eval(verifiedTokens)
        let targetTokens = verifiedTokens.asArray(Int32.self).map(Int.init)

        var accepted: [Int] = []
        var allAccepted = true

        for i in 0 ..< k {
            if targetTokens[i] == draftTokens[i] {
                // Match: accept the draft token.
                accepted.append(draftTokens[i])
            } else {
                // Rejection: use the target's token instead.
                accepted.append(targetTokens[i])
                allAccepted = false
                break
            }
        }

        // Roll back KV state for rejected tokens so the cache reflects exactly
        // the tokens that were accepted. Both target and draft caches need
        // the same trim:
        //   - target wrote K entries during verify; keep `accepted.count`
        //     (= i accepted draft + 1 target replacement, but the
        //     replacement is itself NOT in the cache, so we keep i+1
        //     entries from the K verify writes, which is `accepted.count`).
        //   - draft wrote K entries during its loop; keep the same
        //     `accepted.count` so the draft cache matches "everything
        //     generated up to but not including the new lastToken
        //     (= the rejection replacement)".
        // Without trimming the draft cache, the next round's draft would
        // see a context that includes proposals which the target rejected,
        // and propose continuations of a fictional sequence.
        if !allAccepted {
            let acceptedLength = previousLength + accepted.count
            for cache in targetCaches {
                cache.truncate(to: acceptedLength)
            }
            let acceptedDraftLength = postDraftLength - k + accepted.count
            for cache in draftCaches {
                cache.truncate(to: acceptedDraftLength)
            }
        }

        // If all K were accepted, we get a bonus token from position K.
        // The draft cache currently holds K-1 of the K accepted draft
        // tokens (the K-th, `d_{K-1}`, was generated from logits but never
        // re-forwarded into the draft KV). The target's bonus forward
        // catches the target cache up to position K. We mirror that on
        // the draft cache so both end the round with KV for the same
        // generated prefix; otherwise the next round's draft would skip a
        // position and propose against the wrong context.
        if allAccepted {
            let bonusInput = MLXArray([Int32(draftTokens.last!)]).reshaped(1, 1)
            let bonusLogits = targetModel.forward(bonusInput, targetCaches)
            MLX.eval(bonusLogits)
            let bonusToken = sampler.sample(bonusLogits)
            accepted.append(bonusToken)

            let draftBonusLogits = draftModel.forward(bonusInput, draftCaches)
            MLX.eval(draftBonusLogits)
        }

        recordVerification(acceptedTokenCount: accepted.count, proposedTokenCount: k)

        return accepted
    }

    /// Run one **n-gram (prompt-lookup)** speculative step. The draft comes from
    /// `proposer.propose()` (a context match), not a draft model — so there is no
    /// draft loop, no draft model, and no draft cache. The verify / accept /
    /// rollback / bonus logic is otherwise identical to `step`, operating on the
    /// target cache only.
    ///
    /// Greedy parity: every returned token is the target model's greedy argmax at
    /// its position, computed from a cache holding exactly the accepted prefix.
    /// The proposal *source* is irrelevant because each proposed token is verified
    /// against that argmax. Output matches standard decode token-for-token except
    /// where the width-K verify forward's argmax differs from the width-1 decode
    /// forward's at an fp16 near-tie — the same nondeterminism the batched decoder
    /// already exhibits, not a divergence in the speculation logic.
    ///
    /// - Returns: 1 to K+1 accepted tokens (1 on a no-match plain step).
    public func ngramStep(
        lastToken: Int,
        targetCaches: [RestorableKVCache],
        proposer: NgramProposer
    ) -> [Int] {
        guard let targetModel else {
            preconditionFailure("SpeculativeDecoder.ngramStep requires a target model")
        }

        let draftTokens = proposer.propose()
        let k = draftTokens.count

        // No match: a single plain decode step. Forwarding just [lastToken] and
        // taking its argmax is byte-identical to one standard decode step, so the
        // floor is exactly 1.0x on non-repetitive stretches.
        if k == 0 {
            let input = MLXArray([Int32(lastToken)]).reshaped(1, 1)
            let logits = targetModel.forward(input, targetCaches)
            MLX.eval(logits)
            let tok = sampler.sample(logits)
            proposer.append([tok])
            // A no-match round saved nothing beyond the single decoded token —
            // feed 0 to the stall monitor so non-echo stretches drive it down.
            proposer.recordRound(extraTokens: 0)
            recordVerification(acceptedTokenCount: 1, proposedTokenCount: 0, doAdapt: false)
            return [tok]
        }

        // Verify all k proposed tokens in one batched forward:
        // input = [lastToken] + draftTokens[0..<k-1]  (k positions).
        var verifyTokens = [Int32(lastToken)]
        for t in draftTokens.dropLast() {
            verifyTokens.append(Int32(t))
        }
        let verifyInput = MLXArray(verifyTokens).reshaped(1, verifyTokens.count)

        // Length before verify so we can roll back rejected tokens. Only the
        // target cache exists (no draft cache to keep in sync).
        let previousLength = targetCaches.first?.sequenceLength ?? 0

        let targetLogits = targetModel.forward(verifyInput, targetCaches)
        MLX.eval(targetLogits)

        // Greedy-gated: argmax all k positions in one batched op (see `step`).
        let verifiedTokens = argMax(targetLogits, axis: -1).asType(.int32)
        MLX.eval(verifiedTokens)
        let targetTokens = verifiedTokens.asArray(Int32.self).map(Int.init)

        var accepted: [Int] = []
        var allAccepted = true
        for i in 0 ..< k {
            if targetTokens[i] == draftTokens[i] {
                accepted.append(draftTokens[i])
            } else {
                accepted.append(targetTokens[i])
                allAccepted = false
                break
            }
        }

        // Roll back the target cache to exactly the accepted prefix. The verify
        // wrote k rows; keep `accepted.count` (the rejection replacement is itself
        // never forwarded, so it is not in the cache). Identical math to `step`'s
        // target-cache trim, minus the (absent) draft cache.
        if !allAccepted {
            let acceptedLength = previousLength + accepted.count
            for cache in targetCaches {
                cache.truncate(to: acceptedLength)
            }
        }

        // Full acceptance yields a bonus token from position K. The target bonus
        // forward catches the target cache up by one row; there is no draft cache
        // to mirror.
        if allAccepted {
            let bonusInput = MLXArray([Int32(draftTokens.last!)]).reshaped(1, 1)
            let bonusLogits = targetModel.forward(bonusInput, targetCaches)
            MLX.eval(bonusLogits)
            accepted.append(sampler.sample(bonusLogits))
        }

        // Feed the adaptive cap: on full acceptance all k drafts were correct;
        // on rejection at index i, accepted.count-1 drafts matched (the last is
        // the target replacement, not a draft token).
        let acceptedDraft = allAccepted ? k : accepted.count - 1
        proposer.recordOutcome(acceptedDraft: acceptedDraft, proposed: k)
        // Extra tokens this round = accepted drafts (the trailing bonus/replacement
        // is the token a plain decode step would have produced anyway).
        proposer.recordRound(extraTokens: acceptedDraft)
        proposer.append(accepted)
        recordVerification(acceptedTokenCount: accepted.count, proposedTokenCount: k, doAdapt: false)
        return accepted
    }

    @discardableResult
    private func recordVerification(
        acceptedTokenCount: Int, proposedTokenCount k: Int, doAdapt: Bool = true
    ) -> Double {
        // -1 because the last returned token is either a rejection replacement or a bonus token.
        // A k==0 round (n-gram no-match plain decode) contributes no acceptance signal.
        let rate = k > 0 ? Double(acceptedTokenCount - 1) / Double(k) : 0
        if k > 0 {
            acceptanceHistory.append(rate)
            if acceptanceHistory.count > historyWindow {
                acceptanceHistory.removeFirst()
            }
        }
        totalAccepted += acceptedTokenCount
        totalRounds += 1

        // K is proposer-driven for the n-gram path, so it does not adapt K here.
        if doAdapt { adaptK() }

        return rate
    }

    /// Reset internal state for a new generation. Restores adaptive K to
    /// the value the decoder was created with so a fresh generation does
    /// not inherit a stalled or saturated K from a prior run.
    public func reset() {
        acceptanceHistory.removeAll()
        totalAccepted = 0
        totalRounds = 0
        adaptiveK = initialK
    }

    private func adaptK() {
        let rate = acceptanceRate
        if rate < 0.4 && adaptiveK > minK {
            adaptiveK -= 1
        } else if rate > 0.8 && adaptiveK < maxK {
            adaptiveK += 1
        }
    }
}

// MARK: - Draft Pair Registry

/// Curated draft model pairs for speculative decoding.
/// Maps target model names to their recommended draft models.
public let draftPairs: [String: String] = [
    "llama-3.1-8b": "llama-3.2-1b",
    "llama-3.2-3b": "llama-3.2-1b",
    "qwen2.5-7b": "qwen2.5-0.5b",
    "qwen2.5-14b": "qwen2.5-1.5b",
    "qwen2.5-3b": "qwen2.5-0.5b",
    "gemma-2-9b": "gemma-2-2b",
    // Gemma 4: use Gemma 2 2B as drafter (compatible vocab, smaller architecture)
    "gemma-4-e4b": "gemma-2-2b",
]

/// Find the recommended draft model for a given target.
public func recommendedDraft(for targetModel: String) -> String? {
    draftPairs[targetModel]
}
