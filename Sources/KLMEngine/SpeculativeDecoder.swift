import Foundation
import MLX
import KLMCore
import KLMCache
import KLMSampler

/// Speculative decoding scheduler.
///
/// Uses a small draft model to propose K tokens, then verifies all K in a single
/// batched forward pass of the target model. Accepts tokens up to the first
/// rejection, yielding 1.5-3x decode speedup on draftable workloads.
///
/// Adaptive K: rolls acceptance rate over last 16 verifications. If acceptance
/// rate < 0.4, drops K to 2. If > 0.8, raises K to 6.
public final class SpeculativeDecoder: @unchecked Sendable {
    private let targetModel: LoadedModel
    private let draftModel: LoadedModel
    private let sampler: Sampler

    private var adaptiveK: Int
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

    public init(
        targetModel: LoadedModel,
        draftModel: LoadedModel,
        initialK: Int = 4,
        temperature: Float = 0.0
    ) {
        self.targetModel = targetModel
        self.draftModel = draftModel
        self.adaptiveK = initialK
        self.sampler = Sampler(params: SamplingParams(temperature: temperature))
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
        targetCaches: [KVCache],
        draftCaches: [KVCache]
    ) -> [Int] {
        let k = adaptiveK

        // Draft: generate K tokens greedily
        var draftTokens: [Int] = []
        var draftLogits: [MLXArray] = []
        var currentToken = lastToken

        for _ in 0 ..< k {
            let input = MLXArray([Int32(currentToken)]).reshaped(1, 1)
            let logits = draftModel.forward(input, draftCaches)
            MLX.eval(logits)
            draftLogits.append(logits)
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

        // Record sequence length before verification so we can roll back on rejection.
        let previousLength = targetCaches.first?.sequenceLength ?? 0

        let targetLogits = targetModel.forward(verifyInput, targetCaches)
        MLX.eval(targetLogits)

        // Accept/reject: compare target vs draft at each position
        var accepted: [Int] = []
        var allAccepted = true

        for i in 0 ..< k {
            // Target logits at position i correspond to the prediction after token i
            let targetLogitSlice = targetLogits[0..., i, 0...]
            let targetToken = sampler.sample(expandedDimensions(targetLogitSlice, axis: 0))

            if targetToken == draftTokens[i] {
                // Match: accept the draft token
                accepted.append(draftTokens[i])
            } else {
                // Rejection: use target's token instead
                accepted.append(targetToken)
                allAccepted = false
                break
            }
        }

        // Roll back KV state for rejected tokens so the cache reflects exactly
        // the tokens that were accepted.
        if !allAccepted {
            let acceptedLength = previousLength + accepted.count
            for cache in targetCaches {
                cache.truncate(to: acceptedLength)
            }
        }

        // If all K were accepted, we get a bonus token from position K
        if allAccepted {
            let bonusInput = MLXArray([Int32(draftTokens.last!)]).reshaped(1, 1)
            let bonusLogits = targetModel.forward(bonusInput, targetCaches)
            MLX.eval(bonusLogits)
            let bonusToken = sampler.sample(bonusLogits)
            accepted.append(bonusToken)
        }

        // Update stats
        let rate = Double(accepted.count - 1) / Double(k) // -1 because last is either rejection replacement or bonus
        acceptanceHistory.append(rate)
        if acceptanceHistory.count > historyWindow {
            acceptanceHistory.removeFirst()
        }
        totalAccepted += accepted.count
        totalRounds += 1

        // Adapt K
        adaptK()

        return accepted
    }

    /// Reset internal state for a new generation.
    public func reset() {
        acceptanceHistory.removeAll()
        totalAccepted = 0
        totalRounds = 0
        adaptiveK = 4
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
]

/// Find the recommended draft model for a given target.
public func recommendedDraft(for targetModel: String) -> String? {
    draftPairs[targetModel]
}
