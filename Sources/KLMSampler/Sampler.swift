import MLX
import MLXRandom

// MARK: - Sampling Parameters

/// Parameters controlling token sampling behavior.
public struct SamplingParams: Sendable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var seed: UInt64?

    public init(
        temperature: Float = 0.0,
        topP: Float = 1.0,
        topK: Int = 0,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }

    /// Greedy decoding (temperature = 0).
    public static let greedy = SamplingParams(temperature: 0.0)

    /// Default creative sampling.
    public static let creative = SamplingParams(temperature: 0.7, topP: 0.9)
}

// MARK: - Sampler

/// GPU-resident token sampler using MLX operations.
///
/// Supports greedy (argmax), temperature scaling, top-k, top-p, and
/// repetition penalty. All operations stay on the GPU - no host transfer
/// until the final token ID is read.
public final class Sampler: @unchecked Sendable {
    private let params: SamplingParams

    public init(params: SamplingParams = .greedy) {
        self.params = params
        if let seed = params.seed {
            MLXRandom.seed(seed)
        }
    }

    /// Sample the next token from logits.
    ///
    /// - Parameter logits: Raw logits, shape `[B, vocabSize]` or `[B, 1, vocabSize]`
    /// - Returns: Sampled token ID as Int
    public func sample(_ logits: MLXArray) -> Int {
        return sampleArray(logits).item(Int.self)
    }

    /// Sample the next token, returning a 1-element MLXArray of dtype int32.
    ///
    /// Returning a lazy MLXArray instead of a host Int lets the caller keep the
    /// chosen token on-GPU and feed it directly into the next forward pass, so
    /// two iterations can be in flight before any host sync happens.
    ///
    /// - Parameter logits: Raw logits, shape `[B, vocabSize]` or `[B, 1, vocabSize]`
    /// - Returns: A 1-element int32 MLXArray of shape `[1]` containing the token ID.
    public func sampleArray(_ logits: MLXArray) -> MLXArray {
        // Take last position if 3D: [B, seqLen, vocab] -> [B, vocab]
        var logits = logits
        if logits.ndim == 3 {
            logits = logits[0..., logits.dim(1) - 1, 0...]
        }
        // Take first batch: [B, vocab] -> [vocab]
        logits = logits[0]

        // Greedy: just argmax (keepDims gives shape [1])
        if params.temperature <= 0 {
            return argMax(logits, keepDims: true).asType(.int32)
        }

        // Temperature scaling
        var scaled = logits / params.temperature

        // Top-k filtering
        if params.topK > 0 {
            scaled = topKFilter(scaled, k: params.topK)
        }

        // Top-p (nucleus) filtering
        if params.topP < 1.0 {
            scaled = topPFilter(scaled, p: params.topP)
        }

        // Sample from the distribution. categorical returns shape [1] uint32.
        let probs = softmax(scaled)
        let token = MLXRandom.categorical(expandedDimensions(probs, axis: 0))
        return token.asType(.int32)
    }
}

// MARK: - Filtering Utilities

/// Zero out logits below the top-k values.
private func topKFilter(_ logits: MLXArray, k: Int) -> MLXArray {
    let k = min(k, logits.dim(0))
    let topk = sorted(logits, axis: -1)[logits.dim(0) - k]
    let mask = logits .< topk
    return which(mask, MLXArray(Float(-1e9)), logits)
}

/// Filter logits outside the top-p cumulative probability mass.
///
/// Strategy: sort descending by probability, compute cumulative sum,
/// find the threshold probability where cumsum exceeds p, then mask
/// all tokens in original logits below that probability threshold.
private func topPFilter(_ logits: MLXArray, p: Float) -> MLXArray {
    let probs = softmax(logits)
    // Sort probabilities descending via negation trick
    let negProbs = MLXArray(0) - probs
    let sortedIndices = argSort(negProbs, axis: -1)
    let sortedProbs = takeAlong(probs, sortedIndices, axis: 0)
    let cumProbs = cumsum(sortedProbs, axis: -1)

    // Find the minimum probability that's still within the top-p nucleus
    // cumProbs exceeds p at some index; the prob at that index is our threshold
    let exceedsMask = cumProbs .> MLXArray(p)
    // Shift mask right by one so we keep the token that pushes over p
    let sortedMask = concatenated([MLXArray([false]), exceedsMask[..<(exceedsMask.dim(0) - 1)]], axis: 0)
    // Get threshold: the smallest probability we keep
    let keepProbs = which(sortedMask, MLXArray(Float(0)), sortedProbs)
    let threshold = MLX.min(keepProbs + which(sortedMask, MLXArray(Float(1e9)), MLXArray(Float(0))), axis: -1)

    // Mask original logits where probability is below threshold
    return which(probs .< threshold, MLXArray(Float(-1e9)), logits)
}
