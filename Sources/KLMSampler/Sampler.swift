import Foundation
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
    /// Min-p (relative) nucleus cutoff: keep tokens with prob >= minP * pMax.
    /// 0 disables. (WS-D D3 / T2-10)
    public var minP: Float
    /// OpenAI-style presence penalty (flat, applied once per seen token).
    public var presencePenalty: Float
    /// OpenAI-style frequency penalty (scaled by occurrence count).
    public var frequencyPenalty: Float
    /// How many trailing tokens the penalties consider (Ollama
    /// `repeat_last_n`; <0 = whole context, 0 = disabled). Default 64.
    public var repeatLastN: Int
    /// Mirostat mode: 0 off, 1 (v1) or 2 (v2). With τ/η.
    public var mirostat: Int
    public var mirostatTau: Float
    public var mirostatEta: Float

    /// True only when some penalty/mirostat is non-neutral. The decode loop
    /// uses this to skip *all* history tracking + extra GPU work on the
    /// default path, so the speed/memory gate path is byte-for-byte
    /// unchanged unless a client explicitly opts in.
    public var penaltiesActive: Bool {
        repetitionPenalty != 1.0 || presencePenalty != 0.0
            || frequencyPenalty != 0.0 || mirostat != 0
    }

    public init(
        temperature: Float = 0.0,
        topP: Float = 1.0,
        topK: Int = 0,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil,
        minP: Float = 0.0,
        presencePenalty: Float = 0.0,
        frequencyPenalty: Float = 0.0,
        repeatLastN: Int = 64,
        mirostat: Int = 0,
        mirostatTau: Float = 5.0,
        mirostatEta: Float = 0.1
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.minP = minP
        self.presencePenalty = presencePenalty
        self.frequencyPenalty = frequencyPenalty
        self.repeatLastN = repeatLastN
        self.mirostat = mirostat
        self.mirostatTau = mirostatTau
        self.mirostatEta = mirostatEta
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
    /// Mirostat running estimate (v1/v2). Per-generation state - `Sampler`
    /// is constructed once per request so this is request-scoped.
    private var mirostatMu: Float

    public init(params: SamplingParams = .greedy) {
        self.params = params
        self.mirostatMu = 2.0 * params.mirostatTau
        if let seed = params.seed {
            MLXRandom.seed(seed)
        }
    }

    /// Whether the decode loop must track recent tokens for this request.
    public var needsHistory: Bool { params.penaltiesActive }

    /// Sample the next token from logits.
    ///
    /// - Parameter logits: Raw logits, shape `[B, vocabSize]` or `[B, 1, vocabSize]`
    /// - Returns: Sampled token ID as Int
    public func sample(_ logits: MLXArray) -> Int {
        return sampleArray(logits).item(Int.self)
    }

    /// Penalty-aware variants. `recent` is the trailing token window the
    /// decode loop maintains *only* when `needsHistory` is true; on the
    /// default path these are never called, so the hot path is unchanged.
    public func sample(_ logits: MLXArray, recent: [Int]) -> Int {
        sampleArray(logits, recent: recent).item(Int.self)
    }

    public func sampleArray(_ logits: MLXArray, recent: [Int]) -> MLXArray {
        let l = applyPenalties(to1D(logits), recent: recent)
        return sampleFrom(l)
    }

    /// Reduce `[B, seq, vocab]` / `[B, vocab]` / `[vocab]` to 1-D `[vocab]`.
    private func to1D(_ logits: MLXArray) -> MLXArray {
        var l = logits
        if l.ndim == 3 { l = l[0..., l.dim(1) - 1, 0...] }
        if l.ndim == 2 { l = l[0] }
        return l
    }

    /// repetition_penalty (divide positive / multiply negative logits at
    /// seen tokens), presence_penalty (flat per distinct seen token), and
    /// frequency_penalty (× occurrence count). Applied via a single
    /// scatter over the small unique-recent set - O(window), not O(vocab).
    private func applyPenalties(_ logits: MLXArray, recent: [Int]) -> MLXArray {
        let n = params.repeatLastN
        let window: [Int]
        if n < 0 { window = recent }
        else if n == 0 { window = [] }
        else { window = Array(recent.suffix(n)) }
        guard !window.isEmpty else { return logits }

        var counts: [Int: Int] = [:]
        for t in window { counts[t, default: 0] += 1 }
        let ids = Array(counts.keys)
        let idx = MLXArray(ids.map { Int32($0) })
        let current = take(logits, idx, axis: 0)

        var adjusted = current
        if params.repetitionPenalty != 1.0 {
            let rp = params.repetitionPenalty
            adjusted = MLX.where(adjusted .> 0, adjusted / rp, adjusted * rp)
        }
        if params.presencePenalty != 0.0 {
            adjusted = adjusted - MLXArray(params.presencePenalty)
        }
        if params.frequencyPenalty != 0.0 {
            let freq = MLXArray(ids.map { Float(counts[$0] ?? 0) })
            adjusted = adjusted - freq * MLXArray(params.frequencyPenalty)
        }
        // Scatter the adjusted values back to their vocab positions.
        let updated = updatedAt(logits, indices: idx, values: adjusted)
        return updated
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
        return sampleFrom(to1D(logits))
    }

    /// Shared sampling tail operating on 1-D `[vocab]` logits.
    private func sampleFrom(_ logits1D: MLXArray) -> MLXArray {
        // Greedy: just argmax (keepDims gives shape [1])
        if params.temperature <= 0 && params.mirostat == 0 {
            return argMax(logits1D, keepDims: true).asType(.int32)
        }

        // Temperature scaling
        let temp = params.temperature <= 0 ? 1.0 : params.temperature
        var scaled = logits1D / temp

        // Mirostat (v1/v2): adaptive top-k truncation toward target surprise.
        if params.mirostat != 0 {
            return mirostatSample(scaled)
        }

        // Top-k filtering
        if params.topK > 0 {
            scaled = topKFilter(scaled, k: params.topK)
        }

        // Top-p (nucleus) filtering
        if params.topP < 1.0 {
            scaled = topPFilter(scaled, p: params.topP)
        }

        // Min-p filtering (relative to the peak probability).
        if params.minP > 0.0 {
            scaled = minPFilter(scaled, minP: params.minP)
        }

        // Sample from the distribution. categorical returns shape [1] uint32.
        let probs = softmax(scaled)
        let token = MLXRandom.categorical(expandedDimensions(probs, axis: 0))
        return token.asType(.int32)
    }

    /// Mirostat v2 (and a v1 approximation): keep the running surprise
    /// estimate `mu`, truncate the sorted distribution where surprise
    /// exceeds `mu`, sample, then update `mu` by `eta * (tau - observed)`.
    private func mirostatSample(_ scaled: MLXArray) -> MLXArray {
        let probs = softmax(scaled)
        let sortedIdx = argSort(MLXArray(0) - probs, axis: -1)
        let sortedProbs = takeAlong(probs, sortedIdx, axis: 0)
        let surprise = -MLX.log(sortedProbs + 1e-10) / Float.log(2.0)
        // Keep the prefix whose surprise stays under mu (at least 1 token).
        let keepMask = surprise .<= MLXArray(mirostatMu)
        let keepCount = max(1, MLX.sum(keepMask.asType(.int32)).item(Int.self))
        let head = sortedIdx[0 ..< keepCount]
        let headProbs = sortedProbs[0 ..< keepCount]
        let renorm = headProbs / MLX.sum(headProbs)
        let pick = MLXRandom.categorical(expandedDimensions(renorm, axis: 0)).item(Int.self)
        let chosen = head[pick].item(Int.self)
        // Update mu from the observed surprise of the chosen token.
        let obs = -Float.log(max(1e-10, sortedProbs[pick].item(Float.self))) / Float.log(2.0)
        mirostatMu += params.mirostatEta * (params.mirostatTau - obs)
        return MLXArray([Int32(chosen)])
    }
}

private extension Float {
    static func log(_ x: Float) -> Float { Foundation.log(x) }
}

/// Scatter `values` into `base` at `indices` (1-D), returning a new array.
private func updatedAt(_ base: MLXArray, indices: MLXArray, values: MLXArray) -> MLXArray {
    var out = base
    out[indices] = values
    return out
}

// MARK: - Filtering Utilities

/// Zero out logits below the top-k values.
private func topKFilter(_ logits: MLXArray, k: Int) -> MLXArray {
    let k = min(k, logits.dim(0))
    let topk = sorted(logits, axis: -1)[logits.dim(0) - k]
    let mask = logits .< topk
    return which(mask, MLXArray(Float(-1e9)), logits)
}

/// Min-p filter: keep only tokens whose probability is at least
/// `minP * max(prob)`. A relative cutoff that adapts to confidence -
/// stricter when the model is peaked, looser when it is flat.
private func minPFilter(_ logits: MLXArray, minP: Float) -> MLXArray {
    let probs = softmax(logits)
    let pMax = MLX.max(probs)
    let threshold = pMax * MLXArray(minP)
    return which(probs .< threshold, MLXArray(Float(-1e9)), logits)
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
