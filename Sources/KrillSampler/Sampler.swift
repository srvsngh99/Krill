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

    /// True when this request decodes greedily (pure argmax). Combined with
    /// `!needsHistory`, an epoch of all-greedy rows can batch its sampling into
    /// a single `argMax` + one host sync (the batched-decode overlap path).
    public var isGreedy: Bool { params.temperature <= 0 && params.mirostat == 0 }

    /// Sample the next token from logits.
    ///
    /// - Parameter logits: Raw logits, shape `[B, vocabSize]` or `[B, 1, vocabSize]`
    /// - Returns: Sampled token ID as Int
    /// - Parameter mask: optional additive `[vocab]` logit mask (0 for
    ///   allowed tokens, a large negative bias for forbidden ones), used by
    ///   grammar-constrained decoding. `nil` (the default) leaves the
    ///   sampling path byte-for-byte unchanged.
    public func sample(_ logits: MLXArray, mask: MLXArray? = nil) -> Int {
        return sampleArray(logits, mask: mask).item(Int.self)
    }

    /// Penalty-aware variants. `recent` is the trailing token window the
    /// decode loop maintains *only* when `needsHistory` is true; on the
    /// default path these are never called, so the hot path is unchanged.
    public func sample(_ logits: MLXArray, recent: [Int], mask: MLXArray? = nil) -> Int {
        sampleArray(logits, recent: recent, mask: mask).item(Int.self)
    }

    public func sampleArray(_ logits: MLXArray, recent: [Int], mask: MLXArray? = nil) -> MLXArray {
        let l = applyPenalties(to1D(logits), recent: recent)
        return sampleFrom(l, mask: mask)
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
    public func sampleArray(_ logits: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        return sampleFrom(to1D(logits), mask: mask)
    }

    /// Shared sampling tail operating on 1-D `[vocab]` logits.
    ///
    /// `mask`, when present, is an additive `[vocab]` logit mask applied
    /// FIRST — before greedy argmax and before any temperature / top-p /
    /// top-k / min-p / mirostat step — so a grammar-forbidden token (biased
    /// to roughly `-inf`) can never win regardless of the sampling knobs.
    private func sampleFrom(_ logits1D: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let l = mask.map { logits1D + $0.asType(logits1D.dtype) } ?? logits1D

        // Greedy: just argmax (keepDims gives shape [1])
        if params.temperature <= 0 && params.mirostat == 0 {
            return argMax(l, keepDims: true).asType(.int32)
        }

        // Temperature scaling
        let temp = params.temperature <= 0 ? 1.0 : params.temperature
        var scaled = l / temp

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

        // Grammar-masked sampling: re-add the mask so any token a filter let
        // slip back is at ~-2e9 and can never be drawn.
        if let mask {
            scaled = scaled + mask.asType(scaled.dtype)
        }

        // Sample. `MLXRandom.categorical` interprets its input as LOGITS (it
        // softmaxes internally), so it must be handed `scaled` directly.
        //
        // This previously passed `softmax(scaled)` — probabilities — which made
        // categorical compute softmax(softmax(logits)). That is NOT a harmless
        // double-softmax: probabilities live in [0, 1], so every token collapses
        // into exp([0,1]) = [1, e] and the distribution goes nearly UNIFORM over
        // the whole vocabulary. The filters made it worse rather than better —
        // top-k/top-p/min-p set rejected logits to -1e9, which softmaxes to
        // exactly 0 and then re-weights to exp(0) = 1, the same weight as a
        // token the model gave 0.9 probability (exp(0.9) = 2.46). With a large
        // vocab the rejected mass swamps the nucleus and sampling returns
        // essentially random tokens: every model emitted fluent-looking garbage
        // at any temperature > 0, while greedy (which never reaches this path)
        // stayed correct. Nanbeige's 166k vocab made it unmissable.
        let token = MLXRandom.categorical(expandedDimensions(scaled, axis: 0))
        return token.asType(.int32)
    }

    /// Sample the next token AND compute raw-distribution logprob info for
    /// it, per docs/LOGPROBS_PLAN.md §4.1/§5.1. This is the ONLY sampling
    /// entry point that pays the log-softmax + top-N cost; `sample`/
    /// `sampleArray` above are byte-for-byte unchanged and remain the hot
    /// path when logprobs are not requested (`InferenceEngine` only calls
    /// this method when a request set `wantLogprobs`).
    ///
    /// The sampled token still goes through the normal path (penalties ->
    /// `sampleFrom`, i.e. temperature/top-k/top-p/min-p/grammar mask) —
    /// only the REPORTED logprob is computed from the raw, pre-filter
    /// logits, taken independently so the penalty scatter in
    /// `applyPenalties` cannot alias the array this reads.
    ///
    /// - Parameters:
    ///   - logits: Raw logits, shape `[B, vocabSize]` / `[B, 1, vocabSize]` / `[vocab]`.
    ///   - recent: Trailing token window for penalties (empty when `needsHistory` is false).
    ///   - mask: Optional grammar logit mask, applied only to the sampling path (never to the reported logprobs).
    ///   - topLogprobs: Number of raw top-N alternates to report (0...20).
    /// - Returns: The sampled token ID, that token as a 1-element `MLXArray`
    ///   (so callers needing an on-GPU handle don't have to re-box it), and its logprob info.
    public func sampleWithLogprobs(
        _ logits: MLXArray, recent: [Int] = [], mask: MLXArray? = nil, topLogprobs: Int
    ) -> (token: Int, tokenArray: MLXArray, info: TokenLogprobInfo) {
        // Build the raw log-softmax GRAPH from the raw logits BEFORE
        // `applyPenalties` runs, entirely on the GPU - no host round trip.
        //
        // Why this is safe despite aliasing: `to1D` is a pure pass-through
        // (no slicing) when `logits` is already 1-D, and `MLXArray.asType`
        // short-circuits to `return self` when the dtype already matches
        // (MLXArray.swift), so `raw1D` can literally be the SAME Swift
        // `MLXArray` object (a `final class`, so `let`-binding it is a
        // reference copy) as `logits` itself. `applyPenalties`'s indexed
        // scatter (`updatedAt`, `out[indices] = values`) mutates that
        // object's `ctx` field IN PLACE (`mlx_array_set`) to now describe
        // the scattered result - a later read of `raw1D`/`logits` (e.g.
        // `.item()`) would see the penalized values, not the raw ones
        // (this WAS the bug: SamplerLogprobsTests.
        // testRawLogprobIgnoresActivePenalties failed until the previous fix
        // forced a host round trip to break the alias). But every MLX
        // *operation* (subtract, logSumExp, argSort, take, ...) captures its
        // input's CURRENT underlying array value at the C-API call site into
        // a brand-new, independent result object; it does not keep watching
        // the Swift wrapper for later reassignment. So as long as the
        // log-softmax graph (and the top-N gather over it) is CONSTRUCTED
        // here, before `applyPenalties` is called below, its captured inputs
        // are immune to that later mutation - no independent copy needed,
        // on the host or the GPU. `SamplerLogprobsTests` covers this
        // ordering directly (float32 and non-float32 logits, penalties
        // active). `rawLogSoftmaxAndTopN` is the same logic, factored out
        // (docs/LOGPROBS_PLAN.md Phase 2) so every argMax-only decode path
        // that bypasses `Sampler` entirely (speculative verify, the
        // continuous-batcher's fast paths, Stage-B batched decode) computes
        // logprobs with the IDENTICAL math, not a parallel re-implementation.
        let raw1D = to1D(logits).asType(.float32)
        let (logSoftmax, topIdx, topVals, n) = Sampler.rawLogSoftmaxAndTopN(
            raw1D, topLogprobs: topLogprobs)

        // NOW run the normal sampling path - temperature/top-k/top-p/min-p/
        // penalties/grammar mask - on its own array. `applyPenalties` may
        // mutate `to1D(logits)`'s object in place (see above), but the
        // log-softmax graph above has already captured what it needs.
        let forSampling = recent.isEmpty ? to1D(logits) : applyPenalties(to1D(logits), recent: recent)
        let chosenArr = sampleFrom(forSampling, mask: mask)
        // Gather the chosen token's own raw logprob from the ALREADY-BUILT
        // `logSoftmax` graph node (not from `raw1D`/`logits` again) - safe
        // regardless of `applyPenalties`'s later in-place mutation, same
        // reasoning as above.
        let chosenLogprobArr = Sampler.gatherChosenLogprob(logSoftmax, chosenIds: chosenArr)

        // ONE combined eval + host sync per step for everything this
        // function needs: the chosen token, its logprob, and the top-N.
        if let topIdx, let topVals {
            eval(chosenArr, chosenLogprobArr, topIdx, topVals)
        } else {
            eval(chosenArr, chosenLogprobArr)
        }
        let chosen = chosenArr.item(Int.self)
        let infos = Sampler.logprobInfos(
            chosenLogprobs: chosenLogprobArr, topIdx: topIdx, topVals: topVals, n: n)
        return (chosen, chosenArr, infos[0])
    }

    // MARK: - Shared argMax-path logprobs primitive (Phase 2)

    /// Phase 1 of the raw-logprobs computation: the raw log-softmax + top-N
    /// graph, built strictly from RAW logits with no notion yet of "the
    /// chosen token" (docs/LOGPROBS_PLAN.md §4.1: before any penalty /
    /// temperature / top-k / top-p / min-p / grammar-mask processing).
    /// `sampleWithLogprobs` calls this BEFORE `applyPenalties` runs — see its
    /// own doc comment for why the ORDERING (not a copy) is what protects it
    /// from `applyPenalties`'s in-place scatter. Every argMax-only decode
    /// path (speculative verify, continuous-batcher fast paths, Stage-B
    /// batched decode) that never touches `applyPenalties` at all has no such
    /// hazard and can call this at any point relative to its own `argMax`.
    ///
    /// Batched: pass `[N, vocab]` to compute N independent rows' log-softmax
    /// + top-N in one shot (one GPU round trip for a whole batch step,
    /// matching "one host sync per step" - `docs/LOGPROBS_PLAN.md` Phase 2's
    /// batched requirement). A 1-D `[vocab]` input is treated as N=1.
    ///
    /// - Parameters:
    ///   - logits: `[vocab]` or `[N, vocab]` raw logits.
    ///   - topLogprobs: number of top alternates per row; a value `<= 0`, or
    ///     `>= vocab` after clamping, still returns a valid (possibly empty)
    ///     result — never a precondition failure.
    /// - Returns: `logSoftmax` `[N, vocab]`, and (when the clamped `n > 0`)
    ///   `topIdx`/`topVals` each `[N, n]`, highest logprob first per row —
    ///   all lazy MLXArrays, not yet `eval`'d.
    public static func rawLogSoftmaxAndTopN(
        _ logits: MLXArray, topLogprobs: Int
    ) -> (logSoftmax: MLXArray, topIdx: MLXArray?, topVals: MLXArray?, n: Int) {
        let logits2D = logits.ndim <= 1 ? logits.reshaped(1, -1) : logits
        let raw = logits2D.asType(.float32)
        let logSoftmax = raw - raw.logSumExp(axis: -1, keepDims: true)
        let vocab = logSoftmax.dim(-1)
        let n = Swift.max(0, Swift.min(topLogprobs, vocab))
        guard n > 0 else { return (logSoftmax, nil, nil, 0) }
        // Same O(V) `argPartition` + O(N log N) `argSort`-on-the-candidates
        // approach as the single-row path, batched over axis -1 so every row
        // gets its own independent top-N in the same call.
        let negLogSoftmax = MLXArray(Float(0)) - logSoftmax
        let partitioned = argPartition(negLogSoftmax, kth: n - 1, axis: -1)
        let candIdx = partitioned[0..., 0 ..< n]
        let candVals = takeAlong(logSoftmax, candIdx, axis: -1)
        let order = argSort(MLXArray(Float(0)) - candVals, axis: -1)
        let topIdx = takeAlong(candIdx, order, axis: -1)
        let topVals = takeAlong(candVals, order, axis: -1)
        return (logSoftmax, topIdx, topVals, n)
    }

    /// Phase 2: gather each row's own chosen token's logprob out of an
    /// already-built `logSoftmax` (from `rawLogSoftmaxAndTopN`). Safe to call
    /// at any point afterward — it reads the already-constructed graph node,
    /// never the original raw logits array again, so it cannot observe a
    /// later in-place mutation of that input (e.g. `applyPenalties`'s
    /// scatter in `sampleWithLogprobs`, or an argMax-path caller's own later
    /// reuse of the same logits buffer).
    ///
    /// - Parameters:
    ///   - logSoftmax: `[N, vocab]`, from `rawLogSoftmaxAndTopN`.
    ///   - chosenIds: scalar / `[1]` / `[N]` int token ids, one per row (e.g.
    ///     an `argMax` result — NOT re-derived here).
    /// - Returns: `[N]` lazy chosen logprobs, not yet `eval`'d.
    public static func gatherChosenLogprob(_ logSoftmax: MLXArray, chosenIds: MLXArray) -> MLXArray {
        let chosen2D = chosenIds.asType(.int32).reshaped(-1, 1)
        return takeAlong(logSoftmax, chosen2D, axis: -1).reshaped(-1)
    }

    /// Host-side materialization of `rawLogSoftmaxAndTopN` +
    /// `gatherChosenLogprob`'s lazy outputs into one `TokenLogprobInfo` per
    /// row, in row order. Caller MUST have already `eval()`'d
    /// `chosenLogprobs` (and `topIdx`/`topVals` when `n > 0`) — this only
    /// does host reads (`asArray`), no GPU work.
    public static func logprobInfos(
        chosenLogprobs: MLXArray, topIdx: MLXArray?, topVals: MLXArray?, n: Int
    ) -> [TokenLogprobInfo] {
        let chosenHost = chosenLogprobs.asArray(Float.self)
        guard n > 0, let topIdx, let topVals else {
            return chosenHost.map { TokenLogprobInfo(logprob: $0, topAlternates: []) }
        }
        let idxHost = topIdx.asArray(Int32.self)
        let valHost = topVals.asArray(Float.self)
        return chosenHost.indices.map { i in
            var alts: [TokenAltLogprob] = []
            alts.reserveCapacity(n)
            for j in 0 ..< n {
                alts.append(TokenAltLogprob(tokenId: Int(idxHost[i * n + j]), logprob: valHost[i * n + j]))
            }
            return TokenLogprobInfo(logprob: chosenHost[i], topAlternates: alts)
        }
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
        // categorical takes LOGITS, so pass log(p) - handing it the normalized
        // probabilities directly would flatten the truncated head toward uniform
        // (the same defect fixed in `sampleFrom`). The epsilon keeps log finite
        // for a probability that underflowed to 0.
        let pick = MLXRandom.categorical(
            expandedDimensions(MLX.log(renorm + 1e-20), axis: 0)).item(Int.self)
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

// MARK: - Logprobs (OpenAI/Ollama `logprobs` support, Phase 1)

/// One alternate token's raw log-probability, part of `TokenLogprobInfo.topAlternates`.
public struct TokenAltLogprob: Sendable, Equatable {
    public let tokenId: Int
    public let logprob: Float
    public init(tokenId: Int, logprob: Float) {
        self.tokenId = tokenId
        self.logprob = logprob
    }
}

/// Raw-distribution logprob info for one sampled token: a plain
/// log-softmax of the RAW forward-pass logits, computed in float32,
/// BEFORE `applyPenalties`, temperature scaling, top-k/top-p/min-p
/// truncation, and any grammar mask — see docs/LOGPROBS_PLAN.md §4.1.
/// `nil` on a `TokenEvent` unless the request asked for logprobs, so the
/// default decode path never computes this.
public struct TokenLogprobInfo: Sendable, Equatable {
    /// The sampled token's own raw logprob (identically defined at
    /// temperature 0 / greedy, since it does not depend on the sampling path).
    public let logprob: Float
    /// True top-N alternates of the raw distribution, highest logprob first.
    /// Length equals the request's `top_logprobs` (0...20) — NOT necessarily
    /// including the sampled token unless it is genuinely in the top-N.
    public let topAlternates: [TokenAltLogprob]
    public init(logprob: Float, topAlternates: [TokenAltLogprob]) {
        self.logprob = logprob
        self.topAlternates = topAlternates
    }
}

/// Scatter `values` into `base` at `indices` (1-D), returning a new array.
private func updatedAt(_ base: MLXArray, indices: MLXArray, values: MLXArray) -> MLXArray {
    let out = base
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
