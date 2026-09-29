import Foundation
import MLX
import KrillCore
import KrillCache
import KrillSampler

/// Native decode driver for Qwen3.5-VL (Ornith): a prefill pass followed by an
/// incremental cached decode loop over `Qwen35VLForConditionalGeneration`.
/// Analogous to `Qwen25VLRuntime`, with two Ornith-specific differences:
///
///   * **Heterogeneous caches.** The hybrid decoder mixes full-attn `KVCache`
///     layers with GatedDeltaNet `GatedDeltaCache` (SSM) layers; the model's
///     own `makeCaches()` builds the right per-layer mix.
///   * **No prefix-cache fast path.** The SSM recurrent state is NOT restorable
///     (a partial restore would leave the 24 SSM layers cold while the 8
///     full-attn layers carry history → desync → garbage — the exact text-port
///     bug). This driver always does a full prefill.
///
/// Like the 2.5-VL driver it threads the precise per-step mRoPE offset: after
/// an image prompt the cache length is NOT the next mRoPE position, because the
/// `gridH*gridW` image-pad tokens occupy that many cache slots but only
/// `max(gridH, gridW)` mRoPE positions.
public enum Qwen35VLRuntime {

    public struct Output: Sendable {
        public let tokens: [Int]
        public let prefillSeconds: Double
        public let decodeSeconds: Double
    }

    /// - Parameters:
    ///   - model: the loaded native Qwen3.5-VL model.
    ///   - promptTokens: full prompt token ids (must already contain the
    ///     contiguous image-pad run for an image request).
    ///   - pixelValues: preprocessed flattened patch batch
    ///     `[nPatches, C*T*ph*pw]`, or nil for text-only.
    ///   - grid: full patch grid `(t, h, w)` of the image, or nil for text-only.
    ///   - maxTokens: generation cap (excludes the prompt).
    ///   - stopIds: terminating token ids.
    /// - Parameters:
    ///   - wantLogprobs: When true, every sampled token's raw-distribution
    ///     logprob (and top-N alternates) is computed via `Sampler.
    ///     sampleWithLogprobs` instead of the plain `sampleArray`/`sample`
    ///     path, and threaded to `onToken`'s second argument - the same
    ///     "raw pre-filter logits" semantics as the generic decode loop in
    ///     `InferenceEngine.generate(messages:)` (docs/LOGPROBS_PLAN.md
    ///     §4.1/§5.1). This was the root cause of `logprobs.content` always
    ///     being `[]` for qwen3_5 models: this runtime never computed or
    ///     threaded logprob info at all before, regardless of the request
    ///     (docs/LOGPROBS_QWEN35_EMPTY.md).
    ///   - topLogprobs: Number of raw top-N alternates to report (0...20),
    ///     ignored when `wantLogprobs` is false.
    ///   - onToken: called once per generated token (including the one that
    ///     triggers a stop) with `(tokenId, logprobInfo)`; `logprobInfo` is
    ///     `nil` unless `wantLogprobs` was set.
    public static func generate(
        model: Qwen35VLForConditionalGeneration,
        promptTokens: [Int],
        pixelValues: MLXArray?,
        grid: (t: Int, h: Int, w: Int)?,
        maxTokens: Int,
        stopIds: Set<Int>,
        params: SamplingParams = .greedy,
        mediaHash: String? = nil,
        wantLogprobs: Bool = false,
        topLogprobs: Int = 0,
        onToken: ((Int, TokenLogprobInfo?) -> Void)? = nil
    ) -> Output {
        let caches = model.makeCaches()
        let sampler = Sampler(params: params)
        let promptInt32 = promptTokens.map { Int32($0) }

        // mRoPE frontier: the absolute position the first post-prompt token
        // takes (compressed across the image span).
        let ms = model.config.visionConfig.spatialMergeSize
        let gridHMerged = grid.map { $0.h / ms } ?? 0
        let gridWMerged = grid.map { $0.w / ms } ?? 0
        let frontier = model.nextMRoPEPosition(
            tokenIds: promptInt32, gridHMerged: gridHMerged, gridWMerged: gridWMerged)

        // -- Prefill (always full; no prefix cache for SSM) --
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let promptArray = MLXArray(promptInt32).reshaped(1, promptTokens.count)
        let prefillLogits = model(
            promptArray,
            pixelValues: pixelValues,
            grid: grid,
            caches: caches,
            mropePositionOffset: nil,
            hostTokenIds: promptInt32,
            lastTokenOnly: true,
            mediaHash: mediaHash)
        MLX.eval(prefillLogits)
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        // -- Decode (two-deep on-GPU pipeline; offset = frontier + step) --
        //
        // `pendingLogprobInfo` mirrors the generic decode loop's convention
        // (InferenceEngine.swift's main loop, `pendingLogprobInfo`): the
        // logprob for a token is computed at the moment IT is sampled (one
        // iteration before it is yielded via `onToken`), then carried
        // forward one step so it can be attached when that token is
        // actually reported.
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generated: [Int] = []
        var recent: [Int] = sampler.needsHistory
            ? Array(promptTokens.suffix(512)) : []
        var nextTokenArr: MLXArray
        var pendingLogprobInfo: TokenLogprobInfo?
        if wantLogprobs {
            let (_, tokArr, info) = sampler.needsHistory
                ? sampler.sampleWithLogprobs(prefillLogits, recent: recent, topLogprobs: topLogprobs)
                : sampler.sampleWithLogprobs(prefillLogits, topLogprobs: topLogprobs)
            nextTokenArr = tokArr
            pendingLogprobInfo = info
        } else {
            nextTokenArr = sampler.needsHistory
                ? sampler.sampleArray(prefillLogits, recent: recent)
                : sampler.sampleArray(prefillLogits)
        }
        MLX.asyncEval(nextTokenArr)
        var nextToken = nextTokenArr.item(Int.self)
        let decodePlaceholder: [Int32] = [Int32(0)]
        var step = 0
        while generated.count < maxTokens {
            if stopIds.contains(nextToken) {
                onToken?(nextToken, pendingLogprobInfo)
                generated.append(nextToken)
                break
            }
            let tokenInput = nextTokenArr.reshaped(1, 1)
            let logits = model(
                tokenInput,
                pixelValues: nil,
                grid: nil,
                caches: caches,
                mropePositionOffset: frontier + step,
                hostTokenIds: decodePlaceholder,
                lastTokenOnly: true)
            if sampler.needsHistory { recent.append(nextToken) }
            let nextTokenArr2: MLXArray
            var stepLogprobInfo: TokenLogprobInfo?
            if wantLogprobs {
                let (_, tokArr, info) = sampler.needsHistory
                    ? sampler.sampleWithLogprobs(logits, recent: recent, topLogprobs: topLogprobs)
                    : sampler.sampleWithLogprobs(logits, topLogprobs: topLogprobs)
                nextTokenArr2 = tokArr
                stepLogprobInfo = info
            } else {
                nextTokenArr2 = sampler.needsHistory
                    ? sampler.sampleArray(logits, recent: recent)
                    : sampler.sampleArray(logits)
            }
            MLX.asyncEval(nextTokenArr2)
            onToken?(nextToken, pendingLogprobInfo)
            generated.append(nextToken)
            nextTokenArr = nextTokenArr2
            nextToken = nextTokenArr.item(Int.self)
            pendingLogprobInfo = stepLogprobInfo
            step += 1
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart

        return Output(
            tokens: generated,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds)
    }
}
