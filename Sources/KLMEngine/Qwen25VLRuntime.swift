import Foundation
import MLX
import KLMCore
import KLMCache
import KLMSampler

/// Native decode driver for Qwen 2.5-VL.
///
/// This is the runtime layer on top of the
/// `Qwen25VLForConditionalGeneration` Swift+MLX model: a prefill
/// pass followed by an incremental KV-cached decode loop. It is
/// deliberately token-id-based and tokenizer-free so it can be
/// driven by `InferenceEngine` (which owns the tokenizer) AND
/// unit-tested directly against a synthetic model.
///
/// ## Why a dedicated driver
///
/// The generic `InferenceEngine` decode loop forwards each decode
/// step with no explicit positional offset, which is correct for
/// 1D-RoPE families: the KV-cache length IS the next position. Qwen
/// 2.5-VL uses 3D mRoPE, and after an image prompt the cache length
/// is NOT the next mRoPE position - the `gridH * gridW`
/// `<|image_pad|>` tokens occupy `gridH * gridW` cache slots but
/// only `max(gridH, gridW)` mRoPE positions. This driver threads
/// the precise per-step mRoPE offset (`Qwen25VLForConditionalGeneration`'s
/// `mropePositionOffset`) so decode positioning is correct for
/// image prompts and unchanged for text-only prompts.
public enum Qwen25VLRuntime {

    /// The result of a native Qwen 2.5-VL generation.
    public struct Output: Sendable {
        /// Generated token ids (excludes the prompt).
        public let tokens: [Int]
        /// Wall-clock seconds spent in prefill (prompt forward).
        public let prefillSeconds: Double
        /// Wall-clock seconds spent in the decode loop.
        public let decodeSeconds: Double
    }

    /// Run native prefill + incremental decode.
    ///
    /// - Parameters:
    ///   - model: the loaded native Qwen 2.5-VL model.
    ///   - promptTokens: the full prompt token ids. For an image
    ///     request this MUST already contain the contiguous
    ///     `<|image_pad|>` run (`gridH * gridW` of them).
    ///   - pixelValues: preprocessed per-patch image batch
    ///     `[n_patches, T, ps, ps, C]`, or nil for a text-only
    ///     prompt.
    ///   - imageGridMerged: post-spatial-merge `(gridH, gridW)` of
    ///     the image, or nil for a text-only prompt. Its product
    ///     must equal the `<|image_pad|>` count in `promptTokens`.
    ///   - maxTokens: generation cap (excludes the prompt).
    ///   - stopIds: token ids that terminate generation.
    ///   - params: sampling parameters.
    ///   - onToken: invoked with each generated token id as it is
    ///     produced (before the stop check), so a streaming caller
    ///     can emit it incrementally.
    public static func generate(
        model: Qwen25VLForConditionalGeneration,
        promptTokens: [Int],
        pixelValues: MLXArray?,
        imageGridMerged: (Int, Int)?,
        maxTokens: Int,
        stopIds: Set<Int>,
        params: SamplingParams = .greedy,
        onToken: ((Int) -> Void)? = nil
    ) -> Output {
        let caches = makeKVCaches(numLayers: model.config.numHiddenLayers)
        let sampler = Sampler(params: params)
        let promptInt32 = promptTokens.map { Int32($0) }

        // -- Prefill --
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let promptArray = MLXArray(promptInt32)
            .reshaped(1, promptTokens.count)
        // `hostTokenIds` lets the forward skip the mid-forward
        // eval(tokens) host sync (we already have the host array).
        // `lastTokenOnly` drops the vocab matmul to the single
        // position we actually sample - exact for that token.
        let prefillLogits = model(
            promptArray,
            pixelValues: pixelValues,
            imageGridMerged: imageGridMerged,
            caches: caches,
            mropePositionOffset: nil,
            hostTokenIds: promptInt32,
            lastTokenOnly: true)
        MLX.eval(prefillLogits)
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        // The mRoPE frontier: the absolute position the FIRST
        // post-prompt token takes. For a text-only prompt this is
        // the prompt length; for an image prompt it is smaller
        // (the image span is compressed - see the type doc). Decode
        // step `k` forwards its token with offset `frontier + k`.
        let frontier = Int(Qwen25VLPositions.compute(
            tokenIds: promptInt32,
            imageTokenId: model.config.imageTokenId,
            gridHMerged: imageGridMerged?.0 ?? 0,
            gridWMerged: imageGridMerged?.1 ?? 0).nextPos)

        // -- Decode --
        // Two-deep on-GPU pipeline (mirrors InferenceEngine's dense
        // decode loop). The sampled token stays as a lazy MLXArray
        // and feeds the next forward without a host roundtrip; the
        // next forward + sample is `asyncEval`d while we yield the
        // current token; we sync once per step on a 1-element int32.
        // For a non-image single decode token the host id is never
        // consulted (the image-pad scan is skipped because
        // pixelValues is nil, and the mRoPE coords for a single
        // text token are always (0,0,0) regardless of id), so we
        // pass a fixed placeholder host id and keep the real token
        // on-GPU.
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generated: [Int] = []
        // Penalty sampling needs the trailing token window; only
        // maintained when the request opts into penalties.
        var recent: [Int] = sampler.needsHistory
            ? Array(promptTokens.suffix(512)) : []
        var nextTokenArr: MLXArray = sampler.needsHistory
            ? sampler.sampleArray(prefillLogits, recent: recent)
            : sampler.sampleArray(prefillLogits)
        MLX.asyncEval(nextTokenArr)
        var nextToken = nextTokenArr.item(Int.self)
        let decodePlaceholder: [Int32] = [Int32(0)]
        var step = 0
        while generated.count < maxTokens {
            onToken?(nextToken)
            generated.append(nextToken)
            if stopIds.contains(nextToken) { break }
            if generated.count >= maxTokens { break }

            // Reuse the on-GPU sampled token as the next forward
            // input; mRoPE position is `frontier + step` because a
            // single non-image token's computed coords are all zero.
            let tokenInput = nextTokenArr.reshaped(1, 1)
            let logits = model(
                tokenInput,
                pixelValues: nil,
                imageGridMerged: nil,
                caches: caches,
                mropePositionOffset: frontier + step,
                hostTokenIds: decodePlaceholder,
                lastTokenOnly: true)
            let nextTokenArr2: MLXArray
            if sampler.needsHistory {
                recent.append(nextToken)
                nextTokenArr2 = sampler.sampleArray(logits, recent: recent)
            } else {
                nextTokenArr2 = sampler.sampleArray(logits)
            }
            // Kick GPU work for step k+1 before reading step k's
            // host int below, so kernel launch overlaps execution.
            MLX.asyncEval(nextTokenArr2)
            nextTokenArr = nextTokenArr2
            nextToken = nextTokenArr.item(Int.self)
            step += 1
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart

        return Output(
            tokens: generated,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds)
    }
}
