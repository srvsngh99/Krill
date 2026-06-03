import Foundation
import MLX
import KLMCore
import KLMCache
import KLMSampler

/// Native decode driver for Llama-3.2-Vision (mllama).
///
/// Like `Qwen25VLRuntime`, this is the runtime layer on top of the
/// `Llama32VisionForCausalLM` Swift+MLX model: an image (or text-only) prefill
/// followed by an incremental KV-cached decode loop. It is token-id-based and
/// tokenizer-free so `InferenceEngine` (which owns the tokenizer) can drive it
/// and unit tests can exercise it against a synthetic model.
///
/// ## Why a dedicated driver
///
/// Unlike LLaVA, mllama does NOT splice image features into the embedding
/// prefix; the image enters through gated **cross-attention** on the
/// `cross_attention_layers`, which every decode step must re-attend to. The
/// generic decode loop has no place to thread that, so this driver:
///   - computes the vision K/V once at prefill into a `MllamaCrossKVCache`,
///     reused unchanged on every decode step (the image is consumed in the
///     prompt, so the vision K/V never grow);
///   - builds the sparse `cross_attention_mask` from the `<|image|>` token
///     positions and applies the LAST prompt row's mask on every decode step
///     (decode tokens follow the last image, so they attend to all images,
///     matching HF's `cross_attention_mask[:, -1:]` slice).
public enum MllamaRuntime {

    /// The result of a native mllama generation.
    public struct Output: Sendable {
        /// Generated token ids (excludes the prompt).
        public let tokens: [Int]
        /// Wall-clock seconds spent in prefill (prompt forward).
        public let prefillSeconds: Double
        /// Wall-clock seconds spent in the decode loop.
        public let decodeSeconds: Double
    }

    /// Run native image (or text-only) prefill + incremental decode.
    ///
    /// - Parameters:
    ///   - model: the loaded native mllama model.
    ///   - promptTokens: the full prompt token ids, already containing one
    ///     `<|image|>` token per supplied image (in order).
    ///   - vision: preprocessed multi-image inputs, or nil for a text-only
    ///     prompt (which runs the decoder with its zero-gated cross-attention).
    ///   - maxTokens: generation cap (excludes the prompt).
    ///   - stopIds: token ids that terminate generation.
    ///   - params: sampling parameters.
    ///   - onToken: invoked with each generated token id as it is produced
    ///     (before the stop check) so a streaming caller can emit incrementally.
    public static func generate(
        model: Llama32VisionForCausalLM,
        promptTokens: [Int],
        vision: MllamaProcessing.VisionInputs?,
        maxTokens: Int,
        stopIds: Set<Int>,
        params: SamplingParams = .greedy,
        onToken: ((Int) -> Void)? = nil
    ) -> Output {
        let numLayers = model.config.textConfig.numHiddenLayers
        let caches = makeKVCaches(numLayers: numLayers)
        let sampler = Sampler(params: params)
        let crossKV = MllamaCrossKVCache()

        // Build the additive cross-attention mask from the prompt's image-token
        // positions. The full mask drives prefill; decode steps (which follow
        // the last image) reuse the final prompt row.
        var prefillCross: MLXArray? = nil
        var prefillFullRow: MLXArray? = nil
        var decodeCross: MLXArray? = nil
        var decodeFullRow: MLXArray? = nil
        if let vision {
            let length = promptTokens.count
            let maxImages = vision.numTiles.count
            let maxTiles = model.config.visionConfig.maxNumTiles
            let numVisionTokens = model.config.visionConfig.numPatches
            let tokenMask = MllamaProcessing.crossAttentionTokenMask(
                inputIds: promptTokens, imageTokenId: model.config.imageTokenIndex)
            let dense = MllamaProcessing.denseCrossMask(
                tokenMask: tokenMask, numTiles: vision.numTiles,
                maxImages: maxImages, maxTiles: maxTiles, length: length)
            let (cross, fullRow) = MllamaProcessing.prepareCrossMask(
                dense: dense, length: length, maxImages: maxImages,
                maxTiles: maxTiles, numVisionTokens: numVisionTokens)
            prefillCross = cross
            prefillFullRow = fullRow
            decodeCross = cross[0..., 0..., (length - 1) ..< length, 0...]
            decodeFullRow = fullRow[0..., (length - 1) ..< length, 0...]
        }

        // -- Prefill --
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let promptArray = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)
        let prefillLogits: MLXArray
        if let vision {
            prefillLogits = model(
                promptArray, pixelValues: vision.pixelValues,
                aspectRatioIds: vision.aspectRatioIds, aspectRatioMask: vision.aspectRatioMask,
                caches: caches, crossKV: crossKV,
                crossMask: prefillCross, fullRowMask: prefillFullRow,
                lastTokenOnly: true)
        } else {
            prefillLogits = model(promptArray, caches: caches, lastTokenOnly: true)
        }
        MLX.eval(prefillLogits)
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        // -- Decode --
        // Two-deep on-GPU pipeline (mirrors the dense decode loop and
        // Qwen25VLRuntime): keep the sampled token on-GPU, asyncEval the next
        // forward + sample while yielding the current token, sync once per step.
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generated: [Int] = []
        var recent: [Int] = sampler.needsHistory ? Array(promptTokens.suffix(512)) : []
        var nextTokenArr: MLXArray = sampler.needsHistory
            ? sampler.sampleArray(prefillLogits, recent: recent)
            : sampler.sampleArray(prefillLogits)
        MLX.asyncEval(nextTokenArr)
        var nextToken = nextTokenArr.item(Int.self)
        while generated.count < maxTokens {
            if stopIds.contains(nextToken) {
                onToken?(nextToken)
                generated.append(nextToken)
                break
            }
            let tokenInput = nextTokenArr.reshaped(1, 1)
            let logits: MLXArray
            if vision != nil {
                logits = model(
                    tokenInput, caches: caches, crossKV: crossKV,
                    crossMask: decodeCross, fullRowMask: decodeFullRow,
                    lastTokenOnly: true)
            } else {
                logits = model(tokenInput, caches: caches, lastTokenOnly: true)
            }
            if sampler.needsHistory { recent.append(nextToken) }
            let nextTokenArr2: MLXArray = sampler.needsHistory
                ? sampler.sampleArray(logits, recent: recent)
                : sampler.sampleArray(logits)
            MLX.asyncEval(nextTokenArr2)
            onToken?(nextToken)
            generated.append(nextToken)
            nextTokenArr = nextTokenArr2
            nextToken = nextTokenArr.item(Int.self)
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decodeStart

        return Output(
            tokens: generated, prefillSeconds: prefillSeconds, decodeSeconds: decodeSeconds)
    }
}
