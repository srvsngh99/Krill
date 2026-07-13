import Foundation
import MLX
import KrillCore
import KrillCache
import KrillSampler

/// Native decode driver for NVIDIA LocateAnything-3B.
///
/// A multimodal prefill (MoonViT features spliced into the Qwen2.5 token
/// embeddings at the `<IMG_CONTEXT>` run) followed by a standard KV-cached AR
/// decode. Unlike the Qwen 2.5/3.5-VL drivers, LocateAnything uses plain 1D
/// RoPE, so decode needs NO per-step positional offset — the KV-cache length is
/// the next position, exactly like the dense text path. The dedicated driver
/// exists only because the per-image MoonViT `(h,w)` grid must be threaded into
/// the vision tower (it is not recoverable from the patch count, so the generic
/// six-arg `multimodalForward` closure cannot carry it).
///
/// Token-id based and tokenizer-free so `InferenceEngine` (which owns the
/// tokenizer) can drive it and unit tests can exercise it against a synthetic
/// model.
public enum LocateAnythingRuntime {

    public struct Output: Sendable {
        public let tokens: [Int]
        public let prefillSeconds: Double
        public let decodeSeconds: Double
    }

    /// Run native prefill + incremental decode.
    ///
    /// - Parameters:
    ///   - model: the loaded native LocateAnything model.
    ///   - promptTokens: full prompt ids. For an image request this MUST contain
    ///     the contiguous `<IMG_CONTEXT>` run (`(gridH*gridW)/(mergeH*mergeW)`
    ///     of them) that the spliced vision features replace.
    ///   - pixelValues: preprocessed flattened patch batch `[N, C*ph*pw]`, or nil
    ///     for a text-only prompt.
    ///   - grid: pre-merge `(gridH, gridW)` of the image, or nil for text-only.
    ///   - onToken: invoked with each generated token id (before the stop check).
    public static func generate(
        model: LocateAnythingForConditionalGeneration,
        promptTokens: [Int],
        pixelValues: MLXArray?,
        grid: (Int, Int)?,
        maxTokens: Int,
        stopIds: Set<Int>,
        params: SamplingParams = .greedy,
        onToken: ((Int) -> Void)? = nil
    ) -> Output {
        let caches = makeKVCaches(numLayers: model.config.textConfig.numHiddenLayers)
        let sampler = Sampler(params: params)
        let promptInt32 = promptTokens.map { Int32($0) }

        // -- Prefill --
        let prefillStart = CFAbsoluteTimeGetCurrent()
        let promptArray = MLXArray(promptInt32).reshaped(1, promptTokens.count)
        let prefillLogits: MLXArray
        if let pixelValues, let grid {
            prefillLogits = model(
                promptArray,
                pixelValues: pixelValues,
                grids: [(h: grid.0, w: grid.1)],
                caches: caches,
                lastTokenOnly: true)
        } else {
            prefillLogits = model(promptArray, caches: caches, lastTokenOnly: true)
        }
        MLX.eval(prefillLogits)
        let prefillSeconds = CFAbsoluteTimeGetCurrent() - prefillStart

        // -- Decode -- (two-deep on-GPU pipeline, mirrors the dense loop)
        let decodeStart = CFAbsoluteTimeGetCurrent()
        var generated: [Int] = []
        var recent: [Int] = sampler.needsHistory
            ? Array(promptTokens.suffix(512)) : []
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
            let logits = model(tokenInput, caches: caches, lastTokenOnly: true)
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
            tokens: generated,
            prefillSeconds: prefillSeconds,
            decodeSeconds: decodeSeconds)
    }
}
