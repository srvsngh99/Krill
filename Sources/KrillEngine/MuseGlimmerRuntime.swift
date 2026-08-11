import Foundation
import MLX
import KrillCore
import KrillCache
import KrillSampler

/// Native decode driver for Muse Glimmer 30B image requests.
///
/// A multimodal prefill (perception-encoder features spliced into the token
/// embeddings at the `<|patch|>` run) followed by a standard KV-cached AR
/// decode. Like LocateAnything and unlike the Qwen VL families, Muse Glimmer
/// uses ordinary 1-D positions — no mRoPE — so decode needs NO per-step
/// positional offset: after the splice, the decode path IS the text path.
///
/// The dedicated driver exists because the generic six-argument
/// `multimodalForward` closure cannot carry the per-image patch GRID, and a
/// non-square grid is not recoverable from the patch count alone.
///
/// Two things here differ from the LocateAnything driver and must not be
/// "simplified" to match it:
///   * caches come from the family's `cacheSpec` — 39 of 52 layers are
///     `.rotating(window: 2048)`, so the spec-less `makeKVCaches(numLayers:)`
///     would allocate full-history caches and diverge from the served model.
///   * `hostTokenIds` is passed so the splice can locate the `<|patch|>` run
///     without a GPU->CPU sync on the prompt array.
///
/// Token-id based and tokenizer-free, so `InferenceEngine` (which owns the
/// tokenizer) drives it and tests can exercise it against a synthetic model.
public enum MuseGlimmerRuntime {

    public struct Output: Sendable {
        public let tokens: [Int]
        public let prefillSeconds: Double
        public let decodeSeconds: Double
    }

    /// Run native prefill + incremental decode.
    ///
    /// - Parameters:
    ///   - model: the loaded native Muse Glimmer model.
    ///   - promptTokens: full prompt ids. For an image request this MUST contain
    ///     the contiguous `<|patch|>` run — `(gridH/merge) * (gridW/merge)` ids —
    ///     that the spliced vision features replace. A count mismatch is a
    ///     precondition failure inside the model, not a silent wrong answer.
    ///   - pixelValues: preprocessed patch batch `[N, patchDim]`, or nil for text.
    ///   - grid: the FULL (pre-merge) patch grid, or nil for text-only.
    public static func generate(
        model: MuseGlimmerForConditionalGeneration,
        promptTokens: [Int],
        pixelValues: MLXArray?,
        grid: (t: Int, h: Int, w: Int)?,
        maxTokens: Int,
        stopIds: Set<Int>,
        params: SamplingParams = .greedy,
        mediaHash: String? = nil,
        onToken: ((Int) -> Void)? = nil
    ) -> Output {
        let caches = model.makeCaches()
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
                grids: [grid],
                caches: caches,
                hostTokenIds: promptInt32,
                lastTokenOnly: true,
                mediaHash: mediaHash)
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

    /// Number of `<|patch|>` placeholders a preprocessed image occupies: the
    /// full patch grid divided by the merge factor on BOTH axes (the projector
    /// pixel-shuffles `merge x merge` patches into one LM token).
    public static func placeholderCount(
        grid: (t: Int, h: Int, w: Int), mergeSize: Int
    ) -> Int {
        let m = Swift.max(1, mergeSize)
        return grid.t * (grid.h / m) * (grid.w / m)
    }
}
