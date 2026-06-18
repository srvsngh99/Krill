import Foundation
import MLX
import MLXNN
import MLXFast
import KrillCache

// MARK: - Gemma 4 "Unified" (encoder-free) multimodal model
//
// Gemma 4 12B (`model_type: "gemma4_unified"`,
// `Gemma4UnifiedForConditionalGeneration`) is the first Gemma 4 SKU that
// is *encoder-free*: it has no SigLIP vision tower and no USM Conformer
// audio encoder. Instead it projects raw image patches and raw audio
// sample-frames straight into the language-model embedding space through
// thin linear pipelines:
//
//   vision: patchify(48x48) -> LayerNorm -> Linear -> LayerNorm
//           -> +2D pos-embedding -> LayerNorm        (`vision_embedder.*`)
//           -> RMSNorm(no-scale) -> Linear           (`embed_vision.*`)
//   audio : raw 640-sample frames
//           -> RMSNorm(no-scale) -> Linear           (`embed_audio.*`)
//
// The TEXT backbone is the *identical* dense Gemma 4 decoder we already
// ship (`Gemma4ForCausalLM` / `Gemma4TextModel` / `Gemma4Block`): 48
// layers, 3840 hidden, 16/8 heads, head_dim 256, sliding-window 1024 with
// every-6th full-attention, proportional RoPE on full layers, K-eq-V on
// full layers (`attention_k_eq_v=true`, `num_global_key_value_heads=1`),
// logit softcap 30. No PLE, no MoE, no KV-sharing. The text `Gemma4Config`
// already parses all of these from the nested `text_config`, and
// `Gemma4Attention.useKEqV` already reuses K-as-V on full layers only -
// which is exactly what the 12B checkpoint ships (sliding layers keep
// `v_proj`; the every-6th full layers drop it).
//
// This file therefore reuses every text module unchanged and only adds
// the two encoder-free media front-ends plus a wrapper that injects their
// outputs at the image / audio placeholder positions. The projector
// (`embed_vision` / `embed_audio`) reuses the existing `MultimodalEmbedder`
// (RMSNormNoScale + Linear) - the checkpoint key path
// `embed_{vision,audio}.embedding_projection.*` matches it byte for byte.

// MARK: - Unified vision config

/// `config.json["vision_config"]` for `gemma4_unified_vision`. Encoder-free:
/// no transformer-tower fields, just the patch projector geometry. All
/// values are read from the checkpoint; defaults match the released 12B.
public struct Gemma4UnifiedVisionConfig: Sendable {
    public let patchSize: Int          // 16  (sub-patch; used by the processor)
    public let poolingKernelSize: Int  // 3   (used by the processor)
    public let modelPatchSize: Int     // 48  (= patchSize * poolingKernelSize)
    public let mmEmbedDim: Int         // 3840
    public let mmPosembSize: Int       // 1120 (position-embedding table length)
    public let numSoftTokens: Int      // 280  (max soft tokens per image)
    public let outputProjDims: Int     // 3840 (embed_vision input dim)
    public let rmsNormEps: Float       // 1e-6

    /// Flattened raw-patch dimension fed to `patch_dense`:
    /// `model_patch_size^2 * 3` (48*48*3 = 6912).
    public var patchDim: Int { modelPatchSize * modelPatchSize * 3 }

    public init(
        patchSize: Int = 16, poolingKernelSize: Int = 3, modelPatchSize: Int = 48,
        mmEmbedDim: Int = 3840, mmPosembSize: Int = 1120, numSoftTokens: Int = 280,
        outputProjDims: Int = 3840, rmsNormEps: Float = 1e-6
    ) {
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.modelPatchSize = modelPatchSize
        self.mmEmbedDim = mmEmbedDim
        self.mmPosembSize = mmPosembSize
        self.numSoftTokens = numSoftTokens
        self.outputProjDims = outputProjDims
        self.rmsNormEps = rmsNormEps
    }

    /// Parse from the raw `vision_config` dict. Honors `model_patch_size`
    /// directly, or derives it from `patch_size * pooling_kernel_size`.
    public init(from dict: [String: Any]?) {
        let d = dict ?? [:]
        let patch = d["patch_size"] as? Int ?? 16
        let pool = d["pooling_kernel_size"] as? Int ?? 3
        let modelPatch = d["model_patch_size"] as? Int ?? (patch * pool)
        let embed = d["mm_embed_dim"] as? Int ?? 3840
        self.init(
            patchSize: patch,
            poolingKernelSize: pool,
            modelPatchSize: modelPatch,
            mmEmbedDim: embed,
            mmPosembSize: d["mm_posemb_size"] as? Int ?? 1120,
            numSoftTokens: d["num_soft_tokens"] as? Int ?? 280,
            outputProjDims: d["output_proj_dims"] as? Int ?? embed,
            rmsNormEps: (d["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-6)
    }
}

// MARK: - Unified audio config

/// `config.json["audio_config"]` for `gemma4_unified_audio`. Encoder-free:
/// raw audio is reshaped into `audio_samples_per_token`-sized frames and
/// projected directly. The mel/fft fields some checkpoints carry in the
/// *processor* config are vestigial and unused by the model - the
/// projection input dim is `output_proj_dims == audio_embed_dim == 640`,
/// which is the raw frame width, not a mel-bin count.
public struct Gemma4UnifiedAudioConfig: Sendable {
    public let audioSamplesPerToken: Int  // 640
    public let audioEmbedDim: Int         // 640
    public let outputProjDims: Int        // 640 (embed_audio input dim)
    public let rmsNormEps: Float          // 1e-6

    public init(
        audioSamplesPerToken: Int = 640, audioEmbedDim: Int = 640,
        outputProjDims: Int = 640, rmsNormEps: Float = 1e-6
    ) {
        self.audioSamplesPerToken = audioSamplesPerToken
        self.audioEmbedDim = audioEmbedDim
        self.outputProjDims = outputProjDims
        self.rmsNormEps = rmsNormEps
    }

    public init(from dict: [String: Any]?) {
        let d = dict ?? [:]
        let embed = d["audio_embed_dim"] as? Int ?? 640
        self.init(
            audioSamplesPerToken: d["audio_samples_per_token"] as? Int ?? embed,
            audioEmbedDim: embed,
            outputProjDims: d["output_proj_dims"] as? Int ?? embed,
            rmsNormEps: (d["rms_norm_eps"] as? Double).map(Float.init) ?? 1e-6)
    }
}

// MARK: - Encoder-free vision embedder (`vision_embedder.*`)

/// The entire Gemma 4 unified vision "tower". Operates per-patch (no
/// attention): each row is independent, so it tolerates either the
/// reference's fixed `[B, num_soft_tokens, patch_dim]` padded input or a
/// trimmed `[1, num_valid_patches, patch_dim]` serving input.
///
/// Forward (matches mlx-vlm `gemma4_unified.VisionEmbedder`):
///   `pos_norm( patch_ln2( patch_dense( patch_ln1(x) ) ) + pos[positions] )`
/// where the 2D position embedding adds the per-axis table rows for the
/// patch's (x, y) grid coordinates, masking out padded (-1) positions.
public class Gemma4UnifiedVisionEmbedder: Module {
    @ModuleInfo(key: "patch_ln1") var patchLn1: LayerNorm
    @ModuleInfo(key: "patch_dense") var patchDense: Linear
    @ModuleInfo(key: "patch_ln2") var patchLn2: LayerNorm
    @ParameterInfo(key: "pos_embedding") var posEmbedding: MLXArray
    @ModuleInfo(key: "pos_norm") var posNorm: LayerNorm

    let patchDim: Int
    let mmEmbedDim: Int

    public init(_ config: Gemma4UnifiedVisionConfig) {
        self.patchDim = config.patchDim
        self.mmEmbedDim = config.mmEmbedDim
        _patchLn1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.patchDim, eps: config.rmsNormEps),
            key: "patch_ln1")
        _patchDense = ModuleInfo(
            wrappedValue: Linear(config.patchDim, config.mmEmbedDim, bias: true),
            key: "patch_dense")
        _patchLn2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.mmEmbedDim, eps: config.rmsNormEps),
            key: "patch_ln2")
        _posEmbedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([config.mmPosembSize, 2, config.mmEmbedDim]),
            key: "pos_embedding")
        _posNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.mmEmbedDim, eps: config.rmsNormEps),
            key: "pos_norm")
    }

    /// - Parameters:
    ///   - pixelValues: `[B, N, patchDim]` flattened raw patches (or
    ///     `[B, N, kh, kw, c]` which is reshaped to `[B, N, patchDim]`).
    ///   - positionIds: `[B, N, 2]` int grid coords per patch; `-1` marks
    ///     a padded patch (its position contribution is zeroed). Pass `nil`
    ///     to skip the position embedding (positions all zero).
    /// - Returns: `[B, N, mmEmbedDim]` patch embeddings (pre-projection).
    public func callAsFunction(
        _ pixelValues: MLXArray, positionIds: MLXArray? = nil
    ) -> MLXArray {
        var x = pixelValues
        if x.ndim > 3 {
            x = x.reshaped(x.dim(0), x.dim(1), patchDim)
        }
        var h = patchLn1(x)
        h = patchDense(h)
        h = patchLn2(h)

        if let pos = positionIds {
            let B = h.dim(0); let N = h.dim(1); let D = h.dim(2)
            let clamped = MLX.maximum(pos, MLXArray(Int32(0))).asType(.int32)  // [B,N,2]
            let valid = (pos .!= MLXArray(Int32(-1))).asType(h.dtype)          // [B,N,2]
            // pos_embedding: [P, 2, D]. Take axis-1 slice per coordinate,
            // then gather rows by the (clamped) grid index.
            let posX = posEmbedding[0..., 0, 0...]   // [P, D]  (x axis table)
            let posY = posEmbedding[0..., 1, 0...]   // [P, D]  (y axis table)
            let xIdx = clamped[0..., 0..., 0].flattened()   // [B*N]
            let yIdx = clamped[0..., 0..., 1].flattened()   // [B*N]
            let xPos = posX.take(xIdx, axis: 0).reshaped(B, N, D)
            let yPos = posY.take(yIdx, axis: 0).reshaped(B, N, D)
            let validX = expandedDimensions(valid[0..., 0..., 0], axis: -1)  // [B,N,1]
            let validY = expandedDimensions(valid[0..., 0..., 1], axis: -1)
            h = h + (xPos * validX + yPos * validY)
        }

        return posNorm(h)
    }
}

// MARK: - Image preprocessing (encoder-free patchify)

/// Preprocess an image for the Gemma 4 unified vision front-end and pack it
/// for the `multimodalForward` seam.
///
/// Pipeline (matches HF `Gemma4UnifiedImageProcessor` /
/// mlx-vlm `_convert_image_to_model_patches`):
///  1. decode + aspect-ratio-preserving resize so H, W are multiples of
///     `modelPatchSize` (48), rescaled to `[0, 1]` (`do_normalize=false`).
///     This is exactly what the shared `preprocessImage(targetSize:)` does,
///     and its `(H/48)*(W/48)` patch count already matches the engine's
///     `computeImageTokenCount`, so the placeholder run lines up 1:1.
///  2. patchify `[1, 3, H, W]` into `[N, modelPatchSize^2 * 3]` (channel
///     innermost), where `N = (H/48)*(W/48)`.
///  3. append each patch's `(x, y)` grid position as 2 trailing columns so
///     the positions ride through the single-`MLXArray` seam
///     (`Gemma4UnifiedModel.unpackImage` splits them back out).
///
/// - Returns: `[1, N, modelPatchSize^2*3 + 2]` float32.
public func preprocessGemma4UnifiedImage(
    _ imageData: Data, modelPatchSize: Int = 48
) throws -> MLXArray {
    // Use the shared `preprocessImage` with its DEFAULT target size so the
    // resulting `(H/48)*(W/48)` patch count is identical to the engine's
    // `computeImageTokenCount` (which calls `preprocessImage` the same way).
    // `modelPatchSize` is the patchify granularity, NOT the resize target -
    // conflating them would shrink large images to a single patch and
    // desync the placeholder count.
    let chw = try preprocessImage(imageData)  // [1,3,H,W] in [0,1], H/W multiples of 48
    let c = chw.dim(1)
    let h = chw.dim(2)
    let w = chw.dim(3)
    let p = modelPatchSize
    let pH = h / p
    let pW = w / p
    let n = pH * pW

    // [1,C,H,W] -> [C, pH, p, pW, p] -> [pH, pW, p, p, C] -> [N, p*p*C]
    var patches = chw.reshaped(c, pH, p, pW, p)
    patches = patches.transposed(1, 3, 2, 4, 0)
    patches = patches.reshaped(n, p * p * c)

    // Grid positions, row-major (y outer, x inner): position[i] = [x, y].
    // `indexing="xy"` in the reference yields the same (x, y) ordering.
    var posList = [Float](repeating: 0, count: n * 2)
    var i = 0
    for y in 0 ..< pH {
        for x in 0 ..< pW {
            posList[i * 2] = Float(x)
            posList[i * 2 + 1] = Float(y)
            i += 1
        }
    }
    let positions = MLXArray(posList).reshaped(n, 2)
    let packed = concatenated([patches, positions], axis: -1)  // [N, p*p*C + 2]
    return packed.reshaped(1, n, p * p * c + 2)
}

// MARK: - Audio preprocessing (encoder-free raw frames)

/// Reshape a mono 16 kHz waveform into Gemma 4 unified audio frames:
/// pad to a multiple of `samplesPerToken` (640 = 40 ms at 16 kHz), then
/// reshape to `[1, N, samplesPerToken]`. Matches mlx-vlm
/// `Gemma4UnifiedAudioFeatureExtractor._extract_waveform_features`. The
/// model projects each raw frame directly (no mel, no encoder).
///
/// - Parameter waveform: mono PCM samples in `[-1, 1]` at 16 kHz.
/// - Returns: `[1, N, samplesPerToken]` float32, where
///   `N = ceil(len / samplesPerToken)`.
public func preprocessGemma4UnifiedAudio(
    _ waveform: [Float], samplesPerToken: Int = 640
) -> MLXArray {
    var samples = waveform
    // Pad up to at least one full frame. This also covers an empty waveform,
    // which would otherwise reshape 0 elements into a non-empty frame and trap.
    let pad = samples.isEmpty
        ? samplesPerToken
        : (samplesPerToken - samples.count % samplesPerToken) % samplesPerToken
    if pad != 0 {
        samples.append(contentsOf: [Float](repeating: 0, count: pad))
    }
    let n = samples.count / samplesPerToken
    return MLXArray(samples).reshaped(1, n, samplesPerToken)
}

/// Number of audio soft tokens for a waveform of `sampleCount` mono samples:
/// `ceil(sampleCount / samplesPerToken)`. The engine inserts exactly this
/// many `<|audio|>` placeholders so the projected frames scatter 1:1.
public func gemma4UnifiedAudioTokenCount(
    sampleCount: Int, samplesPerToken: Int = 640
) -> Int {
    max(1, (sampleCount + samplesPerToken - 1) / samplesPerToken)
}

// MARK: - Gemma4UnifiedModel (top-level)

/// Encoder-free Gemma 4 unified multimodal model. Weight key hierarchy:
///   `language_model.*` - dense text decoder (`Gemma4ForCausalLM`)
///   `vision_embedder.*` - encoder-free patch projector
///   `embed_vision.*` - vision -> text-hidden projection
///   `embed_audio.*` - audio -> text-hidden projection
public class Gemma4UnifiedModel: Module {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4ForCausalLM
    @ModuleInfo(key: "vision_embedder") var visionEmbedder: Gemma4UnifiedVisionEmbedder
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder
    @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder

    public let config: Gemma4Config
    public let visionConfig: Gemma4UnifiedVisionConfig
    public let audioConfig: Gemma4UnifiedAudioConfig
    public let imageTokenId: Int
    public let audioTokenId: Int

    public let visionCache: VisionEncoderCache = VisionEncoderCache()

    public init(
        _ config: Gemma4Config,
        visionConfig: Gemma4UnifiedVisionConfig,
        audioConfig: Gemma4UnifiedAudioConfig,
        imageTokenId: Int = 258880,
        audioTokenId: Int = 258881
    ) {
        self.config = config
        self.visionConfig = visionConfig
        self.audioConfig = audioConfig
        self.imageTokenId = imageTokenId
        self.audioTokenId = audioTokenId

        _languageModel = ModuleInfo(
            wrappedValue: Gemma4ForCausalLM(config), key: "language_model")
        _visionEmbedder = ModuleInfo(
            wrappedValue: Gemma4UnifiedVisionEmbedder(visionConfig),
            key: "vision_embedder")
        _embedVision = ModuleInfo(
            wrappedValue: MultimodalEmbedder(
                embeddingDim: visionConfig.outputProjDims,
                textHiddenSize: config.hiddenSize,
                eps: visionConfig.rmsNormEps),
            key: "embed_vision")
        _embedAudio = ModuleInfo(
            wrappedValue: MultimodalEmbedder(
                embeddingDim: audioConfig.outputProjDims,
                textHiddenSize: config.hiddenSize,
                eps: audioConfig.rmsNormEps),
            key: "embed_audio")
    }

    /// Vision soft-token features ready to inject: `[1, N, hidden]`.
    /// `pixelValues` is `[1, N, patchDim]`, `positionIds` `[1, N, 2]`.
    public func encodeImage(
        _ pixelValues: MLXArray, positionIds: MLXArray? = nil
    ) -> MLXArray {
        embedVision(visionEmbedder(pixelValues, positionIds: positionIds))
    }

    /// Audio soft-token features ready to inject: `[1, N, hidden]`.
    /// `audioFeatures` is `[1, N, audioSamplesPerToken]` raw frames.
    public func encodeAudio(_ audioFeatures: MLXArray) -> MLXArray {
        embedAudio(audioFeatures)
    }

    // MARK: Text-only forwards (mirror Gemma4ForCausalLM overloads)

    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil
    ) -> MLXArray {
        languageModel(tokens, caches: caches, lastTokenOnly: false)
    }

    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil, lastTokenOnly: Bool
    ) -> MLXArray {
        languageModel(tokens, caches: caches, lastTokenOnly: lastTokenOnly)
    }

    // MARK: Multimodal forward

    /// Full multimodal forward. `pixelValues` / `imagePositionIds` drive the
    /// vision front-end; `audioFeatures` drives the audio front-end. Each is
    /// optional; nil falls through to text-only. `mediaHash` keys a per-image
    /// vision-feature cache (pure-image requests only).
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        pixelValues: MLXArray? = nil,
        imagePositionIds: MLXArray? = nil,
        audioFeatures: MLXArray? = nil,
        mediaHash: String? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        if pixelValues == nil && audioFeatures == nil {
            return languageModel(tokens, caches: caches, lastTokenOnly: lastTokenOnly)
        }

        var imageEmbeddings: MLXArray? = nil
        if let pixels = pixelValues {
            if let key = mediaHash, let cached = visionCache.lookup(key) {
                imageEmbeddings = cached
            } else {
                let computed = encodeImage(pixels, positionIds: imagePositionIds)
                if let key = mediaHash, audioFeatures == nil {
                    MLX.eval(computed)
                    visionCache.store(key, value: computed)
                }
                imageEmbeddings = computed
            }
        }

        var audioEmbeddings: MLXArray? = nil
        if let feats = audioFeatures {
            audioEmbeddings = encodeAudio(feats)
        }

        return languageModel(
            tokens, caches: caches,
            imageEmbeddings: imageEmbeddings,
            audioEmbeddings: audioEmbeddings,
            imageTokenId: imageTokenId,
            audioTokenId: audioTokenId,
            lastTokenOnly: lastTokenOnly)
    }

    /// Stage B/C batched ragged-decode (text-only; decode never carries
    /// media placeholders). Forwards to the shared text path.
    public func batchedDecode(
        _ tokens: MLXArray, caches: [KVCacheProtocol], mask: MLXArray, rowOffsets: [Int],
        slidingMask: MLXArray? = nil
    ) -> MLXArray {
        languageModel.batchedDecode(
            tokens, caches: caches, mask: mask, rowOffsets: rowOffsets,
            slidingMask: slidingMask)
    }

    public var numLayers: Int { config.numHiddenLayers }

    /// Split a packed image tensor `[1, N, patchDim+2]` back into raw
    /// patches `[1, N, patchDim]` and integer grid positions `[1, N, 2]`.
    /// The trailing 2 columns carry the (x, y) grid coordinates as floats
    /// (`-1` marks a padded patch). This is the inverse of the packing the
    /// `Gemma4UnifiedImagePreprocessor` does so per-patch positions can ride
    /// through the single-`MLXArray` multimodal seam. Returns `(nil, nil)`
    /// for a nil/empty input.
    public static func unpackImage(
        _ packed: MLXArray?, patchDim: Int
    ) -> (pixels: MLXArray?, positionIds: MLXArray?) {
        guard let packed else { return (nil, nil) }
        if packed.dim(-1) == patchDim {
            return (packed, nil)   // already unpacked (no positions)
        }
        let pixels = packed[0..., 0..., 0 ..< patchDim]
        let posF = packed[0..., 0..., patchDim ..< (patchDim + 2)]
        // Round-to-nearest before truncating to int (the columns hold exact
        // small integers, but go through bf16/fp16 on the wire).
        let posIds = MLX.round(posF).asType(.int32)
        return (pixels, posIds)
    }
}
