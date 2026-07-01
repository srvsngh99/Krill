import Foundation
import MLX
import MLXNN
import MLXFast
import KrillCache

// Native Qwen3.5-VL multimodal glue: the interleaved sectioned mRoPE, the 3D
// position computation (`get_rope_index`), and the top-level
// `Qwen35VLForConditionalGeneration` that runs the native vision tower
// (`Qwen35VLVisionModel`) + the native hybrid text decoder
// (`Qwen35ForCausalLM`). Parity oracle: `mlx_vlm.models.qwen3_5`.

// MARK: - Interleaved sectioned 3D mRoPE

/// Builds the 3D multimodal-RoPE `(cos, sin)` tables for the Qwen3.5 text
/// decoder. Qwen3.5 uses the **interleaved** mRoPE style: the per-head rotary
/// dim is `head_dim * partial_rotary_factor` (64 of 256 for Ornith), giving
/// `half = 32` frequencies; each frequency is assigned to one of the three
/// position axes (t/h/w) by an interleaved selector derived from
/// `mrope_section [11, 11, 10]`. For text-only positions (t == h == w) every
/// axis carries the same position, so this collapses exactly to the standard
/// partial RoPE the text decoder already uses.
public struct Qwen35VLMRoPE {
    public let rotaryDim: Int
    public let half: Int
    let invFreq: MLXArray          // [half]
    let selector: MLXArray         // [half] Int32, values in {0,1,2}

    public init(headDim: Int, partialRotaryFactor: Float, mropeSection: [Int], theta: Float) {
        let rot = Int(Float(headDim) * partialRotaryFactor)
        let h = rot / 2
        // inv_freq[i] = 1 / theta^(2i / rotaryDim), i in 0..<half.
        let invF = (0 ..< h).map { i -> Float in
            1.0 / powf(theta, Float(2 * i) / Float(rot))
        }
        // Interleaved selector (mlx_vlm `_interleaved_position_selector`):
        //   default axis 0 (temporal); axis 1 (height) at idx ≡ 1 (mod 3) while
        //   idx < section[1]*3; axis 2 (width) at idx ≡ 2 (mod 3) while
        //   idx < section[2]*3. `freq_dim` is `half`.
        var sel = [Int32](repeating: 0, count: h)
        if mropeSection.count >= 3 {
            var idx = 1
            while idx < min(mropeSection[1] * 3, h) { sel[idx] = 1; idx += 3 }
            idx = 2
            while idx < min(mropeSection[2] * 3, h) { sel[idx] = 2; idx += 3 }
        }
        rotaryDim = rot
        half = h
        invFreq = MLXArray(invF)
        selector = MLXArray(sel)
    }

    /// `positions3`: `[3, L]` (t, h, w positions per token; absolute, so any
    /// decode offset is already folded in by the caller). Returns
    /// `(cos, sin)` of shape `[1, 1, L, half]` ready for `applyPartialMRoPE`.
    public func buildCosSin(_ positions3: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        // Select each frequency's position axis, then outer-product with the
        // per-frequency inverse frequency (mlx_vlm `_selected_mrope_freqs`).
        let taken = take(positions3.asType(.float32), selector, axis: 0)  // [half, L]
        let angles = taken.transposed(1, 0) * invFreq                    // [L, half]
        let cos = MLX.cos(angles).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let sin = MLX.sin(angles).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        return (cos, sin)
    }
}

// MARK: - VL config

/// Top-level Ornith / qwen3_5 VL config: the hybrid text decoder
/// (`text_config`) + the vision tower (`vision_config`) + the media token ids.
public struct Qwen35VLConfig: Decodable {
    public let textConfig: Qwen35Config
    public let visionConfig: Qwen35VLVisionConfig
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let visionStartTokenId: Int
    public let visionEndTokenId: Int
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(Qwen35Config.self, forKey: .textConfig)
        visionConfig = try c.decode(Qwen35VLVisionConfig.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 248056
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 248057
        visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 248053
        visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 248054
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - Qwen3.5-VL ForConditionalGeneration

/// Native Qwen3.5-VL multimodal model: the native vision tower + the native
/// hybrid (GatedDeltaNet + full-attn) text decoder, wired with image-feature
/// scatter and 3D interleaved mRoPE. Module keys match the mlx_vlm-format
/// checkpoint:
///   - `vision_tower.*`            patch embed, pos embed, blocks, merger
///   - `language_model.model.*`    token embedding, hybrid layers, norm
///   - `language_model.lm_head.*`  projection head
public final class Qwen35VLForConditionalGeneration: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: Qwen35VLVisionModel
    @ModuleInfo(key: "language_model") var languageModel: Qwen35ForCausalLM

    public let config: Qwen35VLConfig
    let mrope: Qwen35VLMRoPE
    public let visionCache: VisionEncoderCache = VisionEncoderCache()

    public init(_ config: Qwen35VLConfig) {
        self.config = config
        let tc = config.textConfig
        _visionTower = ModuleInfo(wrappedValue: Qwen35VLVisionModel(config.visionConfig), key: "vision_tower")
        _languageModel = ModuleInfo(wrappedValue: Qwen35ForCausalLM(tc), key: "language_model")
        self.mrope = Qwen35VLMRoPE(
            headDim: tc.headDim, partialRotaryFactor: tc.partialRotaryFactor,
            mropeSection: [11, 11, 10], theta: tc.ropeTheta)
    }

    /// Text-only forward (no image): straight through the hybrid decoder with
    /// standard RoPE (mropeCosSin nil). Keeps the parity-green text path.
    public func callAsFunction(_ tokens: MLXArray, caches: [KVCacheProtocol]? = nil) -> MLXArray {
        languageModel(tokens, caches: caches)
    }

    /// Heterogeneous per-layer caches for the hybrid decoder (KVCache for
    /// full-attn layers, GatedDeltaCache for GatedDeltaNet layers).
    public func makeCaches() -> [KVCacheProtocol] { languageModel.makeCaches() }

    /// Splice `[nMerged, H]` vision features into the contiguous image-token
    /// span of `[1, L, H]` input embeds. Mirrors Qwen 2.5-VL's injector.
    static func injectVisionEmbeds(
        inputEmbeds: MLXArray, visionEmbeds: MLXArray, imagePadStart: Int
    ) -> MLXArray {
        let L = inputEmbeds.dim(1)
        let n = visionEmbeds.dim(0)
        precondition(imagePadStart >= 0 && imagePadStart + n <= L,
            "vision-embed span [\(imagePadStart), \(imagePadStart + n)) exceeds seq len \(L)")
        let before = inputEmbeds[0..., 0 ..< imagePadStart, 0...]
        let after = inputEmbeds[0..., (imagePadStart + n) ..< L, 0...]
        let mid = visionEmbeds.expandedDimensions(axis: 0)
        return MLX.concatenated([before, mid, after], axis: 1)
    }

    private func imagePadSpanLength(_ tokenIds: [Int32], from start: Int) -> Int {
        let pad = Int32(config.imageTokenId)
        var n = 0, i = start
        while i < tokenIds.count && tokenIds[i] == pad { n += 1; i += 1 }
        return n
    }

    /// Multimodal forward.
    /// - Parameters:
    ///   - tokens: prompt token ids `[1, L]`.
    ///   - pixelValues: preprocessed patch batch `[nPatches, C*T*ph*pw]`, or nil.
    ///   - grid: full patch grid `(t, h, w)` for the single image, or nil.
    ///   - caches: per-layer heterogeneous caches (KVCache / GatedDeltaCache).
    ///   - mropePositionOffset: base offset for the 3D positions. nil → cache
    ///     length (correct for text-only / prefill); a decode step after an
    ///     image MUST pass the prefill's final mRoPE position + 1 (the image
    ///     span compresses `h*w` placeholders to `max(h,w)` positions).
    public func callAsFunction(
        _ tokens: MLXArray,
        pixelValues: MLXArray? = nil,
        grid: (t: Int, h: Int, w: Int)? = nil,
        caches: [KVCacheProtocol]? = nil,
        mropePositionOffset: Int? = nil,
        hostTokenIds: [Int32]? = nil,
        lastTokenOnly: Bool = false,
        mediaHash: String? = nil
    ) -> MLXArray {
        var inputEmbeds = languageModel.embed(tokens)  // [1, L, H]

        let tokenIds: [Int32]
        if let hostTokenIds {
            tokenIds = hostTokenIds
        } else {
            eval(tokens)
            tokenIds = tokens.asArray(Int32.self)
        }

        let ms = config.visionConfig.spatialMergeSize
        var gridHMerged = 0, gridWMerged = 0
        if let pixelValues, let grid {
            gridHMerged = grid.h / ms
            gridWMerged = grid.w / ms
            let visionEmbeds: MLXArray
            if let hash = mediaHash, let cached = visionCache.lookup(hash) {
                visionEmbeds = cached
            } else {
                let computed = visionTower(pixelValues, grids: [grid])  // [nMerged, H]
                if let hash = mediaHash { MLX.eval(computed); visionCache.store(hash, value: computed) }
                visionEmbeds = computed
            }
            if let start = tokenIds.firstIndex(of: Int32(config.imageTokenId)) {
                let spanLen = imagePadSpanLength(tokenIds, from: start)
                let nMerged = visionEmbeds.dim(0)
                precondition(spanLen == nMerged,
                    "<|image_pad|> span (\(spanLen)) must equal merged vision-embed count (\(nMerged))")
                inputEmbeds = Self.injectVisionEmbeds(
                    inputEmbeds: inputEmbeds, visionEmbeds: visionEmbeds, imagePadStart: start)
            }
        }

        // 3D mRoPE positions (get_rope_index) — reuses the Qwen 2.5-VL host
        // implementation (the algorithm is identical across Qwen-VL families).
        let coords = Qwen25VLPositions.compute(
            tokenIds: tokenIds, imageTokenId: config.imageTokenId,
            gridHMerged: gridHMerged, gridWMerged: gridWMerged)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let offset = Int32(mropePositionOffset ?? cacheLen)
        let posT = coords.t.map { $0 + offset }
        let posH = coords.h.map { $0 + offset }
        let posW = coords.w.map { $0 + offset }
        let positions3 = MLXArray(posT + posH + posW).reshaped(3, tokenIds.count)
        let cosSin = mrope.buildCosSin(positions3)

        var hidden = languageModel.hiddenStates(
            embeds: inputEmbeds, caches: caches, mropeCosSin: cosSin)
        // Slice to the last position BEFORE the wide vocab projection (the KV
        // caches are already filled), so prefill projects one row, not L.
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        return languageModel.project(hidden)
    }

    /// The mRoPE position the first decode token must take after a prompt
    /// (= `get_rope_index`'s max position + 1). Threaded as `mropePositionOffset`.
    public func nextMRoPEPosition(tokenIds: [Int32], gridHMerged: Int, gridWMerged: Int) -> Int {
        Int(Qwen25VLPositions.compute(
            tokenIds: tokenIds, imageTokenId: config.imageTokenId,
            gridHMerged: gridHMerged, gridWMerged: gridWMerged).nextPos)
    }
}

// MARK: - Image preprocessing

/// Qwen3.5-VL image preprocessing. Reuses the Qwen 2.5-VL geometry
/// (`smartResize`, `decode`, `toConv3DInput` merge-block patchify) but with
/// Ornith's `[0.5, 0.5, 0.5]` mean/std (NOT CLIP) and the flattened
/// `[nPatches, C*T*ph*pw]` layout the native `Qwen35VLPatchEmbed` consumes.
public enum Qwen35VLImagePreprocessor {
    /// Ornith normalization: simple `(x - 0.5) / 0.5` → `[-1, 1]`.
    public static func normalize(_ pixels: MLXArray) -> MLXArray {
        (pixels - MLXArray(Float(0.5))) / MLXArray(Float(0.5))
    }

    /// Decode + resize + normalize + patchify into the flattened Conv3d batch,
    /// and report the FULL patch grid `(t, h, w)` (t == 1 for a still image).
    /// `gridHMerged = h/merge`, `gridWMerged = w/merge` give the image-token
    /// count (`gridHMerged * gridWMerged`).
    public static func preprocess(
        _ imageData: Data, vision: Qwen35VLVisionConfig
    ) throws -> (patches: MLXArray, grid: (t: Int, h: Int, w: Int)) {
        let pixels = try Qwen25VLImagePreprocessor.decode(
            imageData, patchSize: vision.patchSize, spatialMergeSize: vision.spatialMergeSize)
        let H = pixels.dim(0), W = pixels.dim(1)
        // [N, T, ph, pw, C] merge-block order (shared patchify).
        let chLast = Qwen25VLImagePreprocessor.toConv3DInput(
            normalize(pixels),
            patchSize: vision.patchSize,
            temporalPatchSize: vision.temporalPatchSize,
            spatialMergeSize: vision.spatialMergeSize)
        // -> flattened [N, C*T*ph*pw] in (C, T, ph, pw) order (mlx_vlm layout;
        //    the native patch embed reshapes it back to [N, C, T, ph, pw]).
        let n = chLast.dim(0)
        let flat = chLast.movedAxis(source: 4, destination: 1).reshaped(n, -1)
        return (patches: flat, grid: (t: 1, h: H / vision.patchSize, w: W / vision.patchSize))
    }
}
