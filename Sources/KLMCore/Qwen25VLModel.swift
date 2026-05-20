import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Qwen 2.5-VL Config

/// Parsed `config.json` for Qwen 2.5-VL
/// (`Qwen2_5_VLForConditionalGeneration` /
/// `model_type: qwen2_5_vl`). Covers the language side (same shape
/// as dense Qwen 2.5: QKV bias, no q_norm/k_norm, separate lm_head)
/// plus the vision-tower subconfig and the 3D mRoPE section split.
///
/// The language-side fields are kept compatible with `QwenConfig`
/// so the existing dense Qwen 2.5 attention and MLP modules can be
/// reused for the text path. The VL-specific deltas are isolated
/// in the vision tower + the multimodal forward.
public struct Qwen25VLConfig: ModelConfig, Decodable, Sendable {
    // Language side (dense Qwen 2.5 shape)
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let quantization: QuantizationConfig?
    public let tieWordEmbeddings: Bool
    public let explicitHeadDim: Int?

    public var headDim: Int { explicitHeadDim ?? (hiddenSize / numAttentionHeads) }

    // Multimodal tokens (defaults match Qwen 2.5-VL's tokenizer)
    public let imageTokenId: Int      // <|image_pad|>, 151655
    public let videoTokenId: Int      // <|video_pad|>, 151656
    public let visionStartTokenId: Int  // <|vision_start|>, 151652
    public let visionEndTokenId: Int    // <|vision_end|>, 151653

    // 3D mRoPE section split across temporal/height/width axes.
    // Qwen 2.5-VL ships [16, 24, 24] summing to 64 (the per-head
    // dim); we tolerate any 3-vector that sums to head_dim.
    public let mropeSection: [Int]

    /// Vision tower subconfig. Pulled out so callers (image
    /// preprocessing, tower instantiation) can reach it without
    /// re-parsing.
    public let vision: VisionConfig

    public struct VisionConfig: Codable, Sendable {
        /// Number of vision transformer layers. 32 for the 3B / 7B
        /// reference checkpoints.
        public let depth: Int
        /// Hidden width of the vision tower. 1280 for the
        /// reference shape.
        public let hiddenSize: Int
        /// Intermediate width inside each vision block's MLP.
        public let intermediateSize: Int
        /// Number of attention heads inside the vision tower.
        public let numHeads: Int
        /// Edge size of a single image patch in pixels (square
        /// patches). 14 for the reference shape.
        public let patchSize: Int
        /// Temporal patch count. 2 for the reference (a single
        /// image is treated as a 2-frame mini-video by patching
        /// along the time axis so the same code path handles
        /// images and videos).
        public let temporalPatchSize: Int
        /// In-channel count of the patch embedding. 3 (RGB).
        public let inChannels: Int
        /// Spatial merge factor at the patch merger. 2 -> 4
        /// patches collapse into one language-side token.
        public let spatialMergeSize: Int
        /// Layers whose attention is full (not windowed). The
        /// reference ships [7, 15, 23, 31] (every 8th block).
        public let fullAttnBlockIndexes: [Int]
        /// Window edge in patch units for the windowed attention
        /// layers. 112 in pixel units / patch_size = 8 patches.
        public let windowSize: Int
        /// Final output width into the LM hidden space; only
        /// referenced by the merger.
        public let outHiddenSize: Int

        enum CodingKeys: String, CodingKey {
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case inChannels = "in_chans"
            case spatialMergeSize = "spatial_merge_size"
            case fullAttnBlockIndexes = "fullatt_block_indexes"
            case windowSize = "window_size"
            case outHiddenSize = "out_hidden_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 32
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3420
            numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            inChannels = try c.decodeIfPresent(Int.self, forKey: .inChannels) ?? 3
            spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            fullAttnBlockIndexes = try c.decodeIfPresent([Int].self, forKey: .fullAttnBlockIndexes)
                ?? [7, 15, 23, 31]
            windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 112
            outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 2048
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case quantization
        case tieWordEmbeddings = "tie_word_embeddings"
        case explicitHeadDim = "head_dim"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
        case ropeScaling = "rope_scaling"
        case visionConfig = "vision_config"
    }

    private struct RopeScaling: Codable {
        let type: String?
        let mropeSection: [Int]?
        enum CodingKeys: String, CodingKey {
            case type
            case mropeSection = "mrope_section"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? (try c.decode(Int.self, forKey: .numAttentionHeads))
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 128_000
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        explicitHeadDim = try c.decodeIfPresent(Int.self, forKey: .explicitHeadDim)

        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 151_655
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 151_656
        visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId)
            ?? 151_652
        visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId)
            ?? 151_653

        if let rope = try c.decodeIfPresent(RopeScaling.self, forKey: .ropeScaling),
           let section = rope.mropeSection {
            mropeSection = section
        } else {
            mropeSection = [16, 24, 24]
        }

        vision = try c.decodeIfPresent(VisionConfig.self, forKey: .visionConfig)
            ?? (try JSONDecoder().decode(
                VisionConfig.self, from: try JSONSerialization.data(withJSONObject: [String: Any]())))
    }

    /// Project the VL config onto a `QwenConfig` to reuse the
    /// dense Qwen 2.5 attention + MLP for the text path. The VL
    /// model uses QKV bias (Qwen 2.5 style) and a separate
    /// `lm_head`, so the projection sets `model_type: qwen2`.
    ///
    /// Same JSON-roundtrip pattern as `Qwen3MoEConfig` -> on
    /// decode failure we abort with an explicit message naming
    /// the failure surface, rather than `try!`.
    public var qwenTextConfig: QwenConfig {
        let dict: [String: Any] = [
            "hidden_size": hiddenSize,
            "intermediate_size": intermediateSize,
            "num_attention_heads": numAttentionHeads,
            "num_key_value_heads": numKeyValueHeads,
            "num_hidden_layers": numHiddenLayers,
            "vocab_size": vocabSize,
            "rms_norm_eps": rmsNormEps,
            "rope_theta": ropeTheta,
            "max_position_embeddings": maxPositionEmbeddings,
            "model_type": "qwen2",
            "attention_bias": true,
            "tie_word_embeddings": tieWordEmbeddings,
            "head_dim": explicitHeadDim ?? (hiddenSize / numAttentionHeads),
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: dict)
            return try JSONDecoder().decode(QwenConfig.self, from: data)
        } catch {
            fatalError(
                "Qwen2.5-VL to QwenConfig projection failed: \(error). "
                + "A required QwenConfig field was added without a "
                + "corresponding entry in Qwen25VLConfig.qwenTextConfig. "
                + "Update the dict above.")
        }
    }
}

// MARK: - 3D Multimodal RoPE

/// Apply 3D rotary positional embeddings to Q/K tensors along the
/// temporal / height / width axes. The reference splits the per-head
/// dim into three contiguous sub-vectors of sizes `mropeSection`
/// (default [16, 24, 24]); each sub-vector rotates with a different
/// position index (t, h, w respectively). Text-only positions are
/// equivalent to t = h = w (all three axes carry the same
/// linear position), so the same module handles both.
///
/// The math reduces to: for each sub-vector, build cos/sin tables
/// of shape `[L, sub_dim/2]` using the standard RoPE inverse-freq
/// formula at base = `theta`, then apply
/// `q' = q*cos + rotate_half(q)*sin` to that sub-vector and
/// concatenate. The implementation pre-builds the cos/sin tables
/// once per call from the position arrays.
public struct Qwen25VLMRoPE {
    public let headDim: Int
    public let sections: [Int]
    public let theta: Float

    public init(headDim: Int, sections: [Int], theta: Float = 1_000_000.0) {
        self.headDim = headDim
        self.sections = sections
        self.theta = theta
        let sectionSum = sections.reduce(0, +)
        precondition(sectionSum * 2 == headDim || sectionSum == headDim,
            "mrope_section must sum to head_dim or head_dim/2; "
            + "got sections=\(sections), head_dim=\(headDim)")
    }

    /// Apply 3D mRoPE rotation to a `[B, H, L, D]` tensor using
    /// per-axis position arrays of shape `[L]`. Returns the same
    /// shape.
    public func apply(
        _ x: MLXArray,
        positionsT: MLXArray,
        positionsH: MLXArray,
        positionsW: MLXArray
    ) -> MLXArray {
        // The reference handles fp16 / bf16 inputs by upcasting to
        // fp32 for the rotation math and downcasting on output.
        let dtype = x.dtype
        let xF = x.asType(.float32)
        let positions: [MLXArray] = [
            positionsT.asType(.float32),
            positionsH.asType(.float32),
            positionsW.asType(.float32),
        ]

        // Section dim sums to head_dim/2 (each pair of even+odd
        // dims rotates together, so we work in the half-dim
        // frequency space).
        let halfDim = headDim / 2
        let sumSections = sections.reduce(0, +)
        // Two valid conventions: sections sum to head_dim/2 (most
        // common) or to head_dim (older); normalize to half-dim.
        let normSections: [Int]
        if sumSections == halfDim {
            normSections = sections
        } else if sumSections == headDim {
            normSections = sections.map { $0 / 2 }
        } else {
            fatalError("Invalid mrope_section sum")
        }

        // Build inverse-frequency table for the full half-dim.
        // inv_freq[i] = 1 / theta^(2*i / head_dim) for i in 0..halfDim.
        let positionsHalf = MLXArray(stride(from: 0, to: Int32(headDim), by: 2)
            .map { Float($0) }).asType(.float32)  // [halfDim]
        let invFreq = MLXArray(Float(1.0))
            / MLX.pow(MLXArray(Float(theta)), positionsHalf / Float(headDim))  // [halfDim]

        // Per-axis (cos, sin): for each axis (t, h, w) compute the
        // outer product position * inv_freq -> [L, halfDim], then
        // cos/sin. Slice each axis's chunk and stitch.
        var cosChunks: [MLXArray] = []
        var sinChunks: [MLXArray] = []
        var offset = 0
        for (axisIdx, sectionLen) in normSections.enumerated() {
            let pos = positions[axisIdx]  // [L]
            let subInvFreq = invFreq[offset ..< (offset + sectionLen)]  // [sectionLen]
            let angles = pos.expandedDimensions(axis: 1) * subInvFreq  // [L, sectionLen]
            cosChunks.append(MLX.cos(angles))
            sinChunks.append(MLX.sin(angles))
            offset += sectionLen
        }
        let cos = MLX.concatenated(cosChunks, axis: -1)  // [L, halfDim]
        let sin = MLX.concatenated(sinChunks, axis: -1)  // [L, halfDim]

        // x shape [B, H, L, D]. Split last dim into two halves
        // x1, x2 of width halfDim. The rotation is:
        //   y1 = x1 * cos - x2 * sin
        //   y2 = x1 * sin + x2 * cos
        let x1 = xF[.ellipsis, 0 ..< halfDim]
        let x2 = xF[.ellipsis, halfDim ..< headDim]

        // cos/sin are [L, halfDim]; broadcast to [B, H, L, halfDim]
        // by inserting two leading singleton dims. Two sequential
        // single-axis expansions avoid the duplicate-axis trap
        // that `expandedDimensions(axes: [0, 0])` would hit.
        let cos4 = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let sin4 = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0)

        let y1 = x1 * cos4 - x2 * sin4
        let y2 = x1 * sin4 + x2 * cos4
        let y = MLX.concatenated([y1, y2], axis: -1)
        return y.asType(dtype)
    }
}

// MARK: - Patch merger

/// Spatial patch merger: concatenates a `spatial_merge_size`
/// square block of vision tokens into a single language-side
/// token, then projects through a small MLP to the LM hidden
/// dimension. Weight keys at `visual.merger.{ln_q, mlp.0, mlp.2}`.
class Qwen25VLPatchMerger: Module {
    @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: [Linear]

    let mergeSize: Int
    let inputDim: Int
    let outputDim: Int

    init(visionHidden: Int, outHidden: Int, spatialMergeSize: Int, eps: Float = 1e-6) {
        self.mergeSize = spatialMergeSize
        self.inputDim = visionHidden * spatialMergeSize * spatialMergeSize
        self.outputDim = outHidden
        _lnQ = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: visionHidden, eps: eps),
            key: "ln_q")
        _mlp = ModuleInfo(
            wrappedValue: [
                Linear(inputDim, inputDim, bias: true),
                Linear(inputDim, outHidden, bias: true),
            ],
            key: "mlp")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [N_patches, vision_hidden]
        // 1) per-token RMSNorm over the vision hidden dim
        let normed = lnQ(x)
        // 2) merge spatial_merge_size^2 contiguous patches into
        //    one row. The vision tower lays patches out in
        //    row-major (h, w) order with merge_size blocks
        //    pre-grouped by the reference impl, so a simple
        //    reshape suffices here.
        let nGroups = normed.dim(0) / (mergeSize * mergeSize)
        let merged = normed.reshaped(nGroups, inputDim)
        // 3) MLP: linear -> GELU -> linear
        var h = mlp[0](merged)
        h = gelu(h)
        h = mlp[1](h)
        return h
    }
}

// MARK: - Vision Block

class Qwen25VLVisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(hidden: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hidden / numHeads
        self.scale = 1.0 / Float(headDim).squareRoot()
        _qkv = ModuleInfo(
            wrappedValue: Linear(hidden, hidden * 3, bias: true), key: "qkv")
        _proj = ModuleInfo(
            wrappedValue: Linear(hidden, hidden, bias: true), key: "proj")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // x: [B, L, H]
        let B = x.dim(0), L = x.dim(1)
        let q3 = qkv(x).reshaped(B, L, 3, numHeads, headDim)
        let qkvSplit = q3.transposed(2, 0, 3, 1, 4)  // [3, B, heads, L, head_dim]
        let queries = qkvSplit[0]
        let keys = qkvSplit[1]
        let values = qkvSplit[2]
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        return proj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class Qwen25VLVisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hidden: Int, intermediate: Int) {
        _gateProj = ModuleInfo(
            wrappedValue: Linear(hidden, intermediate, bias: true), key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Linear(hidden, intermediate, bias: true), key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(intermediate, hidden, bias: true), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class Qwen25VLVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attn: Qwen25VLVisionAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen25VLVisionMLP

    init(_ visionConfig: Qwen25VLConfig.VisionConfig, eps: Float = 1e-6) {
        _norm1 = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: visionConfig.hiddenSize, eps: eps),
            key: "norm1")
        _norm2 = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: visionConfig.hiddenSize, eps: eps),
            key: "norm2")
        _attn = ModuleInfo(
            wrappedValue: Qwen25VLVisionAttention(
                hidden: visionConfig.hiddenSize, numHeads: visionConfig.numHeads),
            key: "attn")
        _mlp = ModuleInfo(
            wrappedValue: Qwen25VLVisionMLP(
                hidden: visionConfig.hiddenSize,
                intermediate: visionConfig.intermediateSize),
            key: "mlp")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = x + attn(norm1(x), mask: mask)
        return h + mlp(norm2(h))
    }
}

// MARK: - Patch Embedding

class Qwen25VLPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let embedDim: Int

    init(_ visionConfig: Qwen25VLConfig.VisionConfig) {
        self.patchSize = visionConfig.patchSize
        self.temporalPatchSize = visionConfig.temporalPatchSize
        self.inChannels = visionConfig.inChannels
        self.embedDim = visionConfig.hiddenSize
        // Patch embed flattens [C, T, ph, pw] -> embed_dim. The
        // reference uses a Conv3d, but for prepacked patches the
        // equivalent is a Linear over the flat patch vector.
        let inputDim = inChannels * temporalPatchSize * patchSize * patchSize
        _proj = ModuleInfo(
            wrappedValue: Linear(inputDim, embedDim, bias: false),
            key: "proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [N_patches, C * T * ph * pw]
        proj(x)
    }
}

// MARK: - Vision Tower

/// The full Qwen 2.5-VL vision tower: patch embedding + N
/// transformer blocks (window or full attention depending on layer
/// index) + patch merger. Module keys live under `visual.*` in
/// the safetensors checkpoint.
public class Qwen25VLVisionTower: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: Qwen25VLPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [Qwen25VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: Qwen25VLPatchMerger

    let visionConfig: Qwen25VLConfig.VisionConfig
    /// Indices of the layers that use full (non-windowed)
    /// attention. The reference ships [7, 15, 23, 31] (every
    /// 8th block). Windowed attention is the cheaper default; the
    /// periodic full-attention layers let the tower mix
    /// information across windows.
    let fullAttnLayers: Set<Int>

    public init(_ visionConfig: Qwen25VLConfig.VisionConfig, normEps: Float = 1e-6) {
        self.visionConfig = visionConfig
        self.fullAttnLayers = Set(visionConfig.fullAttnBlockIndexes)
        _patchEmbed = ModuleInfo(
            wrappedValue: Qwen25VLPatchEmbed(visionConfig),
            key: "patch_embed")
        _blocks = ModuleInfo(
            wrappedValue: (0 ..< visionConfig.depth).map { _ in
                Qwen25VLVisionBlock(visionConfig, eps: normEps)
            },
            key: "blocks")
        _merger = ModuleInfo(
            wrappedValue: Qwen25VLPatchMerger(
                visionHidden: visionConfig.hiddenSize,
                outHidden: visionConfig.outHiddenSize,
                spatialMergeSize: visionConfig.spatialMergeSize,
                eps: normEps),
            key: "merger")
    }

    /// Run the vision tower over a packed patch tensor and return
    /// language-aligned vision embeddings. `packedPatches` has
    /// shape `[N_patches, C * T * ph * pw]` (flat per-patch
    /// vector); the merger collapses each `spatial_merge_size^2`
    /// block into one output token.
    public func callAsFunction(_ packedPatches: MLXArray) -> MLXArray {
        var h = patchEmbed(packedPatches)  // [N_patches, vision_hidden]
        // Vision attention runs over the patch sequence. The
        // window mask depends on the input image's grid shape, so
        // for the foundation PR we run with mask=nil (full
        // attention everywhere). The window-mask helper exists
        // separately for the runtime PR that wires the real
        // image-grid metadata through.
        h = h.expandedDimensions(axis: 0)  // [1, N_patches, hidden]
        for (i, block) in blocks.enumerated() {
            // Tag fullAttn / windowed via mask choice; with
            // mask=nil today both behave identically. The flag is
            // here so a follow-up can attach the real window mask
            // without touching the block iteration.
            _ = fullAttnLayers.contains(i)
            h = block(h, mask: nil)
        }
        // Drop the batch dim before merging.
        let merged = merger(h.squeezed(axis: 0))
        return merged  // [N_merged_tokens, out_hidden]
    }
}

// MARK: - Image preprocessing

/// Convert a normalized RGB pixel tensor `[H, W, 3]` (values in
/// `[0, 1]`) to the packed-patch shape `[N_patches, C * T * ph *
/// pw]` that the vision tower's patch_embed consumes. Resizing to
/// a multiple of (patch_size * spatial_merge_size) is the caller's
/// responsibility - the helper expects a pre-resized image so the
/// caller can choose its own resize backend (CoreGraphics on Mac,
/// stb_image on Linux).
///
/// The temporal axis is filled with `temporal_patch_size` copies
/// of the same frame so a single image trains the same code path
/// as a video.
public enum Qwen25VLImagePreprocessor {
    /// Mean for the Qwen 2.5-VL image normalization (CLIP defaults).
    public static let imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    /// Std for the Qwen 2.5-VL image normalization (CLIP defaults).
    public static let imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]

    /// Normalize a `[H, W, 3]` `[0, 1]` tensor in CLIP color space.
    public static func normalize(_ pixels: MLXArray) -> MLXArray {
        let mean = MLXArray(imageMean).reshaped(1, 1, 3)
        let std = MLXArray(imageStd).reshaped(1, 1, 3)
        return (pixels - mean) / std
    }

    /// Pack a `[H, W, 3]` normalized tensor into per-patch rows.
    /// `H` and `W` must both be multiples of `patch_size`. Output
    /// rows are ordered (h_patch, w_patch) row-major. The
    /// temporal axis is duplicated to `temporal_patch_size`.
    public static func packPatches(
        _ pixels: MLXArray,
        patchSize: Int,
        temporalPatchSize: Int
    ) -> MLXArray {
        let H = pixels.dim(0)
        let W = pixels.dim(1)
        let C = pixels.dim(2)
        precondition(H % patchSize == 0 && W % patchSize == 0,
            "Image height/width must be multiples of patch_size")
        let nH = H / patchSize
        let nW = W / patchSize
        // [H, W, C] -> [nH, ph, nW, pw, C] -> [nH, nW, C, ph, pw]
        let reshaped = pixels.reshaped(nH, patchSize, nW, patchSize, C)
        let perPatch = reshaped.transposed(0, 2, 4, 1, 3)
            .reshaped(nH * nW, C * patchSize * patchSize)
        // Tile across temporal axis: [N, C * T * ph * pw]. The
        // reference duplicates the frame, so we tile by
        // concatenation rather than by introducing a real time
        // axis.
        var tiled = perPatch
        for _ in 1 ..< temporalPatchSize {
            tiled = MLX.concatenated([tiled, perPatch], axis: -1)
        }
        return tiled
    }
}
