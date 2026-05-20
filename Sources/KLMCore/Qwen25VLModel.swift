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
/// dimension. Weight keys at `visual.merger.{ln_q, mlp.0, mlp.2}`
/// in the shipped `mlx-community/Qwen2.5-VL-*-Instruct-4bit`
/// checkpoints. The GELU activation occupies index 1 and has no
/// trainable parameters, but its slot in the `mlp` array is what
/// makes the indices line up. Declaring the array as
/// `[Linear, GELU, Linear]` rather than `[Linear, Linear]` is
/// what binds `mlp.0.weight` -> `Linear[0]` and
/// `mlp.2.weight` -> `Linear[2]` at safetensors load time;
/// without the GELU placeholder the second Linear lands at
/// `mlp.1.weight` and the actual `mlp.2.*` keys are never
/// assigned to a module.
class Qwen25VLPatchMerger: Module {
    @ModuleInfo(key: "ln_q") var lnQ: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: [UnaryLayer]

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
                GELU(),
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
        // 3) Sequentially apply Linear -> GELU -> Linear via the
        //    array. The indices match the checkpoint keys
        //    (`mlp.0.*` and `mlp.2.*`); the GELU at index 1
        //    contributes no parameters.
        var h = merged
        for layer in mlp {
            h = layer(h)
        }
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

/// Patch embedding for Qwen 2.5-VL. The checkpoint ships
/// `patch_embed.proj.weight` as a rank-5 Conv3d weight tensor of
/// shape `[embed_dim, temporal_patch_size, patch_size, patch_size,
/// in_channels]` (MLX channels-last `NDHWC` layout). The reference
/// (HF / mlx-vlm) is `nn.Conv3d(in_channels, embed_dim,
/// kernel=[T, ph, pw], stride=[T, ph, pw], bias=False)`. We mirror
/// that exactly so the shipped weights are assignable without any
/// load-time reshape.
class Qwen25VLPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d

    let patchSize: Int
    let temporalPatchSize: Int
    let inChannels: Int
    let embedDim: Int

    init(_ visionConfig: Qwen25VLConfig.VisionConfig) {
        self.patchSize = visionConfig.patchSize
        self.temporalPatchSize = visionConfig.temporalPatchSize
        self.inChannels = visionConfig.inChannels
        self.embedDim = visionConfig.hiddenSize
        _proj = ModuleInfo(
            wrappedValue: Conv3d(
                inputChannels: inChannels,
                outputChannels: embedDim,
                kernelSize: .init((temporalPatchSize, patchSize, patchSize)),
                stride: .init((temporalPatchSize, patchSize, patchSize)),
                bias: false),
            key: "proj")
    }

    /// Input: per-patch batched tensor of shape `[N_patches, T,
    /// patch_size, patch_size, C]` (the preprocessor produces
    /// this in merge-block row-major order). Each "image" in the
    /// batch is a single patch; Conv3d's kernel == stride == the
    /// full patch size, so the spatial output is `1x1x1` and the
    /// channel axis carries the embedding.
    ///
    /// Output: `[N_patches, embed_dim]` in the same merge-block
    /// order. The merger then fuses every `spatial_merge_size^2`
    /// consecutive patches into one language-side token.
    ///
    /// Per-patch batching (rather than running Conv3d once over
    /// the full image and reshaping) matches the HF reference
    /// (`Qwen2_5_VisionPatchEmbed.forward` in HF transformers)
    /// and preserves the merge-aware ordering established by the
    /// preprocessor.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // [N_patches, T, ph, pw, C] -> [N_patches, 1, 1, 1, embed_dim]
        let y = proj(x)
        let n = y.dim(0)
        let embed = y.dim(4)
        return y.reshaped(n, embed)
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

    /// Run the vision tower over a per-patch batched tensor and
    /// return language-aligned vision embeddings.
    ///
    /// Input: `[N_patches, T, patch_size, patch_size, C]` in
    /// merge-block row-major order (the order produced by
    /// `Qwen25VLImagePreprocessor.toConv3DInput`). The contract
    /// with the preprocessor is what guarantees that the merger
    /// fuses 2D spatial blocks (not row-adjacent patches).
    ///
    /// Output: `[N_merged_tokens, out_hidden]` where
    /// `N_merged_tokens = N_patches / spatial_merge_size^2`.
    ///
    /// Window-attention masking depends on the per-image grid
    /// shape (h_patches, w_patches); the foundation runs all
    /// blocks with mask=nil (full attention) so the modules are
    /// exercised end-to-end. The runtime PR wires per-image
    /// window masks through `fullAttnLayers` without changing
    /// the iteration shape here. (Issue tracked in WS5 follow-up.)
    public func callAsFunction(_ patchBatch: MLXArray) -> MLXArray {
        var h = patchEmbed(patchBatch)  // [N_patches, vision_hidden]
        h = h.expandedDimensions(axis: 0)  // [1, N_patches, hidden]
        for (i, block) in blocks.enumerated() {
            _ = fullAttnLayers.contains(i)
            h = block(h, mask: nil)
        }
        let merged = merger(h.squeezed(axis: 0))
        return merged  // [N_merged_tokens, out_hidden]
    }
}

// MARK: - Image preprocessing

/// Convert a normalized RGB pixel tensor `[H, W, 3]` (values in
/// `[0, 1]`) into the per-patch batched
/// `[N_patches, T, patch_size, patch_size, C]` channels-last
/// input shape that the Conv3d-based patch embedding consumes
/// (per-patch batching matches the HF / mlx-vlm reference). The
/// temporal axis is duplicated to `temporal_patch_size` so a
/// single image drives the same code path as a video. Patches
/// are produced in merge-block row-major order so the downstream
/// merger fuses `spatial_merge_size^2` consecutive patches as a
/// true 2D spatial block. Resizing to a multiple of (`patch_size
/// * spatial_merge_size`) is the caller's responsibility. The
/// helper expects a pre-resized image so the caller can choose
/// its own resize backend (CoreGraphics on Mac, stb_image on
/// Linux).
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

    /// Shape `[H, W, 3]` normalized pixels into the per-patch
    /// batched layout `[N_patches, T, patch_size, patch_size, 3]`
    /// in merge-block row-major order. This matches the HF and
    /// mlx-vlm reference image processor exactly: the preprocessor
    /// permutes the image so that patches inside a single
    /// `spatial_merge_size x spatial_merge_size` block are
    /// consecutive in the patch dimension. The downstream merger
    /// then fuses every `spatial_merge_size^2` consecutive
    /// patches into one language-side token, and the resulting
    /// token represents a contiguous 2D block of the original
    /// image (not 4 horizontally adjacent patches, which a naive
    /// row-major reshape would yield).
    ///
    /// Preconditions: `H` and `W` must both be multiples of
    /// `patch_size * spatial_merge_size`. The temporal axis is
    /// duplicated to `temporal_patch_size` so a still image and
    /// a video share the same Conv3d code path.
    public static func toConv3DInput(
        _ pixels: MLXArray,
        patchSize: Int,
        temporalPatchSize: Int,
        spatialMergeSize: Int
    ) -> MLXArray {
        precondition(temporalPatchSize >= 1,
            "temporal_patch_size must be >= 1")
        precondition(spatialMergeSize >= 1,
            "spatial_merge_size must be >= 1")
        let H = pixels.dim(0)
        let W = pixels.dim(1)
        let C = pixels.dim(2)
        let ps = patchSize
        let ms = spatialMergeSize
        precondition(H % (ps * ms) == 0 && W % (ps * ms) == 0,
            "Image height/width must be multiples of "
            + "patch_size * spatial_merge_size; got H=\(H), W=\(W), "
            + "patch_size=\(ps), spatial_merge_size=\(ms)")
        let gridH = H / ps
        let gridW = W / ps

        // [H, W, C] -> [grid_h, ph, grid_w, pw, C]
        var x = pixels.reshaped(gridH, ps, gridW, ps, C)
        // Split grid into outer/inner-merge axes:
        // [grid_h/ms, ms, ph, grid_w/ms, ms, pw, C]
        x = x.reshaped(gridH / ms, ms, ps, gridW / ms, ms, ps, C)
        // Move axes so a single 2D merge-block is consecutive in
        // the patch dimension. The reference order in the patch
        // batch is (out_h, out_w, in_h, in_w):
        //   target [grid_h/ms, grid_w/ms, ms, ms, ph, pw, C]
        //   source axes (0,1,2,3,4,5,6) = (out_h, in_h, ph, out_w, in_w, pw, C)
        // -> transpose to (out_h, out_w, in_h, in_w, ph, pw, C)
        //                 = source axes (0, 3, 1, 4, 2, 5, 6)
        x = x.transposed(0, 3, 1, 4, 2, 5, 6)
        // Flatten to [N_patches, ph, pw, C] in merge-block order.
        let nPatches = (gridH / ms) * (gridW / ms) * ms * ms
        x = x.reshaped(nPatches, ps, ps, C)
        // Insert temporal axis and tile to T frames:
        // [N_patches, 1, ph, pw, C] -> [N_patches, T, ph, pw, C].
        x = x.expandedDimensions(axis: 1)
        if temporalPatchSize > 1 {
            var frames: [MLXArray] = []
            for _ in 0 ..< temporalPatchSize {
                frames.append(x)
            }
            x = MLX.concatenated(frames, axis: 1)
        }
        return x
    }
}
