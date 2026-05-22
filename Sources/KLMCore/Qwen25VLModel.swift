import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache
import KLMKernels
#if canImport(CoreGraphics)
import CoreGraphics
#endif
#if canImport(ImageIO)
import ImageIO
#endif

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
    /// Inverse-frequency table `[halfDim]`, precomputed once at init.
    /// Identical across calls (depends only on `headDim` / `theta`),
    /// so rebuilding it every `apply` was pure overhead.
    let invFreq: MLXArray
    /// Sections normalized to the half-dim convention; precomputed.
    let normSections: [Int]
    /// Cumulative per-axis offsets into `invFreq` so the slicing in
    /// `apply` is one host-side lookup per axis instead of a running
    /// sum.
    let sectionOffsets: [Int]

    public init(headDim: Int, sections: [Int], theta: Float = 1_000_000.0) {
        self.headDim = headDim
        self.sections = sections
        self.theta = theta
        let sectionSum = sections.reduce(0, +)
        precondition(sectionSum * 2 == headDim || sectionSum == headDim,
            "mrope_section must sum to head_dim or head_dim/2; "
            + "got sections=\(sections), head_dim=\(headDim)")
        let halfDim = headDim / 2
        // Two valid conventions: sections sum to head_dim/2 (most
        // common) or to head_dim (older); normalize to half-dim.
        if sectionSum == halfDim {
            self.normSections = sections
        } else {
            self.normSections = sections.map { $0 / 2 }
        }
        var offsets: [Int] = [0]
        for s in normSections { offsets.append(offsets.last! + s) }
        self.sectionOffsets = offsets
        // inv_freq[i] = 1 / theta^(2*i / head_dim) for i in 0..halfDim.
        let positionsHalf = MLXArray(stride(from: 0, to: Int32(headDim), by: 2)
            .map { Float($0) }).asType(.float32)
        self.invFreq = MLXArray(Float(1.0))
            / MLX.pow(MLXArray(Float(theta)),
                      positionsHalf / Float(headDim))
    }

    /// Build broadcast-ready cos/sin tables for a given set of
    /// per-axis position arrays. The text-side mRoPE positions are
    /// the same across all 36 transformer layers within one forward,
    /// so the cos/sin work (per-axis outer product + cos/sin +
    /// concat) is identical at every layer. Computing it once in
    /// the text model and reusing it cuts ~10 elementwise op
    /// dispatches per layer (35 redundant rebuilds).
    ///
    /// Returns `(cos, sin)` of shape `[1, 1, L, halfDim]`, already
    /// expanded with the leading batch/head singleton axes so the
    /// downstream `apply` can broadcast directly against `[B, H,
    /// L, D]` queries/keys.
    public func buildCosSin(
        positionsT: MLXArray,
        positionsH: MLXArray,
        positionsW: MLXArray
    ) -> (cos: MLXArray, sin: MLXArray) {
        let positions: [MLXArray] = [
            positionsT.asType(.float32),
            positionsH.asType(.float32),
            positionsW.asType(.float32),
        ]
        var cosChunks: [MLXArray] = []
        var sinChunks: [MLXArray] = []
        for (axisIdx, sectionLen) in normSections.enumerated() {
            let pos = positions[axisIdx]  // [L]
            let start = sectionOffsets[axisIdx]
            let subInvFreq = invFreq[start ..< (start + sectionLen)]  // [sectionLen]
            let angles = pos.expandedDimensions(axis: 1) * subInvFreq  // [L, sectionLen]
            cosChunks.append(MLX.cos(angles))
            sinChunks.append(MLX.sin(angles))
        }
        let cos = MLX.concatenated(cosChunks, axis: -1)  // [L, halfDim]
        let sin = MLX.concatenated(sinChunks, axis: -1)  // [L, halfDim]
        // Two sequential single-axis expansions avoid the duplicate
        // axis trap that `expandedDimensions(axes: [0, 0])` would hit.
        return (
            cos.expandedDimensions(axis: 0).expandedDimensions(axis: 0),
            sin.expandedDimensions(axis: 0).expandedDimensions(axis: 0))
    }

    /// Apply 3D mRoPE rotation to a `[B, H, L, D]` tensor using
    /// precomputed cos/sin tables of shape `[1, 1, L, halfDim]`.
    public func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xF = x.asType(.float32)
        let halfDim = headDim / 2
        // x shape [B, H, L, D]. Split last dim into two halves
        // x1, x2 of width halfDim. The rotation is:
        //   y1 = x1 * cos - x2 * sin
        //   y2 = x1 * sin + x2 * cos
        let x1 = xF[.ellipsis, 0 ..< halfDim]
        let x2 = xF[.ellipsis, halfDim ..< headDim]
        let y1 = x1 * cos - x2 * sin
        let y2 = x1 * sin + x2 * cos
        let y = MLX.concatenated([y1, y2], axis: -1)
        return y.asType(dtype)
    }

    /// Back-compat: build cos/sin internally then apply. Used by
    /// any caller that has not been threaded through the hoisted
    /// table path.
    public func apply(
        _ x: MLXArray,
        positionsT: MLXArray,
        positionsH: MLXArray,
        positionsW: MLXArray
    ) -> MLXArray {
        let (cos, sin) = buildCosSin(
            positionsT: positionsT, positionsH: positionsH, positionsW: positionsW)
        return apply(x, cos: cos, sin: sin)
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
        // Fused SwiGLU: silu(gate) * up in one Metal kernel pass,
        // avoiding the intermediate fp16 silu materialization. Same
        // fusion the dense Llama FeedForward already uses.
        let gate = gateProj(x)
        let up = upProj(x)
        return downProj(KLMKernels.fusedSwiGLU(gate: gate, up: up))
    }
}

class Qwen25VLVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: RMSNorm
    @ModuleInfo(key: "norm2") var norm2: RMSNorm
    @ModuleInfo(key: "attn") var attn: Qwen25VLVisionAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen25VLVisionMLP

    /// Lazily-built `MLX.compile`'d forwards. The vision block has
    /// no KV cache and is the analogue of GGML's fused Metal
    /// kernels for the vision-tower repeated subgraph: compiling
    /// once and replaying across the 32 vision-block invocations
    /// per prefill fuses the residual / norm / SwiGLU elementwise
    /// ops and cuts kernel-launch overhead. Two variants because
    /// MLX.compile keys on input arity / shape signature: the
    /// periodic full-attention layers pass `mask == nil`, the
    /// windowed layers pass the additive `[1, 1, L, L]` mask.
    private var _compiledMasked: ((MLXArray, MLXArray) -> MLXArray)?
    private var _compiledUnmasked: ((MLXArray) -> MLXArray)?

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

    private func uncompiledForward(_ x: MLXArray, _ mask: MLXArray?) -> MLXArray {
        let h = x + attn(norm1(x), mask: mask)
        return h + mlp(norm2(h))
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        // The compiled closure captures `self` weakly because storing it
        // on `self` would otherwise leak the block for the program's
        // lifetime. The only way `self` is nil when the closure runs is
        // a programmer bug (calling the compiled forward on a block
        // already torn down by the parent module hierarchy); crash
        // loudly rather than silently pass `xi` through, which would
        // skip the entire transformer block.
        if let mask {
            if _compiledMasked == nil {
                _compiledMasked = MLX.compile(inputs: [self]) {
                    [weak self] (xi, mi) -> MLXArray in
                    guard let self else {
                        fatalError(
                            "Qwen25VLVisionBlock compiled forward "
                            + "invoked after the block was deallocated")
                    }
                    return self.uncompiledForward(xi, mi)
                }
            }
            return _compiledMasked!(x, mask)
        }
        if _compiledUnmasked == nil {
            _compiledUnmasked = MLX.compile(inputs: [self]) {
                [weak self] xi -> MLXArray in
                guard let self else {
                    fatalError(
                        "Qwen25VLVisionBlock compiled forward "
                        + "invoked after the block was deallocated")
                }
                return self.uncompiledForward(xi, nil)
            }
        }
        return _compiledUnmasked!(x)
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
        // The Conv3d kernel size equals the stride equals the full
        // patch, so the spatial output is 1x1x1 and the convolution
        // collapses to a per-patch linear projection: each output
        // channel is the dot product of the flattened patch with the
        // flattened kernel row. Running the equivalent matmul avoids
        // the Conv3d kernel-launch overhead and lets the result
        // participate in the surrounding compiled graph (an MLX
        // `Conv3d` call is its own kernel boundary).
        //
        // MLX `Conv3d.weight` shape is `[O, kT, kH, kW, I]`; flatten
        // to `[O, kT*kH*kW*I]`. Input `[N, T, ph, pw, C]` flattens to
        // `[N, T*ph*pw*C]` in the same row-major order, so the
        // resulting `xFlat @ wFlat.T` is bit-identical to the
        // single-output-position Conv3d cross-correlation. An
        // equivalence unit test guards the swap.
        let n = x.dim(0)
        let w = proj.weight
        let wFlat = w.reshaped(embedDim, -1)
        let xFlat = x.reshaped(n, -1)
        return MLX.matmul(xFlat, wFlat.transposed(1, 0))
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

    /// Build the additive window-attention mask for the windowed
    /// vision blocks.
    ///
    /// Qwen 2.5-VL's vision tower runs most blocks with *windowed*
    /// attention: a patch attends only to patches in the same
    /// `vit_merger_window_size` square window of the post-merge
    /// (LLM) grid; the periodic `fullatt_block_indexes` layers run
    /// full attention so information mixes across windows. The HF
    /// reference implements this by permuting the patch sequence so
    /// window members are contiguous and then attending with
    /// variable-length `cu_window_seqlens` boundaries. Because
    /// attention is permutation-equivariant, an equivalent result
    /// is obtained WITHOUT any permutation by a block-diagonal
    /// additive mask over the original patch order: patch *i*
    /// attends *j* iff they share a window.
    ///
    /// `gridHFull` / `gridWFull` are the FULL patch-grid dimensions
    /// (`spatial_merge_size *` the LLM grid). Patches are in
    /// `Qwen25VLImagePreprocessor.toConv3DInput` merge-block
    /// row-major order, so patch `p` belongs to LLM-grid (merged)
    /// token `p / spatial_merge_size^2`, laid out row-major over
    /// the LLM grid. Returns a `[1, 1, L, L]` additive mask, or
    /// `nil` when the whole image fits in a single window (the mask
    /// would then be all-zero, i.e. full attention).
    static func windowAttentionMask(
        gridHFull: Int, gridWFull: Int,
        vision: Qwen25VLConfig.VisionConfig, dtype: DType
    ) -> MLXArray? {
        let ms = max(1, vision.spatialMergeSize)
        let llmH = gridHFull / ms
        let llmW = gridWFull / ms
        guard llmH > 0, llmW > 0 else { return nil }
        // vit_merger_window_size: window edge in LLM-grid (merged)
        // tokens. window_size is in pixels; / patch_size gives
        // patches, / spatial_merge_size gives merged tokens.
        let vitWin = max(1, vision.windowSize / vision.patchSize / ms)
        let numWinW = (llmW + vitWin - 1) / vitWin
        let numWinH = (llmH + vitWin - 1) / vitWin
        // One window covers the whole image -> windowed attention
        // is identical to full attention; skip the mask entirely.
        if numWinH * numWinW <= 1 { return nil }

        let patchesPerMerge = ms * ms
        let nPatches = gridHFull * gridWFull
        var winId = [Int32](repeating: 0, count: nPatches)
        for p in 0 ..< nPatches {
            let merged = p / patchesPerMerge
            let r = merged / llmW
            let c = merged % llmW
            winId[p] = Int32((r / vitWin) * numWinW + (c / vitWin))
        }
        let ids = MLXArray(winId)  // [L]
        let same = ids.expandedDimensions(axis: 1)
            .== ids.expandedDimensions(axis: 0)  // [L, L] bool
        // -1e4 (not -inf): fp16-safe and the value Gemma 4's vision
        // encoder uses for the same purpose.
        let mask = MLX.where(
            same, MLXArray(Float(0)), MLXArray(Float(-1e4)))
        return mask.reshaped(1, 1, nPatches, nPatches).asType(dtype)
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
    /// - Parameter gridHWFull: the FULL patch-grid `(h, w)` of the
    ///   image. When supplied, windowed blocks attend with the
    ///   `windowAttentionMask` and `fullAttnBlockIndexes` layers
    ///   attend globally - matching the HF reference. When `nil`,
    ///   every block runs full attention (the back-compatible path
    ///   used by module-level tests that do not have a grid).
    public func callAsFunction(
        _ patchBatch: MLXArray, gridHWFull: (Int, Int)? = nil
    ) -> MLXArray {
        var h = patchEmbed(patchBatch)  // [N_patches, vision_hidden]
        h = h.expandedDimensions(axis: 0)  // [1, N_patches, hidden]
        let windowMask: MLXArray? = gridHWFull.flatMap { grid in
            Self.windowAttentionMask(
                gridHFull: grid.0, gridWFull: grid.1,
                vision: visionConfig, dtype: h.dtype)
        }
        for (i, block) in blocks.enumerated() {
            // Full-attention layers see the whole image; windowed
            // layers see only same-window patches.
            let mask = fullAttnLayers.contains(i) ? nil : windowMask
            h = block(h, mask: mask)
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

    /// Smart-resize target dimensions for an image of the given
    /// size: both a multiple of `factor` (`patch_size *
    /// spatial_merge_size`), aspect ratio preserved as closely as
    /// the `factor` rounding allows, total pixel count clamped into
    /// `[minPixels, maxPixels]`. Mirrors the HF Qwen 2.5-VL image
    /// processor `smart_resize`.
    static func smartResize(
        height: Int, width: Int, factor: Int,
        minPixels: Int, maxPixels: Int
    ) -> (h: Int, w: Int) {
        let hF = Double(height), wF = Double(width)
        func roundTo(_ v: Double) -> Int {
            max(factor, Int((v / Double(factor)).rounded()) * factor)
        }
        var hBar = roundTo(hF)
        var wBar = roundTo(wF)
        if hBar * wBar > maxPixels {
            let beta = (hF * wF / Double(maxPixels)).squareRoot()
            hBar = max(factor,
                Int((hF / beta / Double(factor)).rounded(.down)) * factor)
            wBar = max(factor,
                Int((wF / beta / Double(factor)).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / (hF * wF)).squareRoot()
            hBar = max(factor,
                Int((hF * beta / Double(factor)).rounded(.up)) * factor)
            wBar = max(factor,
                Int((wF * beta / Double(factor)).rounded(.up)) * factor)
        }
        return (hBar, wBar)
    }

    /// Upper bound on the resized pixel count. Caps the vision
    /// sequence length L (and hence the L x L window-attention
    /// mask) at a runtime-sane size: 768x768 / 14^2 ~= 3000 patches.
    static let maxPixels = 768 * 768

    /// Decode PNG/JPEG image data into a normalized `[H, W, 3]`
    /// float32 tensor in `[0, 1]`, resized (via `smartResize`) so
    /// `H` and `W` are multiples of `patch_size * spatial_merge_size`.
    /// CoreGraphics-backed; throws `imagePreprocessingUnavailable`
    /// on platforms without it.
    public static func decode(
        _ imageData: Data, patchSize: Int, spatialMergeSize: Int
    ) throws -> MLXArray {
        guard !imageData.isEmpty else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        #if canImport(CoreGraphics) && canImport(ImageIO)
        let factor = patchSize * spatialMergeSize
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        let (newH, newW) = smartResize(
            height: cgImage.height, width: cgImage.width, factor: factor,
            minPixels: factor * factor * 4, maxPixels: maxPixels)

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: newW, height: newH,
            bitsPerComponent: 8, bytesPerRow: newW * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        // Resize by drawing into the full target rect (CoreGraphics
        // interpolates). smartResize already preserved the aspect
        // ratio to within the `factor` rounding.
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newW, height: newH))
        guard let data = context.data else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        // RGBA -> channel-last [H, W, 3] in [0, 1]. CGContext stores
        // rows bottom-to-top; flip so row 0 is the top of the image.
        let pixelCount = newH * newW
        var floats = [Float](repeating: 0, count: pixelCount * 3)
        let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)
        for row in 0 ..< newH {
            let flippedRow = newH - 1 - row
            for col in 0 ..< newW {
                let srcIdx = (flippedRow * newW + col) * 4
                let dstIdx = (row * newW + col) * 3
                floats[dstIdx] = Float(ptr[srcIdx]) / 255.0
                floats[dstIdx + 1] = Float(ptr[srcIdx + 1]) / 255.0
                floats[dstIdx + 2] = Float(ptr[srcIdx + 2]) / 255.0
            }
        }
        return MLXArray(floats, [newH, newW, 3])
        #else
        throw MultimodalPreprocessingError.imagePreprocessingUnavailable
        #endif
    }

    /// Full image -> patch-batch pipeline: decode + resize,
    /// CLIP-normalize, pack into the Conv3d per-patch batch, and
    /// report the post-spatial-merge image grid.
    ///
    /// The returned `(gridHMerged, gridWMerged)` is what the caller
    /// uses to size the `<|image_pad|>` placeholder run
    /// (`gridHMerged * gridWMerged` tokens) and to drive the 3D
    /// mRoPE positions. The grid is generally NOT square.
    public static func preprocess(
        _ imageData: Data, vision: Qwen25VLConfig.VisionConfig
    ) throws -> (patches: MLXArray, gridHMerged: Int, gridWMerged: Int) {
        let pixels = try decode(
            imageData, patchSize: vision.patchSize,
            spatialMergeSize: vision.spatialMergeSize)
        let H = pixels.dim(0)
        let W = pixels.dim(1)
        let patches = toConv3DInput(
            normalize(pixels),
            patchSize: vision.patchSize,
            temporalPatchSize: vision.temporalPatchSize,
            spatialMergeSize: vision.spatialMergeSize)
        let mergeFactor = vision.patchSize * vision.spatialMergeSize
        return (patches, H / mergeFactor, W / mergeFactor)
    }
}

// MARK: - mRoPE position computation

/// 3D mRoPE position ids for a Qwen 2.5-VL prompt that contains at
/// most one image span.
///
/// Each token gets three position coordinates (t, h, w):
///   - Text tokens advance all three axes together (t == h == w),
///     so they behave like standard 1D RoPE.
///   - Image-placeholder tokens (`<|image_pad|>`) span a
///     `gridHMerged x gridWMerged` 2D grid. The temporal axis is
///     constant across the image (`startPos`); the height axis
///     ranges `startPos ..< startPos + gridHMerged`; the width
///     axis ranges `startPos ..< startPos + gridWMerged`.
///   - After the image, text resumes at
///     `startPos + max(gridHMerged, gridWMerged)`, matching the
///     reference `get_rope_index` in HF transformers.
///
/// `gridHMerged` / `gridWMerged` are the post-spatial-merge grid
/// dimensions (the count of `<|image_pad|>` tokens is their
/// product). Computed on the host because the token sequence is
/// known; returns three `[L]` Int32 `MLXArray`s.
public enum Qwen25VLPositions {
    public struct Coords: Sendable {
        public let t: [Int32]
        public let h: [Int32]
        public let w: [Int32]
        /// The mRoPE position the FIRST token after this prompt must
        /// take. For a pure-text prompt this equals the prompt
        /// length; for an image prompt it is smaller than the token
        /// count, because the `gridH * gridW` `<|image_pad|>` tokens
        /// occupy only `max(gridH, gridW)` positions. A native
        /// decode loop threads `nextPos + k` as the
        /// `mropePositionOffset` of decode step `k`.
        ///
        /// Computed pre-offset: a caller that prefilled with a
        /// non-zero `mropePositionOffset` must add that offset back.
        public let nextPos: Int32
    }

    /// - Parameters:
    ///   - tokenIds: the flat prompt token ids (host array).
    ///   - imageTokenId: the `<|image_pad|>` id.
    ///   - gridHMerged: post-merge image grid height (0 if no image).
    ///   - gridWMerged: post-merge image grid width (0 if no image).
    public static func compute(
        tokenIds: [Int32],
        imageTokenId: Int,
        gridHMerged: Int,
        gridWMerged: Int
    ) -> Coords {
        var t: [Int32] = []
        var h: [Int32] = []
        var w: [Int32] = []
        t.reserveCapacity(tokenIds.count)
        h.reserveCapacity(tokenIds.count)
        w.reserveCapacity(tokenIds.count)

        var nextPos: Int32 = 0
        var i = 0
        let imgTok = Int32(imageTokenId)
        while i < tokenIds.count {
            if tokenIds[i] == imgTok && gridHMerged > 0 && gridWMerged > 0 {
                // Image span. Consume exactly gridHMerged*gridWMerged
                // placeholder tokens (or until the run of image
                // tokens ends, whichever is shorter, so a malformed
                // prompt cannot walk off the end).
                let start = nextPos
                var consumed = 0
                let total = gridHMerged * gridWMerged
                while i < tokenIds.count && tokenIds[i] == imgTok && consumed < total {
                    let row = Int32(consumed / gridWMerged)
                    let col = Int32(consumed % gridWMerged)
                    t.append(start)
                    h.append(start + row)
                    w.append(start + col)
                    consumed += 1
                    i += 1
                }
                nextPos = start + Int32(max(gridHMerged, gridWMerged))
            } else {
                t.append(nextPos)
                h.append(nextPos)
                w.append(nextPos)
                nextPos += 1
                i += 1
            }
        }
        return Coords(t: t, h: h, w: w, nextPos: nextPos)
    }
}

// MARK: - Text-side attention with 3D mRoPE

/// Qwen 2.5-VL text attention. Identical to dense Qwen 2.5
/// attention (QKV bias, no q_norm/k_norm, GQA) EXCEPT that the
/// rotary embedding is 3D mRoPE applied with explicit per-axis
/// position arrays, rather than the standard 1D `RoPE` module.
class Qwen25VLTextAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let mrope: Qwen25VLMRoPE

    init(_ config: Qwen25VLConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()
        self.mrope = Qwen25VLMRoPE(
            headDim: config.headDim,
            sections: config.mropeSection,
            theta: config.ropeTheta)
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: true), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: true), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: true), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray, sin: MLXArray,
        mask: MLXArray? = nil, cache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // 3D mRoPE on Q and K with precomputed cos/sin tables. The
        // cos/sin work is identical across all 36 text layers, so
        // the caller hoists it out of the layer loop and shares one
        // pair of tables per forward.
        queries = mrope.apply(queries, cos: cos, sin: sin)
        keys = mrope.apply(keys, cos: cos, sin: sin)

        var k = keys
        var v = values
        if let cache {
            (k, v) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: k, values: v, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Text-side transformer block

class Qwen25VLTextBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen25VLTextAttention
    @ModuleInfo(key: "mlp") var mlp: QwenMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: Qwen25VLConfig) {
        _selfAttn = ModuleInfo(
            wrappedValue: Qwen25VLTextAttention(config), key: "self_attn")
        _mlp = ModuleInfo(
            wrappedValue: QwenMLP(config.qwenTextConfig), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(
        _ x: MLXArray,
        cos: MLXArray, sin: MLXArray,
        mask: MLXArray? = nil, cache: KVCache? = nil
    ) -> MLXArray {
        let h = x + selfAttn(
            inputLayernorm(x),
            cos: cos, sin: sin,
            mask: mask, cache: cache)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - Text-side model

/// Qwen 2.5-VL language model: token embedding, mRoPE transformer
/// blocks, final norm. Weight keys live under `model.*` in the
/// checkpoint. `callAsFunction` accepts pre-computed input
/// embeddings (rather than token ids) so the multimodal forward
/// can inject vision embeddings before the transformer stack.
class Qwen25VLTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen25VLTextBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Qwen25VLConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in
                Qwen25VLTextBlock(config)
            },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(
        inputEmbeds: MLXArray,
        positionsT: MLXArray, positionsH: MLXArray, positionsW: MLXArray,
        caches: [KVCache]? = nil
    ) -> MLXArray {
        var x = inputEmbeds
        let seqLen = x.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let mask = createCachedCausalMask(
            newLen: seqLen, cacheLen: cacheLen, dtype: x.dtype)
        // Build the 3D mRoPE cos/sin tables once and share across
        // every layer (all 36 attention layers use the same per-axis
        // positions). The first layer's mrope owns the constants
        // (invFreq, section offsets); every layer was constructed
        // from the same config so the tables are identical.
        let (cos, sin) = layers[0].selfAttn.mrope.buildCosSin(
            positionsT: positionsT, positionsH: positionsH, positionsW: positionsW)
        for (i, layer) in layers.enumerated() {
            x = layer(
                x,
                cos: cos, sin: sin,
                mask: mask, cache: caches?[i])
        }
        return norm(x)
    }
}

// MARK: - Language-model wrapper

/// Wraps the Qwen 2.5-VL text tower + LM head under the
/// `language_model.*` key prefix that the mlx-vlm checkpoint
/// layout uses (`language_model.model.*` for the transformer,
/// `language_model.lm_head.*` for the projection head).
class Qwen25VLLanguageModel: Module {
    @ModuleInfo(key: "model") var model: Qwen25VLTextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Qwen25VLConfig) {
        _model = ModuleInfo(wrappedValue: Qwen25VLTextModel(config), key: "model")
        if !config.tieWordEmbeddings {
            _lmHead = ModuleInfo(
                wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
                key: "lm_head")
        }
    }
}

// MARK: - Qwen 2.5-VL ForConditionalGeneration

/// Native Qwen 2.5-VL multimodal model: vision tower + text tower
/// + LM head.
///
/// Module key layout matches the mlx-vlm checkpoint convention
/// used by the mlx-community Qwen2.5-VL-*-Instruct-*bit repos:
///   - `vision_tower.*`           patch embed, blocks, merger
///   - `language_model.model.*`   token embedding, text layers, norm
///   - `language_model.lm_head.*` projection head (absent when
///      embeddings are tied)
///
/// The multimodal forward (`callAsFunction`):
///   1. Embed the text tokens.
///   2. If `pixelValues` is provided, run the vision tower to get
///      `[n_merged, hidden]` vision embeddings, then splice them
///      into the input-embedding sequence at the contiguous
///      `<|image_pad|>` span.
///   3. Compute 3D mRoPE position ids for the merged sequence.
///   4. Run the text transformer stack and project to logits.
public class Qwen25VLForConditionalGeneration: Module {
    @ModuleInfo(key: "vision_tower") var visual: Qwen25VLVisionTower
    @ModuleInfo(key: "language_model") var languageModel: Qwen25VLLanguageModel

    public let config: Qwen25VLConfig

    public init(_ config: Qwen25VLConfig) {
        self.config = config
        _visual = ModuleInfo(
            wrappedValue: Qwen25VLVisionTower(config.vision, normEps: config.rmsNormEps),
            key: "vision_tower")
        _languageModel = ModuleInfo(
            wrappedValue: Qwen25VLLanguageModel(config), key: "language_model")
    }

    /// Splice vision embeddings into the text input-embedding
    /// sequence at the `<|image_pad|>` span.
    ///
    /// `inputEmbeds` is `[1, L, H]`; `visionEmbeds` is
    /// `[n_merged, H]`. `imagePadStart` is the host index of the
    /// first `<|image_pad|>` token; the span `[imagePadStart,
    /// imagePadStart + n_merged)` is replaced. Returns `[1, L, H]`.
    ///
    /// The span bounds are validated: a mismatch between the
    /// vision-embed count and the placeholder span (a malformed
    /// prompt, or a grid that disagrees with the image) would
    /// otherwise either crash on an out-of-range slice or silently
    /// corrupt the sequence. The caller is expected to have
    /// verified the span is genuinely `<|image_pad|>` tokens.
    static func injectVisionEmbeds(
        inputEmbeds: MLXArray, visionEmbeds: MLXArray, imagePadStart: Int
    ) -> MLXArray {
        let L = inputEmbeds.dim(1)
        let n = visionEmbeds.dim(0)
        precondition(imagePadStart >= 0,
            "imagePadStart must be non-negative; got \(imagePadStart)")
        precondition(imagePadStart + n <= L,
            "Vision-embed span [\(imagePadStart), \(imagePadStart + n)) "
            + "exceeds the input sequence length \(L). The vision-embed "
            + "count (\(n)) must match the <|image_pad|> placeholder span.")
        let before = inputEmbeds[0..., 0 ..< imagePadStart, 0...]
        let after = inputEmbeds[0..., (imagePadStart + n) ..< L, 0...]
        let mid = visionEmbeds.expandedDimensions(axis: 0)  // [1, n, H]
        return MLX.concatenated([before, mid, after], axis: 1)
    }

    /// Count the contiguous run of `<|image_pad|>` tokens starting
    /// at `start` in the host token-id list.
    private func imagePadSpanLength(_ tokenIds: [Int32], from start: Int) -> Int {
        let pad = Int32(config.imageTokenId)
        var n = 0
        var i = start
        while i < tokenIds.count && tokenIds[i] == pad {
            n += 1
            i += 1
        }
        return n
    }

    /// Multimodal forward.
    ///
    /// - Parameters:
    ///   - tokens: prompt token ids `[1, L]`.
    ///   - pixelValues: preprocessed per-patch batch
    ///     `[n_patches, T, ps, ps, C]` for the single image, or nil
    ///     for a text-only prompt.
    ///   - imageGridMerged: post-spatial-merge `(gridH, gridW)` of
    ///     the image, or nil for a text-only prompt. The product
    ///     must equal the number of `<|image_pad|>` tokens.
    ///   - caches: optional per-layer KV caches.
    ///   - mropePositionOffset: explicit base offset added to the
    ///     computed 3D mRoPE position ids. Pass `nil` for a fresh
    ///     prefill (offset 0). For a decode step the caller MUST
    ///     pass the prefill's final mRoPE position + 1: the cache
    ///     length is NOT a valid offset after an image prompt
    ///     because the image span compresses
    ///     `gridH * gridW` placeholder tokens down to
    ///     `max(gridH, gridW)` positions. Threading the precise
    ///     decode offset is the server-integration follow-up; for
    ///     a text-only prompt `nil` falls back to the cache length,
    ///     which is correct there.
    public func callAsFunction(
        _ tokens: MLXArray,
        pixelValues: MLXArray? = nil,
        imageGridMerged: (Int, Int)? = nil,
        caches: [KVCache]? = nil,
        mropePositionOffset: Int? = nil,
        hostTokenIds: [Int32]? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        let textModel = languageModel.model
        var inputEmbeds = textModel.embedTokens(tokens)  // [1, L, H]

        // Host-side token id list for image-span location + mRoPE.
        // When the caller already has it (the runtime constructs
        // tokens from a host `[Int32]` array), threading it through
        // avoids the `eval(tokens); asArray(...)` GPU->host stall
        // that would otherwise sit in the middle of the forward and
        // block whole-graph fusion.
        let tokenIds: [Int32]
        if let hostTokenIds {
            precondition(hostTokenIds.count == tokens.dim(1),
                "hostTokenIds must match tokens.dim(1); got "
                + "\(hostTokenIds.count) vs \(tokens.dim(1))")
            tokenIds = hostTokenIds
        } else {
            eval(tokens)
            tokenIds = tokens.asArray(Int32.self)
        }

        var gridH = 0
        var gridW = 0
        if let pixelValues, let grid = imageGridMerged {
            gridH = grid.0
            gridW = grid.1
            // The vision tower needs the FULL patch grid (the merged
            // grid times spatial_merge_size) to build window masks.
            let ms = config.vision.spatialMergeSize
            let visionEmbeds = visual(
                pixelValues, gridHWFull: (gridH * ms, gridW * ms))  // [n_merged, H]
            if let start = tokenIds.firstIndex(of: Int32(config.imageTokenId)) {
                // The contiguous <|image_pad|> span must match the
                // vision-embed count exactly; otherwise the splice
                // would corrupt the sequence or trip the
                // injectVisionEmbeds bounds precondition.
                let spanLen = imagePadSpanLength(tokenIds, from: start)
                let nMerged = visionEmbeds.dim(0)
                precondition(spanLen == nMerged,
                    "The <|image_pad|> span (\(spanLen) tokens) must "
                    + "equal the merged vision-embed count (\(nMerged) "
                    + "= n_patches / spatial_merge_size^2). The image "
                    + "grid passed to the forward disagrees with the "
                    + "prompt's placeholder count.")
                inputEmbeds = Self.injectVisionEmbeds(
                    inputEmbeds: inputEmbeds,
                    visionEmbeds: visionEmbeds,
                    imagePadStart: start)
            }
        }

        // 3D mRoPE position ids. With no image, gridH/gridW are 0
        // and every token advances all three axes together.
        let coords = Qwen25VLPositions.compute(
            tokenIds: tokenIds,
            imageTokenId: config.imageTokenId,
            gridHMerged: gridH,
            gridWMerged: gridW)
        // Prefill (cacheLen 0) -> offset 0. For a decode step the
        // caller passes the explicit mRoPE offset; falling back to
        // the cache length is correct ONLY for a text-only prompt.
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let offset = Int32(mropePositionOffset ?? cacheLen)
        let posT = MLXArray(coords.t.map { $0 + offset })
        let posH = MLXArray(coords.h.map { $0 + offset })
        let posW = MLXArray(coords.w.map { $0 + offset })

        var hidden = textModel(
            inputEmbeds: inputEmbeds,
            positionsT: posT, positionsH: posH, positionsW: posW,
            caches: caches)
        // The runtime samples from the last position only, so for
        // prefill the lm_head's `[1, L, vocab]` projection is
        // wasteful work over ~L-1 unused rows: the 2048 x 151936
        // (or tied-embedding equivalent) matmul dominates the head
        // cost. Slicing `[:, -1:, :]` before the projection drops
        // the vocab matmul to a single position. The KV cache was
        // already filled by the attention layers above, so the
        // sliced path is bit-exact for the sampled token. Decode
        // steps forward a single token so `hidden.dim(1)` is 1 and
        // the slice is a no-op.
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        if let lmHead = languageModel.lmHead {
            return lmHead(hidden)
        }
        return textModel.embedTokens.asLinear(hidden)
    }
}
