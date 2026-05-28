import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Gemma 4 Vision Config

/// Vision tower config parsed from `config.json["vision_config"]`.
/// Gemma 4 e2b/e4b ship a SigLIP-base tower (hidden 768, 12 heads, 64
/// head_dim); 26B-A4B ships a larger tower (hidden 1152, 16 heads, 72
/// head_dim). Prior to PR #79 the `VisionEncoder` instance was built
/// with hardcoded e2b shapes, so 26B-A4B crashed at load with a 1.5x
/// reshape mismatch in the patch embed (#76). All values come from the
/// checkpoint; defaults match the prior hardcoded e2b shapes so an
/// older checkpoint with missing fields still loads.
public struct Gemma4VisionConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let poolingKernelSize: Int
    public let positionEmbeddingSize: Int
    public let defaultOutputLength: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case positionEmbeddingSize = "position_embedding_size"
        case defaultOutputLength = "default_output_length"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 16
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        poolingKernelSize = try c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3
        positionEmbeddingSize = try c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240
        defaultOutputLength = try c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6

        // `rope_theta` can be either a flat field or nested under
        // `rope_parameters.rope_theta`. Gemma 4 e2b/e4b use the flat
        // form; 26B-A4B uses the nested form. Honor both.
        if let theta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeTheta = theta
        } else if let nested = try? c.nestedContainer(
                keyedBy: CodingKeys.self, forKey: .ropeParameters),
                let theta = try? nested.decode(Float.self, forKey: .ropeTheta) {
            ropeTheta = theta
        } else {
            ropeTheta = 100.0
        }
    }

    /// e2b / e4b SigLIP-base defaults, used as the fallback when a
    /// checkpoint has no parseable `vision_config` block.
    public static let defaults = Gemma4VisionConfig(
        hiddenSize: 768, intermediateSize: 3072,
        numHiddenLayers: 16, numAttentionHeads: 12,
        numKeyValueHeads: 12, headDim: 64,
        patchSize: 16, poolingKernelSize: 3,
        positionEmbeddingSize: 10240, defaultOutputLength: 280,
        ropeTheta: 100.0, rmsNormEps: 1e-6)

    public init(
        hiddenSize: Int, intermediateSize: Int, numHiddenLayers: Int,
        numAttentionHeads: Int, numKeyValueHeads: Int, headDim: Int,
        patchSize: Int, poolingKernelSize: Int, positionEmbeddingSize: Int,
        defaultOutputLength: Int, ropeTheta: Float, rmsNormEps: Float
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.positionEmbeddingSize = positionEmbeddingSize
        self.defaultOutputLength = defaultOutputLength
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
    }
}

// MARK: - Gemma 4 Config

/// Configuration for Gemma 4 model family (E2B, E4B, 12B, 27B, 31B).
/// Complex architecture: PLE, dual head_dim, KV sharing, 4-norm blocks,
/// and (26B-A4B only) a per-layer sparse MoE branch alongside the dense
/// MLP plus K-eq-V global attention.
public struct Gemma4Config: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int

    // Gemma 4 specific
    public let headDim: Int              // 256 for sliding layers
    public let globalHeadDim: Int        // 512 for full attention layers
    public let slidingWindow: Int
    public let slidingWindowPattern: Int  // every Nth layer is full attention (fallback only)
    /// Canonical per-layer attention type list from `config.json`
    /// (`"full_attention"` or `"sliding_attention"`). Authoritative when
    /// present; we fall back to the `slidingWindowPattern` modulo only
    /// when it isn't. All Gemma 4 SKUs released today (e2b, e4b,
    /// 26B-A4B, 31B) publish this list and use *different* full-layer
    /// strides (e2b: every 5th; e4b / 26B-A4B: every 6th), so trusting
    /// the modulo alone gets the wrong head_dim on every non-e2b
    /// variant and the attention reshape crashes at first inference.
    public let layerTypes: [String]?
    public let hiddenSizePerLayerInput: Int  // PLE dimension (256); 0 disables PLE (26B-A4B)
    public let numKVSharedLayers: Int    // layers sharing KV from earlier layers; 0 disables (26B-A4B)
    public let useDoubleWideMlp: Bool
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool
    public let quantization: QuantizationConfig?
    /// Parsed `vision_config` from the checkpoint, when present.
    /// Multimodal Gemma 4 checkpoints (e2b/e4b/26B-A4B) ship this;
    /// text-only checkpoints (12B/27B) do not.
    public let visionConfig: Gemma4VisionConfig?

    // MoE fields (26B-A4B only; defaults disable the MoE branch).
    /// Toggles the per-layer sparse MoE branch in parallel with the
    /// dense MLP. True on Gemma 4 26B-A4B, false everywhere else.
    public let enableMoeBlock: Bool
    public let numExperts: Int?
    public let topKExperts: Int?
    public let moeIntermediateSize: Int?

    // K-eq-V global attention (26B-A4B only). When true, full-attention
    // layers reuse the K projection as V and use
    // `numGlobalKeyValueHeads` heads instead of `numKeyValueHeads`.
    public let attentionKEqV: Bool
    public let numGlobalKeyValueHeads: Int?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case layerTypes = "layer_types"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
        case quantizationConfig = "quantization_config"
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case enableMoeBlock = "enable_moe_block"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case attentionKEqV = "attention_k_eq_v"
        case numGlobalKeyValueHeads = "num_global_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        // Parse from nested text_config or flat
        let tc: KeyedDecodingContainer<CodingKeys>
        if let nested = try? c.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {
            tc = nested
        } else {
            tc = c
        }

        hiddenSize = try tc.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try tc.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try tc.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try tc.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        numHiddenLayers = try tc.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try tc.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try tc.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings = try tc.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        headDim = try tc.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try tc.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        slidingWindow = try tc.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try tc.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        layerTypes = try tc.decodeIfPresent([String].self, forKey: .layerTypes)
        hiddenSizePerLayerInput = try tc.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        numKVSharedLayers = try tc.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 20
        useDoubleWideMlp = try tc.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        finalLogitSoftcapping = try tc.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        tieWordEmbeddings = try tc.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Quantization at top level
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
            ?? (try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig))

        enableMoeBlock = try tc.decodeIfPresent(Bool.self, forKey: .enableMoeBlock) ?? false
        numExperts = try tc.decodeIfPresent(Int.self, forKey: .numExperts)
        topKExperts = try tc.decodeIfPresent(Int.self, forKey: .topKExperts)
        moeIntermediateSize = try tc.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        attentionKEqV = try tc.decodeIfPresent(Bool.self, forKey: .attentionKEqV) ?? false
        numGlobalKeyValueHeads = try tc.decodeIfPresent(Int.self, forKey: .numGlobalKeyValueHeads)

        // vision_config is at top level for all Gemma 4 multimodal SKUs.
        // A Bool sentinel (some text-only checkpoints use literal `false`
        // to signal "no vision tower") is treated as missing. An object
        // form must decode STRICTLY so a malformed vision_config surfaces
        // an error rather than silently falling back to the SigLIP-base
        // defaults. The latter would re-introduce the original #76 crash
        // class on a partly-malformed 26B-A4B vision_config.
        if (try? c.decode(Bool.self, forKey: .visionConfig)) != nil {
            visionConfig = nil
        } else {
            visionConfig = try c.decodeIfPresent(
                Gemma4VisionConfig.self, forKey: .visionConfig)
        }
    }

    /// Whether a layer uses full attention (vs sliding window).
    ///
    /// Prefers the canonical `layer_types` list from `config.json` and
    /// falls back to the `(idx + 1) % slidingWindowPattern == 0` modulo
    /// only when that key is missing. The modulo default of 5 matched
    /// Gemma 4 e2b by coincidence; e4b and 26B-A4B publish the same
    /// `layer_types` shape but skip the modulo key entirely, and their
    /// actual pattern is every 6th layer.
    public func isFullAttention(layerIdx: Int) -> Bool {
        if let types = layerTypes,
           layerIdx >= 0, layerIdx < types.count {
            return types[layerIdx] == "full_attention"
        }
        return (layerIdx + 1) % slidingWindowPattern == 0
    }

    /// Whether a layer shares KV from an earlier layer.
    public func isKVShared(layerIdx: Int) -> Bool {
        layerIdx >= (numHiddenLayers - numKVSharedLayers)
    }

    /// First layer index that shares KV.
    public var firstKVSharedLayer: Int { numHiddenLayers - numKVSharedLayers }

    /// Whether this checkpoint enables per-layer-input (PLE) embeddings.
    /// True for Gemma 4 e2b/e4b (`hidden_size_per_layer_input=256`),
    /// false for 26B-A4B (`hidden_size_per_layer_input=0`).
    public var hasPerLayerInputs: Bool { hiddenSizePerLayerInput > 0 }

    /// KV head count for a given layer. Full-attention layers on
    /// 26B-A4B use `num_global_key_value_heads=2` (versus 8 for sliding
    /// layers) when `attention_k_eq_v` is set; everywhere else returns
    /// the per-config `numKeyValueHeads`.
    public func kvHeads(layerIdx: Int) -> Int {
        if attentionKEqV, isFullAttention(layerIdx: layerIdx),
           let g = numGlobalKeyValueHeads {
            return g
        }
        return numKeyValueHeads
    }

    /// True for layers where K is reused as V (no v_proj weights ship
    /// in the checkpoint). Only Gemma 4 26B-A4B with
    /// `attention_k_eq_v` and a full-attention layer.
    public func useKEqV(layerIdx: Int) -> Bool {
        attentionKEqV && isFullAttention(layerIdx: layerIdx)
    }

    // Conform to ModelConfig
    public var ropeTheta: Float { 10_000.0 }
}

// Explicit ModelConfig conformance
extension Gemma4Config: ModelConfig {}

// MARK: - Gemma 4 Attention

class Gemma4Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let layerHeadDim: Int
    let isFullAttn: Bool
    /// True on 26B-A4B full-attention layers when the checkpoint sets
    /// `attention_k_eq_v`. In that mode v_proj is absent and we reuse
    /// the K projection as V (matching the mlx-lm reference at
    /// `gemma4_text.Attention.has_kv / use_k_eq_v`).
    let useKEqV: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    /// Optional: omitted when `useKEqV` is true.
    @ModuleInfo(key: "v_proj") var vProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE
    let rmsNormEps: Float

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.kvHeads(layerIdx: layerIdx)
        self.isFullAttn = config.isFullAttention(layerIdx: layerIdx)
        self.layerHeadDim = isFullAttn ? config.globalHeadDim : config.headDim
        self.rmsNormEps = config.rmsNormEps
        self.useKEqV = config.useKEqV(layerIdx: layerIdx)

        let qSize = numHeads * layerHeadDim
        let kvSize = numKVHeads * layerHeadDim

        _qProj = ModuleInfo(wrappedValue: Linear(dim, qSize, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(dim, kvSize, bias: false), key: "k_proj")
        if !useKEqV {
            _vProj = ModuleInfo(wrappedValue: Linear(dim, kvSize, bias: false), key: "v_proj")
        } else {
            _vProj = ModuleInfo(wrappedValue: nil, key: "v_proj")
        }
        _oProj = ModuleInfo(wrappedValue: Linear(qSize, dim, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: layerHeadDim, eps: config.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: layerHeadDim, eps: config.rmsNormEps), key: "k_norm")

        // RoPE configuration:
        // Sliding layers: standard RoPE on all 256 dims, base=10000
        // Full layers: ProportionalRoPE - only 128 dims rotated but with freqs from 512-dim space
        //   MLX RoPE(dimensions=128) rotates first 128 dims with freqs base^(-2i/128)
        //   We need freqs base^(-2i/512) but only applied to 128 dims.
        //   Trick: RoPE(dimensions=128) with adjusted base that produces equivalent freqs.
        //   freq_i = base^(-2i/dims). We want base'^(-2i/128) = 1e6^(-2i/512)
        //   => base' = 1e6^(128/512) = 1e6^0.25 = 31.623
        //   This gives the same frequency values as 1e6 with dims=512.
        if isFullAttn {
            // ProportionalRoPE: rotate 128 dims with freqs from 512-dim space.
            // Adjusted base: 1e6^(128/512) = 31.623 gives correct frequency spacing.
            // Note: this rotates dims [0:128] sequentially. The reference rotates
            // [0:64]+[256:320] (split halves). For short sequences (<512 tokens),
            // the positional info is dominated by the lower-frequency components
            // which are the same in both orderings.
            let adjustedBase = powf(1_000_000.0, 128.0 / Float(layerHeadDim))
            self.rope = RoPE(dimensions: 128, traditional: false, base: adjustedBase)
        } else {
            self.rope = RoPE(dimensions: layerHeadDim, traditional: false, base: 10_000.0)
        }
    }

    /// Apply RoPE, handling proportional (split-half) for full attention layers.
    private func applyRoPE(_ x: MLXArray, offset: Int) -> MLXArray {
        if isFullAttn {
            // ProportionalRoPE: split head into halves, rotate first 64 dims of each
            // x shape: [B, H, L, headDim=512]
            let halfDim = layerHeadDim / 2  // 256
            let rotateDims = 64  // 128 total / 2 halves

            let leftHalf = x[0..., 0..., 0..., ..<halfDim]      // [B,H,L,256]
            let rightHalf = x[0..., 0..., 0..., halfDim...]      // [B,H,L,256]

            // Extract dims to rotate
            let leftRotate = leftHalf[0..., 0..., 0..., ..<rotateDims]   // [B,H,L,64]
            let leftKeep = leftHalf[0..., 0..., 0..., rotateDims...]     // [B,H,L,192]
            let rightRotate = rightHalf[0..., 0..., 0..., ..<rotateDims] // [B,H,L,64]
            let rightKeep = rightHalf[0..., 0..., 0..., rotateDims...]   // [B,H,L,192]

            // Concatenate the dims to rotate and apply RoPE
            let toRotate = concatenated([leftRotate, rightRotate], axis: -1) // [B,H,L,128]
            let rotated = rope(toRotate, offset: offset)                     // [B,H,L,128]

            // Split back
            let leftRotated = rotated[0..., 0..., 0..., ..<rotateDims]    // [B,H,L,64]
            let rightRotated = rotated[0..., 0..., 0..., rotateDims...]   // [B,H,L,64]

            // Reassemble
            let newLeft = concatenated([leftRotated, leftKeep], axis: -1)   // [B,H,L,256]
            let newRight = concatenated([rightRotated, rightKeep], axis: -1) // [B,H,L,256]
            return concatenated([newLeft, newRight], axis: -1)               // [B,H,L,512]
        } else {
            // Standard RoPE for sliding attention
            return rope(x, offset: offset)
        }
    }

    /// - Parameter sharedCache: If non-nil, use this cache's K/V instead of computing new ones (KV sharing)
    func callAsFunction(
        _ x: MLXArray, maskMode: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCacheProtocol? = nil, sharedCache: KVCacheProtocol? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Always compute Q
        var q = qProj(x).reshaped(B, L, numHeads, layerHeadDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        let offset = cache?.sequenceLength ?? (sharedCache?.sequenceLength ?? 0)
        q = applyRoPE(q, offset: offset)

        let k: MLXArray
        let v: MLXArray

        if let shared = sharedCache, let sharedSnap = shared.snapshot() {
            // KV-shared layer: reuse K/V from the donor layer directly.
            // Do NOT compute new K/V — the donor's accumulated K/V IS the context.
            k = sharedSnap.keys
            v = sharedSnap.values
        } else {
            // Normal layer: compute K/V and write to own cache.
            // K-eq-V mode (26B-A4B full layers): no v_proj weight ships
            // in the checkpoint; values are derived from the same
            // projection as keys before k_norm / RoPE are applied. The
            // reference normalizes V with a scale-less RMSNorm, then
            // skips RoPE on V (positional info lives only on Q/K).
            var newK = kProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            var newV: MLXArray
            if useKEqV {
                newV = newK
            } else if let vProj {
                newV = vProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            } else {
                fatalError("Gemma4Attention: v_proj is nil but useKEqV is false (layer \(numKVHeads) kv heads)")
            }
            newK = kNorm(newK)
            let vVar = MLX.mean(newV * newV, axis: -1, keepDims: true)
            newV = newV * MLX.rsqrt(vVar + MLXArray(rmsNormEps).asType(newV.dtype))
            newK = applyRoPE(newK, offset: offset)
            if let cache {
                (k, v) = cache.update(keys: newK, values: newV)
            } else {
                k = newK
                v = newV
            }
        }

        // Attention with scale=1.0 (Gemma 4 uses unit scale)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: maskMode)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Gemma 4 MLP (GeGLU, variable width)

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        // KV-shared layers use double-wide MLP
        let inter: Int
        if config.useDoubleWideMlp && config.isKVShared(layerIdx: layerIdx) {
            inter = config.intermediateSize * 2
        } else {
            inter = config.intermediateSize
        }

        _gateProj = ModuleInfo(wrappedValue: Linear(dim, inter, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(wrappedValue: Linear(dim, inter, bias: false), key: "up_proj")
        _downProj = ModuleInfo(wrappedValue: Linear(inter, dim, bias: false), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Gemma 4 MoE Router (26B-A4B)

/// Router for the Gemma 4 26B-A4B MoE block. Forward:
///
///   1. Apply RMSNorm with weight = `scale * hidden_size^(-0.5)` (the
///      reference fuses the 1/sqrt(H) factor into the weight; reproduced
///      here so the math matches bit for bit).
///   2. Linear `proj`: hidden -> num_experts. The projection is
///      quantized in 26B-A4B; the standard `Linear` module is replaced
///      by `QuantizedLinear` during `quantize(...)` at load time.
///   3. Top-K via `argPartition` (kth = -topK, last topK columns), then
///      softmax over the topK logits.
///   4. Multiply by `per_expert_scale[topK_indices]` (Gemma-specific
///      learned per-expert multiplicative term applied to the topK
///      softmax output).
///
/// Returns the topK expert ids and the per-token topK routing weights.
/// Both shaped `[N, topK]` where `N = B*L` after flattening upstream.
class Gemma4MoERouter: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray
    @ParameterInfo(key: "per_expert_scale") var perExpertScale: MLXArray

    let topK: Int
    let numExperts: Int
    let hiddenSize: Int
    let rmsNormEps: Float
    /// `hidden_size^(-0.5)` cached as a host scalar; mixed into the RMS
    /// norm weight at forward time so the kernel sees a single fused
    /// scale vector instead of an extra elementwise multiply.
    let rootScale: Float

    init(_ config: Gemma4Config) {
        guard let numExperts = config.numExperts, numExperts > 0,
              let topK = config.topKExperts, topK > 0 else {
            fatalError(
                "Gemma4MoERouter requires `num_experts` and `top_k_experts` "
                + "in the checkpoint config; got "
                + "num_experts=\(String(describing: config.numExperts)) "
                + "top_k_experts=\(String(describing: config.topKExperts))")
        }
        self.numExperts = numExperts
        self.topK = topK
        self.hiddenSize = config.hiddenSize
        self.rmsNormEps = config.rmsNormEps
        self.rootScale = 1.0 / Float(config.hiddenSize).squareRoot()

        _proj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, numExperts, bias: false),
            key: "proj")
        _scale = ParameterInfo(
            wrappedValue: MLXArray.ones([config.hiddenSize]),
            key: "scale")
        _perExpertScale = ParameterInfo(
            wrappedValue: MLXArray.ones([numExperts]),
            key: "per_expert_scale")
    }

    /// - Parameter flat: Hidden states reshaped to `[N, H]` (caller
    ///   already flattened batch/sequence).
    /// - Returns: `(topKIdx [N, topK] Int32, topKWeight [N, topK])`.
    func callAsFunction(_ flat: MLXArray) -> (topKIdx: MLXArray, topKWeight: MLXArray) {
        // RMSNorm with weight = scale * 1/sqrt(H), eps = config.rms_norm_eps.
        let fusedWeight = scale * MLXArray(rootScale).asType(scale.dtype)
        let normed = MLXFast.rmsNorm(flat, weight: fusedWeight, eps: rmsNormEps)

        // Router logits: [N, E].
        let logits = proj(normed)

        // Top-K via argPartition (kth = -topK pivots so the last topK
        // entries along axis -1 are the top scores; their internal
        // ordering is undefined, matching the mlx-lm reference).
        // Cast to int32 -- mlx's gather_qmm requires the rhs index
        // tensor to be int32 specifically, not the uint32 the
        // arg-sort family returns by default.
        let partIdx = argPartition(logits, kth: numExperts - topK, axis: -1)
        let topKIdx = partIdx[0..., (numExperts - topK)..<numExperts]
            .asType(.int32)                                              // [N, topK]
        let topKLogits = takeAlong(logits, topKIdx, axis: -1)            // [N, topK]
        var topKWeight = softmax(topKLogits, axis: -1)                   // [N, topK]

        // Gemma-specific per-expert scale on the routed weights.
        // The take/gather kernels in MLX require int32 specifically;
        // topKIdx is already int32 from the cast above.
        let gathered = perExpertScale.take(topKIdx.flattened(), axis: 0)
            .reshaped(topKIdx.shape)                                     // [N, topK]
        topKWeight = topKWeight * gathered.asType(topKWeight.dtype)

        return (topKIdx, topKWeight)
    }
}

// MARK: - Gemma 4 MoE Experts (26B-A4B)

/// One stacked SwitchLinear inside the Gemma 4 SwitchGLU block. Holds
/// the quantized `[numExperts, outputDims, inputDims_packed]` weight
/// tensor plus per-expert scales and biases, and dispatches across the
/// chosen top-K experts in a single `gatherQuantizedMM` call instead
/// of per-expert matmuls. The scatter-dispatch path that landed in
/// the first MoE PR walked the experts in a Swift `for` loop, which
/// forced a per-layer host sync (the loop bounds came from a CPU read
/// of per-expert token counts); decoding 1 token per step paid the
/// sync once per layer, dominating the FFN math itself and putting
/// 26B-A4B's decode behind Ollama. The `gather_qmm` form keeps the
/// dispatch entirely on the GPU and matches `mlx_lm/models/
/// switch_layers.QuantizedSwitchLinear` bit for bit.
///
/// Parameter layout (matches mlx-community's packed format directly,
/// so the loader no longer has to unpack `experts.switch_glu.*`):
///   - `weight: [E, O, I/(32/bits)]` int-packed
///   - `scales: [E, O, I/groupSize]`
///   - `biases: [E, O, I/groupSize]`
class Gemma4QuantizedSwitchedLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int
    let groupSize: Int
    let bits: Int

    init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        groupSize: Int, bits: Int
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts
        self.groupSize = groupSize
        self.bits = bits

        // Pre-allocate the parameter tensors with the SAME shape the
        // mlx-community checkpoint ships so the loader's
        // `model.update(parameters:)` binds them via shape match. The
        // initial fill values are placeholders and are overwritten by
        // the checkpoint at load time.
        let packedIn = inputDims * bits / 32
        let groupsIn = inputDims / groupSize
        _weight = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, packedIn], dtype: .uint32),
            key: "weight")
        _scales = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, groupsIn], dtype: .bfloat16),
            key: "scales")
        _biases = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, groupsIn], dtype: .bfloat16),
            key: "biases")
    }

    /// Per-token expert dispatch.
    /// - Parameters:
    ///   - x: Input activations shaped so the last two dims feed
    ///        `gather_qmm`'s `[..., M, K]` matmul slot. The
    ///        SwitchGLU caller expands to `[..., 1, 1, I]` so each
    ///        token contributes one M=1 row per chosen expert.
    ///   - indices: `[..., K]` Int32 expert ids; flat indices into the
    ///        weight tensor's leading `numExperts` batch dim.
    ///   - sortedIndices: When true the caller has pre-sorted
    ///        `indices` by expert id so MLX's gather kernel can use
    ///        the faster sorted-indices path.
    func callAsFunction(
        _ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        return gatherQuantizedMM(
            x, weight,
            scales: scales, biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize, bits: bits, mode: .affine,
            sortedIndices: sortedIndices)
    }
}

/// SwitchGLU: three stacked SwitchLinears (`gate_proj`, `up_proj`,
/// `down_proj`) plus the GeGLU activation. Mirrors mlx-lm's
/// `switch_layers.SwitchGLU` (with `activation=GeGLU()`), so the
/// in-checkpoint key path `experts.switch_glu.{proj}.{weight,scales,
/// biases}` lines up with the module hierarchy directly. No
/// per-expert weight unpacking required.
///
/// Forward pass:
///   1. Reshape input from `[N, H]` to `[N, 1, 1, H]` so each row
///      participates in `topK` expert matmuls (one per chosen expert).
///   2. `gate_proj(x, indices)` and `up_proj(x, indices)` via
///      `gatherQuantizedMM` on the stacked weight tensors. Each call
///      lands a single device-side kernel that picks the right
///      expert per (token, slot) pair and contracts the M=1 row.
///      Output shape: `[N, topK, 1, moeIntermediate]`.
///   3. GeGLU activation: `gelu_approx(gate) * up`.
///   4. `down_proj` back to `[N, topK, 1, H]`.
///   5. Squeeze the M=1 axis to `[N, topK, H]`. The caller does the
///      topK weighted sum.
///
/// Note: mlx-lm's `SwitchGLU` adds an optional sort step at high
/// token counts (`indices.size >= 64`) that pre-arranges
/// `(token, expert)` assignments by expert id, so each expert's
/// kernel slice is contiguous in the gather op. The unsorted path
/// already beats Ollama by ~38% wall-time on 26B-A4B (see the
/// 2026-05-28 benchmark report) and the sort path requires figuring
/// out a gather_qmm shape contract that differs between mlx-swift
/// and the Python reference; left as a follow-up perf delta.
class Gemma4SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Gemma4QuantizedSwitchedLinear
    @ModuleInfo(key: "up_proj") var upProj: Gemma4QuantizedSwitchedLinear
    @ModuleInfo(key: "down_proj") var downProj: Gemma4QuantizedSwitchedLinear

    init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        groupSize: Int, bits: Int
    ) {
        _gateProj = ModuleInfo(
            wrappedValue: Gemma4QuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Gemma4QuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Gemma4QuantizedSwitchedLinear(
                inputDims: hiddenDims, outputDims: inputDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "down_proj")
    }

    /// - Parameters:
    ///   - x: `[N, H]` flattened token activations (caller has already
    ///        reshaped any batch / sequence axes into `N`).
    ///   - indices: `[N, topK]` Int32 expert ids per token, in router
    ///        score order.
    /// - Returns: `[N, topK, H]` per-expert outputs. The caller does
    ///   the topK weighted sum (Gemma's `Experts.__call__` shape).
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let N = x.dim(0)
        let H = x.dim(1)

        // Expand to [N, 1, 1, H] so each (token, slot) sees an M=1
        // row inside the gather. The [N, 1] outer batch combined with
        // the [N, topK] indices yields [N, topK, 1, H_out] for each
        // projection -- no Swift loop, no per-layer host sync, the
        // entire dispatch lands in one device kernel per projection.
        let xExp = x.reshaped(N, 1, 1, H)
        let idx = indices.asType(.int32)

        let xUp = upProj(xExp, indices: idx)
        let xGate = gateProj(xExp, indices: idx)
        let activated = geluApproximate(xGate) * xUp
        let out = downProj(activated, indices: idx)

        // out: [N, topK, 1, H_out] -- squeeze the M=1 inner axis.
        return out.squeezed(axis: -2)
    }
}

/// Holder module that owns the `switch_glu` sub-module under the
/// block's `experts.` key. Matches the in-checkpoint key path:
/// `experts.switch_glu.{gate_proj,up_proj,down_proj}.{weight,scales,
/// biases}`. Optional inside the block so non-MoE layers can leave it
/// nil; MoE layers allocate the SwitchGLU sized from the config.
class Gemma4MoEExpertsHolder: Module {
    @ModuleInfo(key: "switch_glu") var switchGLU: Gemma4SwitchGLU

    init(_ config: Gemma4Config, groupSize: Int, bits: Int) {
        let hidden = config.hiddenSize
        let inter = config.moeIntermediateSize ?? 704
        let numExperts = config.numExperts ?? 0
        _switchGLU = ModuleInfo(
            wrappedValue: Gemma4SwitchGLU(
                inputDims: hidden, hiddenDims: inter,
                numExperts: numExperts,
                groupSize: groupSize, bits: bits),
            key: "switch_glu")
    }
}

// MARK: - Gemma 4 Transformer Block (4 norms + optional PLE / MoE)

/// Per-layer block. Three modes share this class via Optional submodules:
///
///   1. **PLE-only** (e2b / e4b): dense GeGLU MLP with
///      `pre_feedforward_layernorm` + `post_feedforward_layernorm`,
///      followed by per-layer-input gating. No router/experts.
///   2. **MoE + no PLE** (26B-A4B): parallel dense MLP + sparse MoE
///      paths summed inside the 4-norm block, no PLE gating.
///   3. **MoE + PLE** (hypothetical future SKU): both branches active.
///
/// The shared 4-norm scaffolding plus the conditional `Optional`
/// submodules let one `Gemma4Block` cover all three; the per-SKU module
/// hierarchy lines up bit-exact with the checkpoint keys after the
/// `switch_glu` unpacker rewrites the stacked expert tensors.
class Gemma4Block: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFfnNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFfnNorm: RMSNorm
    // MoE-only norms (26B-A4B).
    @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFfnNorm2: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_1") var postFfnNorm1: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm_2") var postFfnNorm2: RMSNorm?
    // PLE-only modules (e2b / e4b).
    @ModuleInfo(key: "per_layer_input_gate") var pleGate: Linear?
    @ModuleInfo(key: "per_layer_projection") var pleProj: Linear?
    @ModuleInfo(key: "post_per_layer_input_norm") var pleNorm: RMSNorm?
    // MoE-only submodules. The experts holder owns a single
    // `Gemma4SwitchGLU` that does per-expert dispatch via
    // `gatherQuantizedMM` on stacked weights -- one kernel per
    // (gate, up, down) projection across all topK chosen experts,
    // with no per-layer Swift loop or host sync. The original
    // scatter-dispatch path that owned `[Gemma4MoEExpert]` (one
    // module per expert) walked the experts in Swift and read a
    // device-side count tensor into the host every layer; on decode
    // (1 token x topK slots, well below the per-expert FFN
    // threshold) that sync dominated the math and put 26B-A4B's
    // decode behind Ollama. The gather form keeps every layer
    // entirely on the GPU.
    @ModuleInfo(key: "router") var moeRouter: Gemma4MoERouter?
    @ModuleInfo(key: "experts") var moeExperts: Gemma4MoEExpertsHolder?
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    /// Cached config-driven flags so the forward avoids reading them
    /// off the Module property bag every token.
    let moeEnabled: Bool
    let pleEnabled: Bool
    let topK: Int
    let numExperts: Int

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        self.moeEnabled = config.enableMoeBlock
        self.pleEnabled = config.hasPerLayerInputs
        self.topK = config.topKExperts ?? 0
        self.numExperts = config.numExperts ?? 0

        _selfAttn = ModuleInfo(
            wrappedValue: Gemma4Attention(config, layerIdx: layerIdx),
            key: "self_attn")
        _mlp = ModuleInfo(
            wrappedValue: Gemma4MLP(config, layerIdx: layerIdx), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttnNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
        _preFfnNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "pre_feedforward_layernorm")
        _postFfnNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "post_feedforward_layernorm")

        if moeEnabled {
            _preFfnNorm2 = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
                key: "pre_feedforward_layernorm_2")
            _postFfnNorm1 = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
                key: "post_feedforward_layernorm_1")
            _postFfnNorm2 = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
                key: "post_feedforward_layernorm_2")
            _moeRouter = ModuleInfo(
                wrappedValue: Gemma4MoERouter(config), key: "router")
            // Quantization for the experts comes from the config's
            // top-level (groupSize, bits). On 26B-A4B that's the 4-bit
            // / 64-group default; the per-module overrides in the
            // checkpoint cover only `mlp.{proj}` (dense MLP) and
            // `router.proj`, never the experts. Falling back to (64,
            // 4) when no quantization block is present keeps the
            // module instantiable for unit-test fixtures.
            let groupSize = config.quantization?.groupSize ?? 64
            let bits = config.quantization?.bits ?? 4
            _moeExperts = ModuleInfo(
                wrappedValue: Gemma4MoEExpertsHolder(
                    config, groupSize: groupSize, bits: bits),
                key: "experts")
        } else {
            _preFfnNorm2 = ModuleInfo(wrappedValue: nil, key: "pre_feedforward_layernorm_2")
            _postFfnNorm1 = ModuleInfo(wrappedValue: nil, key: "post_feedforward_layernorm_1")
            _postFfnNorm2 = ModuleInfo(wrappedValue: nil, key: "post_feedforward_layernorm_2")
            _moeRouter = ModuleInfo(wrappedValue: nil, key: "router")
            _moeExperts = ModuleInfo(wrappedValue: nil, key: "experts")
        }

        if pleEnabled {
            let pleDim = config.hiddenSizePerLayerInput
            _pleGate = ModuleInfo(
                wrappedValue: Linear(dim, pleDim, bias: false),
                key: "per_layer_input_gate")
            _pleProj = ModuleInfo(
                wrappedValue: Linear(pleDim, dim, bias: false),
                key: "per_layer_projection")
            _pleNorm = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
                key: "post_per_layer_input_norm")
        } else {
            _pleGate = ModuleInfo(wrappedValue: nil, key: "per_layer_input_gate")
            _pleProj = ModuleInfo(wrappedValue: nil, key: "per_layer_projection")
            _pleNorm = ModuleInfo(wrappedValue: nil, key: "post_per_layer_input_norm")
        }

        // Learned per-layer scalar (ranges from ~0.018 to ~0.87)
        _layerScalar = ParameterInfo(
            wrappedValue: MLXArray([Float(1.0)]), key: "layer_scalar")
    }

    /// Dispatch the MoE branch by reading the router/experts modules
    /// directly so the parent's forward stays compact. `h` is the
    /// post-attention residual stream; this returns the summed
    /// dense+sparse FFN output ready to add back onto the residual.
    private func moeForward(_ h: MLXArray) -> MLXArray {
        guard let router = moeRouter,
              let experts = moeExperts,
              let preFfnNorm2 = preFfnNorm2,
              let postFfnNorm1 = postFfnNorm1,
              let postFfnNorm2 = postFfnNorm2 else {
            fatalError("Gemma4Block: moeEnabled but MoE submodules are nil")
        }

        // Dense branch (same gate/up/down weights as e2b/e4b path).
        var h1 = preFfnNorm(h)
        h1 = mlp(h1)
        h1 = postFfnNorm1(h1)

        // Sparse branch: router input is the pre-pre-norm hidden state.
        let B = h.dim(0); let L = h.dim(1); let H = h.dim(2)
        let N = B * L
        let routerFlat = h.reshaped(N, H)
        let (topKIdx, topKWeight) = router(routerFlat)   // [N, topK] each

        // Expert input goes through its own layernorm, then the
        // SwitchGLU runs the per-expert gate/up/down via gather_qmm
        // on stacked weights. Each call is a single device-side
        // matmul over the [N * topK] assignment list -- no Swift
        // loop, no per-layer host sync. The return is shaped
        // [N, topK, H], one row per (token, slot).
        let h2In = preFfnNorm2(h).reshaped(N, H)
        let perSlot = experts.switchGLU(h2In, indices: topKIdx)

        // Weight each slot by its router probability and sum the
        // topK contributions per token.
        let weighted = perSlot * topKWeight
            .reshaped(N, topK, 1).asType(perSlot.dtype)
        var h2 = weighted.sum(axis: 1).reshaped(B, L, H)
        h2 = postFfnNorm2(h2)

        return h1 + h2
    }

    func callAsFunction(
        _ x: MLXArray, maskMode: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCacheProtocol? = nil, sharedCache: KVCacheProtocol? = nil,
        perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // Attention with post-norm
        var h = x + postAttnNorm(
            selfAttn(
                inputLayernorm(x), maskMode: maskMode,
                cache: cache, sharedCache: sharedCache))

        // FFN: either parallel dense+sparse (MoE) or dense-only.
        let ffnOut: MLXArray
        if moeEnabled {
            ffnOut = moeForward(h)
        } else {
            ffnOut = mlp(preFfnNorm(h))
        }
        h = h + postFfnNorm(ffnOut)

        // PLE gating (e2b / e4b only)
        if pleEnabled, let ple = perLayerInput,
           let pleGate = pleGate, let pleProj = pleProj, let pleNorm = pleNorm {
            let gate = geluApproximate(pleGate(h))   // [B, L, pleDim]
            let gated = gate * ple
            let projected = pleProj(gated)           // [B, L, hidden]
            h = h + pleNorm(projected)
        }

        // Per-layer scaling (critical for output quality)
        return h * layerScalar
    }
}

// MARK: - Gemma 4 Full Model

class Gemma4TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    // PLE-only sub-modules (e2b/e4b). Absent on 26B-A4B which sets
    // `hidden_size_per_layer_input=0`.
    @ModuleInfo(key: "embed_tokens_per_layer") var embedPerLayer: Embedding?
    @ModuleInfo(key: "per_layer_model_projection") var perLayerProj: Linear?
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerNorm: RMSNorm?
    @ModuleInfo(key: "layers") var layers: [Gemma4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: Gemma4Config

    // Pre-computed scalar constants (avoids per-call allocation)
    let embedScale: MLXArray
    let pleScale: MLXArray
    let projScale: MLXArray
    let combineScale: MLXArray

    // KV sharing: donor layer indices (computed once at init)
    let lastSlidingDonor: Int
    let lastFullDonor: Int
    let hasPLE: Bool
    let kvSharingEnabled: Bool

    init(_ config: Gemma4Config) {
        self.config = config
        let dim = config.hiddenSize
        let pleDim = config.hiddenSizePerLayerInput
        let pleTotal = config.numHiddenLayers * pleDim
        self.hasPLE = config.hasPerLayerInputs
        self.kvSharingEnabled = config.numKVSharedLayers > 0

        // Pre-compute BF16 scalars. `pleScale` is unused when PLE is
        // disabled but `MLXArray(0)` is cheap to allocate; we still
        // honor the same hidden_size^0.5 / hidden_size^-0.5 factors so
        // the e2b/e4b path is unchanged.
        self.embedScale = MLXArray(Float(dim).squareRoot()).asType(.bfloat16)
        self.pleScale = MLXArray(
            hasPLE ? Float(pleDim).squareRoot() : 1.0
        ).asType(.bfloat16)
        self.projScale = MLXArray(1.0 / Float(dim).squareRoot()).asType(.bfloat16)
        self.combineScale = MLXArray(Float(0.7071067811865476)).asType(.bfloat16)

        // Pre-compute KV sharing donor indices. `firstKVSharedLayer ==
        // numHiddenLayers` (i.e. `num_kv_shared_layers == 0`) leaves
        // the donor indices at 0 unused; the forward gates the sharing
        // path on `kvSharingEnabled` so 26B-A4B (no KV sharing) takes
        // the dense per-layer KV path.
        let firstShared = config.firstKVSharedLayer
        var slidingDonor = 0
        var fullDonor = 0
        for i in 0 ..< firstShared {
            if config.isFullAttention(layerIdx: i) {
                fullDonor = i
            } else {
                slidingDonor = i
            }
        }
        self.lastSlidingDonor = slidingDonor
        self.lastFullDonor = fullDonor

        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: dim),
            key: "embed_tokens")
        if hasPLE {
            _embedPerLayer = ModuleInfo(
                wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: pleTotal),
                key: "embed_tokens_per_layer")
            _perLayerProj = ModuleInfo(
                wrappedValue: Linear(dim, pleTotal, bias: false),
                key: "per_layer_model_projection")
            _perLayerNorm = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: pleDim, eps: config.rmsNormEps),
                key: "per_layer_projection_norm")
        } else {
            _embedPerLayer = ModuleInfo(wrappedValue: nil, key: "embed_tokens_per_layer")
            _perLayerProj = ModuleInfo(wrappedValue: nil, key: "per_layer_model_projection")
            _perLayerNorm = ModuleInfo(wrappedValue: nil, key: "per_layer_projection_norm")
        }
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { i in Gemma4Block(config, layerIdx: i) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "norm")
    }

    /// Forward pass with optional multimodal embedding injection.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, L]
    ///   - caches: Per-layer KV caches
    ///   - imageEmbeddings: Vision encoder output to replace image_token_id positions [1, numImageTokens, hiddenSize]
    ///   - audioEmbeddings: Audio encoder output to replace audio_token_id positions [1, numAudioTokens, hiddenSize]
    ///   - imageTokenId: Token ID for image placeholder (default 258880)
    ///   - audioTokenId: Token ID for audio placeholder (default 258881)
    func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        imageEmbeddings: MLXArray? = nil, audioEmbeddings: MLXArray? = nil,
        imageTokenId: Int = 258880, audioTokenId: Int = 258881
    ) -> MLXArray {
        let B = tokens.dim(0)
        let L = tokens.dim(1)
        let numLayers = config.numHiddenLayers
        let pleDim = config.hiddenSizePerLayerInput

        // Main embeddings (scaled by sqrt(hidden_size))
        var h = embedTokens(tokens) * embedScale

        // Inject multimodal embeddings at placeholder token positions
        if let imgEmb = imageEmbeddings {
            h = injectEmbeddings(h, tokens: tokens, embeddings: imgEmb, tokenId: imageTokenId)
        }
        if let audEmb = audioEmbeddings {
            h = injectEmbeddings(h, tokens: tokens, embeddings: audEmb, tokenId: audioTokenId)
        }

        // PLE: compute per-layer inputs (e2b/e4b only).
        // For multimodal: zero out image/audio token IDs so PLE only computes
        // for text tokens. Image/audio positions get zero PLE contribution.
        var combinedPLE: MLXArray? = nil
        if hasPLE, let embedPerLayer = embedPerLayer,
           let perLayerProj = perLayerProj, let perLayerNorm = perLayerNorm {
            let pleTokens: MLXArray
            if imageEmbeddings != nil || audioEmbeddings != nil {
                let imgMask = tokens .== MLXArray(Int32(imageTokenId))
                let audMask = tokens .== MLXArray(Int32(audioTokenId))
                let mmMask = (imgMask.asType(.int32) + audMask.asType(.int32)) .> MLXArray(Int32(0))
                pleTokens = MLX.where(mmMask, MLXArray(Int32(0)), tokens)
            } else {
                pleTokens = tokens
            }
            let pleEmbed = embedPerLayer(pleTokens) * pleScale
            let projection = perLayerProj(h) * projScale
            let projNormed = perLayerNorm(projection.reshaped(B * L * numLayers, pleDim))
                .reshaped(B, L, numLayers * pleDim)
            combinedPLE = (projNormed + pleEmbed) * combineScale
        }

        // Evaluate embedding and PLE computation before entering layer loop.
        // This flushes the graph so each layer starts with materialized inputs,
        // preventing quadratic graph growth during prefill.
        if L > 1 {
            if let combinedPLE = combinedPLE {
                MLX.eval(h, combinedPLE)
            } else {
                MLX.eval(h)
            }
        }

        // Causal mask. Same shape rules as the dense models: empty cache
        // gets a square (L, L) mask; non-empty cache + multi-token forward
        // (spec-decode verify, partial prefix resume) gets the shifted
        // (L, cacheLen + L) mask. `caches.first` may be a quantized cache
        // when int8 KV is active - its `sequenceLength` semantics match
        // the fp16 path.
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let mask = createCachedCausalMask(
            newLen: L, cacheLen: cacheLen, dtype: .bfloat16)
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask != nil ? .array(mask!) : .none

        // KV sharing donor indices (pre-computed at init)
        let firstShared = config.firstKVSharedLayer

        // Forward through layers. Shared layers pass the donor's cache
        // as sharedCache so attention reuses the donor's accumulated
        // K/V (e2b/e4b). 26B-A4B has `num_kv_shared_layers=0` so this
        // branch is never taken.
        for (i, layer) in layers.enumerated() {
            let pleSlice: MLXArray? = combinedPLE.map { ple in
                ple[0..., 0..., (i * pleDim) ..< ((i + 1) * pleDim)]
            }

            if kvSharingEnabled, i >= firstShared, let caches {
                let donorIdx = config.isFullAttention(layerIdx: i) ? lastFullDonor : lastSlidingDonor
                h = layer(h, maskMode: maskMode, cache: caches[i],
                         sharedCache: caches[donorIdx], perLayerInput: pleSlice)
            } else {
                h = layer(h, maskMode: maskMode, cache: caches?[i], perLayerInput: pleSlice)
            }

            // Evaluate every 5 layers during prefill to bound graph size
            if L > 1 && (i + 1) % 5 == 0 {
                MLX.eval(h)
            }
        }

        return norm(h)
    }
}

// MARK: - Gemma4ForCausalLM

public class Gemma4ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Gemma4TextModel

    public let config: Gemma4Config

    public init(_ config: Gemma4Config) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Gemma4TextModel(config), key: "model")
    }

    /// Output projection: reuse embed_tokens as the LM head (tied weights).
    /// This matches the Python implementation which uses `embed_tokens.as_linear()`.
    private func lmHead(_ hidden: MLXArray) -> MLXArray {
        model.embedTokens.asLinear(hidden)
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCacheProtocol]? = nil) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    /// `lastTokenOnly` slices hidden to the last position before
    /// the tied head and softcap. See `LlamaForCausalLM` for the
    /// rationale; the softcap is per-element so slicing commutes
    /// with it.
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        lastTokenOnly: Bool
    ) -> MLXArray {
        var hidden = model(tokens, caches: caches)
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        let logits = lmHead(hidden)
        let cap = config.finalLogitSoftcapping
        return MLX.tanh(logits / cap) * cap
    }

    /// Forward pass with multimodal embedding injection.
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        imageEmbeddings: MLXArray? = nil, audioEmbeddings: MLXArray? = nil,
        imageTokenId: Int = 258880, audioTokenId: Int = 258881,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        var hidden = model(
            tokens, caches: caches,
            imageEmbeddings: imageEmbeddings, audioEmbeddings: audioEmbeddings,
            imageTokenId: imageTokenId, audioTokenId: audioTokenId)
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        let logits = lmHead(hidden)
        let cap = config.finalLogitSoftcapping
        return MLX.tanh(logits / cap) * cap
    }
}

// MARK: - Multimodal Embedding Injection

/// Replace token embeddings at placeholder positions with encoder outputs
/// using masked_scatter semantics.
///
/// The first True mask position gets source[0], the second True gets source[1], etc.
/// This matches PyTorch's masked_scatter / the Gemma4 reference implementation.
func maskedScatter(
    _ inputTensor: MLXArray, mask: MLXArray, source: MLXArray
) -> MLXArray {
    // mask: [B, L] or [B, L, D] (expanded), source: [1, N, D]
    // Flatten mask to [B*L] or [B*L*D]
    let maskFlat = mask.flattened().asType(.int32)
    // cumsum gives sequential indices: first True → 0, second True → 1, etc.
    let indices = MLX.cumsum(maskFlat, axis: 0) - 1
    // Flatten source and index with modular arithmetic
    let sourceFlat = source.flattened()
    let aligned = sourceFlat.take(indices % MLXArray(Int32(sourceFlat.size)), axis: 0)
    // Where mask is True use aligned value, else keep original
    let result = MLX.where(maskFlat, aligned, inputTensor.flattened())
    return result.reshaped(inputTensor.shape)
}

/// Replace token embeddings at placeholder positions with encoder outputs.
///
/// Uses masked_scatter: replacement[0] goes to first mask position,
/// replacement[1] to second, etc.
private func injectEmbeddings(
    _ embeddings: MLXArray, tokens: MLXArray,
    embeddings replacement: MLXArray, tokenId: Int
) -> MLXArray {
    // tokens: [B, L], embeddings: [B, L, D], replacement: [1, N, D]
    let mask = tokens .== MLXArray(Int32(tokenId))  // [B, L]
    let maskExpanded = expandedDimensions(mask, axis: -1)
    let mask3D = MLX.broadcast(maskExpanded, to: embeddings.shape)
    return maskedScatter(embeddings, mask: mask3D, source: replacement)
}

// MARK: - MultimodalEmbedder

/// Projects soft tokens from vision/audio encoders into language model space.
///
/// Pipeline: RMSNormNoScale -> Linear projection (4-bit quantized)
///
/// Weight keys: `embed_vision.embedding_projection.*` or `embed_audio.embedding_projection.*`
public class MultimodalEmbedder: Module {
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    let norm: VisionRMSNormNoScale

    public init(embeddingDim: Int, textHiddenSize: Int, eps: Float = 1e-6) {
        _embeddingProjection = ModuleInfo(
            wrappedValue: Linear(embeddingDim, textHiddenSize, bias: false),
            key: "embedding_projection")
        self.norm = VisionRMSNormNoScale(eps: eps)
    }

    public func callAsFunction(_ inputsEmbeds: MLXArray) -> MLXArray {
        let normed = norm(inputsEmbeds)
        return embeddingProjection(normed)
    }
}

// MARK: - Gemma4MultimodalModel

/// Top-level multimodal model wrapper matching the safetensors key structure.
///
/// Weight key hierarchy:
///   `language_model.*` — text model (Gemma4ForCausalLM)
///   `vision_tower.*` — SigLIP2 vision encoder
///   `embed_vision.*` — vision → language projection
///   `audio_tower.*` — native USM Conformer audio encoder
///   `embed_audio.*` — audio → language projection
public class Gemma4MultimodalModel: Module {
    @ModuleInfo(key: "language_model") var languageModel: Gemma4ForCausalLM
    @ModuleInfo(key: "vision_tower") var visionTower: VisionEncoder
    @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder
    @ModuleInfo(key: "audio_tower") var audioTower: AudioEncoder
    @ModuleInfo(key: "embed_audio") var embedAudio: MultimodalEmbedder

    public let config: Gemma4Config
    public let imageTokenId: Int
    public let audioTokenId: Int

    // Per-instance cache: invalidated automatically when the model is unloaded.
    public let visionCache: VisionEncoderCache = VisionEncoderCache()

    public init(_ config: Gemma4Config, imageTokenId: Int = 258880,
                audioTokenId: Int = 258881,
                audioConfig: AudioConfig = AudioConfig()) {
        self.config = config
        self.imageTokenId = imageTokenId
        self.audioTokenId = audioTokenId

        _languageModel = ModuleInfo(
            wrappedValue: Gemma4ForCausalLM(config), key: "language_model")

        _audioTower = ModuleInfo(
            wrappedValue: AudioEncoder(audioConfig), key: "audio_tower")
        _embedAudio = ModuleInfo(
            wrappedValue: MultimodalEmbedder(
                embeddingDim: audioConfig.outputProjDims,
                textHiddenSize: config.hiddenSize,
                eps: audioConfig.rmsNormEps),
            key: "embed_audio")

        // Vision tower shapes come from the checkpoint's `vision_config`
        // when present. Falling back to the e2b/e4b SigLIP-base defaults
        // keeps a `visionConfig: nil` checkpoint loadable (legacy
        // text-only path with a stub vision tower); 26B-A4B ships a
        // larger tower (hidden 1152, 16 heads, head_dim 72) that the
        // prior hardcoded constructor crashed loading (#76).
        let vc = config.visionConfig ?? Gemma4VisionConfig.defaults
        _visionTower = ModuleInfo(
            wrappedValue: VisionEncoder(
                hiddenSize: vc.hiddenSize, intermediateSize: vc.intermediateSize,
                numLayers: vc.numHiddenLayers, numHeads: vc.numAttentionHeads,
                numKVHeads: vc.numKeyValueHeads, headDim: vc.headDim,
                patchSize: vc.patchSize, poolingKernelSize: vc.poolingKernelSize,
                defaultOutputLength: vc.defaultOutputLength,
                positionEmbeddingSize: vc.positionEmbeddingSize,
                ropeTheta: vc.ropeTheta, eps: vc.rmsNormEps),
            key: "vision_tower")

        // Vision -> language projection input dim must match the tower's
        // hidden_size, not the hardcoded 768.
        _embedVision = ModuleInfo(
            wrappedValue: MultimodalEmbedder(
                embeddingDim: vc.hiddenSize, textHiddenSize: config.hiddenSize,
                eps: vc.rmsNormEps),
            key: "embed_vision")
    }

    /// Text-only forward pass.
    public func callAsFunction(_ tokens: MLXArray, caches: [KVCacheProtocol]? = nil) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    /// Text-only forward pass with the prefill `lastTokenOnly`
    /// shortcut. Mirrors the inner Gemma4ForCausalLM overload.
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        lastTokenOnly: Bool
    ) -> MLXArray {
        languageModel(tokens, caches: caches, lastTokenOnly: lastTokenOnly)
    }

    /// Multimodal forward pass with image pixel values.
    ///
    /// If `imageBytesHash` is non-nil it is treated as a stable identifier for the
    /// raw image bytes the caller derived `pixelValues` from; the encoder output
    /// for that key is cached and reused on subsequent calls. Pass `nil` to
    /// bypass the cache (e.g. for multi-image batches or any non-image input).
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        pixelValues: MLXArray? = nil,
        imageBytesHash: String? = nil
    ) -> MLXArray {
        callAsFunction(tokens, caches: caches, pixelValues: pixelValues,
                       audioMel: nil, audioValidMask: nil,
                       mediaHash: imageBytesHash)
    }

    /// Multimodal forward supporting image and/or native audio. `audioMel`
    /// is `[1,T,128]` log-mel from `AudioPreprocessor`, `audioValidMask` is
    /// `[1,T]` bool (true = real audio). When both image and audio are
    /// present they run in one native Swift generation pass (no bridge).
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil,
        pixelValues: MLXArray? = nil,
        audioMel: MLXArray? = nil,
        audioValidMask: MLXArray? = nil,
        mediaHash: String? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        if pixelValues == nil && audioMel == nil {
            return languageModel(
                tokens, caches: caches, lastTokenOnly: lastTokenOnly)
        }

        var imageEmbeddings: MLXArray? = nil
        if let pixels = pixelValues {
            if let key = mediaHash, let cached = visionCache.lookup(key) {
                imageEmbeddings = cached
            } else {
                let computed = embedVision(visionTower(pixels))
                // Only cache when this is a pure-image request (the hash is
                // the image-bytes key); combined audio requests pass nil.
                if let key = mediaHash, audioMel == nil {
                    MLX.eval(computed)
                    visionCache.store(key, value: computed)
                }
                imageEmbeddings = computed
            }
        }

        var audioEmbeddings: MLXArray? = nil
        if let mel = audioMel, let mask = audioValidMask {
            let (enc, _) = audioTower(mel, validMask: mask)
            audioEmbeddings = embedAudio(enc)
        }

        return languageModel(
            tokens, caches: caches,
            imageEmbeddings: imageEmbeddings,
            audioEmbeddings: audioEmbeddings,
            imageTokenId: imageTokenId,
            audioTokenId: audioTokenId,
            lastTokenOnly: lastTokenOnly)
    }

    /// Number of transformer layers (for KV cache creation).
    public var numLayers: Int { config.numHiddenLayers }
}
