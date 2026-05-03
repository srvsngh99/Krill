import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Gemma 4 Config

/// Configuration for Gemma 4 model family (E2B, E4B, 12B, 27B, 31B).
/// Complex architecture: PLE, dual head_dim, KV sharing, 4-norm blocks.
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
    public let slidingWindowPattern: Int  // every Nth layer is full attention
    public let hiddenSizePerLayerInput: Int  // PLE dimension (256)
    public let numKVSharedLayers: Int    // layers sharing KV from earlier layers
    public let useDoubleWideMlp: Bool
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool
    public let quantization: QuantizationConfig?

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
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
        case quantizationConfig = "quantization_config"
        case textConfig = "text_config"
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
        hiddenSizePerLayerInput = try tc.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        numKVSharedLayers = try tc.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 20
        useDoubleWideMlp = try tc.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        finalLogitSoftcapping = try tc.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        tieWordEmbeddings = try tc.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Quantization at top level
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
            ?? (try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig))
    }

    /// Whether a layer uses full attention (vs sliding window).
    public func isFullAttention(layerIdx: Int) -> Bool {
        (layerIdx + 1) % slidingWindowPattern == 0
    }

    /// Whether a layer shares KV from an earlier layer.
    public func isKVShared(layerIdx: Int) -> Bool {
        layerIdx >= (numHiddenLayers - numKVSharedLayers)
    }

    /// First layer index that shares KV.
    public var firstKVSharedLayer: Int { numHiddenLayers - numKVSharedLayers }

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

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE
    let rmsNormEps: Float

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.isFullAttn = config.isFullAttention(layerIdx: layerIdx)
        self.layerHeadDim = isFullAttn ? config.globalHeadDim : config.headDim
        self.rmsNormEps = config.rmsNormEps

        let qSize = numHeads * layerHeadDim
        let kvSize = numKVHeads * layerHeadDim

        _qProj = ModuleInfo(wrappedValue: Linear(dim, qSize, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(dim, kvSize, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(dim, kvSize, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(qSize, dim, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: layerHeadDim, eps: config.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: layerHeadDim, eps: config.rmsNormEps), key: "k_norm")

        // RoPE: full dims for sliding, partial (25%) for full attention
        let ropeDim: Int
        let ropeBase: Float
        if isFullAttn {
            ropeDim = layerHeadDim / 4  // partial_rotary_factor = 0.25
            ropeBase = 1_000_000.0
        } else {
            ropeDim = layerHeadDim
            ropeBase = 10_000.0
        }
        self.rope = RoPE(dimensions: ropeDim, traditional: false, base: ropeBase)
    }

    /// - Parameter sharedCache: If non-nil, use this cache's K/V instead of computing new ones (KV sharing)
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil,
        cache: KVCache? = nil, sharedCache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Always compute Q
        var q = qProj(x).reshaped(B, L, numHeads, layerHeadDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        let offset = cache?.sequenceLength ?? (sharedCache?.sequenceLength ?? 0)
        q = rope(q, offset: offset)

        let k: MLXArray
        let v: MLXArray

        if let sharedCache {
            // KV-shared layer: read K/V from donor cache (no new K/V computed)
            // The donor cache already has accumulated K/V from its own layer
            // We just need to read it for attention
            var newK = kProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            var newV = vProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            newK = kNorm(newK)
            let vVar = MLX.mean(newV * newV, axis: -1, keepDims: true)
            newV = newV * MLX.rsqrt(vVar + MLXArray(rmsNormEps))
            newK = rope(newK, offset: offset)
            // Use own K/V but don't write to cache (shared layers still compute K/V for their own input)
            if let cache {
                (k, v) = cache.update(keys: newK, values: newV)
            } else {
                k = newK
                v = newV
            }
        } else {
            // Normal layer: compute K/V and write to own cache
            var newK = kProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            var newV = vProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            newK = kNorm(newK)
            let vVar = MLX.mean(newV * newV, axis: -1, keepDims: true)
            newV = newV * MLX.rsqrt(vVar + MLXArray(rmsNormEps))
            newK = rope(newK, offset: offset)
            if let cache {
                (k, v) = cache.update(keys: newK, values: newV)
            } else {
                k = newK
                v = newV
            }
        }

        // Attention with scale=1.0 (Gemma 4 uses unit scale)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask)

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
        downProj(gelu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Gemma 4 Transformer Block (4 norms + PLE gating)

class Gemma4Block: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFfnNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFfnNorm: RMSNorm
    @ModuleInfo(key: "per_layer_input_gate") var pleGate: Linear
    @ModuleInfo(key: "per_layer_projection") var pleProj: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var pleNorm: RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        let pleDim = config.hiddenSizePerLayerInput

        _selfAttn = ModuleInfo(wrappedValue: Gemma4Attention(config, layerIdx: layerIdx), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: Gemma4MLP(config, layerIdx: layerIdx), key: "mlp")
        _inputLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "input_layernorm")
        _postAttnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "post_attention_layernorm")
        _preFfnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "pre_feedforward_layernorm")
        _postFfnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "post_feedforward_layernorm")
        _pleGate = ModuleInfo(wrappedValue: Linear(dim, pleDim, bias: false), key: "per_layer_input_gate")
        _pleProj = ModuleInfo(wrappedValue: Linear(pleDim, dim, bias: false), key: "per_layer_projection")
        _pleNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "post_per_layer_input_norm")
        // Learned per-layer scalar (ranges from ~0.018 to ~0.87)
        _layerScalar = ModuleInfo(wrappedValue: MLXArray([Float(1.0)]), key: "layer_scalar")
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
        sharedCache: KVCache? = nil, perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // Attention with post-norm
        var h = x + postAttnNorm(selfAttn(inputLayernorm(x), mask: mask, cache: cache, sharedCache: sharedCache))

        // FFN with pre+post norm
        h = h + postFfnNorm(mlp(preFfnNorm(h)))

        // PLE gating
        if let ple = perLayerInput {
            let gate = gelu(pleGate(h))  // [B, L, 256]
            let gated = gate * ple        // element-wise with PLE slice
            let projected = pleProj(gated) // [B, L, 1536]
            h = h + pleNorm(projected)
        }

        // Per-layer scaling (critical for output quality)
        return h * layerScalar
    }
}

// MARK: - Gemma 4 Full Model

class Gemma4TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerProj: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Gemma4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: Gemma4Config

    init(_ config: Gemma4Config) {
        self.config = config
        let dim = config.hiddenSize
        let pleDim = config.hiddenSizePerLayerInput
        let pleTotal = config.numHiddenLayers * pleDim

        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: dim),
            key: "embed_tokens")
        _embedPerLayer = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: pleTotal),
            key: "embed_tokens_per_layer")
        _perLayerProj = ModuleInfo(
            wrappedValue: Linear(dim, pleTotal, bias: false),
            key: "per_layer_model_projection")
        _perLayerNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: pleDim, eps: config.rmsNormEps),
            key: "per_layer_projection_norm")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { i in Gemma4Block(config, layerIdx: i) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        let B = tokens.dim(0)
        let L = tokens.dim(1)
        let numLayers = config.numHiddenLayers
        let pleDim = config.hiddenSizePerLayerInput

        // Main embeddings (scaled by sqrt(hidden_size))
        var h = embedTokens(tokens) * MLXArray(Float(config.hiddenSize).squareRoot())

        // PLE: compute per-layer inputs
        let pleEmbed = embedPerLayer(tokens) * MLXArray(Float(pleDim).squareRoot())

        let projection = perLayerProj(h) * MLXArray(1.0 / Float(config.hiddenSize).squareRoot())
        let projNormed = perLayerNorm(projection.reshaped(B * L * numLayers, pleDim))
            .reshaped(B, L, numLayers * pleDim)

        let combinedPLE = (projNormed + pleEmbed) * MLXArray(Float(0.7071067811865476))

        // Causal mask
        let mask: MLXArray? = L > 1 ? createAdditiveCausalMask(L) : nil

        // KV sharing: find the donor cache indices for shared layers.
        // Layers 0..<firstKVSharedLayer have their own caches.
        // Layers firstKVSharedLayer..< numLayers reuse KV from earlier donor layers.
        // Donor for sliding: last non-shared sliding layer
        // Donor for full: last non-shared full layer
        let firstShared = config.firstKVSharedLayer
        var lastSlidingDonor = 0
        var lastFullDonor = 0
        for i in 0 ..< firstShared {
            if config.isFullAttention(layerIdx: i) {
                lastFullDonor = i
            } else {
                lastSlidingDonor = i
            }
        }

        // Forward through layers
        for (i, layer) in layers.enumerated() {
            let pleSlice = combinedPLE[0..., 0..., (i * pleDim) ..< ((i + 1) * pleDim)]

            if i < firstShared {
                // Non-shared layer: has its own cache
                h = layer(h, mask: mask, cache: caches?[i], perLayerInput: pleSlice)
            } else {
                // KV-shared layer: still has its own cache for accumulation,
                // but conceptually shares the KV computation pattern
                h = layer(h, mask: mask, cache: caches?[i], perLayerInput: pleSlice)
            }
        }

        return norm(h)
    }
}

// MARK: - Gemma4ForCausalLM

public class Gemma4ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Gemma4TextModel
    // Separate lm_head for logit projection (loaded from embed_tokens weights)
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4Config

    public init(_ config: Gemma4Config) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Gemma4TextModel(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        let hidden = model(tokens, caches: caches)
        let logits = lmHead(hidden)
        // Logit soft-capping
        let cap = config.finalLogitSoftcapping
        return MLX.tanh(logits / cap) * cap
    }
}
