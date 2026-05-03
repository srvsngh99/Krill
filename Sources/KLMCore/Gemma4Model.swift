import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Gemma 4 Config

/// Configuration for Gemma 4 model family.
/// Key differences from Gemma 2: hybrid sliding/global attention, partial RoPE,
/// GELU activation (not GeGLU), dual KV head counts.
public struct Gemma4Config: ModelConfig, Decodable, Sendable {
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

    // Gemma 4 specific
    public let slidingWindowSize: Int
    public let numGlobalKVHeads: Int
    public let partialRotaryFactor: Float
    public let globalRopeTheta: Float

    // Gemma 4 has an explicit head_dim that may differ from hidden_size/num_heads
    public let _explicitHeadDim: Int?
    public var headDim: Int { _explicitHeadDim ?? (hiddenSize / numAttentionHeads) }

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
        case quantizationConfig = "quantization_config"
        case slidingWindowSize = "sliding_window_size"
        case numGlobalKVHeads = "num_global_key_value_heads"
        case partialRotaryFactor = "partial_rotary_factor"
        case globalRopeTheta = "global_rope_theta"
        case textConfig = "text_config"
        case explicitHeadDim = "head_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        // Gemma 4 config is nested: top-level has text_config with decoder params
        // Try nested first, fall back to flat
        if let textContainer = try? c.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {
            hiddenSize = try textContainer.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try textContainer.decode(Int.self, forKey: .intermediateSize)
            numAttentionHeads = try textContainer.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try textContainer.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
                ?? numAttentionHeads
            numHiddenLayers = try textContainer.decode(Int.self, forKey: .numHiddenLayers)
            vocabSize = try textContainer.decode(Int.self, forKey: .vocabSize)
            rmsNormEps = try textContainer.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try textContainer.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
            maxPositionEmbeddings = try textContainer.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
            slidingWindowSize = try textContainer.decodeIfPresent(Int.self, forKey: .slidingWindowSize) ?? 1024
            numGlobalKVHeads = try textContainer.decodeIfPresent(Int.self, forKey: .numGlobalKVHeads) ?? 4
            partialRotaryFactor = try textContainer.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            globalRopeTheta = try textContainer.decodeIfPresent(Float.self, forKey: .globalRopeTheta) ?? 1_000_000.0
            _explicitHeadDim = try textContainer.decodeIfPresent(Int.self, forKey: .explicitHeadDim)
        } else {
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
                ?? numAttentionHeads
            numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
            vocabSize = try c.decode(Int.self, forKey: .vocabSize)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
            maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
            slidingWindowSize = try c.decodeIfPresent(Int.self, forKey: .slidingWindowSize) ?? 1024
            numGlobalKVHeads = try c.decodeIfPresent(Int.self, forKey: .numGlobalKVHeads) ?? 4
            partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            globalRopeTheta = try c.decodeIfPresent(Float.self, forKey: .globalRopeTheta) ?? 1_000_000.0
            _explicitHeadDim = try c.decodeIfPresent(Int.self, forKey: .explicitHeadDim)
        }
        // Quantization can be at top level or in text_config
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
            ?? (try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig))
    }
}

// MARK: - Gemma 4 Attention (hybrid sliding + global)

class Gemma4Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let isGlobal: Bool
    let ropeDim: Int

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        // Gemma 4 alternates: 5 sliding-window layers then 1 global layer
        self.isGlobal = (layerIdx % 6 == 5)
        self.numKVHeads = isGlobal ? config.numGlobalKVHeads : config.numKeyValueHeads

        // Partial RoPE: only rotate a fraction of head dimensions
        self.ropeDim = Int(Float(config.headDim) * config.partialRotaryFactor)

        let ropeTheta = isGlobal ? config.globalRopeTheta : config.ropeTheta
        self.rope = RoPE(dimensions: ropeDim, traditional: false, base: ropeTheta)

        // Note: Q/K/V sizes use headDim (explicit, may differ from hidden_size/num_heads)
        let qSize = numHeads * headDim
        let kvSize = numKVHeads * headDim
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, qSize, bias: false), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, kvSize, bias: false), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, kvSize, bias: false), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(qSize, dim, bias: false), key: "o_proj")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Apply partial RoPE (only to first ropeDim dimensions)
        let offset = cache?.sequenceLength ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Gemma 4 MLP (GELU, not GeGLU)

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4Config) {
        _gateProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: false),
            key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: false),
            key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: false),
            key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(gelu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Gemma 4 Transformer Block

class Gemma4TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: Gemma4Config, layerIdx: Int) {
        _selfAttn = ModuleInfo(
            wrappedValue: Gemma4Attention(config, layerIdx: layerIdx), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: Gemma4MLP(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - Gemma 4 Full Model

class Gemma4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Gemma4TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let hiddenSize: Int

    init(_ config: Gemma4Config) {
        self.hiddenSize = config.hiddenSize
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { i in
                Gemma4TransformerBlock(config, layerIdx: i)
            },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        // Gemma scales embeddings by sqrt(hidden_size)
        var x = embedTokens(tokens) * MLXArray(Float(hiddenSize).squareRoot())
        let seqLen = x.dim(1)
        let mask: MLXArray? = seqLen > 1 ? createAdditiveCausalMask(seqLen) : nil

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: caches?[i])
        }
        return norm(x)
    }
}

public class Gemma4ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Gemma4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4Config

    public init(_ config: Gemma4Config) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Gemma4ModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        lmHead(model(tokens, caches: caches))
    }
}
