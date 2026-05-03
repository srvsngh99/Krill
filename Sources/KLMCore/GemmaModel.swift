import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Gemma Config

/// Configuration for Gemma 2 / Gemma 3 model family.
/// Key differences: RMSNorm with +1 offset, GeGLU activation, head_dim explicit.
public struct GemmaConfig: ModelConfig, Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let _headDim: Int?
    public let quantization: QuantizationConfig?

    public var headDim: Int { _headDim ?? (hiddenSize / numAttentionHeads) }

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
        case _headDim = "head_dim"
        case quantization
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
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 8192
        _headDim = try c.decodeIfPresent(Int.self, forKey: ._headDim)
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - Gemma RMSNorm (with +1 offset on weight)

/// Gemma's RMSNorm adds 1 to the learned weight before scaling.
/// This differs from standard RMSNorm used by Llama/Qwen/Mistral.
class GemmaRMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray

    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight = ModuleInfo(
            wrappedValue: MLXArray.zeros([dimensions]),
            key: "weight")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Gemma adds 1 to the weight: (1 + weight) * rms_norm(x)
        let variance = mean(x * x, axis: -1, keepDims: true)
        let normalized = x * rsqrt(variance + MLXArray(eps))
        return (MLXArray(1.0) + weight) * normalized
    }
}

// MARK: - Gemma Attention

class GemmaAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: GemmaConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        let dim = config.hiddenSize
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

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

// MARK: - Gemma MLP (GeGLU)

/// Gemma uses GeGLU: gelu(gate) * up, then down.
class GemmaMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: GemmaConfig) {
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

// MARK: - Gemma Transformer Block

class GemmaTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: GemmaAttention
    @ModuleInfo(key: "mlp") var mlp: GemmaMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: GemmaRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: GemmaRMSNorm

    init(_ config: GemmaConfig) {
        _selfAttn = ModuleInfo(wrappedValue: GemmaAttention(config), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: GemmaMLP(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - Gemma Full Model

class GemmaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [GemmaTransformerBlock]
    @ModuleInfo(key: "norm") var norm: GemmaRMSNorm

    let hiddenSize: Int

    init(_ config: GemmaConfig) {
        self.hiddenSize = config.hiddenSize
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in GemmaTransformerBlock(config) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: GemmaRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
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

public class GemmaForCausalLM: Module {
    @ModuleInfo(key: "model") var model: GemmaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: GemmaConfig

    public init(_ config: GemmaConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: GemmaModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        lmHead(model(tokens, caches: caches))
    }
}
