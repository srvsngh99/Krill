import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - GLM-4 Config

/// Configuration for GLM-4 / ChatGLM model family (Zhipu AI).
/// Key differences: post-layer norm, QKV bias, GQA with 2 KV heads,
/// very small layernorm epsilon, high rope_ratio.
public struct GLMConfig: ModelConfig, Codable, Sendable {
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

    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "ffn_hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "multi_query_group_num"
        case numHiddenLayers = "num_layers"
        case vocabSize = "padded_vocab_size"
        case rmsNormEps = "layernorm_epsilon"
        case ropeTheta = "rope_ratio"
        case maxPositionEmbeddings = "seq_length"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 2
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 151552
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1.5625e-7
        // GLM uses rope_ratio as a multiplier on base theta (10000 * ratio)
        let ratio = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500.0
        ropeTheta = 10000.0 * ratio
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - GLM Attention (GQA with bias on QKV)

class GLMAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    // GLM uses a fused QKV projection with bias
    @ModuleInfo(key: "query_key_value") var qkvProj: Linear
    @ModuleInfo(key: "dense") var oProj: Linear

    let rope: RoPE

    init(_ config: GLMConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        // Fused QKV: projects to (numHeads + 2*numKVHeads) * headDim
        let qkvSize = (numHeads + 2 * numKVHeads) * headDim
        _qkvProj = ModuleInfo(
            wrappedValue: Linear(dim, qkvSize, bias: true), key: "query_key_value")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "dense")

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Fused QKV projection
        let qkv = qkvProj(x)

        // Split into Q, K, V
        let qSize = numHeads * headDim
        let kvSize = numKVHeads * headDim

        let q = qkv[0..., 0..., ..<qSize]
        let k = qkv[0..., 0..., qSize ..< (qSize + kvSize)]
        let v = qkv[0..., 0..., (qSize + kvSize)...]

        var queries = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = k.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = v.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

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

// MARK: - GLM MLP (SwiGLU, fused gate+up)

class GLMMLP: Module {
    @ModuleInfo(key: "dense_h_to_4h") var gateUpProj: Linear
    @ModuleInfo(key: "dense_4h_to_h") var downProj: Linear

    let intermediateSize: Int

    init(_ config: GLMConfig) {
        self.intermediateSize = config.intermediateSize
        // GLM fuses gate and up into one projection (2 * intermediate)
        _gateUpProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false),
            key: "dense_h_to_4h")
        _downProj = ModuleInfo(
            wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: false),
            key: "dense_4h_to_h")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        let gate = gateUp[0..., 0..., ..<intermediateSize]
        let up = gateUp[0..., 0..., intermediateSize...]
        return downProj(silu(gate) * up)
    }
}

// MARK: - GLM Transformer Block (post-layer norm!)

/// GLM uses post-layer normalization (norm after attention/FFN, not before).
class GLMTransformerBlock: Module {
    @ModuleInfo(key: "self_attention") var selfAttn: GLMAttention
    @ModuleInfo(key: "mlp") var mlp: GLMMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: GLMConfig) {
        _selfAttn = ModuleInfo(wrappedValue: GLMAttention(config), key: "self_attention")
        _mlp = ModuleInfo(wrappedValue: GLMMLP(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        // GLM post-norm: norm applied to output, not input
        let attnOut = selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        let h = x + attnOut
        let mlpOut = mlp(postAttentionLayernorm(h))
        return h + mlpOut
    }
}

// MARK: - GLM Full Model

class GLMModelInner: Module {
    @ModuleInfo(key: "embedding") var embedding: GLMEmbedding
    @ModuleInfo(key: "encoder") var encoder: GLMEncoder
    @ModuleInfo(key: "output_layer") var outputLayer: Linear

    init(_ config: GLMConfig) {
        _embedding = ModuleInfo(
            wrappedValue: GLMEmbedding(config), key: "embedding")
        _encoder = ModuleInfo(
            wrappedValue: GLMEncoder(config), key: "encoder")
        _outputLayer = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "output_layer")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        let hidden = embedding(tokens)
        let encoded = encoder(hidden, caches: caches)
        return outputLayer(encoded)
    }
}

class GLMEmbedding: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding

    init(_ config: GLMConfig) {
        _wordEmbeddings = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "word_embeddings")
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        wordEmbeddings(tokens)
    }
}

class GLMEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [GLMTransformerBlock]
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm

    init(_ config: GLMConfig) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in GLMTransformerBlock(config) },
            key: "layers")
        _finalNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "final_layernorm")
    }

    func callAsFunction(_ x: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        var hidden = x
        let seqLen = hidden.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let mask = createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen)

        for (i, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: caches?[i])
        }
        return finalNorm(hidden)
    }
}

public class GLMForCausalLM: Module {
    @ModuleInfo(key: "transformer") var transformer: GLMModelInner

    public let config: GLMConfig

    public init(_ config: GLMConfig) {
        self.config = config
        _transformer = ModuleInfo(wrappedValue: GLMModelInner(config), key: "transformer")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        transformer(tokens, caches: caches)
    }
}
