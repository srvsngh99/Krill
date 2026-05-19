import MLX
import MLXNN
import KLMCache

// MARK: - Mistral Config

/// Configuration for Mistral 7B model family.
/// Architecturally identical to Llama (GQA, RoPE, SwiGLU, RMSNorm, no bias).
/// Sliding window attention is a v0.1/v0.2 feature, not used in instruct models.
public struct MistralConfig: ModelConfig, Codable, Sendable {
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
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
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
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 32768
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - Mistral Full Model

/// Mistral uses the exact same architecture as Llama.
/// The LlamaForCausalLM implementation handles Mistral weights correctly
/// since both share identical layer naming and structure.
///
/// This type alias enables explicit family dispatch while reusing Llama code.
/// Config differences (rope_theta, num_kv_heads, etc.) are handled by
/// MistralConfig providing correct defaults.
public class MistralForCausalLM: Module {
    @ModuleInfo(key: "model") var model: MistralModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: MistralConfig

    public init(_ config: MistralConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: MistralModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        lmHead(model(tokens, caches: caches))
    }
}

// Reuses the same Llama-style architecture internally
class MistralModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: MistralConfig) {
        // Create a LlamaConfig from MistralConfig (architecturally identical)
        let llamaConfig = LlamaConfig(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            numAttentionHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            numHiddenLayers: config.numHiddenLayers,
            vocabSize: config.vocabSize,
            rmsNormEps: config.rmsNormEps,
            ropeTheta: config.ropeTheta,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
            quantization: config.quantization
        )

        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in TransformerBlock(llamaConfig) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        var x = embedTokens(tokens)
        let seqLen = x.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let mask = createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen)

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: caches?[i])
        }
        return norm(x)
    }
}
