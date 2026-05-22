import MLX
import MLXNN
import MLXFast
import KLMCache
import KLMKernels

// MARK: - Qwen Config

/// Configuration for Qwen 2.5 / Qwen 3 dense model family.
///
/// Differences this config captures between the two:
///   - Qwen 2.5: QKV projections have bias; no q_norm / k_norm; no
///     tied embeddings.
///   - Qwen 3:   QKV projections have NO bias; per-head RMSNorm on
///     Q and K before RoPE (`q_norm`, `k_norm`); tied embeddings
///     (no separate `lm_head.weight`); `head_dim` set explicitly in
///     config rather than derived from `hidden_size /
///     num_attention_heads`.
///
/// The two variants share the same module hierarchy; flags below
/// switch the per-module behavior without forcing a duplicate file.
public struct QwenConfig: ModelConfig, Codable, Sendable {
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
    /// Whether Q/K/V projections have a bias term. True for Qwen 2.5,
    /// false for Qwen 3. Defaults to true so pre-WS4 Qwen 2.5
    /// configs (which omit `attention_bias`) keep their old behavior.
    public let attentionBias: Bool
    /// Whether the attention module applies per-head RMSNorm to Q and
    /// K before RoPE. True for Qwen 3, false for Qwen 2.5. Detected
    /// from `model_type == "qwen3"`.
    public let hasQKNorm: Bool
    /// When true, the LM head reuses `embed_tokens` (no separate
    /// `lm_head` weight is loaded). True for Qwen 3, false for Qwen
    /// 2.5.
    public let tieWordEmbeddings: Bool
    /// Explicit head_dim from config. Qwen 3 sets this independently
    /// of `hidden_size / num_attention_heads`. Qwen 2.5 omits it; we
    /// fall back to the derived value.
    public let explicitHeadDim: Int?
    /// HuggingFace `model_type`. We persist it (rather than deriving
    /// other flags from it on the fly) so `Codable` synthesis stays
    /// well-formed: every CodingKey case must map to a stored
    /// property for encode-side synthesis.
    public let modelType: String

    public var headDim: Int { explicitHeadDim ?? (hiddenSize / numAttentionHeads) }

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
        case attentionBias = "attention_bias"
        case modelType = "model_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case explicitHeadDim = "head_dim"
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
            ?? 32768
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2"
        let isQwen3 = modelType == "qwen3"
        // Qwen 2.5 omits `attention_bias` and ships QKV biases; Qwen 3
        // sets it to false explicitly. When the key is present we
        // honor it; otherwise default to the family's historical
        // behavior.
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias)
            ?? !isQwen3
        hasQKNorm = isQwen3
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
            ?? isQwen3
        explicitHeadDim = try c.decodeIfPresent(Int.self, forKey: .explicitHeadDim)
    }
}

// MARK: - Qwen Attention (with bias on Q/K/V projections)

class QwenAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    // Qwen 3: per-head RMSNorm on Q and K before RoPE. Optional so
    // Qwen 2.5 checkpoints (which have no q_norm / k_norm weight)
    // do not gain unused parameters at load time.
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

    let rope: RoPE

    init(_ config: QwenConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        let bias = config.attentionBias
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: bias), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: bias), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: bias), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")

        if config.hasQKNorm {
            _qNorm = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps),
                key: "q_norm")
            _kNorm = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps),
                key: "k_norm")
        }

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Qwen 3: apply per-head RMSNorm to Q and K *before* RoPE.
        // The norm operates on the last dimension (headDim) so we
        // apply it while the tensors are still in [B, L, heads, dim]
        // layout, then transpose for attention.
        if let qNorm { queries = qNorm(queries) }
        if let kNorm { keys = kNorm(keys) }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)

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

// MARK: - Qwen Transformer Block

class QwenTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: QwenAttention
    @ModuleInfo(key: "mlp") var mlp: QwenMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: QwenConfig) {
        _selfAttn = ModuleInfo(wrappedValue: QwenAttention(config), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: QwenMLP(config), key: "mlp")
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

// MARK: - Qwen MLP (SwiGLU, same as Llama)

class QwenMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: QwenConfig) {
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
        // Fused SwiGLU: silu(gate) * up in one Metal kernel pass.
        // Same fusion the dense Llama FeedForward already uses; the
        // kernel writes through Metal's implicit conversion to the
        // output buffer's element type so fp16, bf16, and fp32
        // activations are all handled correctly.
        let gate = gateProj(x)
        let up = upProj(x)
        return downProj(KLMKernels.fusedSwiGLU(gate: gate, up: up))
    }
}

// MARK: - Qwen Full Model

class QwenModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [QwenTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: QwenConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in QwenTransformerBlock(config) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        var x = embedTokens(tokens)
        let seqLen = x.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        // Mask must promote to the hidden state's dtype. Qwen 3
        // checkpoints run inference in bf16 (configs declare
        // `torch_dtype: "bfloat16"`); Qwen 2.5 4-bit MLX checkpoints
        // run in fp16. `MLXFast.scaledDotProductAttention` errors if
        // the mask dtype cannot be promoted, so we source the dtype
        // from the embedding output itself rather than guessing.
        let mask = createCachedCausalMask(
            newLen: seqLen, cacheLen: cacheLen, dtype: x.dtype)

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: caches?[i])
        }
        return norm(x)
    }
}

public class QwenForCausalLM: Module {
    @ModuleInfo(key: "model") var model: QwenModelInner
    // Optional so Qwen 3 (tied embeddings, no separate `lm_head`
    // weight in the safetensors) loads cleanly. When nil, the LM
    // projection reuses `model.embed_tokens` via `asLinear`, matching
    // the Gemma 4 tied-embedding path.
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: QwenConfig

    public init(_ config: QwenConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: QwenModelInner(config), key: "model")
        if !config.tieWordEmbeddings {
            _lmHead = ModuleInfo(
                wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
                key: "lm_head")
        }
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        let hidden = model(tokens, caches: caches)
        if let lmHead {
            return lmHead(hidden)
        }
        // Tied embeddings: reuse embed_tokens. `asLinear` does
        // matmul(hidden, embed_tokens.weight.T) and respects quantization.
        return model.embedTokens.asLinear(hidden)
    }
}
