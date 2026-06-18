import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - GLM-4 (Glm4ForCausalLM, model_type "glm4") Config
//
// The GLM-4-0414 / GLM-Z1 generation. A WHOLLY DIFFERENT architecture from the
// legacy ChatGLM runtime in `GLMModel.swift`:
//   - separate q/k/v/o projections, with an additive bias on q/k/v ONLY
//   - a four-RMSNorm "sandwich": input_layernorm + post_self_attn_layernorm
//     (on the attention output, before its residual add) + post_attention_layernorm
//     + post_mlp_layernorm (on the MLP output, before its residual add)
//   - partial RoPE (only the first `partial_rotary_factor * head_dim` dims rotate)
//   - fused gate_up_proj SwiGLU, untied lm_head, standard `model.layers.*` naming
//
// The `quantization` field is required by the `ModelConfig` protocol.
public struct Glm4Config: ModelConfig, Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let headDimValue: Int
    public let partialRotaryFactor: Float
    public let attentionBias: Bool
    public let quantization: QuantizationConfig?

    // GLM-4 ships an explicit `head_dim` that need not equal hidden/heads.
    public var headDim: Int { headDimValue }

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
        case headDimValue = "head_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case attentionBias = "attention_bias"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        // GLM-4 uses rope_theta directly (NOT the legacy ChatGLM `10000 * rope_ratio`).
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        let heads = numAttentionHeads
        headDimValue = try c.decodeIfPresent(Int.self, forKey: .headDimValue) ?? (hiddenSize / heads)
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - GLM-4 Attention (separate q/k/v/o; bias on q/k/v only; partial RoPE)

class Glm4Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPE

    init(_ config: Glm4Config) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        // GLM-4 puts an additive bias on q/k/v but NOT on o_proj.
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: config.attentionBias), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: config.attentionBias), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: config.attentionBias), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")

        // Partial RoPE: only the first `partial_rotary_factor * head_dim` dims
        // are rotated; MLX's RoPE passes the remaining trailing dims through
        // unchanged when `dimensions < head_dim`. GLM uses the traditional
        // (interleaved) rotation.
        let rotaryDims = Int(config.partialRotaryFactor * Float(config.headDim))
        self.rope = RoPE(dimensions: rotaryDims, traditional: true, base: config.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = queries.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

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

// MARK: - GLM-4 MLP (fused gate+up SwiGLU)

class Glm4MLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let intermediateSize: Int

    init(_ config: Glm4Config) {
        self.intermediateSize = config.intermediateSize
        _gateUpProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false),
            key: "gate_up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: false),
            key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        let gate = gateUp[0..., 0..., ..<intermediateSize]
        let up = gateUp[0..., 0..., intermediateSize...]
        return downProj(silu(gate) * up)
    }
}

// MARK: - GLM-4 Decoder Layer (four-RMSNorm sandwich)

class Glm4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Glm4Attention
    @ModuleInfo(key: "mlp") var mlp: Glm4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttnLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "post_mlp_layernorm") var postMlpLayernorm: RMSNorm

    init(_ config: Glm4Config) {
        _selfAttn = ModuleInfo(wrappedValue: Glm4Attention(config), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: Glm4MLP(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postSelfAttnLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_self_attn_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
        _postMlpLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_mlp_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        // GLM-4 sandwich norm: the post_self_attn / post_mlp norms wrap each
        // sublayer's OUTPUT before it is added back to the residual stream.
        var r = x
        var h = inputLayernorm(x)
        h = selfAttn(h, mask: mask, cache: cache)
        h = postSelfAttnLayernorm(h)
        h = r + h

        r = h
        var m = postAttentionLayernorm(h)
        m = mlp(m)
        m = postMlpLayernorm(m)
        return r + m
    }
}

// MARK: - GLM-4 Full Model

class Glm4ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Glm4DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Glm4Config) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in Glm4DecoderLayer(config) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    /// `lastTokenOnly` slices the hidden state to the final position before the
    /// caller's vocab projection, so prefill skips the lm_head matmul over the
    /// unused prefix rows. See `LlamaForCausalLM` for the rationale.
    func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil, lastTokenOnly: Bool = false
    ) -> MLXArray {
        var h = embedTokens(tokens)
        let seqLen = h.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        // Build the additive mask in the ACTIVATION dtype. Quant mode decides the
        // dequant dtype: affine 4-bit dequants to fp16, nvfp4 to bf16. SDPA
        // requires the mask dtype to promote to the output (activation) dtype, so
        // a fixed fp16 mask crashes the bf16 nvfp4 path with "Mask type must
        // promote to output type bfloat16".
        let mask = createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen, dtype: h.dtype)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches?[i])
        }
        h = norm(h)
        if lastTokenOnly {
            let last = h.dim(1) - 1
            h = h[0..., last ..< (last + 1), 0...]
        }
        return h
    }
}

public class Glm4ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Glm4ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Glm4Config

    public init(_ config: Glm4Config) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Glm4ModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil, lastTokenOnly: Bool
    ) -> MLXArray {
        lmHead(model(tokens, caches: caches, lastTokenOnly: lastTokenOnly))
    }
}
