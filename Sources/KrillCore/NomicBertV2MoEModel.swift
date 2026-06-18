import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Config

/// Configuration for nomic-embed-text-v2-moe: a `nomic_bert` encoder (RoPE,
/// fused biased `Wqkv`, post-norm) whose MLP alternates dense GELU and a top-2
/// mixture of experts. Distinct from the dense `NomicBertConfig` by its MoE
/// fields (`num_experts`, `moe_every_n_layers`, `moe_top_k`). The XLM-R
/// vocabulary (250k) is shared with bge-m3; positions come from RoPE.
public struct NomicBertV2Config: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let layerNormEps: Float
    public let ropeBase: Float
    public let rotaryFraction: Float
    public let maxTokens: Int
    public let numExperts: Int
    public let moeEveryNLayers: Int
    public let moeTopK: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// MoE replaces the dense MLP on layer `i` when `i % moe_every_n_layers == 1`
    /// (reference `NomicBertEncoder`: `moe = i % every_n == 1`).
    public func isMoELayer(_ i: Int) -> Bool {
        moeEveryNLayers > 0 && i % moeEveryNLayers == 1
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "n_embd"
        case numHiddenLayers = "n_layer"
        case numAttentionHeads = "n_head"
        case intermediateSize = "n_inner"
        case vocabSize = "vocab_size"
        case layerNormEps = "layer_norm_epsilon"
        case ropeBase = "rotary_emb_base"
        case rotaryFraction = "rotary_emb_fraction"
        case maxPositionEmbeddings = "max_trained_positions"
        case numExperts = "num_experts"
        case moeEveryNLayers = "moe_every_n_layers"
        case moeTopK = "moe_top_k"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps)) ?? 1e-5
        ropeBase = (try? c.decode(Float.self, forKey: .ropeBase)) ?? 10000
        rotaryFraction = (try? c.decode(Float.self, forKey: .rotaryFraction)) ?? 1.0
        maxTokens = min((try? c.decode(Int.self, forKey: .maxPositionEmbeddings)) ?? 2048, 2048)
        numExperts = (try? c.decode(Int.self, forKey: .numExperts)) ?? 0
        moeEveryNLayers = (try? c.decode(Int.self, forKey: .moeEveryNLayers)) ?? 0
        moeTopK = (try? c.decode(Int.self, forKey: .moeTopK)) ?? 2
    }

    /// True when the checkpoint is an MoE nomic-bert (v2), not the dense v1/v1.5.
    public var isMoE: Bool { numExperts > 0 && moeEveryNLayers > 0 }
}

// MARK: - Embeddings (word + token-type, no positions)

final class NomicV2Embeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding

    init(_ cfg: NomicBertV2Config) {
        _word = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "word_embeddings")
        // v2 ships a single token type (table [1, H]); index 0 for every token.
        _tokenType = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: 1, dimensions: cfg.hiddenSize),
            key: "token_type_embeddings")
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let T = tokens.dim(1)
        return word(tokens) + tokenType(MLXArray.zeros([1, T], dtype: .int32))
    }
}

// MARK: - Attention (bidirectional, fused biased Wqkv, RoPE)

final class NomicV2Attention: Module {
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ cfg: NomicBertV2Config) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        scale = 1.0 / Float(headDim).squareRoot()
        // v2 is biased (qkv_proj_bias=true), unlike v1.5's bias-free projections.
        _wqkv = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 3 * cfg.hiddenSize, bias: true), key: "Wqkv")
        _outProj = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize, bias: true), key: "out_proj")
        let rotaryDim = Int(Float(headDim) * cfg.rotaryFraction)
        rope = RoPE(dimensions: rotaryDim, traditional: false, base: cfg.ropeBase)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        // Fused qkv laid out as [qkv, heads, headDim] (reference rearranges
        // "(three h d)"), so index 0/1/2 of axis 2 picks q/k/v.
        let qkv = wqkv(x).reshaped(B, T, 3, numHeads, headDim)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        q = rope(q, offset: 0)
        k = rope(k, offset: 0)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim))
    }
}

// MARK: - MLP: dense GELU or top-2 MoE (one per layer)

/// The router projects to per-expert logits (`router.layer`, no bias).
final class NomicV2Router: Module {
    @ModuleInfo(key: "layer") var layer: Linear

    init(_ cfg: NomicBertV2Config) {
        _layer = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.numExperts, bias: false), key: "layer")
    }
}

/// Stacked expert weights (`experts.mlp.w1`/`w2`, each `[E*F, H]`). Per expert
/// `e`, the MLP is the non-gated `gelu(x @ w1[e]^T) @ w2[e]` (reference
/// `NomicExpertMLP`); `w2[e]` is `[F, H]` and used directly, not transposed.
final class NomicV2ExpertWeights: Module {
    @ParameterInfo(key: "w1") var w1: MLXArray
    @ParameterInfo(key: "w2") var w2: MLXArray

    init(experts: Int, ffn: Int, hidden: Int) {
        _w1 = ParameterInfo(wrappedValue: MLXArray.zeros([experts * ffn, hidden]), key: "w1")
        _w2 = ParameterInfo(wrappedValue: MLXArray.zeros([experts * ffn, hidden]), key: "w2")
    }
}

final class NomicV2Experts: Module {
    @ModuleInfo(key: "mlp") var mlp: NomicV2ExpertWeights
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(_ cfg: NomicBertV2Config) {
        _mlp = ModuleInfo(
            wrappedValue: NomicV2ExpertWeights(
                experts: cfg.numExperts, ffn: cfg.intermediateSize, hidden: cfg.hiddenSize),
            key: "mlp")
        _bias = ParameterInfo(wrappedValue: MLXArray.zeros([cfg.hiddenSize]), key: "bias")
    }
}

/// One layer's MLP. A layer is either dense (`fc1`/`fc2`, GELU) or MoE
/// (`router` + `experts`); the unused branch stays nil so the flattened
/// parameter keys match the checkpoint exactly under strict verify.
final class NomicV2MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear?
    @ModuleInfo(key: "fc2") var fc2: Linear?
    @ModuleInfo(key: "router") var router: NomicV2Router?
    @ModuleInfo(key: "experts") var experts: NomicV2Experts?

    let isMoE: Bool
    let numExperts: Int
    let topK: Int
    let ffn: Int
    let hidden: Int

    init(_ cfg: NomicBertV2Config, isMoE: Bool) {
        self.isMoE = isMoE
        self.numExperts = cfg.numExperts
        self.topK = cfg.moeTopK
        self.ffn = cfg.intermediateSize
        self.hidden = cfg.hiddenSize
        if isMoE {
            _router = ModuleInfo(wrappedValue: NomicV2Router(cfg), key: "router")
            _experts = ModuleInfo(wrappedValue: NomicV2Experts(cfg), key: "experts")
            _fc1 = ModuleInfo(wrappedValue: nil, key: "fc1")
            _fc2 = ModuleInfo(wrappedValue: nil, key: "fc2")
        } else {
            _fc1 = ModuleInfo(
                wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize, bias: true), key: "fc1")
            _fc2 = ModuleInfo(
                wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize, bias: true), key: "fc2")
            _router = ModuleInfo(wrappedValue: nil, key: "router")
            _experts = ModuleInfo(wrappedValue: nil, key: "experts")
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if !isMoE {
            return fc2!(gelu(fc1!(x)))
        }
        guard let router, let experts else { return x }
        let B = x.dim(0), T = x.dim(1)
        // Router: softmax over ALL experts, then keep the top-k probs WITHOUT
        // renormalizing (config moe_normalize_expert_weights=false).
        let probs = softmax(router.layer(x), axis: -1)  // [B, T, E]
        // The k-th largest prob per row is the ascending element at index E-k;
        // keep every expert whose prob is >= it (exactly top-k for distinct fp32
        // logits). Masked probs are the unnormalized combine weights.
        let sorted = MLX.sorted(probs, axis: -1)
        let thresh = sorted[0..., 0..., (numExperts - topK) ..< (numExperts - topK + 1)]
        let weights = probs * (probs .>= thresh).asType(probs.dtype)  // [B, T, E]

        let w1 = experts.mlp.w1.reshaped(numExperts, ffn, hidden)  // [E, F, H]
        let w2 = experts.mlp.w2.reshaped(numExperts, ffn, hidden)  // [E, F, H]
        var out = MLXArray.zeros([B, T, hidden], dtype: x.dtype)
        for e in 0 ..< numExperts {
            // expert e: gelu(x @ w1[e]^T) @ w2[e]; w2[e] is [F, H], used directly.
            let he = matmul(gelu(matmul(x, w1[e].transposed())), w2[e])  // [B, T, H]
            out = out + weights[0..., 0..., e ..< (e + 1)] * he
        }
        return out + experts.bias
    }
}

// MARK: - Block (post-norm)

final class NomicV2Layer: Module {
    @ModuleInfo(key: "attn") var attn: NomicV2Attention
    @ModuleInfo(key: "mlp") var mlp: NomicV2MLP
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(_ cfg: NomicBertV2Config, isMoE: Bool) {
        _attn = ModuleInfo(wrappedValue: NomicV2Attention(cfg), key: "attn")
        _mlp = ModuleInfo(wrappedValue: NomicV2MLP(cfg, isMoE: isMoE), key: "mlp")
        _norm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps), key: "norm1")
        _norm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps), key: "norm2")
    }

    /// Post-norm: `h = norm1(attn(x) + x)`, then `norm2(mlp(h) + h)`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = norm1(attn(x, mask: mask) + x)
        return norm2(mlp(h) + h)
    }
}

final class NomicV2Encoder: Module {
    @ModuleInfo(key: "layers") var layers: [NomicV2Layer]

    init(_ cfg: NomicBertV2Config) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { NomicV2Layer(cfg, isMoE: cfg.isMoELayer($0)) },
            key: "layers")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        for l in layers { h = l(h, mask: mask) }
        return h
    }
}

// MARK: - Top-level model

/// nomic-embed-text-v2-moe `NomicBertModel`: `embeddings` -> `emb_ln` ->
/// `encoder`. Parameter keys match the checkpoint exactly so weights load with
/// `strictVerify` and no key rewrite. Mean-pooled.
public final class NomicBertV2MoEModel: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embeddings") var embeddings: NomicV2Embeddings
    @ModuleInfo(key: "emb_ln") var embLn: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: NomicV2Encoder

    public let config: NomicBertV2Config

    public init(_ cfg: NomicBertV2Config) {
        self.config = cfg
        _embeddings = ModuleInfo(wrappedValue: NomicV2Embeddings(cfg), key: "embeddings")
        _embLn = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps), key: "emb_ln")
        _encoder = ModuleInfo(wrappedValue: NomicV2Encoder(cfg), key: "encoder")
    }

    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        encoder(embLn(embeddings(tokens)))
    }
}
