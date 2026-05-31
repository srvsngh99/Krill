import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Config

/// Configuration for a GTE-v1.5 encoder (Alibaba-NLP `NewModel`,
/// `model_type: "new"`). A RoPE encoder like nomic-bert, but with biased fused
/// QKV / output projections, a GeGLU MLP (gelu), post-norm `attn_ln`/`mlp_ln`,
/// and no token-type embeddings. CLS-pooled.
public struct GTEConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let typeVocabSize: Int
    public let layerNormEps: Float
    public let ropeBase: Float
    public let maxTokens: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        // gte-base is type_vocab_size 0 (no token-type table); some variants
        // (e.g. gte-large) ship a 2-entry table that must be summed in.
        typeVocabSize = (try? c.decode(Int.self, forKey: .typeVocabSize)) ?? 0
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps)) ?? 1e-12
        ropeBase = (try? c.decode(Float.self, forKey: .ropeTheta)) ?? 10000
        // Cap a single embed forward; deepkrill-scale chunks sit far under it.
        maxTokens = min((try? c.decode(Int.self, forKey: .maxPositionEmbeddings)) ?? 8192, 8192)
    }
}

// MARK: - Embeddings (word + LayerNorm, no positions / token-type)

final class GTEEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    /// Present only when `type_vocab_size > 0` (e.g. gte-large); the checkpoint
    /// has no such tensor when it is 0 (gte-base), and a strict-verify load
    /// would reject an unconsumed module, so this stays nil there.
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding?
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: GTEConfig) {
        _word = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "word_embeddings")
        _tokenType = ModuleInfo(
            wrappedValue: cfg.typeVocabSize > 0
                ? Embedding(embeddingCount: cfg.typeVocabSize, dimensions: cfg.hiddenSize)
                : nil,
            key: "token_type_embeddings")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var e = word(tokens)
        if let tokenType {
            let T = tokens.dim(1)
            e = e + tokenType(MLXArray.zeros([1, T], dtype: .int32))
        }
        return norm(e)
    }
}

// MARK: - Attention (fused qkv with bias, RoPE)

final class GTEAttention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ cfg: GTEConfig) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        scale = 1.0 / Float(headDim).squareRoot()
        _qkvProj = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 3 * cfg.hiddenSize), key: "qkv_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "o_proj")
        rope = RoPE(dimensions: headDim, traditional: false, base: cfg.ropeBase)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        // Fused projection -> [B, T, 3, heads, headDim]; index 0/1/2 picks q/k/v.
        let qkv = qkvProj(x).reshaped(B, T, 3, numHeads, headDim)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        q = rope(q, offset: 0)
        k = rope(k, offset: 0)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim))
    }
}

// MARK: - GeGLU MLP

/// `down_proj(up * gelu(gate))`. The fused `up_gate_proj` packs `[up, gate]`;
/// the activation (gelu) is applied to the gate half. `up_gate_proj` has no
/// bias; `down_proj` does.
final class GTEMLP: Module {
    @ModuleInfo(key: "up_gate_proj") var upGate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ cfg: GTEConfig) {
        _upGate = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 2 * cfg.intermediateSize, bias: false),
            key: "up_gate_proj")
        _down = ModuleInfo(
            wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(upGate(x), parts: 2, axis: -1)
        return down(parts[0] * gelu(parts[1]))
    }
}

// MARK: - Block (post-norm)

final class GTELayer: Module {
    @ModuleInfo(key: "attention") var attention: GTEAttention
    @ModuleInfo(key: "mlp") var mlp: GTEMLP
    @ModuleInfo(key: "attn_ln") var attnLn: LayerNorm
    @ModuleInfo(key: "mlp_ln") var mlpLn: LayerNorm

    init(_ cfg: GTEConfig) {
        _attention = ModuleInfo(wrappedValue: GTEAttention(cfg), key: "attention")
        _mlp = ModuleInfo(wrappedValue: GTEMLP(cfg), key: "mlp")
        _attnLn = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "attn_ln")
        _mlpLn = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "mlp_ln")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = attnLn(attention(x, mask: mask) + x)
        return mlpLn(mlp(h) + h)
    }
}

final class GTEEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [GTELayer]

    init(_ cfg: GTEConfig) {
        _layer = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { _ in GTELayer(cfg) },
            key: "layer")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        for l in layer { h = l(h, mask: mask) }
        return h
    }
}

// MARK: - Top-level model

/// GTE-v1.5 `NewModel`: `embeddings` -> `encoder`. Parameter keys match the
/// checkpoint exactly, so weights load with `strictVerify` and no key rewrite.
public final class GTEEmbeddingModel: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embeddings") var embeddings: GTEEmbeddings
    @ModuleInfo(key: "encoder") var encoder: GTEEncoder

    public let config: GTEConfig

    public init(_ cfg: GTEConfig) {
        self.config = cfg
        _embeddings = ModuleInfo(wrappedValue: GTEEmbeddings(cfg), key: "embeddings")
        _encoder = ModuleInfo(wrappedValue: GTEEncoder(cfg), key: "encoder")
    }

    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        encoder(embeddings(tokens))
    }
}
