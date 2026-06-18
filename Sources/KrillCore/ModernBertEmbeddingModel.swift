import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Config

/// Configuration for a ModernBERT encoder (`model_type: "modernbert"`;
/// gte-modernbert, nomic-modernbert-embed). A pre-norm RoPE encoder with:
/// alternating global/local attention (every Nth layer is global with a large
/// RoPE theta; the rest are local with a sliding window and a small theta), a
/// GeGLU MLP, weight-only LayerNorms, and no biases anywhere. Layer 0's
/// attention norm is an identity (the embeddings are already normed).
public struct ModernBertConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let normEps: Float
    public let globalRopeTheta: Float
    public let localRopeTheta: Float
    public let globalAttnEveryNLayers: Int
    public let localAttention: Int
    public let maxTokens: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }
    /// Local layers attend within +/- `localAttention / 2` tokens.
    public var localWindow: Int { localAttention / 2 }
    public func isGlobal(_ layer: Int) -> Bool { layer % globalAttnEveryNLayers == 0 }
    public func ropeTheta(_ layer: Int) -> Float {
        isGlobal(layer) ? globalRopeTheta : localRopeTheta
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case normEps = "norm_eps"
        case globalRopeTheta = "global_rope_theta"
        case localRopeTheta = "local_rope_theta"
        case globalAttnEveryNLayers = "global_attn_every_n_layers"
        case localAttention = "local_attention"
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        normEps = (try? c.decode(Float.self, forKey: .normEps)) ?? 1e-5
        globalRopeTheta = (try? c.decode(Float.self, forKey: .globalRopeTheta)) ?? 160000
        localRopeTheta = (try? c.decode(Float.self, forKey: .localRopeTheta)) ?? 10000
        globalAttnEveryNLayers = (try? c.decode(Int.self, forKey: .globalAttnEveryNLayers)) ?? 3
        localAttention = (try? c.decode(Int.self, forKey: .localAttention)) ?? 128
        maxTokens = min((try? c.decode(Int.self, forKey: .maxPositionEmbeddings)) ?? 8192, 8192)
    }
}

// MARK: - Weight-only LayerNorm

/// LayerNorm with a scale but no bias (`norm_bias: false`). A plain `weight`
/// MLXArray property is auto-registered as the `weight` parameter, matching the
/// checkpoint's `*.norm.weight` / `final_norm.weight` keys.
final class ModernBertNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(_ dimensions: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mu = MLX.mean(x, axis: -1, keepDims: true)
        let v = MLX.variance(x, axis: -1, keepDims: true)
        return weight * (x - mu) * MLX.rsqrt(v + eps)
    }
}

// MARK: - Attention (fused Wqkv, per-layer RoPE, optional sliding-window mask)

final class ModernBertAttention: Module {
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ cfg: ModernBertConfig, ropeBase: Float) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        scale = 1.0 / Float(headDim).squareRoot()
        _wqkv = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 3 * cfg.hiddenSize, bias: false), key: "Wqkv")
        _wo = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize, bias: false), key: "Wo")
        rope = RoPE(dimensions: headDim, traditional: false, base: ropeBase)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let qkv = wqkv(x).reshaped(B, T, 3, numHeads, headDim)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        q = rope(q, offset: 0)
        k = rope(k, offset: 0)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return wo(out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim))
    }
}

// MARK: - GeGLU MLP

/// `Wo(gelu(input) * gate)` where the fused `Wi` packs `[input, gate]`.
final class ModernBertMLP: Module {
    @ModuleInfo(key: "Wi") var wi: Linear
    @ModuleInfo(key: "Wo") var wo: Linear

    init(_ cfg: ModernBertConfig) {
        _wi = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 2 * cfg.intermediateSize, bias: false), key: "Wi")
        _wo = ModuleInfo(
            wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false), key: "Wo")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(wi(x), parts: 2, axis: -1)
        return wo(gelu(parts[0]) * parts[1])
    }
}

// MARK: - Block (pre-norm; layer 0 attn norm is identity)

final class ModernBertLayer: Module {
    /// nil for layer 0 (the embeddings are already normed) - the checkpoint
    /// ships no `layers.0.attn_norm.weight`.
    @ModuleInfo(key: "attn_norm") var attnNorm: ModernBertNorm?
    @ModuleInfo(key: "attn") var attn: ModernBertAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: ModernBertNorm
    @ModuleInfo(key: "mlp") var mlp: ModernBertMLP

    init(_ cfg: ModernBertConfig, layer: Int) {
        _attnNorm = ModuleInfo(
            wrappedValue: layer == 0 ? nil : ModernBertNorm(cfg.hiddenSize, eps: cfg.normEps),
            key: "attn_norm")
        _attn = ModuleInfo(
            wrappedValue: ModernBertAttention(cfg, ropeBase: cfg.ropeTheta(layer)), key: "attn")
        _mlpNorm = ModuleInfo(
            wrappedValue: ModernBertNorm(cfg.hiddenSize, eps: cfg.normEps), key: "mlp_norm")
        _mlp = ModuleInfo(wrappedValue: ModernBertMLP(cfg), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = attnNorm?(x) ?? x
        var h = x + attn(normed, mask: mask)
        h = h + mlp(mlpNorm(h))
        return h
    }
}

// MARK: - Top-level model

/// HF `ModernBertModel`: `embeddings` (tok + norm) -> `layers` -> `final_norm`.
/// Global layers see full attention; local layers see a `[-window, +window]`
/// sliding-window mask (a no-op when the sequence is shorter than the window).
public final class ModernBertEmbeddingModel: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embeddings") var embeddings: ModernBertEmbeddings
    @ModuleInfo(key: "layers") var layers: [ModernBertLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: ModernBertNorm

    public let config: ModernBertConfig

    public init(_ cfg: ModernBertConfig) {
        self.config = cfg
        _embeddings = ModuleInfo(wrappedValue: ModernBertEmbeddings(cfg), key: "embeddings")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { ModernBertLayer(cfg, layer: $0) },
            key: "layers")
        _finalNorm = ModuleInfo(
            wrappedValue: ModernBertNorm(cfg.hiddenSize, eps: cfg.normEps), key: "final_norm")
    }

    /// Additive `[1, 1, T, T]` mask, 0 within +/- window, -inf outside. Cast to
    /// `dtype` so scaled_dot_product_attention can promote it to the q/k/v type.
    private func slidingWindowMask(_ T: Int, dtype: DType) -> MLXArray {
        let rows = MLXArray(Int32(0) ..< Int32(T))
        let dist = MLX.abs(rows.reshaped(T, 1) - rows.reshaped(1, T))
        let allowed = dist .<= MLXArray(Int32(config.localWindow))
        return MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            .reshaped(1, 1, T, T).asType(dtype)
    }

    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        var h = embeddings(tokens)
        let localMask = slidingWindowMask(tokens.dim(1), dtype: h.dtype)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: config.isGlobal(i) ? nil : localMask)
        }
        return finalNorm(h)
    }
}

final class ModernBertEmbeddings: Module {
    @ModuleInfo(key: "tok_embeddings") var tok: Embedding
    @ModuleInfo(key: "norm") var norm: ModernBertNorm

    init(_ cfg: ModernBertConfig) {
        _tok = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "tok_embeddings")
        _norm = ModuleInfo(wrappedValue: ModernBertNorm(cfg.hiddenSize, eps: cfg.normEps), key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        norm(tok(tokens))
    }
}
