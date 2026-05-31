import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Sentence-encoder protocol

/// A dedicated sentence-embedding encoder: maps `tokens [1, T]` to a
/// last-hidden-state `[1, T, H]` that `poolSentenceEmbedding` reduces to a
/// single vector. Implemented by both the vanilla `BertEmbeddingModel`
/// (learned positions) and `NomicBertEmbeddingModel` (RoPE), so the
/// `EmbeddingEngine` can hold either behind one type.
public protocol SentenceEmbeddingEncoder: Module {
    func lastHiddenState(_ tokens: MLXArray) -> MLXArray
}

extension BertEmbeddingModel: SentenceEmbeddingEncoder {
    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray { self(tokens) }
}

// MARK: - Config

/// Configuration for a `nomic_bert` encoder (nomic-embed-text v1 / v1.5).
///
/// Differs from vanilla BERT (`BertEmbeddingConfig`): rotary position
/// embeddings instead of learned ones, a fused `Wqkv` projection, a SwiGLU
/// gated MLP, and post-norm (`prenorm=false`) residual structure. The
/// `position_embeddings` table is absent from the checkpoint.
public struct NomicBertConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let typeVocabSize: Int
    public let layerNormEps: Float
    public let ropeBase: Float
    public let rotaryFraction: Float
    /// Token cap for embedding (matches Ollama's default num_ctx for nomic).
    public let maxTokens: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "n_embd"
        case hiddenSizeAlt = "hidden_size"
        case numHiddenLayers = "n_layer"
        case numHiddenLayersAlt = "num_hidden_layers"
        case numAttentionHeads = "n_head"
        case numAttentionHeadsAlt = "num_attention_heads"
        case intermediateSize = "n_inner"
        case intermediateSizeAlt = "intermediate_size"
        case vocabSize = "vocab_size"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_epsilon"
        case layerNormEpsAlt = "layer_norm_eps"
        case ropeBase = "rotary_emb_base"
        case rotaryFraction = "rotary_emb_fraction"
        case maxPositionEmbeddings = "max_position_embeddings"
        case nPositions = "n_positions"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // nomic configs use GPT-2-style keys (n_embd, n_layer, n_head, n_inner)
        // but tolerate the HF BERT-style aliases too.
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? c.decode(Int.self, forKey: .hiddenSizeAlt)
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers)
            ?? c.decode(Int.self, forKey: .numHiddenLayersAlt)
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
            ?? c.decode(Int.self, forKey: .numAttentionHeadsAlt)
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? c.decode(Int.self, forKey: .intermediateSizeAlt)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        typeVocabSize = (try? c.decode(Int.self, forKey: .typeVocabSize)) ?? 2
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps))
            ?? (try? c.decode(Float.self, forKey: .layerNormEpsAlt)) ?? 1e-12
        ropeBase = (try? c.decode(Float.self, forKey: .ropeBase)) ?? 1000
        rotaryFraction = (try? c.decode(Float.self, forKey: .rotaryFraction)) ?? 1.0
        // Cap at max_position_embeddings (the trained window, 2048 for
        // nomic-embed-text); n_positions (8192) is the extended ceiling.
        maxTokens = (try? c.decode(Int.self, forKey: .maxPositionEmbeddings)) ?? 2048
    }
}

// MARK: - Embeddings

/// `word + token_type` lookup. No learned position embeddings (RoPE supplies
/// position in attention). The `emb_ln` LayerNorm is applied at the model
/// level, matching the checkpoint's top-level `emb_ln.*` keys.
final class NomicBertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding

    init(_ cfg: NomicBertConfig) {
        _word = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "word_embeddings")
        _tokenType = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: max(cfg.typeVocabSize, 1), dimensions: cfg.hiddenSize),
            key: "token_type_embeddings")
    }

    /// `tokens [1, T] -> embeddings [1, T, H]` (pre-LayerNorm).
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let T = tokens.dim(1)
        let typeIds = MLXArray.zeros([1, T], dtype: .int32)
        return word(tokens) + tokenType(typeIds)
    }
}

// MARK: - Attention (bidirectional, fused Wqkv, RoPE)

final class NomicBertAttention: Module {
    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ cfg: NomicBertConfig) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        scale = 1.0 / Float(headDim).squareRoot()
        // Fused QKV projection: out = 3 * hidden, no bias (qkv_proj_bias=false).
        _wqkv = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 3 * cfg.hiddenSize, bias: false), key: "Wqkv")
        _outProj = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize, bias: false), key: "out_proj")
        // rotary_emb_fraction=1.0 -> rotate the full head dim. traditional:false
        // matches flash-attn's non-interleaved (GPT-NeoX) rotary convention.
        let rotaryDim = Int(Float(headDim) * cfg.rotaryFraction)
        rope = RoPE(dimensions: rotaryDim, traditional: false, base: cfg.ropeBase)
    }

    /// Bidirectional self-attention over `x [B, T, H]`. `mask` is an optional
    /// additive key-padding mask (batch-1 callers pass nil — every token real).
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        // Fused projection -> [B, T, 3, numHeads, headDim]. flash-attn lays the
        // 3*hidden axis out as [qkv, heads, headDim], so index 0/1/2 picks q/k/v.
        let qkv = wqkv(x).reshaped(B, T, 3, numHeads, headDim)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)  // [B,H,T,D]
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)

        // Encoder positions are 0..<T, so a scalar offset of 0 applies.
        q = rope(q, offset: 0)
        k = rope(k, offset: 0)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        let merged = out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim)
        return outProj(merged)
    }
}

// MARK: - Gated SwiGLU MLP

/// `fc2(fc11(x) * silu(fc12(x)))`. The activation is applied to the *gate*
/// projection (`fc12`), then multiplied by `fc11`; nomic-embed-text ships no
/// intermediate MLP norm, so none is applied. All projections are bias-free.
final class NomicBertGatedMLP: Module {
    @ModuleInfo(key: "fc11") var fc11: Linear
    @ModuleInfo(key: "fc12") var fc12: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ cfg: NomicBertConfig) {
        _fc11 = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false), key: "fc11")
        _fc12 = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false), key: "fc12")
        _fc2 = ModuleInfo(
            wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false), key: "fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(fc11(x) * silu(fc12(x)))
    }
}

// MARK: - Block (post-norm)

final class NomicBertLayer: Module {
    @ModuleInfo(key: "attn") var attn: NomicBertAttention
    @ModuleInfo(key: "mlp") var mlp: NomicBertGatedMLP
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm

    init(_ cfg: NomicBertConfig) {
        _attn = ModuleInfo(wrappedValue: NomicBertAttention(cfg), key: "attn")
        _mlp = ModuleInfo(wrappedValue: NomicBertGatedMLP(cfg), key: "mlp")
        _norm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "norm1")
        _norm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "norm2")
    }

    /// Post-norm: `h = norm1(attn(h) + h)`, then `h = norm2(mlp(h) + h)`.
    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let h = norm1(attn(x, mask: mask) + x)
        return norm2(mlp(h) + h)
    }
}

final class NomicBertEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NomicBertLayer]

    init(_ cfg: NomicBertConfig) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { _ in NomicBertLayer(cfg) },
            key: "layers")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        for l in layers { h = l(h, mask: mask) }
        return h
    }
}

// MARK: - Top-level model

/// `NomicBertModel`: `embeddings` -> `emb_ln` -> `encoder`. Parameter keys
/// match the checkpoint exactly (`embeddings.*`, `emb_ln.*`, `encoder.layers.*`)
/// so weights load with `strictVerify` and no key rewrite.
public final class NomicBertEmbeddingModel: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embeddings") var embeddings: NomicBertEmbeddings
    @ModuleInfo(key: "emb_ln") var embLn: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: NomicBertEncoder

    public let config: NomicBertConfig

    public init(_ cfg: NomicBertConfig) {
        self.config = cfg
        _embeddings = ModuleInfo(wrappedValue: NomicBertEmbeddings(cfg), key: "embeddings")
        _embLn = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "emb_ln")
        _encoder = ModuleInfo(wrappedValue: NomicBertEncoder(cfg), key: "encoder")
    }

    /// `tokens [1, T] -> lastHiddenState [1, T, H]`.
    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        encoder(embLn(embeddings(tokens)))
    }
}
