import Foundation
import MLX
import MLXNN

// MARK: - Config

/// Configuration for an MPNet encoder (sentence-transformers/all-mpnet-base-v2).
///
/// MPNet differs from vanilla BERT in three ways the loader must honor:
/// no token-type embeddings, absolute positions offset by `pad_token_id + 1`
/// (RoBERTa-style), and a learned **relative attention bias** added to the
/// attention scores in every layer (T5-style buckets).
public struct MPNetConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let layerNormEps: Float
    public let relativeAttentionNumBuckets: Int
    public let positionOffset: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }
    /// Usable token cap: position table holds `maxPositionEmbeddings` slots,
    /// positions start at `positionOffset`, so a sequence can be at most
    /// `maxPositionEmbeddings - positionOffset` long.
    public var maxTokens: Int { maxPositionEmbeddings - positionOffset }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerNormEps = "layer_norm_eps"
        case relativeAttentionNumBuckets = "relative_attention_num_buckets"
        case padTokenId = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps)) ?? 1e-5
        relativeAttentionNumBuckets =
            (try? c.decode(Int.self, forKey: .relativeAttentionNumBuckets)) ?? 32
        let padId = (try? c.decode(Int.self, forKey: .padTokenId)) ?? 1
        positionOffset = padId + 1
    }
}

// MARK: - Embeddings (word + absolute position, no token_type)

final class MPNetEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "position_embeddings") var position: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    private let positionOffset: Int

    init(_ cfg: MPNetConfig) {
        _word = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "word_embeddings")
        _position = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: cfg.maxPositionEmbeddings, dimensions: cfg.hiddenSize),
            key: "position_embeddings")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
        positionOffset = cfg.positionOffset
    }

    /// `tokens [1, T] -> embeddings [1, T, H]`. Positions are offset by
    /// `pad_token_id + 1`; there are no token-type embeddings.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let T = tokens.dim(1)
        let posIds = MLXArray(Int32(positionOffset) ..< Int32(positionOffset + T))
            .reshaped(1, T)
        return norm(word(tokens) + position(posIds))
    }
}

// MARK: - Self-attention (q/k/v/o, relative bias added to scores)

final class MPNetSelfAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ cfg: MPNetConfig) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        scale = 1.0 / Float(headDim).squareRoot()
        _q = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "q")
        _k = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "k")
        _v = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "v")
        _o = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "o")
    }

    /// `positionBias [1, numHeads, T, T]` is added to the pre-softmax scores.
    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let query = q(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let key = k(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let value = v(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)

        var logits = MLX.matmul(query * scale, key.transposed(0, 1, 3, 2))
        logits = logits + positionBias
        let scores = MLX.softmax(logits, axis: -1)
        let out = MLX.matmul(scores, value)
        let merged = out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim)
        return o(merged)
    }
}

final class MPNetAttention: Module {
    @ModuleInfo(key: "attn") var attn: MPNetSelfAttention
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: MPNetConfig) {
        _attn = ModuleInfo(wrappedValue: MPNetSelfAttention(cfg), key: "attn")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        norm(attn(x, positionBias: positionBias) + x)
    }
}

final class MPNetIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ cfg: MPNetConfig) {
        _dense = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize), key: "dense")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(dense(x)) }
}

final class MPNetOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: MPNetConfig) {
        _dense = ModuleInfo(
            wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize), key: "dense")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
    }

    func callAsFunction(_ hidden: MLXArray, _ input: MLXArray) -> MLXArray {
        norm(dense(hidden) + input)
    }
}

final class MPNetLayer: Module {
    @ModuleInfo(key: "attention") var attention: MPNetAttention
    @ModuleInfo(key: "intermediate") var intermediate: MPNetIntermediate
    @ModuleInfo(key: "output") var output: MPNetOutput

    init(_ cfg: MPNetConfig) {
        _attention = ModuleInfo(wrappedValue: MPNetAttention(cfg), key: "attention")
        _intermediate = ModuleInfo(wrappedValue: MPNetIntermediate(cfg), key: "intermediate")
        _output = ModuleInfo(wrappedValue: MPNetOutput(cfg), key: "output")
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray) -> MLXArray {
        let a = attention(x, positionBias: positionBias)
        return output(intermediate(a), a)
    }
}

// MARK: - Encoder (owns the shared relative attention bias)

final class MPNetEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [MPNetLayer]
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding

    private let numBuckets: Int
    private let numHeads: Int

    init(_ cfg: MPNetConfig) {
        _layer = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { _ in MPNetLayer(cfg) },
            key: "layer")
        _relativeAttentionBias = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: cfg.relativeAttentionNumBuckets, dimensions: cfg.numAttentionHeads),
            key: "relative_attention_bias")
        numBuckets = cfg.relativeAttentionNumBuckets
        numHeads = cfg.numAttentionHeads
    }

    /// T5-style bidirectional bucketing of `relativePosition` (key - query).
    private func relativePositionBucket(
        _ relativePosition: Int, maxDistance: Int = 128
    ) -> Int {
        var ret = 0
        var n = -relativePosition
        let nb = numBuckets / 2
        if n < 0 { ret += nb }
        n = abs(n)
        let maxExact = nb / 2
        if n < maxExact {
            ret += n
        } else {
            let scaled = log(Double(n) / Double(maxExact))
                / log(Double(maxDistance) / Double(maxExact)) * Double(nb - maxExact)
            ret += min(maxExact + Int(scaled), nb - 1)
        }
        return ret
    }

    /// Per-head additive bias `[1, numHeads, T, T]` from the relative-position
    /// buckets. Computed once and shared across all layers.
    private func positionBias(_ T: Int) -> MLXArray {
        var buckets = [Int32](repeating: 0, count: T * T)
        for i in 0 ..< T {        // query position
            for j in 0 ..< T {    // key position
                buckets[i * T + j] = Int32(relativePositionBucket(j - i))
            }
        }
        let idx = MLXArray(buckets).reshaped(T, T)
        let values = relativeAttentionBias(idx)         // [T, T, numHeads]
        return values.transposed(2, 0, 1).reshaped(1, numHeads, T, T)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let bias = positionBias(x.dim(1))
        var h = x
        for l in layer { h = l(h, positionBias: bias) }
        return h
    }
}

// MARK: - Top-level model

/// HF `MPNetModel`: `embeddings` + `encoder`. The checkpoint also ships a
/// `pooler` and a `position_ids` buffer that this encoder does not use; the
/// loader drops those keys before a strict-verify update.
public final class MPNetEmbeddingModel: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embeddings") var embeddings: MPNetEmbeddings
    @ModuleInfo(key: "encoder") var encoder: MPNetEncoder

    public let config: MPNetConfig

    public init(_ cfg: MPNetConfig) {
        self.config = cfg
        _embeddings = ModuleInfo(wrappedValue: MPNetEmbeddings(cfg), key: "embeddings")
        _encoder = ModuleInfo(wrappedValue: MPNetEncoder(cfg), key: "encoder")
    }

    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        encoder(embeddings(tokens))
    }
}
