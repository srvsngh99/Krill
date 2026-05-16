import Foundation
import MLX
import MLXNN

// MARK: - Config

/// Configuration for a BERT-style sentence-embedding encoder
/// (sentence-transformers / BGE / MiniLM / E5 — all `BertModel` or
/// `XLMRobertaModel` architectures). This is a *dedicated embedding model*,
/// distinct from the causal-LM families in `ModelLoader` — it has no
/// `lm_head` and uses bidirectional (non-causal) attention.
public struct BertEmbeddingConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let typeVocabSize: Int
    public let layerNormEps: Float
    /// RoBERTa/XLM-R offset positions by `padding_idx + 1`; plain BERT does not.
    public let positionOffset: Int

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case intermediateSize = "intermediate_size"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabSize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case modelType = "model_type"
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
        typeVocabSize = (try? c.decode(Int.self, forKey: .typeVocabSize)) ?? 2
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps)) ?? 1e-12
        let modelType = (try? c.decode(String.self, forKey: .modelType)) ?? "bert"
        let padId = (try? c.decode(Int.self, forKey: .padTokenId)) ?? 0
        positionOffset = (modelType == "xlm-roberta" || modelType == "roberta")
            ? padId + 1 : 0
    }
}

// MARK: - Embeddings

final class BertEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "position_embeddings") var position: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    private let positionOffset: Int

    init(_ cfg: BertEmbeddingConfig) {
        _word = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize,
                                    dimensions: cfg.hiddenSize),
            key: "word_embeddings")
        _position = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.maxPositionEmbeddings,
                                    dimensions: cfg.hiddenSize),
            key: "position_embeddings")
        _tokenType = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: max(cfg.typeVocabSize, 1),
                                    dimensions: cfg.hiddenSize),
            key: "token_type_embeddings")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
        positionOffset = cfg.positionOffset
    }

    /// `tokens [1, T] -> embeddings [1, T, H]`
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let T = tokens.dim(1)
        let posIds = MLXArray(Int32(positionOffset) ..< Int32(positionOffset + T))
            .reshaped(1, T)
        let typeIds = MLXArray.zeros([1, T], dtype: .int32)
        let e = word(tokens) + position(posIds) + tokenType(typeIds)
        return norm(e)
    }
}

// MARK: - Self-attention (bidirectional, no causal mask)

final class BertSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    let numHeads: Int
    let headDim: Int

    init(_ cfg: BertEmbeddingConfig) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        _query = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "query")
        _key = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "key")
        _value = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "value")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        var q = query(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = key(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = value(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let scale = 1.0 / Float(headDim).squareRoot()
        q = q * scale
        let scores = MLX.softmax(MLX.matmul(q, k.transposed(0, 1, 3, 2)), axis: -1)
        let out = MLX.matmul(scores, v)
        return out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim)
    }
}

final class BertSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: BertEmbeddingConfig) {
        _dense = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "dense")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
    }

    func callAsFunction(_ hidden: MLXArray, _ input: MLXArray) -> MLXArray {
        norm(dense(hidden) + input)
    }
}

final class BertAttention: Module {
    @ModuleInfo(key: "self") var selfAttn: BertSelfAttention
    @ModuleInfo(key: "output") var output: BertSelfOutput

    init(_ cfg: BertEmbeddingConfig) {
        _selfAttn = ModuleInfo(wrappedValue: BertSelfAttention(cfg), key: "self")
        _output = ModuleInfo(wrappedValue: BertSelfOutput(cfg), key: "output")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        output(selfAttn(x), x)
    }
}

final class BertIntermediate: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ cfg: BertEmbeddingConfig) {
        _dense = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize), key: "dense")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(dense(x)) }
}

final class BertOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: BertEmbeddingConfig) {
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

final class BertLayer: Module {
    @ModuleInfo(key: "attention") var attention: BertAttention
    @ModuleInfo(key: "intermediate") var intermediate: BertIntermediate
    @ModuleInfo(key: "output") var output: BertOutput

    init(_ cfg: BertEmbeddingConfig) {
        _attention = ModuleInfo(wrappedValue: BertAttention(cfg), key: "attention")
        _intermediate = ModuleInfo(wrappedValue: BertIntermediate(cfg), key: "intermediate")
        _output = ModuleInfo(wrappedValue: BertOutput(cfg), key: "output")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = attention(x)
        return output(intermediate(a), a)
    }
}

final class BertEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [BertLayer]
    init(_ cfg: BertEmbeddingConfig) {
        _layer = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { _ in BertLayer(cfg) },
            key: "layer")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for l in layer { h = l(h) }
        return h
    }
}

// MARK: - Top-level model

/// HF `BertModel`: `embeddings` + `encoder`. Weight keys load with the
/// `model.` / no prefix depending on the checkpoint; the loader strips a
/// leading `bert.`/`roberta.` if present.
public final class BertEmbeddingModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: BertEmbeddings
    @ModuleInfo(key: "encoder") var encoder: BertEncoder

    public let config: BertEmbeddingConfig

    public init(_ cfg: BertEmbeddingConfig) {
        self.config = cfg
        _embeddings = ModuleInfo(wrappedValue: BertEmbeddings(cfg), key: "embeddings")
        _encoder = ModuleInfo(wrappedValue: BertEncoder(cfg), key: "encoder")
    }

    /// `tokens [1, T] -> lastHiddenState [1, T, H]`
    public func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        encoder(embeddings(tokens))
    }
}

// MARK: - Pooling

public enum EmbeddingPooling: String, Sendable {
    case mean
    case cls

    public static func fromEnv() -> EmbeddingPooling {
        if let v = ProcessInfo.processInfo.environment["KRILL_EMBED_POOLING"],
           let p = EmbeddingPooling(rawValue: v.lowercased()) {
            return p
        }
        return .mean
    }
}

/// Pool a `[1, T, H]` last-hidden-state into a `[H]` sentence vector and
/// L2-normalize. Batch is 1 (callers embed one text at a time, so no
/// padding mask is needed — every token is real).
public func poolSentenceEmbedding(
    _ lastHidden: MLXArray,
    pooling: EmbeddingPooling,
    normalize: Bool = true
) -> [Float] {
    let pooled: MLXArray
    switch pooling {
    case .cls:
        pooled = lastHidden[0, 0, 0...]            // [H]
    case .mean:
        pooled = MLX.mean(lastHidden[0], axis: 0)  // mean over T -> [H]
    }
    var vec = pooled
    if normalize {
        let denom = MLX.sqrt(MLX.sum(vec * vec)) + 1e-12
        vec = vec / denom
    }
    vec.eval()
    return vec.asArray(Float.self)
}
