import Foundation
import MLX
import MLXNN

// MARK: - Config

/// Configuration for a JinaBERT encoder (jina-embeddings-v2; `model_type: "bert"`
/// but `position_embedding_type: "alibi"`, arch `JinaBertForMaskedLM`). A
/// post-norm BERT encoder with no positional embeddings - position is supplied
/// by a per-head ALiBi bias on the attention scores - and a GLU (GeGLU) MLP.
public struct JinaBertConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let intermediateSize: Int
    public let vocabSize: Int
    public let typeVocabSize: Int
    public let layerNormEps: Float
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
        case maxPositionEmbeddings = "max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        typeVocabSize = (try? c.decode(Int.self, forKey: .typeVocabSize)) ?? 2
        layerNormEps = (try? c.decode(Float.self, forKey: .layerNormEps)) ?? 1e-12
        maxTokens = min((try? c.decode(Int.self, forKey: .maxPositionEmbeddings)) ?? 8192, 8192)
    }

    /// Per-head ALiBi slopes (Press et al.), with the non-power-of-2 fallback
    /// (closest lower power of 2, plus every-other slope from the next power).
    public func alibiSlopes() -> [Float] {
        func powerOf2(_ m: Int) -> [Double] {
            let start = pow(2.0, -pow(2.0, -(log2(Double(m)) - 3.0)))
            return (0 ..< m).map { start * pow(start, Double($0)) }
        }
        func slopes(_ n: Int) -> [Double] {
            if log2(Double(n)).truncatingRemainder(dividingBy: 1) == 0 {
                return powerOf2(n)
            }
            let closest = Int(pow(2.0, floor(log2(Double(n)))))
            var out = powerOf2(closest)
            let extra = slopes(2 * closest)
            var i = 0
            while out.count < n {
                out.append(extra[i]); i += 2
            }
            return out
        }
        return slopes(numAttentionHeads).map { Float($0) }
    }
}

// MARK: - Embeddings (word + token_type + LayerNorm, no positions)

final class JinaEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var word: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenType: Embedding
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: JinaBertConfig) {
        _word = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "word_embeddings")
        _tokenType = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: max(cfg.typeVocabSize, 1), dimensions: cfg.hiddenSize),
            key: "token_type_embeddings")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let T = tokens.dim(1)
        let typeIds = MLXArray.zeros([1, T], dtype: .int32)
        return norm(word(tokens) + tokenType(typeIds))
    }
}

// MARK: - Self-attention (separate q/k/v, ALiBi bias added to scores)

final class JinaSelfAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ cfg: JinaBertConfig) {
        numHeads = cfg.numAttentionHeads
        headDim = cfg.hiddenSize / cfg.numAttentionHeads
        scale = 1.0 / Float(headDim).squareRoot()
        _query = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "query")
        _key = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "key")
        _value = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "value")
    }

    /// `alibiBias [1, numHeads, T, T]` is added to the pre-softmax scores.
    func callAsFunction(_ x: MLXArray, alibiBias: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let q = query(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = key(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = value(x).reshaped(B, T, numHeads, headDim).transposed(0, 2, 1, 3)
        var logits = MLX.matmul(q * scale, k.transposed(0, 1, 3, 2))
        logits = logits + alibiBias
        let scores = MLX.softmax(logits, axis: -1)
        let out = MLX.matmul(scores, v)
        return out.transposed(0, 2, 1, 3).reshaped(B, T, numHeads * headDim)
    }
}

final class JinaSelfOutput: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "LayerNorm") var norm: LayerNorm

    init(_ cfg: JinaBertConfig) {
        _dense = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize), key: "dense")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "LayerNorm")
    }

    func callAsFunction(_ hidden: MLXArray, _ input: MLXArray) -> MLXArray {
        norm(dense(hidden) + input)
    }
}

final class JinaAttention: Module {
    @ModuleInfo(key: "self") var selfAttn: JinaSelfAttention
    @ModuleInfo(key: "output") var output: JinaSelfOutput

    init(_ cfg: JinaBertConfig) {
        _selfAttn = ModuleInfo(wrappedValue: JinaSelfAttention(cfg), key: "self")
        _output = ModuleInfo(wrappedValue: JinaSelfOutput(cfg), key: "output")
    }

    func callAsFunction(_ x: MLXArray, alibiBias: MLXArray) -> MLXArray {
        output(selfAttn(x, alibiBias: alibiBias), x)
    }
}

// MARK: - GLU MLP (post-norm, own LayerNorm)

/// `layernorm(wo(gelu(gated) * up) + residual)`, where the fused `gated_layers`
/// packs `[gated, up]`. `gated_layers` has no bias; `wo` does.
final class JinaGLU: Module {
    @ModuleInfo(key: "gated_layers") var gated: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "layernorm") var norm: LayerNorm

    init(_ cfg: JinaBertConfig) {
        _gated = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, 2 * cfg.intermediateSize, bias: false),
            key: "gated_layers")
        _wo = ModuleInfo(
            wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize), key: "wo")
        _norm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "layernorm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(gated(x), parts: 2, axis: -1)
        return norm(wo(gelu(parts[0]) * parts[1]) + x)
    }
}

final class JinaLayer: Module {
    @ModuleInfo(key: "attention") var attention: JinaAttention
    @ModuleInfo(key: "mlp") var mlp: JinaGLU

    init(_ cfg: JinaBertConfig) {
        _attention = ModuleInfo(wrappedValue: JinaAttention(cfg), key: "attention")
        _mlp = ModuleInfo(wrappedValue: JinaGLU(cfg), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray, alibiBias: MLXArray) -> MLXArray {
        mlp(attention(x, alibiBias: alibiBias))
    }
}

final class JinaEncoder: Module {
    @ModuleInfo(key: "layer") var layer: [JinaLayer]

    init(_ cfg: JinaBertConfig) {
        _layer = ModuleInfo(
            wrappedValue: (0 ..< cfg.numHiddenLayers).map { _ in JinaLayer(cfg) },
            key: "layer")
    }

    func callAsFunction(_ x: MLXArray, alibiBias: MLXArray) -> MLXArray {
        var h = x
        for l in layer { h = l(h, alibiBias: alibiBias) }
        return h
    }
}

// MARK: - Top-level model

/// HF `JinaBertModel`: `embeddings` + `encoder`. The checkpoint also ships a
/// `pooler` this encoder does not use; the loader drops it before strict verify.
public final class JinaBertEmbeddingModel: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embeddings") var embeddings: JinaEmbeddings
    @ModuleInfo(key: "encoder") var encoder: JinaEncoder

    public let config: JinaBertConfig
    private let slopes: [Float]

    public init(_ cfg: JinaBertConfig) {
        self.config = cfg
        self.slopes = cfg.alibiSlopes()
        _embeddings = ModuleInfo(wrappedValue: JinaEmbeddings(cfg), key: "embeddings")
        _encoder = ModuleInfo(wrappedValue: JinaEncoder(cfg), key: "encoder")
    }

    /// Symmetric (bidirectional) ALiBi bias `[1, numHeads, T, T]`:
    /// `-slope[h] * |i - j|`. Computed once and shared across layers.
    private func alibiBias(_ T: Int) -> MLXArray {
        let rows = MLXArray(Int32(0) ..< Int32(T))
        let dist = MLX.abs(rows.reshaped(T, 1) - rows.reshaped(1, T)).asType(.float32)
        let s = MLXArray(slopes.map { -$0 }).reshaped(config.numAttentionHeads, 1, 1)
        return (s * dist.reshaped(1, T, T)).reshaped(1, config.numAttentionHeads, T, T)
    }

    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        encoder(embeddings(tokens), alibiBias: alibiBias(tokens.dim(1)))
    }
}
