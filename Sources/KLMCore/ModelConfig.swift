import Foundation

// MARK: - Model Config Protocol

/// Common interface for all model family configurations.
public protocol ModelConfig: Sendable {
    var hiddenSize: Int { get }
    var intermediateSize: Int { get }
    var numAttentionHeads: Int { get }
    var numKeyValueHeads: Int { get }
    var numHiddenLayers: Int { get }
    var vocabSize: Int { get }
    var rmsNormEps: Float { get }
    var ropeTheta: Float { get }
    var maxPositionEmbeddings: Int { get }
    var headDim: Int { get }
    var quantization: QuantizationConfig? { get }
}

// MARK: - Quantization

/// Quantization parameters (present in 4-bit / 8-bit mlx-community models).
public struct QuantizationConfig: Codable, Sendable {
    public let groupSize: Int
    public let bits: Int

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }

    public init(groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
    }
}

// MARK: - Llama Config

/// Configuration for the Llama model family (Llama 3.x, Llama 3.2, etc.).
/// Parsed from config.json in HuggingFace model repos.
public struct LlamaConfig: ModelConfig, Codable, Sendable {
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
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 131_072
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }

    /// Convenience initializer for tests and manual construction.
    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        numHiddenLayers: Int,
        vocabSize: Int,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 500_000.0,
        maxPositionEmbeddings: Int = 131_072,
        quantization: QuantizationConfig? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numHiddenLayers = numHiddenLayers
        self.vocabSize = vocabSize
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.quantization = quantization
    }
}

// MARK: - Config Loading

/// Load a model config from a directory containing config.json.
public func loadConfig(from directory: URL) throws -> LlamaConfig {
    let configURL = directory.appendingPathComponent("config.json")
    let data = try Data(contentsOf: configURL)
    return try JSONDecoder().decode(LlamaConfig.self, from: data)
}
