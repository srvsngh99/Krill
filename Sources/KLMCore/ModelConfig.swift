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
///
/// Two on-disk shapes coexist:
///
/// - **Uniform**: every quantized module shares the same `(group_size, bits)`:
///   ```json
///   "quantization": { "group_size": 64, "bits": 4 }
///   ```
/// - **Mixed precision** (newer mlx-community quants, e.g. Qwen3-Coder MoE):
///   top-level defaults plus per-module overrides keyed by full dotted name:
///   ```json
///   "quantization": {
///     "group_size": 64, "bits": 4,
///     "model.layers.0.mlp.gate": { "group_size": 64, "bits": 8 },
///     ...
///   }
///   ```
///   Qwen3-Coder ships 4-bit base with the MoE gates promoted to 8-bit; loading
///   the gates as 4-bit (the prior behaviour) crashes MLX with a scales-shape
///   mismatch in `quantized_matmul`.
///
/// `moduleOverrides` is the parsed per-module map; `effective(for:)` returns
/// the right `(groupSize, bits)` for a module path (override if present, top
/// level otherwise).
public struct QuantizationConfig: Sendable {
    public let groupSize: Int
    public let bits: Int
    /// Per-module overrides keyed by the module's full dotted name.
    /// Empty for uniform-precision checkpoints.
    public let moduleOverrides: [String: ModuleQuant]

    public struct ModuleQuant: Codable, Sendable, Equatable {
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

    public init(
        groupSize: Int, bits: Int,
        moduleOverrides: [String: ModuleQuant] = [:]
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.moduleOverrides = moduleOverrides
    }

    /// Effective `(groupSize, bits)` for a module at the given dotted path.
    /// Falls back to the top-level defaults when no override is registered.
    public func effective(
        for modulePath: String
    ) -> (groupSize: Int, bits: Int) {
        if let o = moduleOverrides[modulePath] {
            return (o.groupSize, o.bits)
        }
        return (groupSize, bits)
    }
}

extension QuantizationConfig: Codable {
    /// Dynamic keys so we can iterate every entry in the JSON object - the
    /// per-module override keys are arbitrary dotted names, not a fixed set.
    private struct DynamicKey: CodingKey {
        var stringValue: String
        var intValue: Int? { nil }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: DynamicKey.self)
        var groupSize: Int?
        var bits: Int?
        var overrides: [String: ModuleQuant] = [:]
        for key in c.allKeys {
            switch key.stringValue {
            case "group_size":
                groupSize = try c.decode(Int.self, forKey: key)
            case "bits":
                bits = try c.decode(Int.self, forKey: key)
            case "mode", "method":
                // Top-level mode / method labels (e.g. `"mode": "affine"`)
                // are informational; the MLX quantize call uses .affine
                // and the per-module overrides do not carry these.
                continue
            default:
                // Nested object => module override. If the entry is not
                // a `ModuleQuant`-shaped object, leave it alone for
                // forwards-compat with newer config fields.
                if let mq = try? c.decode(ModuleQuant.self, forKey: key) {
                    overrides[key.stringValue] = mq
                }
            }
        }
        // Strict on the top-level scalars: an mlx-community quant block
        // without `group_size` or `bits` is a malformed config we'd
        // rather surface immediately than paper over with silent
        // defaults. The earlier synthesized Codable threw here too;
        // this matches that contract while still supporting the new
        // per-module overrides.
        guard let resolvedGroupSize = groupSize else {
            throw DecodingError.keyNotFound(
                DynamicKey(stringValue: "group_size")!,
                .init(codingPath: decoder.codingPath,
                      debugDescription: "QuantizationConfig requires `group_size`"))
        }
        guard let resolvedBits = bits else {
            throw DecodingError.keyNotFound(
                DynamicKey(stringValue: "bits")!,
                .init(codingPath: decoder.codingPath,
                      debugDescription: "QuantizationConfig requires `bits`"))
        }
        self.groupSize = resolvedGroupSize
        self.bits = resolvedBits
        self.moduleOverrides = overrides
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: DynamicKey.self)
        try c.encode(groupSize, forKey: DynamicKey(stringValue: "group_size")!)
        try c.encode(bits, forKey: DynamicKey(stringValue: "bits")!)
        for (path, mq) in moduleOverrides {
            try c.encode(mq, forKey: DynamicKey(stringValue: path)!)
        }
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
