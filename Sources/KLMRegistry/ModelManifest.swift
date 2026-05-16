import Foundation

/// A model manifest describes an installed model in the local registry.
///
/// Manifests live at ~/.krillm/models/manifests/<name>.json and reference
/// content-addressed blobs in ~/.krillm/models/blobs/.
public struct ModelManifest: Codable, Sendable {
    /// Human-friendly name (e.g., "llama-3.1-8b")
    public let name: String

    /// Model family identifier for architecture dispatch
    public let family: ModelFamily

    /// Parameter count label (e.g., "8B", "3B")
    public let params: String

    /// Quantization identifier (e.g., "4bit", "8bit", "fp16")
    public let quant: String

    /// HuggingFace source repo (e.g., "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    public let source: String

    /// Maximum context length
    public let context: Int

    /// Files in this model (relative paths -> sha256 hashes)
    public let files: [ModelFile]

    /// Optional draft model name for speculative decoding (Phase 3)
    public let draftPair: String?

    /// Chat template identifier
    public let chatTemplate: String

    /// Size on disk in bytes
    public let sizeBytes: Int64

    /// When this model was pulled
    public let pulledAt: Date

    /// Modelfile-derived customizations (WS-C). Optional + decoded with
    /// `decodeIfPresent` (synthesized), so pre-existing manifests without
    /// this key still decode.
    public let overrides: ModelOverrides?

    public init(
        name: String,
        family: ModelFamily,
        params: String,
        quant: String,
        source: String,
        context: Int,
        files: [ModelFile],
        draftPair: String? = nil,
        chatTemplate: String,
        sizeBytes: Int64,
        pulledAt: Date = Date(),
        overrides: ModelOverrides? = nil
    ) {
        self.name = name
        self.family = family
        self.params = params
        self.quant = quant
        self.source = source
        self.context = context
        self.files = files
        self.draftPair = draftPair
        self.chatTemplate = chatTemplate
        self.sizeBytes = sizeBytes
        self.pulledAt = pulledAt
        self.overrides = overrides
    }
}

/// Modelfile-derived overrides layered on top of a base model (WS-C / T1-2).
public struct ModelOverrides: Codable, Sendable, Equatable {
    public var system: String?
    public var template: String?
    public var license: String?
    public var parameters: [String: String]
    public var messages: [[String: String]]

    public init(system: String? = nil, template: String? = nil,
                license: String? = nil,
                parameters: [String: String] = [:],
                messages: [[String: String]] = []) {
        self.system = system
        self.template = template
        self.license = license
        self.parameters = parameters
        self.messages = messages
    }
}

/// A file within a model (weight shard, tokenizer, config, etc.)
public struct ModelFile: Codable, Sendable {
    /// Relative path within the model directory
    public let path: String

    /// SHA256 hash of the file contents
    public let sha256: String

    /// File size in bytes
    public let sizeBytes: Int64

    public init(path: String, sha256: String, sizeBytes: Int64) {
        self.path = path
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
    }
}

/// Supported model families for architecture dispatch.
public enum ModelFamily: String, Codable, Sendable, CaseIterable {
    case llama
    case qwen
    case mistral
    case gemma
    case gemma4
    case phi
    case glm
    case deepseek
    /// Dedicated sentence-embedding encoder (BERT/RoBERTa/MiniLM/BGE/E5).
    /// Not a causal LM — served only via the embeddings endpoints.
    case bert

    /// Detect model family from HuggingFace config.json's `architectures` field.
    public static func detect(from configJSON: [String: Any]) -> ModelFamily? {
        guard let architectures = configJSON["architectures"] as? [String],
              let arch = architectures.first else {
            // Fallback: try model_type
            if let modelType = configJSON["model_type"] as? String {
                return fromModelType(modelType)
            }
            return nil
        }

        let archLower = arch.lowercased()
        // Order matters: check specific before generic
        if archLower.contains("bert") || archLower.contains("roberta") { return .bert }
        if archLower.contains("gemma4") { return .gemma4 }
        if archLower.contains("gemma") { return .gemma }
        if archLower.contains("chatglm") || archLower.contains("glm") { return .glm }
        if archLower.contains("deepseek") { return .deepseek }
        if archLower.contains("llama") { return .llama }
        if archLower.contains("qwen") { return .qwen }
        if archLower.contains("mistral") { return .mistral }
        if archLower.contains("phi") { return .phi }
        return nil
    }

    private static func fromModelType(_ type: String) -> ModelFamily? {
        switch type.lowercased() {
        case "llama": return .llama
        case "qwen2", "qwen3": return .qwen
        case "mistral": return .mistral
        case "gemma", "gemma2", "gemma3": return .gemma
        case "gemma4", "gemma4_text": return .gemma4
        case "phi", "phi3": return .phi
        case "chatglm", "glm", "glm4_moe": return .glm
        case "deepseek_v3": return .deepseek
        case "bert", "roberta", "xlm-roberta", "mpnet", "distilbert": return .bert
        default: return nil
        }
    }
}
