import Foundation
import MLX
import MLXNN
import KLMCache
import KLMRuntime

/// Unified model loader that auto-detects the model family from config.json
/// and instantiates the correct architecture.
///
/// Returns a `LoadedModel` that provides a uniform interface regardless of family.
public struct LoadedModel: @unchecked Sendable {
    /// The underlying MLXNN Module (LlamaForCausalLM, QwenForCausalLM, etc.)
    public let module: Module

    /// Number of transformer layers
    public let numLayers: Int

    /// The detected model family
    public let family: String

    /// Forward pass returning logits
    public let forward: (MLXArray, [KVCache]?) -> MLXArray

    /// Multimodal forward pass (tokens, caches, imageEmbeddings, audioEmbeddings) -> logits
    /// Only set for models that support native multimodal (e.g. Gemma4).
    public let multimodalForward: ((MLXArray, [KVCache]?, MLXArray?, MLXArray?) -> MLXArray)?

    /// Vocab size for validation
    public let vocabSize: Int
}

/// Detect the model family and load the appropriate architecture.
///
/// Reads config.json from the model directory, detects the architecture,
/// instantiates the model, quantizes if needed, and loads weights.
public func loadModel(from directory: URL) throws -> LoadedModel {
    try MLXMetalRuntime.validateForNativeInference()

    let configURL = directory.appendingPathComponent("config.json")
    let configData = try Data(contentsOf: configURL)

    // Parse raw JSON to detect architecture
    guard let configJSON = try JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
        throw ModelLoadError.invalidConfig("Cannot parse config.json")
    }

    // Detect family
    let architectures = configJSON["architectures"] as? [String] ?? []
    let modelType = configJSON["model_type"] as? String ?? ""
    let arch = architectures.first?.lowercased() ?? ""

    // Order matters: check specific patterns before generic ones
    if arch.contains("gemma4") || modelType == "gemma4_text" || modelType == "gemma4" {
        return try loadGemma4(configData: configData, directory: directory)
    } else if arch.contains("chatglm") || arch.contains("glm") || modelType == "chatglm" {
        return try loadGLM(configData: configData, directory: directory)
    } else if arch.contains("deepseek") || modelType == "deepseek_v3" {
        // Full DeepSeek V3 MoE - not yet supported, suggest distill variants
        throw ModelLoadError.unsupportedArchitecture(
            "DeepSeek V3 MoE (671B) requires MoE support (coming soon). "
            + "Use DeepSeek-R1-Distill variants instead: krillm pull deepseek-r1-7b")
    } else if arch.contains("llama") || modelType == "llama" {
        return try loadLlama(configData: configData, directory: directory)
    } else if arch.contains("qwen") || modelType.hasPrefix("qwen") {
        return try loadQwen(configData: configData, directory: directory)
    } else if arch.contains("mistral") || modelType == "mistral" {
        return try loadMistral(configData: configData, directory: directory)
    } else if arch.contains("gemma") || modelType.hasPrefix("gemma") {
        return try loadGemma(configData: configData, directory: directory)
    } else if arch.contains("phi") || modelType.hasPrefix("phi") {
        return try loadPhi(configData: configData, directory: directory)
    } else {
        // Fallback: try as Llama (most common architecture)
        return try loadLlama(configData: configData, directory: directory)
    }
}

// MARK: - Per-Family Loaders

private func loadLlama(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(LlamaConfig.self, from: configData)
    let model = LlamaForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "llama",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadQwen(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(QwenConfig.self, from: configData)
    let model = QwenForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "qwen",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadMistral(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(MistralConfig.self, from: configData)
    let model = MistralForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "mistral",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadGemma(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(GemmaConfig.self, from: configData)
    let model = GemmaForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "gemma",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadPhi(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(PhiConfig.self, from: configData)
    let model = PhiForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "phi",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadGemma4(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Gemma4Config.self, from: configData)
    let model = Gemma4ForCausalLM(config)

    // Gemma 4: quantize with filter to exclude per_layer_model_projection (stays BF16)
    if let q = config.quantization {
        quantize(model: model, groupSize: q.groupSize, bits: q.bits) { name, _ in
            !name.contains("per_layer_model_projection") && !name.contains("per_layer_projection_norm")
        }
    }

    // Load weights with "language_model." prefix stripped
    var flatWeights = try loadWeightArrays(from: directory)
    var stripped: [String: MLXArray] = [:]
    for (key, value) in flatWeights {
        if key.hasPrefix("language_model.") {
            stripped[String(key.dropFirst("language_model.".count))] = value
        }
    }
    if !stripped.isEmpty { flatWeights = stripped }

    // Handle tied embeddings
    let hasLmHead = flatWeights.keys.contains { $0.hasPrefix("lm_head.") }
    if !hasLmHead {
        let embedKeys = flatWeights.keys.filter { $0.hasPrefix("model.embed_tokens.") && !$0.contains("per_layer") }
        for key in embedKeys {
            let lmKey = key.replacingOccurrences(of: "model.embed_tokens.", with: "lm_head.")
            flatWeights[lmKey] = flatWeights[key]
        }
    }

    let tuples = flatWeights.map { ($0.key, $0.value) }
    let nested = ModuleParameters.unflattened(tuples)
    try model.update(parameters: nested, verify: [])

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "gemma4",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: { tokens, caches, imageEmb, audioEmb in
            model(tokens, caches: caches,
                  imageEmbeddings: imageEmb, audioEmbeddings: audioEmb)
        },
        vocabSize: config.vocabSize
    )
}

private func loadGLM(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(GLMConfig.self, from: configData)
    let model = GLMForCausalLM(config)
    try loadWeights(into: model, from: directory, quantization: config.quantization)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "glm",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

// MARK: - Errors

public enum ModelLoadError: Error, CustomStringConvertible {
    case invalidConfig(String)
    case unsupportedArchitecture(String)

    public var description: String {
        switch self {
        case .invalidConfig(let msg): return "Invalid config: \(msg)"
        case .unsupportedArchitecture(let arch): return "Unsupported architecture: \(arch)"
        }
    }
}
