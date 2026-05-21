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

    /// Forward pass returning logits.
    ///
    /// Takes `[KVCacheProtocol]?` so callers can pass either fp16 `KVCache` or `QuantizedKVCache`.
    /// Families that only support fp16 internally (Llama/Qwen/Mistral/Gemma/Phi/GLM) downcast at the
    /// boundary; passing a non-`KVCache` array to those families is a logic error caught by the engine.
    public let forward: (MLXArray, [KVCacheProtocol]?) -> MLXArray

    /// Multimodal forward pass — only set for native-multimodal models
    /// (Gemma 4). Arguments:
    /// `(tokens, caches, pixelValues?, audioMel?, audioValidMask?, mediaHash?)`
    /// where `audioMel` is `[1,T,128]` log-mel, `audioValidMask` is `[1,T]`
    /// bool (true = real audio). `mediaHash` (when non-nil) keys the
    /// per-model vision encoder cache; pass nil to bypass. Image and audio
    /// may be supplied together (combined native path).
    public let multimodalForward: ((MLXArray, [KVCacheProtocol]?, MLXArray?, MLXArray?, MLXArray?, String?) -> MLXArray)?

    /// Vocab size for validation
    public let vocabSize: Int
}

/// Detect the model family and load the appropriate architecture.
///
/// Reads config.json from the model directory, detects the architecture,
/// instantiates the model, quantizes if needed, and loads weights.
public func loadModel(from directory: URL) throws -> LoadedModel {
    try MLXMetalRuntime.validateForNativeInference()

    // Bound MLX's Metal buffer-recycling pool so it does not inflate the
    // process phys_footprint (the figure the release benchmark samples for
    // memory_ratio). Idempotent; safe to call on every load.
    MLXMemoryConfig.apply()

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
    if arch.contains("qwen2_5_vl") || arch.contains("qwen2vl")
        || modelType == "qwen2_5_vl" || modelType == "qwen2_vl"
    {
        // Qwen 2.5-VL native runtime. With `KRILL_NATIVE_QWEN25VL=1`
        // the loader builds the native Swift+MLX
        // `Qwen25VLForConditionalGeneration` (vision tower + 3D
        // mRoPE text tower + multimodal forward) and returns a
        // `LoadedModel`. Without the env var the family routes
        // through the `Qwen25VLEngine` Python bridge
        // (compatible_fallback tier) - the default until the native
        // path is validated against a real checkpoint.
        if ProcessInfo.processInfo.environment["KRILL_NATIVE_QWEN25VL"] == "1" {
            return try loadQwen25VL(configData: configData, directory: directory)
        }
        throw ModelLoadError.unsupportedArchitecture(
            "Qwen 2.5-VL runs through the multimodal bridge (compatible_fallback "
            + "tier). Use POST /api/chat or /v1/chat/completions with an image "
            + "attachment - the server routes VL manifests to Qwen25VLEngine. "
            + "The native Swift+MLX runtime is opt-in via KRILL_NATIVE_QWEN25VL=1 "
            + "until it is validated against a real checkpoint. Detected "
            + "arch=\(arch), model_type=\(modelType).")
    } else if arch.contains("forsequenceclassification") || arch.contains("crossencoder") {
        // Cross-encoder rerankers (BGE Reranker, Cohere Rerank,
        // etc.) are loaded through the dedicated `RerankEngine`,
        // not through the causal-LM dispatcher here. `loadModel`
        // is only the entry point for causal LMs; the reranker
        // family routes through Server / RerankEngine, so we
        // refuse to instantiate one as a causal LM. This keeps
        // /api/generate / /v1/chat callers from accidentally
        // dispatching a reranker through the chat path; they hit
        // a clear "not a causal LM" error here instead of a
        // garbage forward pass.
        throw ModelLoadError.unsupportedArchitecture(
            "Reranker is not a causal LM. Use POST /v1/rerank instead. "
            + "Detected arch=\(arch), model_type=\(modelType).")
    } else if arch.contains("gemma4") || modelType == "gemma4_text" || modelType == "gemma4" {
        return try loadGemma4(configData: configData, directory: directory)
    } else if arch.contains("chatglm") || arch.contains("glm") || modelType == "chatglm" {
        return try loadGLM(configData: configData, directory: directory)
    } else if arch.contains("qwen3moe") || modelType == "qwen3_moe" {
        // Qwen 3 MoE: native Swift+MLX runtime exists, but the
        // forward currently evaluates ALL experts on ALL tokens
        // (correctness-first; the scatter dispatch that runs only
        // the assigned experts per token is the follow-up). For
        // Qwen3-30B-A3B (128 experts, top-8) that is 16x more FFN
        // compute per layer than mlx-lm. Until the scatter
        // dispatch lands and a benchmark confirms parity, the
        // native path is OPT-IN via `KRILL_NATIVE_MOE=1`. Default
        // routes through the Python bridge so existing users keep
        // mlx-lm throughput.
        if ProcessInfo.processInfo.environment["KRILL_NATIVE_MOE"] == "1" {
            return try loadQwen3MoE(configData: configData, directory: directory)
        }
        throw ModelLoadError.unsupportedArchitecture(
            "Qwen 3 MoE native Swift+MLX runtime is opt-in until the "
            + "scatter-dispatch optimization lands. Set KRILL_NATIVE_MOE=1 "
            + "to enable; default routes through the MoE bridge "
            + "(compatible_fallback tier via mlx-lm). Use POST /api/chat "
            + "or /v1/chat/completions - the server routes MoE manifests "
            + "to MoEEngine. Detected arch=\(arch), model_type=\(modelType).")
    } else if arch.contains("mixtral") || arch.contains("qwen2moe")
        || arch.contains("olmoe")
        || modelType == "mixtral" || modelType == "qwen2_moe"
        || modelType == "olmoe"
        || arch.contains("deepseek") || modelType == "deepseek_v3"
    {
        // Remaining MoE families run through `MoEEngine` (Python
        // sidecar / mlx-lm), not the native causal-LM dispatcher.
        // `loadModel` is the entry point for native Swift+MLX
        // runtimes only. Refuse to instantiate these as a dense
        // causal LM so callers that hit /api/generate or
        // /v1/chat on a non-Qwen3-MoE MoE manifest get a clear
        // redirect instead of a garbage forward pass through the
        // dense text loader. Qwen 3 MoE is handled by the
        // dedicated arm above.
        throw ModelLoadError.unsupportedArchitecture(
            "Mixture-of-experts models run through the MoE bridge "
            + "(compatible_fallback tier). Use POST /api/chat or "
            + "/v1/chat/completions - the server routes MoE manifests to "
            + "MoEEngine. Native Swift+MLX router + expert dispatch landed "
            + "for Qwen 3 MoE in WS6; other MoE families are follow-ups. "
            + "Detected arch=\(arch), model_type=\(modelType).")
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
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadQwen(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(QwenConfig.self, from: configData)
    let model = QwenForCausalLM(config)
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        tieWordEmbeddings: config.tieWordEmbeddings)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "qwen",
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadQwen25VL(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Qwen25VLConfig.self, from: configData)
    let model = Qwen25VLForConditionalGeneration(config)
    // `loadWeights` quantizes every Linear when the config carries
    // quantization params and loads the safetensors. The Conv3d in
    // the vision patch embedding is not a Linear, so it stays fp16
    // (matching the checkpoint, which cannot quantize a Conv3d).
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        tieWordEmbeddings: config.tieWordEmbeddings)

    let mergeSize = config.vision.spatialMergeSize

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "qwen25vl",
        forward: { tokens, caches in
            // Text-only path: no image, no vision tower.
            model(tokens, pixelValues: nil, imageGridMerged: nil,
                  caches: caches as? [KVCache])
        },
        multimodalForward: { tokens, caches, pixelValues, _, _, _ in
            // pixelValues is the preprocessed per-patch batch
            // `[n_patches, T, ps, ps, C]`. The post-merge grid is
            // derived assuming a square image (n_patches is a
            // perfect square): gridFull = sqrt(n_patches),
            // gridMerged = gridFull / spatial_merge_size. Non-square
            // images need the grid threaded explicitly - that is
            // the remaining server-integration follow-up.
            guard let pixelValues else {
                return model(tokens, pixelValues: nil, imageGridMerged: nil,
                             caches: caches as? [KVCache])
            }
            let nPatches = pixelValues.dim(0)
            let gridFull = Int(Double(nPatches).squareRoot().rounded())
            let gridMerged = max(1, gridFull / mergeSize)
            return model(tokens, pixelValues: pixelValues,
                         imageGridMerged: (gridMerged, gridMerged),
                         caches: caches as? [KVCache])
        },
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
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
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
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
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
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadGemma4(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Gemma4Config.self, from: configData)

    // Parse image/audio token IDs from config
    let rawConfig = try JSONSerialization.jsonObject(with: configData) as? [String: Any]
    let imageTokenId = rawConfig?["image_token_id"] as? Int ?? 258880
    let audioTokenId = rawConfig?["audio_token_id"] as? Int ?? 258881
    let hasVisionConfig = rawConfig?["vision_config"] != nil

    // Use multimodal model if vision config is present
    if hasVisionConfig {
        let audioConfig = AudioConfig(from: rawConfig?["audio_config"] as? [String: Any])
        let model = Gemma4MultimodalModel(
            config, imageTokenId: imageTokenId, audioTokenId: audioTokenId,
            audioConfig: audioConfig)

        // Quantize language model layers (exclude PLE and vision/audio towers)
        if let q = config.quantization {
            quantize(model: model, groupSize: q.groupSize, bits: q.bits) { name, _ in
                let isLangModel = name.contains("language_model.")
                let isPLE = name.contains("per_layer_model_projection") || name.contains("per_layer_projection_norm")
                let isEmbedProj = name.contains("embed_vision.embedding_projection") || name.contains("embed_audio.embedding_projection")
                // Quantize: language model (except PLE) and embedding projections
                return (isLangModel && !isPLE) || isEmbedProj
            }
        }

        // Load ALL weights — key structure matches module hierarchy directly
        var flatWeights = try loadWeightArrays(from: directory)

        // Strip "model." prefix if present (some checkpoints use it)
        var cleaned: [String: MLXArray] = [:]
        for (key, value) in flatWeights {
            if key.hasPrefix("model.") && !key.hasPrefix("model.embed_tokens") {
                cleaned[String(key.dropFirst("model.".count))] = value
            } else {
                cleaned[key] = value
            }
        }
        flatWeights = cleaned

        // Tied embeddings: Gemma4 uses embed_tokens.as_linear() as the LM head.
        // No separate lm_head weights needed — the model reuses embed_tokens directly.

        // Audio conv weights ship in mlx-vlm/MLX channel-last layout already:
        // Conv2d `subsample_conv_projection.*.conv.weight` is
        // [C_out, kH, kW, C_in] (e.g. [128,3,3,1], [32,3,3,128]) and the
        // depthwise `lconv1d.depthwise_conv1d.weight` is
        // [C_out, kW, C_in/groups] ([1024,5,1]) — exactly what MLXNN.Conv2d
        // and MLX.conv1d expect. No transpose (a prior PyTorch-layout
        // transpose here corrupted these; it was dormant because the audio
        // tower was never instantiated).

        let tuples = flatWeights.map { ($0.key, $0.value) }
        let nested = ModuleParameters.unflattened(tuples)
        try model.update(parameters: nested, verify: [])

        return LoadedModel(
            module: model,
            numLayers: config.numHiddenLayers,
            family: "gemma4",
            forward: { tokens, caches in model(tokens, caches: caches) },
            multimodalForward: { tokens, caches, imageEmb, audioMel, audioMask, mediaHash in
                model(tokens, caches: caches,
                      pixelValues: imageEmb,
                      audioMel: audioMel, audioValidMask: audioMask,
                      mediaHash: mediaHash)
            },
            vocabSize: config.vocabSize
        )
    }

    // Fallback: text-only Gemma4 (no vision_config in checkpoint)
    let model = Gemma4ForCausalLM(config)
    if let q = config.quantization {
        quantize(model: model, groupSize: q.groupSize, bits: q.bits) { name, _ in
            !name.contains("per_layer_model_projection") && !name.contains("per_layer_projection_norm")
        }
    }

    var flatWeights = try loadWeightArrays(from: directory)
    var stripped: [String: MLXArray] = [:]
    for (key, value) in flatWeights {
        if key.hasPrefix("language_model.") {
            stripped[String(key.dropFirst("language_model.".count))] = value
        }
    }
    if !stripped.isEmpty { flatWeights = stripped }

    // Tied embeddings: Gemma4 uses embed_tokens.as_linear() as the LM head.

    let tuples = flatWeights.map { ($0.key, $0.value) }
    let nested = ModuleParameters.unflattened(tuples)
    try model.update(parameters: nested, verify: [])

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "gemma4",
        forward: { tokens, caches in model(tokens, caches: caches) },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadQwen3MoE(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Qwen3MoEConfig.self, from: configData)
    let model = Qwen3MoEForCausalLM(config)
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        tieWordEmbeddings: config.tieWordEmbeddings)

    // `family` returned here must round-trip through
    // `ModelFamily(rawValue:)` so that `InferenceEngine.capabilities`
    // can look up the declared capability set, and so that the tool
    // template selector picks the right format. We return "moe" so
    // the capability lookup hits `.moe` cleanly; the tool format
    // adapter maps `.moe` to `.qwen` (the only native MoE today is
    // Qwen 3 MoE, which uses the Qwen chat / tool template).
    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "moe",
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        multimodalForward: nil,
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
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
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
