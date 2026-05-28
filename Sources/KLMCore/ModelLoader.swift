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

    /// Optional prefill-specialized forward returning logits sliced
    /// to the last position only - shape `[B, 1, vocab]` rather than
    /// `[B, L, vocab]`. Bit-exact for the sampled token (the KV
    /// cache is filled by the attention layers above the head). The
    /// engine prefers this on prefill to drop the
    /// `vocabSize * hidden` matmul over the L-1 unused rows; falls
    /// back to `forward` when nil (text-multimodal paths that have
    /// not yet been wired in).
    public let prefillForward: ((MLXArray, [KVCacheProtocol]?) -> MLXArray)?

    /// Multimodal forward pass - only set for native-multimodal models
    /// (Gemma 4). Arguments:
    /// `(tokens, caches, pixelValues?, audioMel?, audioValidMask?, mediaHash?)`
    /// where `audioMel` is `[1,T,128]` log-mel, `audioValidMask` is `[1,T]`
    /// bool (true = real audio). `mediaHash` (when non-nil) keys the
    /// per-model vision encoder cache; pass nil to bypass. Image and audio
    /// may be supplied together (combined native path).
    public let multimodalForward: ((MLXArray, [KVCacheProtocol]?, MLXArray?, MLXArray?, MLXArray?, String?) -> MLXArray)?

    /// Optional last-token-only multimodal prefill. Same argument
    /// shape as `multimodalForward`, but returns logits sliced to
    /// the last position - bit exact for the sampled token (the
    /// KV cache is populated by the attention layers above the
    /// lm_head) and skips the per-position vocab matmul over the
    /// L-1 unused rows. The engine prefers this on multimodal
    /// prefill when present; falls back to `multimodalForward`.
    public let multimodalPrefillForward: ((MLXArray, [KVCacheProtocol]?, MLXArray?, MLXArray?, MLXArray?, String?) -> MLXArray)?

    /// Vocab size for validation
    public let vocabSize: Int

    /// Explicit memberwise init so `prefillForward` and
    /// `multimodalForward` can default to `nil` for callers that
    /// only set the bare `forward` (the Swift-synthesized
    /// memberwise init does not accept defaults).
    public init(
        module: Module,
        numLayers: Int,
        family: String,
        forward: @escaping (MLXArray, [KVCacheProtocol]?) -> MLXArray,
        prefillForward: ((MLXArray, [KVCacheProtocol]?) -> MLXArray)? = nil,
        multimodalForward: ((MLXArray, [KVCacheProtocol]?, MLXArray?, MLXArray?, MLXArray?, String?) -> MLXArray)? = nil,
        multimodalPrefillForward: ((MLXArray, [KVCacheProtocol]?, MLXArray?, MLXArray?, MLXArray?, String?) -> MLXArray)? = nil,
        vocabSize: Int
    ) {
        self.module = module
        self.numLayers = numLayers
        self.family = family
        self.forward = forward
        self.prefillForward = prefillForward
        self.multimodalForward = multimodalForward
        self.multimodalPrefillForward = multimodalPrefillForward
        self.vocabSize = vocabSize
    }
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
        // Qwen 2.5-VL native Swift+MLX runtime: vision tower + 3D
        // mRoPE text tower + multimodal forward. WS5 retired the
        // `Qwen25VLEngine` Python bridge - the native path is now
        // the only runtime for this family.
        return try loadQwen25VL(configData: configData, directory: directory)
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
        // Qwen 3 MoE: native Swift+MLX runtime is the DEFAULT. Expert
        // dispatch is a single `gatherQuantizedMM` per projection
        // (`Qwen3SwitchGLU`, mirroring Gemma 4's PR #82) -- no Swift
        // per-expert loop, no per-layer host sync. Decode benches 2.7x
        // faster than the old scatter dispatch (24 -> 66 tok/s on
        // 30B-A3B, PR #85), and the #87 sort path recovers long-prompt
        // prefill (229 -> 536 tok/s) so prefill is at parity too -- the
        // precondition the opt-in gate waited on. Native is now the only
        // tested Qwen3-MoE path on this build.
        //
        // `KRILL_NATIVE_MOE=0` is the opt-out: it forces the legacy
        // mlx-lm MoE bridge (compatible_fallback tier) for one release,
        // for anyone who needs to fall back. The server routes that case
        // to `MoEEngine`; native loading is refused here so the bridge
        // handler takes over.
        if ProcessInfo.processInfo.environment["KRILL_NATIVE_MOE"] == "0" {
            throw ModelLoadError.unsupportedArchitecture(
                "Qwen 3 MoE native runtime disabled via KRILL_NATIVE_MOE=0; "
                + "routing through the legacy MoE bridge (mlx-lm, "
                + "compatible_fallback tier). Use POST /api/chat or "
                + "/v1/chat/completions - the server routes MoE manifests to "
                + "MoEEngine. Unset KRILL_NATIVE_MOE for the native default. "
                + "Detected arch=\(arch), model_type=\(modelType).")
        }
        return try loadQwen3MoE(configData: configData, directory: directory)
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
    } else if let specialized = detectSpecializedModelType(arch: arch, modelType: modelType) {
        // WS7 specialized model types (ASR / TTS / diffusion / video /
        // OCR). KrillLM has no native runtime for these; rejecting
        // here with a specific error is the roadmap's "Unsupported
        // tier" - far better than mis-loading them via the Llama
        // fallback below and emitting a garbage forward pass.
        throw ModelLoadError.specializedModelUnsupported(
            "KrillLM does not support \(specialized.displayName) models. "
            + "These are WS7 specialized model types with no native "
            + "runtime in this build (of the WS7 types, only rerankers "
            + "have shipped - use POST /v1/rerank for those; see "
            + "docs/workstreams/WS7_SPECIALIZED_MODEL_TYPES.md). KrillLM "
            + "serves causal text LMs, Gemma 4 vision/audio, embeddings, "
            + "and rerankers. Detected arch=\(arch), model_type=\(modelType).")
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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadQwen25VL(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Qwen25VLConfig.self, from: configData)
    let model = Qwen25VLForConditionalGeneration(config)

    // The mlx-community Qwen2.5-VL 4-bit checkpoints quantize ONLY
    // the language model; the vision tower ships fp16 (verified
    // from the safetensors header - there are no `.scales` tensors
    // under `vision_tower.*`). Quantizing the vision tower's
    // Linears here would make `update(parameters:)` expect packed
    // 4-bit weights the checkpoint does not provide. Restrict the
    // quantization predicate to `language_model.*`, mirroring how
    // `loadGemma4` excludes its vision / PLE towers.
    if let q = config.quantization {
        quantize(model: model, groupSize: q.groupSize, bits: q.bits) { name, _ in
            name.contains("language_model")
        }
    }
    let flatWeights = try loadWeightArrays(from: directory)
    let tuples = flatWeights.map { ($0.key, $0.value) }
    let nested = ModuleParameters.unflattened(tuples)
    // `verify: []` tolerates the checkpoint omitting `lm_head.*`
    // when embeddings are tied (the model then has no lm_head
    // module either, so there is nothing to assign).
    try model.update(parameters: nested, verify: [])

    // `family` must round-trip through `ModelFamily(rawValue:)` so
    // `InferenceEngine.capabilities` and the tool-template selector
    // resolve correctly - `ModelFamily.qwen25vl.rawValue` is
    // `"qwen2_5_vl"`, not `"qwen25vl"`.
    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "qwen2_5_vl",
        forward: { tokens, caches in
            // Text-only path: no image, no vision tower.
            model(tokens, pixelValues: nil, imageGridMerged: nil,
                  caches: caches as? [KVCache])
        },
        // Unreachable sentinel. `InferenceEngine.generate` intercepts
        // every Qwen 2.5-VL request (`loadedModel.module as?
        // Qwen25VLForConditionalGeneration`) and routes it through
        // the native `Qwen25VLRuntime` driver, which threads the real
        // `(gridH, gridW)` grid and the decode-step mRoPE offset -
        // neither of which the generic six-argument closure can carry
        // (a non-square grid is not recoverable from the patch count).
        // This closure exists ONLY so `multimodalForward != nil`
        // keeps the `.visionInput` capability advertised. If it is
        // ever actually invoked, the VL routing has regressed; fail
        // loudly rather than run a wrong-grid forward.
        multimodalForward: { _, _, _, _, _, _ in
            fatalError(
                "Qwen 2.5-VL must run via Qwen25VLRuntime, not the "
                + "generic multimodalForward closure. The VL "
                + "interception in InferenceEngine.generate has "
                + "regressed.")
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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
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

        // Quantization: skip PLE projection and the non-LM towers; honor
        // per-module overrides (26B-A4B ships 4-bit experts + 8-bit dense
        // MLP + 8-bit router projection, encoded as per-module entries
        // in `config.quantization`). The closure-based form lets each
        // module land on its own (groupSize, bits, mode).
        //
        // Also skip the MoE experts (`experts.switch_glu.*`): each
        // `Gemma4QuantizedSwitchedLinear` already owns its quantized
        // parameter tensors at the right shape from
        // `Gemma4MoEExpertsHolder.init`, and MLX's `quantize` walks
        // leaf modules (Linear / Embedding). The switched linears are
        // not `Linear`, so without this skip the call would no-op on
        // them anyway -- the explicit `nil` makes the intent obvious
        // and saves the path-string match.
        if let q = config.quantization {
            quantize(model: model) { path, _ in
                let isLangModel = path.contains("language_model.")
                let isPLE = path.contains("per_layer_model_projection") || path.contains("per_layer_projection_norm")
                let isEmbedProj = path.contains("embed_vision.embedding_projection") || path.contains("embed_audio.embedding_projection")
                let isMoEExpert = path.contains(".experts.switch_glu.")
                let shouldQuant = (isLangModel && !isPLE && !isMoEExpert) || isEmbedProj
                guard shouldQuant else { return nil }
                let eff = q.effective(for: path)
                return (eff.groupSize, eff.bits, .affine)
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

        // The 26B-A4B checkpoint ships its experts as stacked tensors
        // under `experts.switch_glu.{gate_proj,up_proj,down_proj}.*`.
        // The module hierarchy now mirrors that shape exactly via
        // `Gemma4MoEExpertsHolder.switchGLU` -- no unpacking required.
        // (The original implementation rewrote those keys into
        // per-expert `.experts.{e}.{proj}.{field}` form to match an
        // older `[Gemma4MoEExpert]` array module; that path had a
        // per-layer host sync that dominated decode and is gone.)

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
            // Text-only prefill on the multimodal model goes through
            // the text-only overload (no image / audio), which
            // accepts the same `lastTokenOnly` flag the Gemma 4 LM
            // grew below.
            prefillForward: { tokens, caches in
                model(tokens, caches: caches, lastTokenOnly: true)
            },
            multimodalForward: { tokens, caches, imageEmb, audioMel, audioMask, mediaHash in
                model(tokens, caches: caches,
                      pixelValues: imageEmb,
                      audioMel: audioMel, audioValidMask: audioMask,
                      mediaHash: mediaHash)
            },
            multimodalPrefillForward: { tokens, caches, imageEmb, audioMel, audioMask, mediaHash in
                // Multimodal prefill returns logits at the last
                // position only - the engine samples one token from
                // them, so the slice is bit exact and skips the
                // per-position vocab matmul over the L-1 unused
                // rows (Gemma 4 vocab is 262 144). The KV cache is
                // filled inside `languageModel(...)` before the head,
                // so the cache state is identical to the un-sliced
                // path.
                model(tokens, caches: caches,
                      pixelValues: imageEmb,
                      audioMel: audioMel, audioValidMask: audioMask,
                      mediaHash: mediaHash,
                      lastTokenOnly: true)
            },
            vocabSize: config.vocabSize
        )
    }

    // Fallback: text-only Gemma4 (no vision_config in checkpoint)
    let model = Gemma4ForCausalLM(config)
    if let q = config.quantization {
        quantize(model: model) { path, _ in
            // Skip PLE in the text-only path too. After `language_model.`
            // is stripped from the weight keys, the model's module path
            // is e.g. `model.layers.0.mlp.gate_proj`; override keys in
            // `q` may still carry the `language_model.` prefix from the
            // raw checkpoint config. Look up the de-prefixed path first
            // (this is what the multimodal path also sees because that
            // path does not strip the prefix); if no override exists
            // there, try the prefixed form so 26B-A4B's `language_model.
            // model.layers.0.mlp.gate_proj` override still binds when
            // the same checkpoint is loaded text-only.
            if path.contains("per_layer_model_projection")
                || path.contains("per_layer_projection_norm") { return nil }
            let eff: (groupSize: Int, bits: Int)
            if q.moduleOverrides[path] != nil {
                eff = q.effective(for: path)
            } else if q.moduleOverrides["language_model." + path] != nil {
                eff = q.effective(for: "language_model." + path)
            } else {
                eff = (q.groupSize, q.bits)
            }
            return (eff.groupSize, eff.bits, .affine)
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
    // The MoE experts (when present) match the in-checkpoint
    // `experts.switch_glu.*` shape directly; no key rewrite needed.

    let tuples = flatWeights.map { ($0.key, $0.value) }
    let nested = ModuleParameters.unflattened(tuples)
    try model.update(parameters: nested, verify: [])

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "gemma4",
        forward: { tokens, caches in model(tokens, caches: caches) },
        prefillForward: { tokens, caches in
            model(tokens, caches: caches, lastTokenOnly: true)
        },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

private func loadQwen3MoE(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Qwen3MoEConfig.self, from: configData)
    let model = Qwen3MoEForCausalLM(config)
    // No key rewrite: the mlx-community checkpoint ships its experts as
    // stacked `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*` tensors,
    // and the module hierarchy now mirrors that shape directly via
    // `Qwen3MoESparseMLP.switchMLP` (a `Qwen3SwitchGLU` of stacked
    // `Qwen3QuantizedSwitchedLinear`s), dispatched in a single
    // `gatherQuantizedMM` per projection. The earlier path unpacked
    // those keys into per-expert `mlp.experts.{e}.*` form to feed a
    // Swift `for` loop with a per-layer host sync; that stall dominated
    // decode and is gone (mirrors Gemma 4's PR #82 rewrite). The router
    // `mlp.gate` stays a `Linear` and is quantized by `loadWeights`'
    // quantize pass (8-bit per the checkpoint's per-module override);
    // the switched linears are not `Linear`, so that pass skips them and
    // they load their quantized parameters directly.
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        tieWordEmbeddings: config.tieWordEmbeddings,
        strictVerify: true)

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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
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
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        multimodalForward: nil,
        vocabSize: config.vocabSize
    )
}

// MARK: - Errors

public enum ModelLoadError: Error, CustomStringConvertible {
    case invalidConfig(String)
    case unsupportedArchitecture(String)
    /// A WS7 specialized model type (ASR / TTS / diffusion / video /
    /// OCR) with no native runtime in this build. See
    /// `SpecializedModelType` / `detectSpecializedModelType`.
    case specializedModelUnsupported(String)

    public var description: String {
        switch self {
        case .invalidConfig(let msg): return "Invalid config: \(msg)"
        case .unsupportedArchitecture(let arch): return "Unsupported architecture: \(arch)"
        case .specializedModelUnsupported(let msg): return "Unsupported model type: \(msg)"
        }
    }
}
