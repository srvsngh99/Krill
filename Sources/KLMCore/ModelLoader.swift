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

    /// Optional batched ragged-decode forward (Stage B). Arguments:
    /// `(tokens [R,1], caches, mask, rowOffsets)` -> logits `[R, 1, vocab]`.
    /// One new token per row, each rotated at its own next position
    /// (`rowOffsets[r]`), attending under an explicit per-row additive mask
    /// that hides each row's left-padded prefix in the stacked KV cache. Set
    /// for every family that supports true KV-batched concurrent decode: Llama
    /// and Qwen 2.5/3 dense, Gemma 4 (dense e2b/e4b/12b and 26B-A4B MoE), and
    /// Qwen3 MoE. nil means the family falls back to serialized generation.
    /// fp16 `KVCache` only.
    public let batchedDecodeForward: ((MLXArray, [KVCache], MLXArray, [Int]) -> MLXArray)?

    /// Optional int8-quantized batched ragged-decode forward (Stage C4). Same
    /// `(tokens [R,1], caches, mask, rowOffsets) -> [R,1,vocab]` contract as
    /// `batchedDecodeForward`, but the stacked KV caches are `QuantizedKVCache`
    /// (uint8 storage, dequantized inside attention). Set ONLY for the families
    /// whose attention accepts `KVCacheProtocol` and therefore support int8 KV
    /// serially - currently Gemma 4 (the plain-causal Llama/Qwen/MoE attentions
    /// are hard-typed to fp16 `KVCache`). nil means the engine batches this
    /// model with fp16 KV even when `KRILL_KV_CACHE_DTYPE=int8` is set, matching
    /// its serial behavior (those families already fall back to fp16 serially).
    public let batchedDecodeForwardQuantized: ((MLXArray, [QuantizedKVCache], MLXArray, [Int]) -> MLXArray)?

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
        batchedDecodeForward: ((MLXArray, [KVCache], MLXArray, [Int]) -> MLXArray)? = nil,
        batchedDecodeForwardQuantized: ((MLXArray, [QuantizedKVCache], MLXArray, [Int]) -> MLXArray)? = nil,
        vocabSize: Int
    ) {
        self.module = module
        self.numLayers = numLayers
        self.family = family
        self.forward = forward
        self.prefillForward = prefillForward
        self.multimodalForward = multimodalForward
        self.multimodalPrefillForward = multimodalPrefillForward
        self.batchedDecodeForward = batchedDecodeForward
        self.batchedDecodeForwardQuantized = batchedDecodeForwardQuantized
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

    // Detect the architecture and dispatch to the matching family loader.
    // Detection is a declarative, ordered table (`architectureRules`): the
    // first rule whose matcher claims the (arch, model_type) pair wins, so the
    // table is ordered specific-before-generic. Adding a family is a new table
    // row, not a new branch hand-placed in an if/else chain. The hot decode
    // path is untouched -- this runs once, at load.
    let architectures = configJSON["architectures"] as? [String] ?? []
    let modelType = configJSON["model_type"] as? String ?? ""
    let arch = architectures.first?.lowercased() ?? ""

    guard let rule = architectureRules.first(where: { $0.matches(arch, modelType) }) else {
        // Unreachable: the table's last rule (`fallback`) matches any input.
        throw ModelLoadError.unsupportedArchitecture(
            "No architecture rule matched arch=\(arch), model_type=\(modelType).")
    }
    switch rule.action {
    case .load(let loader):
        return try loader(configData, directory)
    case .reject(let makeError):
        throw makeError(arch, modelType)
    }
}

// MARK: - Architecture detection table

/// What an `ArchitectureRule` does once it claims a config: load the
/// checkpoint as a family, or deliberately refuse it.
enum ArchitectureAction: Sendable {
    /// Instantiate + load the checkpoint as this family.
    case load(@Sendable (_ configData: Data, _ directory: URL) throws -> LoadedModel)
    /// Refuse this architecture (not a causal LM, or no native runtime),
    /// building the error from the detected `arch` + `model_type`.
    case reject(@Sendable (_ arch: String, _ modelType: String) -> ModelLoadError)
}

/// One architecture-detection rule: a named matcher plus the action it
/// dispatches to. Rules live in an ordered table evaluated first-match-wins,
/// so the specific-before-generic ordering that was previously an implicit
/// invariant of a hand-written if/else chain is now explicit and testable.
struct ArchitectureRule: Sendable {
    /// Stable identifier for the matched architecture (diagnostics + tests).
    let id: String
    /// True when this rule claims the lowercased `arch` / raw `model_type`.
    let matches: @Sendable (_ arch: String, _ modelType: String) -> Bool
    /// Load or reject.
    let action: ArchitectureAction
}

/// The architecture-detection table, ordered specific-before-generic. The
/// last rule (`fallback`) matches any input, so `loadModel`'s lookup never
/// fails. Order is load-bearing: e.g. `qwen3_moe` must precede the generic
/// `qwen` rule, or a MoE checkpoint would load as a dense Qwen.
let architectureRules: [ArchitectureRule] = [
    // Qwen 2.5-VL native Swift+MLX runtime: vision tower + 3D mRoPE text
    // tower + multimodal forward. WS5 retired the `Qwen25VLEngine` Python
    // bridge - the native path is now the only runtime for this family.
    ArchitectureRule(
        id: "qwen2_5_vl",
        matches: { arch, mt in
            arch.contains("qwen2_5_vl") || arch.contains("qwen2vl")
                || mt == "qwen2_5_vl" || mt == "qwen2_vl"
        },
        action: .load { try loadQwen25VL(configData: $0, directory: $1) }),

    // Cross-encoder rerankers (BGE Reranker, Cohere Rerank, etc.) load
    // through the dedicated `RerankEngine`, not the causal-LM dispatcher.
    // `loadModel` is only the entry point for causal LMs, so we refuse to
    // instantiate a reranker as one. This keeps /api/generate / /v1/chat
    // callers from dispatching a reranker through the chat path; they hit a
    // clear "not a causal LM" error here instead of a garbage forward pass.
    ArchitectureRule(
        id: "reranker",
        matches: { arch, _ in
            arch.contains("forsequenceclassification") || arch.contains("crossencoder")
        },
        action: .reject { arch, mt in
            .unsupportedArchitecture(
                "Reranker is not a causal LM. Use POST /v1/rerank instead. "
                + "Detected arch=\(arch), model_type=\(mt).")
        }),

    // LLaVA-1.5: CLIP ViT vision tower + multi-modal projector + Llama text
    // backbone. Must precede the generic dense rules (its text stack is Llama).
    // The llava-next / llava-bunny variants are NOT supported (different image
    // handling); only model_type "llava".
    ArchitectureRule(
        id: "llava",
        matches: { arch, mt in
            mt == "llava" || arch.contains("llavaforconditionalgeneration")
        },
        action: .load { try loadLlava(configData: $0, directory: $1) }),

    // Llama-3.2-Vision (mllama): tiled ViT vision tower + multi-modal projector
    // + a Llama text decoder whose cross_attention_layers attend to the vision
    // features. Must precede the generic llama rule (its arch is
    // `MllamaForConditionalGeneration`, which does not contain "llama" as the
    // dispatch substring would need, and model_type is "mllama").
    ArchitectureRule(
        id: "mllama",
        matches: { arch, mt in
            arch.contains("mllamaforconditionalgeneration") || mt == "mllama"
        },
        action: .load { try loadLlamaVision(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "gemma4",
        matches: { arch, mt in
            arch.contains("gemma4") || mt == "gemma4_text" || mt == "gemma4"
        },
        action: .load { try loadGemma4(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "glm",
        matches: { arch, mt in
            arch.contains("chatglm") || arch.contains("glm") || mt == "chatglm"
        },
        action: .load { try loadGLM(configData: $0, directory: $1) }),

    // Qwen 3 MoE: native Swift+MLX runtime is the DEFAULT. Expert dispatch is
    // a single `gatherQuantizedMM` per projection (shared `MoESwitchGLU`) --
    // no Swift per-expert loop, no per-layer host sync. Decode benches 2.7x
    // faster than the old scatter dispatch (24 -> 66 tok/s on 30B-A3B, PR
    // #85), and the #87 sort path recovers long-prompt prefill (229 -> 536
    // tok/s) so prefill is at parity too. Native is the only Qwen3-MoE path:
    // the `KRILL_NATIVE_MOE=0` opt-out and the mlx-lm MoE bridge it routed to
    // were removed once every MoE family went native.
    ArchitectureRule(
        id: "qwen3_moe",
        matches: { arch, mt in arch.contains("qwen3moe") || mt == "qwen3_moe" },
        action: .load { try loadQwen3MoE(configData: $0, directory: $1) }),

    // Mixtral: native Swift+MLX sparse-MoE runtime. Mistral attention + a
    // `block_sparse_moe` router and `gatherQuantizedMM` SwitchGLU expert
    // dispatch, mirroring the Qwen 3 MoE path (PR #85/#87). Replaces the
    // legacy mlx-lm MoE bridge for this family.
    ArchitectureRule(
        id: "mixtral",
        matches: { arch, mt in arch.contains("mixtral") || mt == "mixtral" },
        action: .load { try loadMixtral(configData: $0, directory: $1) }),

    // Qwen 2 MoE: native Swift+MLX sparse-MoE runtime. Dense Qwen 2 attention
    // (QKV bias, no q/k-norm) + a `mlp` router, a `gatherQuantizedMM`
    // SwitchGLU for the routed experts, and an always-on sigmoid-gated shared
    // expert. Replaces the mlx-lm bridge for this family.
    ArchitectureRule(
        id: "qwen2_moe",
        matches: { arch, mt in arch.contains("qwen2moe") || mt == "qwen2_moe" },
        action: .load { try loadQwen2MoE(configData: $0, directory: $1) }),

    // OLMoE: native Swift+MLX sparse-MoE runtime. GQA attention with a
    // whole-projection q/k RMSNorm (OLMoE's delta vs Qwen 3's per-head norm)
    // + an `mlp` router and `gatherQuantizedMM` SwitchGLU; no shared expert.
    // Replaces the mlx-lm bridge for this family.
    ArchitectureRule(
        id: "olmoe",
        matches: { arch, mt in arch.contains("olmoe") || mt == "olmoe" },
        action: .load { try loadOLMoE(configData: $0, directory: $1) }),

    // DeepSeek-V2 / V2-Lite: native Swift+MLX runtime. MLA attention (low-rank
    // Q/KV bottleneck, split rope/nope head dims) + YaRN RoPE + a
    // `gatherQuantizedMM` SwitchGLU for the routed experts, an always-on
    // shared expert, a dense-layer prefix (first_k_dense_replace), and softmax
    // / group_limited_greedy gating. `loadDeepSeek` rejects the V3
    // absorbed-MLA layout with a clear message (docs/BACKLOG.md); the V3
    // `noaux_tc` gating is implemented in the shared gate. Replaces the mlx-lm
    // bridge for this family.
    ArchitectureRule(
        id: "deepseek",
        matches: { arch, mt in
            arch.contains("deepseek") || mt == "deepseek_v2" || mt == "deepseek_v3"
        },
        action: .load { try loadDeepSeek(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "llama",
        matches: { arch, mt in arch.contains("llama") || mt == "llama" },
        action: .load { try loadLlama(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "qwen",
        matches: { arch, mt in arch.contains("qwen") || mt.hasPrefix("qwen") },
        action: .load { try loadQwen(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "mistral",
        matches: { arch, mt in arch.contains("mistral") || mt == "mistral" },
        action: .load { try loadMistral(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "gemma",
        matches: { arch, mt in arch.contains("gemma") || mt.hasPrefix("gemma") },
        action: .load { try loadGemma(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "phi",
        matches: { arch, mt in arch.contains("phi") || mt.hasPrefix("phi") },
        action: .load { try loadPhi(configData: $0, directory: $1) }),

    // WS7 specialized model types (ASR / TTS / diffusion / video / OCR).
    // KrillLM has no native runtime for these; rejecting here with a specific
    // error is the roadmap's "Unsupported tier" - far better than mis-loading
    // them via the Llama fallback below and emitting a garbage forward pass.
    ArchitectureRule(
        id: "specialized",
        matches: { arch, mt in detectSpecializedModelType(arch: arch, modelType: mt) != nil },
        action: .reject { arch, mt in
            let specialized = detectSpecializedModelType(arch: arch, modelType: mt)
            let name = specialized?.displayName ?? "specialized"
            return .specializedModelUnsupported(
                "KrillLM does not support \(name) models. "
                + "These are WS7 specialized model types with no native "
                + "runtime in this build (of the WS7 types, only rerankers "
                + "have shipped - use POST /v1/rerank for those; see "
                + "docs/workstreams/WS7_SPECIALIZED_MODEL_TYPES.md). KrillLM "
                + "serves causal text LMs, Gemma 4 vision/audio, embeddings, "
                + "and rerankers. Detected arch=\(arch), model_type=\(mt).")
        }),

    // Fallback: most checkpoints in the wild are Llama-architecture, so an
    // unrecognized config is loaded as Llama. Matches any input, so it must
    // stay last.
    ArchitectureRule(
        id: "fallback",
        matches: { _, _ in true },
        action: .load { try loadLlama(configData: $0, directory: $1) }),
]

/// The id of the architecture rule that claims this `(architectures,
/// model_type)` pair (e.g. `"qwen3_moe"`, `"deepseek"`, `"fallback"`). Pure:
/// no disk, no weights, no model instantiation. Exposed so tests can pin the
/// detection table's ordering without a real checkpoint -- the regression net
/// for "a generic rule shadows a specific one".
public func detectedArchitectureID(architectures: [String], modelType: String) -> String {
    let arch = architectures.first?.lowercased() ?? ""
    // Force-unwrap is safe: the table's last rule matches any input.
    return architectureRules.first(where: { $0.matches(arch, modelType) })!.id
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
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
        vocabSize: config.vocabSize
    )
}

private func loadLlava(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(LlavaConfig.self, from: configData)
    let model = LlavaForCausalLM(config)
    // Real LLaVA 4-bit checkpoints quantize the language + vision Linears
    // (the patch-embed Conv2d stays fp; the quantize predicate only matches
    // Linear/Embedding). The tiny synthetic parity fixture is unquantized.
    // `verify: []` tolerates tied-embedding checkpoints that omit lm_head.
    if let q = config.quantization {
        quantize(model: model, groupSize: q.groupSize, bits: q.bits) { name, module in
            (module is Linear || module is Embedding) && !name.contains("patch_embedding")
        }
    }
    var flatWeights = try loadWeightArrays(from: directory)
    // CLIP's patch embed is a Conv2d. HF checkpoints ship its weight in
    // PyTorch `[out, in, kH, kW]` layout; MLX Conv2d wants `[out, kH, kW, in]`.
    // Transpose when the tensor is not already in MLX layout (mlx-community
    // checkpoints are pre-sanitized; raw HF ones are not). Mirrors mlx-vlm's
    // VisionModel.sanitize / check_array_shape.
    let patchKey = "vision_tower.vision_model.embeddings.patch_embedding.weight"
    if let w = flatWeights[patchKey], w.ndim == 4 {
        let s = w.shape
        let isMLXLayout = s[0] >= s[1] && s[0] >= s[2] && s[1] == s[2]
        if !isMLXLayout { flatWeights[patchKey] = w.transposed(0, 2, 3, 1) }
    }
    let nested = ModuleParameters.unflattened(flatWeights.map { ($0.key, $0.value) })
    // Strict verify: llava-1.5 has a real lm_head (untied), so every model
    // parameter must be set by the checkpoint and no key may go unused -- a
    // mismatch is a load bug, not something to swallow silently.
    try model.update(parameters: nested, verify: [.all])

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "llava",
        // Text-only turns run straight through the Llama backbone.
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        // Prefill slices to the last position before the vocab projection (the
        // sampler reads only that row), so the long image prefill does not pay
        // a full `[1, L, hidden] -> [1, L, vocab]` lm_head matmul over ~L-1
        // unused rows. Matches every dense family's prefill path.
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        // Multimodal: the engine passes the preprocessed `[1, C, H, W]` pixel
        // tensor as the third argument; the model embeds, splices the projected
        // CLIP features at the image-token positions, and runs the text stack.
        multimodalForward: { tokens, caches, pixelValues, _, _, _ in
            guard let pixelValues else {
                return model(tokens, caches: caches as? [KVCache])
            }
            return model(tokens, pixelValues: pixelValues, caches: caches as? [KVCache])
        },
        // Last-token-only multimodal prefill: same lm_head saving as the text
        // path, on the (long, image-bearing) prompt forward.
        multimodalPrefillForward: { tokens, caches, pixelValues, _, _, _ in
            guard let pixelValues else {
                return model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
            }
            return model(
                tokens, pixelValues: pixelValues, caches: caches as? [KVCache],
                lastTokenOnly: true)
        },
        vocabSize: config.vocabSize
    )
}

private func loadLlamaVision(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Llama32VisionConfig.self, from: configData)
    let model = Llama32VisionForCausalLM(config)
    // Quantize the Linear layers and the text token embedding. The vision
    // patch-embed Conv2d, the raw gate / position parameters, and the vision
    // tower's small aspect-ratio / tile / position Embedding tables stay fp
    // (those tables are typically NOT quantized in 4-bit dumps; quantizing them
    // would mismatch a real checkpoint's fp weights at strict load). The
    // synthetic parity fixture is unquantized; the exact real-4bit-checkpoint
    // quantization layout is validated when the image-serving follow-up runs a
    // real checkpoint on a larger box (the 11B run is RAM-blocked here).
    if let q = config.quantization {
        quantize(model: model, groupSize: q.groupSize, bits: q.bits) { name, module in
            guard module is Linear || module is Embedding else { return false }
            if name.contains("patch_embedding") { return false }
            // Skip the vision tower's Embedding tables (tile / aspect-ratio /
            // position), keep the text `embed_tokens`.
            if module is Embedding && name.contains("vision_tower") { return false }
            return true
        }
    }
    var flatWeights = try loadWeightArrays(from: directory)
    // HF ships the vision tower under `vision_model.*`; our module key is
    // `vision_tower.*` (mirrors mlx-vlm's sanitize rename). Also drop the
    // precomputed rotary inv_freq / position_ids buffers mlx-vlm discards.
    var renamed: [String: MLXArray] = [:]
    for (k, v) in flatWeights {
        if k.contains("rotary_emb.inv_freq") || k.contains("position_ids") { continue }
        var key = k
        if key.hasPrefix("vision_model.") {
            key = "vision_tower." + key.dropFirst("vision_model.".count)
        }
        renamed[key] = v
    }
    flatWeights = renamed
    // The vision patch embed is a Conv2d: PyTorch `[out, in, kH, kW]` ->
    // MLX `[out, kH, kW, in]` (transpose when not already in MLX layout).
    let patchKey = "vision_tower.patch_embedding.weight"
    if let w = flatWeights[patchKey], w.ndim == 4 {
        let s = w.shape
        let isMLXLayout = s[0] >= s[1] && s[0] >= s[2] && s[1] == s[2]
        if !isMLXLayout { flatWeights[patchKey] = w.transposed(0, 2, 3, 1) }
    }
    let nested = ModuleParameters.unflattened(flatWeights.map { ($0.key, $0.value) })
    // Strict verify: mllama has an untied lm_head, so every model parameter must
    // be set and no checkpoint key may go unused.
    try model.update(parameters: nested, verify: [.all])

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "llama_vision",
        // Text-only turns run through the decoder (the cross-attention layers
        // contribute ~nothing through their zero-initialized gates). Image
        // serving (tile preprocessing + a cross-KV decode driver) is wired in a
        // follow-up; the model's image forward is reachable via `module`.
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        multimodalForward: nil,
        vocabSize: config.textConfig.vocabSize
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
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
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
    // Phi-4-mini ties its output projection to the input embeddings (no
    // `lm_head.*` in the checkpoint); the flag skips the embed->lm_head copy
    // and the model produces logits from the shared embedding matrix.
    //
    // strictVerify catches the silent weight-drop that hid the original Phi
    // bug: the model declared separate q/k/v_proj while the checkpoint ships a
    // fused qkv_proj, so under lax verify the fused tensor was dropped and
    // attention ran at random init. With strict verify a future module/key
    // mismatch fails loudly at load instead of degenerating into garbage.
    try loadWeights(into: model, from: directory, quantization: config.quantization,
                    tieWordEmbeddings: config.tieWordEmbeddings, strictVerify: true)

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
            // Batched ragged-decode (Stage B/C) for ALL Gemma 4 text paths
            // (e2b/e4b/12b dense AND 26B-A4B MoE). The MoE branch reuses the
            // same Gemma4Block.moeForward the prefill drives - it is token-count
            // (N=B*L) parametric, so the SwitchGLU gather_qmm dispatch runs
            // unchanged at N=R (follow-up #8 Stage C3, proven on Qwen3 MoE).
            batchedDecodeForward: { tokens, caches, mask, rowOffsets in
                model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
            },
            // int8 batched decode (Stage C4): the same Gemma4 batchedDecode
            // accepts a stacked [QuantizedKVCache] - the attention dequantizes
            // per element and the per-row RoPE / left-pad mask apply unchanged.
            batchedDecodeForwardQuantized: { tokens, caches, mask, rowOffsets in
                model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
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
        // Batched ragged-decode (Stage B/C) for all Gemma 4 text paths (dense
        // and MoE). See the multimodal branch for the MoE rationale.
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
        // int8 batched decode (Stage C4): see the multimodal branch.
        batchedDecodeForwardQuantized: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
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
        // Batched ragged-decode (Stage C3): attention is the proven
        // QwenAttention per-row RoPE path and the sparse MoE MLP is N-parametric
        // (same dispatch as prefill, at N=R), so batching reuses both unchanged.
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
        vocabSize: config.vocabSize
    )
}

private func loadMixtral(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(MixtralConfig.self, from: configData)
    let model = MixtralForCausalLM(config)
    // No key rewrite: mlx-community Mixtral checkpoints ship their experts
    // as stacked `block_sparse_moe.switch_mlp.{gate_proj,up_proj,down_proj}.*`
    // tensors (the original HF `block_sparse_moe.experts.{e}.w1/w2/w3` are
    // sanitized into that layout at convert time), and the module hierarchy
    // mirrors it directly via `MixtralSparseMLP.switchMLP`. The router
    // `block_sparse_moe.gate` stays a `Linear` and is quantized by
    // `loadWeights`' quantize pass; the switched linears are not `Linear`,
    // so that pass skips them and they load their packed parameters directly.
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        strictVerify: true)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "moe",
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        multimodalForward: nil,
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
        vocabSize: config.vocabSize
    )
}

private func loadQwen2MoE(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(Qwen2MoEConfig.self, from: configData)
    let model = Qwen2MoEForCausalLM(config)
    // mlx-community Qwen2-MoE checkpoints ship the routed experts as stacked
    // `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*` tensors, which bind
    // directly to `Qwen2MoESparseMLP.switchMLP`. The router `mlp.gate`, the
    // dense `mlp.shared_expert` projections, and `mlp.shared_expert_gate`
    // stay `Linear` and are quantized by `loadWeights`' quantize pass; the
    // switched linears are born quantized and load their packed parameters
    // directly. No key rewrite needed.
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        tieWordEmbeddings: config.tieWordEmbeddings,
        strictVerify: true)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "moe",
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        multimodalForward: nil,
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
        vocabSize: config.vocabSize
    )
}

private func loadOLMoE(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(OLMoEConfig.self, from: configData)
    let model = OLMoEForCausalLM(config)
    // mlx-community OLMoE checkpoints ship the experts as stacked
    // `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*` tensors (mlx-lm's
    // sanitize stacks any per-expert `mlp.experts.{e}.*` at convert time),
    // binding directly to `OLMoESparseMLP.switchMLP`. The router `mlp.gate`
    // and the attention/embedding Linears are quantized by `loadWeights`'
    // quantize pass; the switched linears are born quantized.
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        tieWordEmbeddings: config.tieWordEmbeddings,
        strictVerify: true)

    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "moe",
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        multimodalForward: nil,
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
        },
        vocabSize: config.vocabSize
    )
}

private func loadDeepSeek(configData: Data, directory: URL) throws -> LoadedModel {
    let config = try JSONDecoder().decode(DeepSeekConfig.self, from: configData)
    // DeepSeek-V2 / V2-Lite use the `kv_b_proj` MLA expansion; V3 (and V3.2)
    // use the absorbed `embed_q` / unembed_out per-head linears + a latent KV
    // cache. `DeepSeekDecoderLayer` builds the right attention per
    // `config.usesAbsorbedMLA`; both share the YaRN RoPE, shared expert,
    // dense-layer prefix, and (for V3) the `noaux_tc` sigmoid group gate. The
    // 671B real V3 is RAM-blocked on a small host, but the runtime + synthetic
    // logit parity (`tools/verify_deepseek_parity.py <dir> v3`) run here.
    let model = DeepSeekForCausalLM(config)
    // mlx-community DeepSeek checkpoints ship the routed experts as stacked
    // `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.*` tensors (mlx-lm's
    // sanitize stacks per-expert `mlp.experts.{e}.*` at convert time), and
    // the MLA / shared-expert / dense Linears match by name. The MoE gate
    // (`mlp.gate.weight` + `e_score_correction_bias`) is a raw parameter,
    // not a Linear, so the quantize pass leaves it unquantized (matching
    // mlx-lm's MoEGate). No key rewrite needed.
    try loadWeights(
        into: model, from: directory,
        quantization: config.quantization,
        strictVerify: true)

    // The DeepSeek manifest family is `.deepseek` (dense chat routing), so
    // return that family for the capability / tool-template lookup; the
    // server reaches this loader through the dense engine, not the `.moe`
    // bridge dispatch.
    return LoadedModel(
        module: model,
        numLayers: config.numHiddenLayers,
        family: "deepseek",
        forward: { tokens, caches in model(tokens, caches: caches as? [KVCache]) },
        prefillForward: { tokens, caches in
            model(tokens, caches: caches as? [KVCache], lastTokenOnly: true)
        },
        multimodalForward: nil,
        batchedDecodeForward: { tokens, caches, mask, rowOffsets in
            model.batchedDecode(tokens, caches: caches, mask: mask, rowOffsets: rowOffsets)
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
