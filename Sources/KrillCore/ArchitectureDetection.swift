import Foundation

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

    // NVIDIA LocateAnything-3B: MoonViT vision tower + mlp1 connector + a
    // Qwen2.5-3B text decoder (arch `LocateAnythingForConditionalGeneration`,
    // model_type "locateanything"). Its text backbone is standard Qwen2.5, so
    // this rule MUST precede any generic qwen rule — the arch/model_type strings
    // are unique (no "qwen" substring), but keep it with the VLMs so a future
    // qwen-substring broadening does not hijack it.
    ArchitectureRule(
        id: "locateanything",
        matches: { arch, mt in
            arch.contains("locateanythingforconditionalgeneration") || mt == "locateanything"
        },
        action: .load { try loadLocateAnything(configData: $0, directory: $1) }),

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

    // Gemma 4 12B "unified": ENCODER-FREE multimodal (model_type
    // "gemma4_unified", arch Gemma4UnifiedForConditionalGeneration). No
    // SigLIP/USM towers - raw image patches and raw audio frames project
    // straight into the text embedding space. Its text decoder is the same
    // dense Gemma 4 backbone, so the arch string also contains "gemma4";
    // this rule MUST precede the generic gemma4 rule (first match wins) so
    // a unified checkpoint does not fall into the SigLIP-tower path.
    ArchitectureRule(
        id: "gemma4_unified",
        matches: { arch, mt in
            arch.contains("gemma4unified") || mt == "gemma4_unified"
                || mt == "gemma4_unified_text"
        },
        action: .load { try loadGemma4Unified(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "gemma4",
        matches: { arch, mt in
            arch.contains("gemma4") || mt == "gemma4_text" || mt == "gemma4"
        },
        action: .load { try loadGemma4(configData: $0, directory: $1) }),

    // GLM-4-0414 / GLM-Z1 generation (arch Glm4ForCausalLM, model_type "glm4").
    // A WHOLLY DIFFERENT architecture from the legacy ChatGLM `glm` rule below:
    // separate q/k/v/o projections (bias on q/k/v only), a four-RMSNorm sandwich
    // (input / post_self_attn / post_attention / post_mlp), partial RoPE, fused
    // gate_up MLP, and standard `model.layers.*` naming. MUST precede the `glm`
    // rule: that rule matches `arch.contains("glm")`, so a Glm4 checkpoint would
    // otherwise fall into the legacy ChatGLM loader and emit garbage. First
    // match wins.
    ArchitectureRule(
        id: "glm4",
        // Exclude `Glm4MoeForCausalLM` (GLM-4.5 / GLM-MoE, model_type
        // "glm4_moe"): it also contains "glm4" but is a sparse-MoE arch the
        // dense `loadGlm4` cannot serve. Without the `moe` guard this rule would
        // hijack it and hard-fail mid-forward, and contradict the manifest's
        // `glm4_moe -> .glm` mapping. MoE GLM is an out-of-scope follow-up.
        matches: { arch, mt in (arch.contains("glm4") && !arch.contains("moe")) || mt == "glm4" },
        action: .load { try loadGlm4(configData: $0, directory: $1) }),

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

    // Unlimited-OCR (DeepSeek-OCR): native multimodal load. The language
    // backbone is a DeepSeek-MoE nested under `language_config` (use_mla:false,
    // so plain MHA via DeepSeekStandardAttention); the vision front-end is the
    // native DeepEncoder (SAM-ViT-B + CLIP-L + linear projector). The serve
    // path splices the base-view vision features at the `<image>` block before
    // the LM (base view; gundam tiling is a follow-up). MUST precede the
    // `specialized` rule, which would otherwise reject anything matching "ocr".
    ArchitectureRule(
        id: "unlimited_ocr",
        matches: { arch, mt in arch.contains("unlimitedocr") || mt == "unlimited-ocr" },
        action: .load { try loadUnlimitedOCR(configData: $0, directory: $1) }),

    ArchitectureRule(
        id: "llama",
        matches: { arch, mt in arch.contains("llama") || mt == "llama" },
        action: .load { try loadLlama(configData: $0, directory: $1) }),

    // Qwen 3.5 (Ornith-9B) native Swift+MLX runtime: a Qwen3-Next-class hybrid
    // decoder (GatedDeltaNet plus periodic full attention) and, when the config
    // includes it, the native vision tower. Text-only checkpoints still load
    // the lean text decoder. MUST precede the generic `qwen` rule: it matches
    // `arch.contains("qwen")`, so a qwen3_5 checkpoint would otherwise load as a
    // dense Qwen and emit garbage. The `!moe` guard keeps a future
    // `qwen3_5_moe` (which also contains "qwen3_5") out of this dense loader.
    ArchitectureRule(
        id: "qwen3_5",
        matches: { arch, mt in
            (arch.contains("qwen3_5") && !arch.contains("moe"))
                || mt == "qwen3_5" || mt == "qwen3_5_text"
        },
        // Ornith's config carries BOTH a `text_config` and a `vision_config`.
        // When the vision tower is present, load the native VL model (vision
        // advertised, image inference native); a text-only checkpoint (no
        // `vision_config`) still loads the lean text decoder. VL-as-default was
        // the confirmed rollout choice.
        action: .load { data, dir in
            let hasVision = ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?
                .keys.contains("vision_config") ?? false
            return hasVision
                ? try loadQwen35VL(configData: data, directory: dir)
                : try loadQwen35(configData: data, directory: dir)
        }),

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
    // Krill has no native runtime for these; rejecting here with a specific
    // error is the roadmap's "Unsupported tier" - far better than mis-loading
    // them via the Llama fallback below and emitting a garbage forward pass.
    ArchitectureRule(
        id: "specialized",
        matches: { arch, mt in detectSpecializedModelType(arch: arch, modelType: mt) != nil },
        action: .reject { arch, mt in
            let specialized = detectSpecializedModelType(arch: arch, modelType: mt)
            let name = specialized?.displayName ?? "specialized"
            return .specializedModelUnsupported(
                "Krill does not support \(name) models. "
                + "These are WS7 specialized model types with no native "
                + "runtime in this build (of the WS7 types, only rerankers "
                + "have shipped - use POST /v1/rerank for those; see "
                + "docs/workstreams/WS7_SPECIALIZED_MODEL_TYPES.md). Krill "
                + "serves causal text LMs, supported native multimodal models, "
                + "embeddings, and rerankers. Detected arch=\(arch), model_type=\(mt).")
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
func detectedArchitectureID(architectures: [String], modelType: String) -> String {
    let arch = architectures.first?.lowercased() ?? ""
    // Force-unwrap is safe: the table's last rule matches any input.
    return architectureRules.first(where: { $0.matches(arch, modelType) })!.id
}
