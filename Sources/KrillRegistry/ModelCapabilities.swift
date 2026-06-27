import Foundation

/// What a model family can actually do at runtime.
///
/// `Capability` is the declared per-family contract: it answers "should
/// the server accept this kind of input at all", "is this surface
/// supported", etc., without forcing every call site to ask "is this
/// Gemma 4?". Native availability is the orthogonal axis; see
/// `SupportTier`.
public enum Capability: String, Sendable, CaseIterable, Codable {
    /// Causal text generation (the universal default for the LM
    /// families). Embedding-only models do NOT have this.
    case textGeneration

    /// Image input via the engine's native multimodal forward.
    case visionInput

    /// Audio input via the engine's native multimodal forward.
    case audioInput

    /// Vector embeddings via the dedicated embedding engine.
    case embeddings

    /// Function / tool calling. Today this is "the model emits
    /// tool-call shaped JSON and the server parses it" rather than a
    /// constrained sampler, so it is family-by-family quality dependent.
    case tools

    /// Structured / JSON-schema constrained output via the structured
    /// sampler path.
    case structuredOutput

    /// Mixture-of-experts at runtime (router + selected experts).
    case moe

    /// Reranker (cross-encoder) scoring head.
    case reranker
}

/// How well-supported a model family is in this build.
///
/// Promotion across tiers requires tests, benchmark gates, and docs;
/// the registry should NEVER silently upgrade a family.
public enum SupportTier: String, Sendable, CaseIterable, Codable {
    /// Swift + MLX/Metal path, tests, docs, benchmark gate green.
    case productionNative = "production_native"

    /// Works through a bridge or slower reference path. NOT a
    /// performance claim.
    case compatibleFallback = "compatible_fallback"

    /// Runs on known fixtures but lacks full gates. Developer preview.
    case experimental

    /// Explicit error before execution. Not supported.
    case unsupported
}

/// Per-family declared capabilities and support tier.
///
/// Used by:
///   - `InferenceEngine` instead of hardcoded `family == "gemma4"`
///     branches when checking whether to accept image / audio input.
///   - The Ollama-compat `details` / `show` payloads, so clients can
///     see what each model actually does.
///   - The server's pre-generation media gating, so unsupported media
///     fails before the model is touched.
///
/// Keep this in sync with what the family's loader and engine actually
/// implement. Adding a new family means adding an entry here BEFORE
/// adding an alias.
public enum ModelCapabilities {
    /// The declared capability set for a model family.
    ///
    /// `tools` is only declared for families that ship a parity-tested
    /// native tool chat template (Llama 3.x, Qwen 2.5, Gemma 4 today,
    /// landed in PR #23). Other dense families can still be prompted
    /// for tool calls via raw JSON, but we do NOT advertise the
    /// capability there because there is no parity gate behind it yet.
    /// `structuredOutput` is intentionally not declared here today: the
    /// structured sampler operates on logits and is family-agnostic,
    /// but per-family parity for the structured surface has not been
    /// gated. Adding the capability later will be a deliberate
    /// promotion.
    public static func capabilities(for family: ModelFamily) -> Set<Capability> {
        switch family {
        case .gemma, .glm, .glm4, .deepseek:
            return [.textGeneration]
        case .mistral, .phi:
            // Native tool template + parser (injectMistral/extractMistral,
            // injectPhi/extractPhi) - advertise tools so they can go agentic.
            return [.textGeneration, .tools]
        case .llama, .qwen:
            return [.textGeneration, .tools]
        case .qwen35:
            // Ornith-9B (qwen3_5): native text decoder + Qwen tool template.
            // visionInput is intentionally NOT advertised - the vision tower is
            // deferred to mlx_vlm, so this build serves the text decoder only.
            return [.textGeneration, .tools]
        case .gemma4:
            return [.textGeneration, .visionInput, .audioInput, .tools]
        case .gemma4Unified:
            // Encoder-free multimodal Gemma 4 (12B): text + image + native
            // audio, same tool surface as the rest of the Gemma 4 family.
            return [.textGeneration, .visionInput, .audioInput, .tools]
        case .bert:
            return [.embeddings]
        case .qwen25vl:
            // Qwen 2.5-VL: native Swift+MLX runtime (WS5) - vision
            // tower, patch merger, 3D mRoPE, image preprocessing,
            // and a grid/decode-offset-correct generation loop. The
            // Python bridge was retired once the native path was
            // validated against a real checkpoint and the recorded
            // mlx-vlm oracle.
            return [.textGeneration, .visionInput, .tools]
        case .llava:
            // LLaVA-1.5: native CLIP + projector + Llama runtime (PR #129)
            // with engine image-serving wiring. Vision input is supported;
            // tools are NOT advertised (no parity-tested tool template -
            // the vicuna-style prompt is built directly).
            return [.textGeneration, .visionInput]
        case .llamaVision:
            // Llama-3.2-Vision (mllama) native runtime WITH image serving: tile /
            // aspect-ratio preprocessing, the sparse cross-attention mask, and a
            // cross-KV decode driver (`MllamaRuntime`), validated by synthetic
            // mlx-vlm parity (single + multi-image) and a cross-KV decode
            // self-consistency gate. Multiple images per request are supported.
            // Tools are NOT advertised (no parity-tested tool template).
            return [.textGeneration, .visionInput]
        case .moe:
            // WS6 foundation: the family DECLARES textGeneration +
            // moe + tools (the initial targets - Mixtral, Qwen 3
            // MoE - both inherit the qwen/mistral tool template).
            // The loader rejects instantiation until the router +
            // expert dispatch lands in follow-up PRs.
            return [.textGeneration, .moe, .tools]
        case .reranker:
            // WS7 foundation: cross-encoder rerankers expose ONLY
            // the reranker capability - they are not causal LMs,
            // not embedding encoders (they take a (query, document)
            // pair and produce a single relevance score), and the
            // existing /api/generate / /v1/chat surfaces must
            // refuse them. The `/v1/rerank` endpoint that consumes
            // this capability ships in the follow-up runtime PR.
            return [.reranker]
        }
    }

    /// The support tier for a model family in this build.
    public static func supportTier(for family: ModelFamily) -> SupportTier {
        switch family {
        case .llama, .qwen, .mistral, .gemma, .phi, .glm, .glm4, .deepseek,
             .gemma4, .bert, .qwen25vl:
            // Production-native: native Swift + MLX/Metal load+run
            // path, deterministic smoke tests, and a benchmark
            // report against Ollama or a reference. WS5 promoted
            // Qwen 2.5-VL here once the native vision tower, patch
            // merger, 3D mRoPE, and image preprocessing shipped and
            // passed the recorded mlx-vlm oracle on a real
            // checkpoint; the Python bridge was then retired.
            return .productionNative
        case .qwen35:
            // Ornith-9B (qwen3_5): the native GatedDeltaNet + full-attention
            // hybrid decoder is parity-verified vs mlx_lm (scan + forward +
            // decode-cache gates), but real-checkpoint generation and a serving
            // benchmark are still landing, so `.experimental` until those gate.
            return .experimental
        case .llava:
            // LLaVA-1.5 has a native Swift+MLX load+run path and an
            // mlx-vlm logit-parity gate (LlavaParityTests, both MLX and
            // PyTorch conv layouts). It is `experimental` rather than
            // productionNative until a serving benchmark gate (vs a
            // reference) lands, mirroring how WS5/WS7 families are promoted
            // only after their benchmark gate is green.
            return .experimental
        case .llamaVision:
            // Native runtime + loader + mlx-vlm synthetic logit parity, but no
            // serving benchmark and no real-checkpoint run yet (RAM-blocked on
            // the dev box), so `.experimental`.
            return .experimental
        case .moe:
            // Every MoE family is native Swift+MLX now (Qwen 3 MoE,
            // Mixtral, Qwen2-MoE, OLMoE; DeepSeek-V2 lives under the
            // `.deepseek` family). The mlx-lm sidecar bridge and the
            // per-checkpoint tier refinement it required are gone, so the
            // family reports productionNative directly.
            return .productionNative
        case .reranker:
            // WS7 foundation: same shape as WS5/WS6. Family
            // detection + capability + alias entries + clear
            // rejection exist; the cross-encoder scoring head +
            // `/v1/rerank` endpoint + reference-score parity smoke
            // are pending. Tier promotes once the scoring runtime
            // lands and the parity smoke matches a reference model
            // within tolerance.
            return .experimental
        case .gemma4Unified:
            // Encoder-free Gemma 4 12B: native Swift+MLX load+run path
            // reusing the dense Gemma 4 text backbone, with encoder-free
            // vision/audio embedders gated on numeric parity. `.experimental`
            // until the real-checkpoint serving run + benchmark gate land
            // (mirrors how llava / llamaVision are promoted only after
            // their serving gate is green).
            return .experimental
        }
    }

    /// Support tier for an installed model. No family needs per-checkpoint
    /// refinement anymore: the `.moe` family that once spanned a native
    /// runtime plus mlx-lm-bridge members is now uniformly native
    /// (productionNative), so this delegates to the family-level
    /// `supportTier(for:)`. The `directory` parameter is retained for API
    /// stability (callers like `/api/show` pass the installed path) but is no
    /// longer consulted.
    public static func supportTier(
        for family: ModelFamily, at directory: URL?
    ) -> SupportTier {
        return supportTier(for: family)
    }

    /// True iff the family ships a parity-tested native tool chat
    /// template. Equivalent today to `capabilities(for:).contains(.tools)`;
    /// the helper exists so callers that only care about this one
    /// question read cleanly.
    public static func hasNativeToolTemplate(_ family: ModelFamily) -> Bool {
        capabilities(for: family).contains(.tools)
    }
}

/// String IDs used in Ollama-compatible payloads (`/api/show`
/// `capabilities` array, `/api/tags` `details.capabilities`).
///
/// We emit a stable, snake_case identifier per capability so external
/// tooling can match on it. Ollama itself emits `["completion",
/// "vision"]` today; we keep `completion` as an alias for
/// `textGeneration` to remain drop-in.
public extension Capability {
    var ollamaTag: String {
        switch self {
        case .textGeneration: return "completion"
        case .visionInput: return "vision"
        case .audioInput: return "audio"
        case .embeddings: return "embedding"
        case .tools: return "tools"
        case .structuredOutput: return "structured_output"
        case .moe: return "moe"
        case .reranker: return "reranker"
        }
    }
}
