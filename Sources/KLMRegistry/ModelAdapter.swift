import Foundation

/// The runtime adapter for a model family.
///
/// `ModelAdapter` is the single declarative source of truth for the
/// family-specific decisions the server used to make with scattered
/// `manifest.family == .qwen25vl` style branches:
///
///   - `chatRouting`: which chat handler a request must be routed to
///     (the native dense engine, the VLM bridge, or the MoE path),
///   - `requiresImageInput`: whether a text-only request must be
///     refused before a multi-GB sidecar is spun up,
///   - `chatTemplate`: which tool / function-call chat template the
///     family expects.
///
/// Capability and support-tier facts are a separate concern and stay
/// in `ModelCapabilities` (the sibling per-family table); a caller
/// that needs those reads `ModelCapabilities` directly. `ModelAdapter`
/// is deliberately scoped to the server's *routing and chat-template*
/// decisions - the things that were previously scattered switches.
///
/// Adding a new family should start here: give it a `chatRouting`
/// and a `chatTemplate`, and the server picks it up without a new
/// hand-written branch.
///
/// Scope note (WS3): this adapter is the *server-side* routing and
/// templating contract. The load-time concerns the WS3 design sketch
/// also lists - `detect`, `load`, `tokenizerPolicy`, `cachePolicy`,
/// `benchmarkProfile` - are deliberately NOT folded in here: they
/// would force a rewrite of every loader and engine for an
/// abstraction whose hot-path cost must stay zero. They remain in
/// `ModelLoader` / the per-family engines and can adopt this type
/// incrementally.
public struct ModelAdapter: Sendable, Equatable {
    /// The model family this adapter describes.
    public let family: ModelFamily

    public init(family: ModelFamily) {
        self.family = family
    }

    /// Which server chat handler a request for this family needs.
    public var chatRouting: ChatRouting {
        switch family {
        case .llama, .qwen, .qwen25vl, .llava, .llamaVision, .mistral, .gemma, .gemma4,
             .gemma4Unified, .phi, .glm, .glm4, .deepseek, .bert, .reranker, .moe:
            // Native Swift+MLX path. WS5 made Qwen 2.5-VL native, so
            // a VL manifest routes here too - the standard chat path
            // decodes the image and calls the native engine, exactly
            // like Gemma 4 vision. `.moe` joined this group once every
            // MoE family (Qwen 3 MoE, Mixtral, Qwen2-MoE, OLMoE, plus
            // DeepSeek-V2 under `.deepseek`) went native and the mlx-lm
            // sidecar bridge was deleted; a MoE manifest now loads
            // through `loadModel` on the dense engine like any other
            // native causal LM. (`.bert` / `.reranker` are not causal
            // LMs and are refused by the chat surface on a capability
            // check; they never reach a chat handler, so `.denseEngine`
            // is the correct inert default.)
            return .denseEngine
        }
    }

    /// True iff a chat request for this family MUST carry an image.
    /// No family requires one today: WS5 retired the Qwen 2.5-VL
    /// Python sidecar, and the native VL runtime serves a text-only
    /// turn directly (it just skips the vision tower). The hook is
    /// kept so a future image-only family can opt back in.
    public var requiresImageInput: Bool {
        switch family {
        case .llama, .qwen, .qwen25vl, .llava, .llamaVision, .mistral, .gemma, .gemma4,
             .gemma4Unified, .phi, .glm, .glm4, .deepseek, .bert, .reranker, .moe:
            return false
        }
    }

    /// The tool / function-call chat template this family expects.
    public var chatTemplate: ChatTemplatePolicy {
        switch family {
        case .gemma4, .gemma4Unified:
            return .gemma4
        case .llama:
            return .llama
        case .qwen:
            return .qwen
        case .moe:
            // The only native MoE runtime today (Qwen 3 MoE) uses
            // the Qwen chat / tool template verbatim. A future MoE
            // member needing a different template gets its own case.
            return .qwen
        case .mistral:
            // Mistral's native `[AVAILABLE_TOOLS]…[/AVAILABLE_TOOLS]` /
            // `[TOOL_CALLS]` / `[TOOL_RESULTS]` format (token ids 5-9).
            return .mistral
        case .phi:
            // Phi-3.5 / Phi-4 native `<|tool|>…<|/tool|>` definitions and
            // `<|tool_call|>…<|/tool_call|>` call format.
            return .phi
        case .gemma, .glm, .glm4, .deepseek, .bert, .qwen25vl, .llava, .llamaVision, .reranker:
            // The generic Hermes-style `<tool_call>{…}</tool_call>`
            // prompt: an acceptable fallback, not a native template.
            // LLaVA does not advertise tools; its vicuna-style multimodal
            // prompt is built directly in the engine, not via this policy.
            return .hermes
        }
    }

    // MARK: - Load-time policies (WS3 completion)
    //
    // `detect` landed as the declarative `architectureRules` table in
    // `ModelLoader`; these two properties fold the remaining load-time
    // family-keyed decisions the engine used to make with `family ==
    // "gemma4"` / `"phi"` string compares into the same declarative adapter.
    // They are read ONCE per request (prompt build / cache allocation), never
    // per decode step, so the hot path stays zero-cost. `load` itself is
    // already per-architecture (one loader per `architectureRules` row).

    /// How the engine must build the prompt token ids for this family. The
    /// per-family split exists because the right tokenization path is not
    /// uniform: Gemma 4 needs direct turn-special ids, phi's o200k BPE needs a
    /// render+re-encode round-trip, LLaVA needs the vicuna prompt with the
    /// image-token run placed inline, and everyone else prefers the
    /// swift-transformers direct template with a render+encode fallback.
    public var tokenizerPrompt: TokenizerPromptPolicy {
        switch family {
        case .gemma4, .gemma4Unified:
            // Direct token-id construction keeps the 105/106/107 turn
            // specials that a decode -> re-encode round-trip would drop.
            // The unified (encoder-free) SKU shares the same turn structure;
            // its media placeholder runs are spliced in the engine's
            // multimodal path, not here.
            return .gemma4DirectIds
        case .phi:
            // o200k (GPT-4o / tiktoken) BPE: the swift-transformers DIRECT
            // path mis-tokenizes the body, so render the template to a string
            // and re-encode through the canonical encode path instead.
            return .phiRenderReencode
        case .llava:
            // Vicuna prompt with the per-CLIP-patch image-token run placed
            // inline (`formatLlavaTokenIds`).
            return .llavaVicuna
        case .llama, .qwen, .qwen25vl, .llamaVision, .mistral, .gemma, .glm, .glm4, .deepseek,
             .bert, .reranker, .moe:
            // Try the swift-transformers direct token-id template (keeps
            // ChatML / FIM / tool specials), else render + encode.
            return .directTokenIdsWithRenderFallback
        }
    }

    /// Whether this family supports the int8 quantized KV cache. Only Gemma 4
    /// today: every other family's forward closure downcasts caches to
    /// `[KVCache]`, so handing it a `[QuantizedKVCache]` would crash at the
    /// first attention. (The int8 batched-decode path is separately gated on
    /// the model exposing a `batchedDecodeForwardQuantized` closure, which is
    /// also Gemma-4-only, so it needs no family check.)
    public var kvCacheQuantization: KVCacheQuantizationPolicy {
        switch family {
        case .gemma4:
            return .supportsInt8
        case .gemma4Unified:
            // The unified text decoder is the same Gemma4Attention that
            // accepts `KVCacheProtocol`, so int8 KV is structurally
            // supported, but it is not yet end-to-end verified for this
            // family. Stay fp16-only until that gate lands (follow-up).
            return .fp16Only
        case .llama, .qwen, .qwen25vl, .llava, .llamaVision, .mistral, .gemma, .phi,
             .glm, .glm4, .deepseek, .bert, .reranker, .moe:
            return .fp16Only
        }
    }
}

/// How the engine builds the prompt token ids for a family. A stable,
/// module-neutral identifier; the engine maps each case to its concrete
/// tokenizer call. Declaring a new family's policy here (a compile error until
/// done, since the `switch` is exhaustive) is the single deliberate decision.
public enum TokenizerPromptPolicy: String, Sendable, Equatable, CaseIterable {
    /// Gemma 4: `formatGemma4TokenIds` (direct turn-special ids).
    case gemma4DirectIds
    /// Phi: render the chat template, then `encodeWithoutExtraBOS`.
    case phiRenderReencode
    /// LLaVA-1.5: `formatLlavaTokenIds` (vicuna prompt + inline image run).
    case llavaVicuna
    /// Default: `applyChatTemplateTokens` if available, else render + encode.
    case directTokenIdsWithRenderFallback
}

/// Whether a family supports the int8 quantized KV cache or runs fp16-only.
public enum KVCacheQuantizationPolicy: String, Sendable, Equatable, CaseIterable {
    /// int8 KV cache is safe (and a quantized batched-decode forward exists).
    case supportsInt8
    /// fp16 KV only (the family's forward closure assumes `[KVCache]`).
    case fp16Only
}

/// Which server chat handler a model family's request is routed to.
///
/// There is a single native path today: every supported family (dense text,
/// Qwen 2.5-VL, Gemma 4 vision/audio, and all MoE families) loads through
/// `loadModel` and runs on the native Swift+MLX engine. The historical
/// `mixtureOfExperts` case (which fell back to the mlx-lm Python sidecar) was
/// removed when the last MoE family went native and the bridge was deleted.
public enum ChatRouting: String, Sendable, Equatable, CaseIterable {
    /// Native Swift+MLX dense engine path (`engine.generate`).
    case denseEngine = "dense_engine"
}

/// The tool / function-call chat template a model family expects.
///
/// This is a stable, module-neutral identifier. KLMServer's
/// `ToolCalling.ToolFormat` maps it to the concrete renderer/parser;
/// keeping the policy in the registry means a new family declares
/// its template here, next to its capabilities, rather than in a
/// server-side switch.
public enum ChatTemplatePolicy: String, Sendable, Equatable, CaseIterable {
    /// Generic `<tool_call>{"name",…,"arguments"}</tool_call>` prompt.
    case hermes
    /// Gemma 4 native special-token tool format (token ids 46-51).
    case gemma4
    /// Llama 3.x native tool template.
    case llama
    /// Qwen 2.5 / Qwen 3 native tool template.
    case qwen
    /// Mistral native `[AVAILABLE_TOOLS]` / `[TOOL_CALLS]` / `[TOOL_RESULTS]`
    /// tool template (Mistral 7B Instruct v0.3, Nemo, Small).
    case mistral
    /// Phi-3.5 / Phi-4 native `<|tool|>` definitions + `<|tool_call|>` call
    /// template.
    case phi
}
