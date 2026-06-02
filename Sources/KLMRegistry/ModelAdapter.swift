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
        case .llama, .qwen, .qwen25vl, .llava, .mistral, .gemma, .gemma4,
             .phi, .glm, .deepseek, .bert, .reranker, .moe:
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
        case .llama, .qwen, .qwen25vl, .llava, .mistral, .gemma, .gemma4,
             .phi, .glm, .deepseek, .bert, .reranker, .moe:
            return false
        }
    }

    /// The tool / function-call chat template this family expects.
    public var chatTemplate: ChatTemplatePolicy {
        switch family {
        case .gemma4:
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
        case .gemma, .glm, .deepseek, .bert, .qwen25vl, .llava, .reranker:
            // The generic Hermes-style `<tool_call>{…}</tool_call>`
            // prompt: an acceptable fallback, not a native template.
            // LLaVA does not advertise tools; its vicuna-style multimodal
            // prompt is built directly in the engine, not via this policy.
            return .hermes
        }
    }
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
