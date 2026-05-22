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
        case .moe:
            // Native Swift+MLX runtime when the checkpoint supports
            // it (today: Qwen 3 MoE), else the mlx-lm sidecar bridge.
            return .mixtureOfExperts
        case .llama, .qwen, .qwen25vl, .mistral, .gemma, .gemma4,
             .phi, .glm, .deepseek, .bert, .reranker:
            // Native Swift+MLX path. WS5 made Qwen 2.5-VL native, so
            // a VL manifest routes here too - the standard chat path
            // decodes the image and calls the native engine, exactly
            // like Gemma 4 vision. (`.bert` / `.reranker` are not
            // causal LMs and are refused by the chat surface on a
            // capability check; they never reach a chat handler, so
            // `.denseEngine` is the correct inert default.)
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
        case .llama, .qwen, .qwen25vl, .mistral, .gemma, .gemma4,
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
        case .mistral, .gemma, .phi, .glm, .deepseek, .bert,
             .qwen25vl, .reranker:
            // The generic Hermes-style `<tool_call>{…}</tool_call>`
            // prompt: an acceptable fallback, not a native template.
            return .hermes
        }
    }
}

/// Which server chat handler a model family's request is routed to.
public enum ChatRouting: String, Sendable, Equatable, CaseIterable {
    /// Native Swift+MLX dense engine path (`engine.generate`).
    /// Includes Qwen 2.5-VL since WS5 retired its Python bridge.
    case denseEngine = "dense_engine"
    /// Mixture-of-experts: the native Swift+MLX runtime when the
    /// checkpoint supports it (see `nativeMoEDispatchSupported`),
    /// otherwise the Python-sidecar MoE bridge (`handleMoEChat`).
    case mixtureOfExperts = "mixture_of_experts"
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
}
