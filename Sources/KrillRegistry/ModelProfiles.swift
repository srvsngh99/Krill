import Foundation

/// A curated, hand-written profile for a model family: the editorial half of the
/// `/model` deep-dive (the factual half - params, quant, context, inputs, size -
/// is derived live from the registry/capabilities, never from here). Kept
/// deliberately short and general so it stays accurate without per-SKU churn.
public struct ModelProfile: Sendable {
    public let displayName: String     // wordmark text, e.g. "LLAMA"
    public let vendor: String          // e.g. "Meta"
    public let released: String        // month + year where known, e.g. "Sep 2024 (3.2)"
    public let trainingCutoff: String  // vendor-published cutoff, else "not publicly disclosed"
    public let tagline: String
    public let strengths: [String]
    public let weaknesses: [String]
    public let goodFor: [String]
}

public enum ModelProfiles {
    /// Curated profile for a family, or nil if we have not written one.
    public static func profile(for family: ModelFamily) -> ModelProfile? {
        switch family {
        case .llama, .llava, .llamaVision:
            return ModelProfile(
                displayName: "LLAMA", vendor: "Meta", released: "Sep 2024 (Llama 3.2)",
                trainingCutoff: "Dec 2023",
                tagline: "Open general-purpose models with a deep ecosystem.",
                strengths: ["Strong general reasoning and instruction following",
                            "Long context and wide tooling/community support",
                            "Reliable, well-behaved tool calling"],
                weaknesses: ["No native multimodality in the base text SKUs",
                             "Smaller sizes trail larger frontier models on hard tasks"],
                goodFor: ["General chat and assistants", "Coding and agents", "RAG"])
        case .gemma, .gemma4, .gemma4Unified:
            return ModelProfile(
                displayName: "GEMMA", vendor: "Google", released: "2024-2025",
                trainingCutoff: "not publicly disclosed",
                tagline: "Compact, capable models; Gemma 4 adds native vision + audio.",
                strengths: ["Strong quality for the size", "Native image and audio input (Gemma 4)",
                            "Good structured output and tool calling"],
                weaknesses: ["A built-in thinking channel can ramble if unprompted",
                             "Large SKUs are memory hungry on a 24GB box"],
                goodFor: ["Multimodal chat (image/audio)", "On-device assistants", "Structured extraction"])
        case .qwen, .qwen25vl:
            return ModelProfile(
                displayName: "QWEN", vendor: "Alibaba", released: "Sep 2024 (Qwen2.5)",
                trainingCutoff: "not publicly disclosed",
                tagline: "Broad, multilingual family with strong coding and a VL variant.",
                strengths: ["Strong coding and math", "Excellent multilingual coverage",
                            "Vision-language variant (Qwen2.5-VL)"],
                weaknesses: ["Tool-call format varies by SKU", "Larger sizes are RAM-bound locally"],
                goodFor: ["Coding", "Multilingual tasks", "Vision-language"])
        case .qwen35:
            return ModelProfile(
                displayName: "ORNITH", vendor: "Ornith", released: "2025 (Ornith 1.0)",
                trainingCutoff: "not publicly disclosed",
                tagline: "Qwen3.5-class hybrid (GatedDeltaNet linear-attention + full attention); text served natively.",
                strengths: ["Efficient linear-attention hybrid decoder",
                            "Strong multilingual + coding inheritance from the Qwen3.5 lineage",
                            "Compact 9B that fits a 24GB box at int4"],
                weaknesses: ["Vision tower runs via mlx_vlm, not the native engine yet",
                             "Newer lineage with a smaller ecosystem than base Qwen"],
                goodFor: ["General chat", "Coding", "Multilingual tasks"])
        case .mistral:
            return ModelProfile(
                displayName: "MISTRAL", vendor: "Mistral AI", released: "2023-2024",
                trainingCutoff: "not publicly disclosed",
                tagline: "Efficient European models; strong throughput per parameter.",
                strengths: ["Fast and efficient", "Solid general reasoning",
                            "Native [TOOL_CALLS] function calling"],
                weaknesses: ["No native multimodality", "Shorter context than some peers"],
                goodFor: ["General chat", "Tool use", "Latency-sensitive serving"])
        case .phi:
            return ModelProfile(
                displayName: "PHI", vendor: "Microsoft", released: "2024 (Phi-3 / Phi-4)",
                trainingCutoff: "not publicly disclosed",
                tagline: "Small models punching above their weight via curated data.",
                strengths: ["High quality at very small sizes", "Strong on reasoning benchmarks",
                            "Cheap to run"],
                weaknesses: ["Narrower world knowledge than larger models",
                             "Less robust on open-ended creative tasks"],
                goodFor: ["On-device and edge", "Reasoning at low cost", "Drafting"])
        case .deepseek:
            return ModelProfile(
                displayName: "DEEPSEEK", vendor: "DeepSeek", released: "Dec 2024 (V3)",
                trainingCutoff: "not publicly disclosed",
                tagline: "MoE-heavy models with standout coding and reasoning.",
                strengths: ["Top-tier coding and math", "Efficient mixture-of-experts inference",
                            "Strong long-form reasoning"],
                weaknesses: ["Flagship sizes are far beyond a 24GB box",
                             "MoE memory footprint is large"],
                goodFor: ["Coding", "Math and reasoning", "Agentic workflows"])
        case .glm:
            return ModelProfile(
                displayName: "GLM", vendor: "Zhipu AI", released: "2024",
                trainingCutoff: "not publicly disclosed",
                tagline: "Bilingual (Chinese/English) general-purpose models.",
                strengths: ["Strong Chinese and English", "Good general chat", "Tool calling"],
                weaknesses: ["Smaller ecosystem", "Less coverage on niche English tasks"],
                goodFor: ["Bilingual chat", "General assistants"])
        case .glm4:
            return ModelProfile(
                displayName: "GLM-4", vendor: "Zhipu AI / Z.ai", released: "Apr 2025 (0414)",
                trainingCutoff: "not publicly disclosed",
                tagline: "GLM-4-0414 / GLM-Z1: stronger reasoning and coding, English+Chinese.",
                strengths: ["Strong reasoning (GLM-Z1)", "Good coding", "Bilingual"],
                weaknesses: ["32B is tight on a 24GB box", "Flagship GLM-4.5+ are MoE and far larger"],
                goodFor: ["Reasoning", "Coding", "Bilingual chat"])
        case .bert, .reranker:
            return ModelProfile(
                displayName: "BERT", vendor: "Encoder", released: "various",
                trainingCutoff: "n/a",
                tagline: "Sentence-embedding / reranker encoders (not a chat model).",
                strengths: ["Fast vector embeddings", "Reranking / retrieval scoring"],
                weaknesses: ["Cannot generate text", "Not usable for chat"],
                goodFor: ["Embeddings", "RAG retrieval", "Reranking"])
        case .unlimitedOcr:
            return ModelProfile(
                displayName: "UNLIMITED-OCR", vendor: "DeepSeek-OCR", released: "2025",
                trainingCutoff: "not publicly disclosed",
                tagline: "Native document/image OCR: DeepEncoder vision + DeepSeek-MoE.",
                strengths: ["High-fidelity document and image OCR",
                            "Native Apple-Silicon MLX runtime (no Python)",
                            "Compact nvfp4 mixed-precision footprint"],
                weaknesses: ["Single-purpose OCR, not a general chat model",
                             "Base-view resolution (gundam tiling is a follow-up)"],
                goodFor: ["Document OCR", "Image-to-markdown", "Layout/text extraction"])
        case .moe:
            // Generic MoE bucket - no single curated story; the deep-dive still
            // shows the live specs derived from the registry.
            return nil
        }
    }
}
