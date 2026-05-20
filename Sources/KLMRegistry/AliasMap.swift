import Foundation

/// Curated alias map: short model names -> HuggingFace repos.
///
/// These are the recommended MLX-quantized models from mlx-community
/// for each supported family. krillm pull <alias> resolves through
/// this map before falling back to treating the input as a raw HF repo path.
public struct AliasMap: Sendable {
    /// Resolve a model name to a HuggingFace repo path.
    ///
    /// Priority:
    /// 1. Exact match in alias map
    /// 2. Treat as raw HF repo path if contains "/"
    /// 3. Return nil (not found)
    public static func resolve(_ name: String) -> ResolvedModel? {
        let normalized = name.lowercased().trimmingCharacters(in: .whitespaces)

        // Direct alias match
        if let entry = aliases[normalized] {
            return entry
        }

        // If it looks like a HF repo path (org/name), use directly
        if name.contains("/") {
            return ResolvedModel(
                repo: name,
                name: name.split(separator: "/").last.map(String.init) ?? name,
                family: .llama,  // will be detected from config.json
                params: "?",
                quant: "4bit",
                context: 8192
            )
        }

        return nil
    }

    /// All available aliases for display.
    public static var allAliases: [(shortName: String, model: ResolvedModel)] {
        aliases.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }
}

/// A resolved model reference from the alias map.
public struct ResolvedModel: Sendable {
    public let repo: String
    public let name: String
    public let family: ModelFamily
    public let params: String
    public let quant: String
    public let context: Int
}

// MARK: - Alias Database

private let aliases: [String: ResolvedModel] = [
    // Llama 3.2
    "llama-3.2-1b": ResolvedModel(
        repo: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        name: "llama-3.2-1b", family: .llama, params: "1B", quant: "4bit", context: 131072),
    "llama-3.2-3b": ResolvedModel(
        repo: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        name: "llama-3.2-3b", family: .llama, params: "3B", quant: "4bit", context: 131072),

    // Llama 3.1
    "llama-3.1-8b": ResolvedModel(
        repo: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        name: "llama-3.1-8b", family: .llama, params: "8B", quant: "4bit", context: 131072),

    // Qwen 2.5
    "qwen2.5-0.5b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        name: "qwen2.5-0.5b", family: .qwen, params: "0.5B", quant: "4bit", context: 32768),
    "qwen2.5-1.5b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        name: "qwen2.5-1.5b", family: .qwen, params: "1.5B", quant: "4bit", context: 32768),
    "qwen2.5-3b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-3B-Instruct-4bit",
        name: "qwen2.5-3b", family: .qwen, params: "3B", quant: "4bit", context: 32768),
    "qwen2.5-7b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        name: "qwen2.5-7b", family: .qwen, params: "7B", quant: "4bit", context: 131072),
    "qwen2.5-14b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-14B-Instruct-4bit",
        name: "qwen2.5-14b", family: .qwen, params: "14B", quant: "4bit", context: 131072),

    // Qwen 3 (dense variants; MoE variants tracked under WS6)
    // Architecture delta vs Qwen 2.5: no QKV bias, per-head q_norm/k_norm
    // before RoPE, tied embeddings, explicit head_dim. Same Qwen family
    // dispatch + the QwenConfig flags detected from `model_type: qwen3`.
    "qwen3-0.6b": ResolvedModel(
        repo: "mlx-community/Qwen3-0.6B-4bit",
        name: "qwen3-0.6b", family: .qwen, params: "0.6B", quant: "4bit", context: 40960),
    "qwen3-1.7b": ResolvedModel(
        repo: "mlx-community/Qwen3-1.7B-4bit",
        name: "qwen3-1.7b", family: .qwen, params: "1.7B", quant: "4bit", context: 40960),
    "qwen3-4b": ResolvedModel(
        repo: "mlx-community/Qwen3-4B-4bit",
        name: "qwen3-4b", family: .qwen, params: "4B", quant: "4bit", context: 40960),
    "qwen3-8b": ResolvedModel(
        repo: "mlx-community/Qwen3-8B-4bit",
        name: "qwen3-8b", family: .qwen, params: "8B", quant: "4bit", context: 40960),
    "qwen3-14b": ResolvedModel(
        repo: "mlx-community/Qwen3-14B-4bit",
        name: "qwen3-14b", family: .qwen, params: "14B", quant: "4bit", context: 40960),

    // Qwen 2.5-VL vision-language (WS5 foundation tier: family
    // detection + clear rejection only; native vision tower lands
    // in follow-up PRs).
    "qwen2.5-vl-3b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        name: "qwen2.5-vl-3b", family: .qwen25vl, params: "3B", quant: "4bit", context: 128000),
    "qwen2.5-vl-7b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        name: "qwen2.5-vl-7b", family: .qwen25vl, params: "7B", quant: "4bit", context: 128000),
    "qwen2.5-vl-32b": ResolvedModel(
        repo: "mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
        name: "qwen2.5-vl-32b", family: .qwen25vl, params: "32B", quant: "4bit", context: 128000),

    // Mixture-of-experts (WS6 foundation tier: family detection +
    // clear rejection only; router + expert dispatch lands in
    // follow-up PRs).
    "mixtral-8x7b": ResolvedModel(
        repo: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        name: "mixtral-8x7b", family: .moe, params: "8x7B", quant: "4bit", context: 32768),
    "qwen3-30b-a3b": ResolvedModel(
        repo: "mlx-community/Qwen3-30B-A3B-4bit",
        name: "qwen3-30b-a3b", family: .moe, params: "30B-A3B", quant: "4bit", context: 40960),

    // Mistral
    "mistral-7b": ResolvedModel(
        repo: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        name: "mistral-7b", family: .mistral, params: "7B", quant: "4bit", context: 32768),

    // Gemma 2
    "gemma-2-2b": ResolvedModel(
        repo: "mlx-community/gemma-2-2b-it-4bit",
        name: "gemma-2-2b", family: .gemma, params: "2B", quant: "4bit", context: 8192),
    "gemma-2-9b": ResolvedModel(
        repo: "mlx-community/gemma-2-9b-it-4bit",
        name: "gemma-2-9b", family: .gemma, params: "9B", quant: "4bit", context: 8192),

    // Phi-3/4
    "phi-3-mini": ResolvedModel(
        repo: "mlx-community/Phi-3-mini-4k-instruct-4bit",
        name: "phi-3-mini", family: .phi, params: "3.8B", quant: "4bit", context: 4096),
    "phi-4-mini": ResolvedModel(
        repo: "mlx-community/phi-4-mini-instruct-4bit",
        name: "phi-4-mini", family: .phi, params: "3.8B", quant: "4bit", context: 131072),

    // DeepSeek R1 Distill (dense models - use Qwen/Llama architecture)
    "deepseek-r1-7b": ResolvedModel(
        repo: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        name: "deepseek-r1-7b", family: .qwen, params: "7B", quant: "4bit", context: 131072),
    "deepseek-r1-14b": ResolvedModel(
        repo: "mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        name: "deepseek-r1-14b", family: .qwen, params: "14B", quant: "4bit", context: 131072),
    "deepseek-r1-8b": ResolvedModel(
        repo: "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        name: "deepseek-r1-8b", family: .llama, params: "8B", quant: "4bit", context: 131072),

    // Gemma 4 (all variants - text mode)
    "gemma-4-e2b": ResolvedModel(
        repo: "mlx-community/gemma-4-e2b-it-4bit",
        name: "gemma-4-e2b", family: .gemma4, params: "2B", quant: "4bit", context: 131072),
    "gemma-4-e4b": ResolvedModel(
        repo: "mlx-community/gemma-4-e4b-it-4bit",
        name: "gemma-4-e4b", family: .gemma4, params: "4B", quant: "4bit", context: 131072),
    "gemma-4-12b": ResolvedModel(
        repo: "mlx-community/gemma-4-12b-it-4bit",
        name: "gemma-4-12b", family: .gemma4, params: "12B", quant: "4bit", context: 131072),
    "gemma-4-27b": ResolvedModel(
        repo: "mlx-community/gemma-4-27b-it-4bit",
        name: "gemma-4-27b", family: .gemma4, params: "27B", quant: "4bit", context: 131072),

    // GLM-4
    "glm-4-9b": ResolvedModel(
        repo: "mlx-community/glm-4-9b-chat-4bit",
        name: "glm-4-9b", family: .glm, params: "9B", quant: "4bit", context: 131072),

    // Sentence-embedding models (dedicated encoders for /api/embed,
    // /v1/embeddings). Original sentence-transformers/BAAI repos ship
    // model.safetensors + tokenizer.json, loaded by EmbeddingEngine.
    "all-minilm": ResolvedModel(
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        name: "all-minilm", family: .bert, params: "23M", quant: "fp32", context: 512),
    "all-minilm-l6-v2": ResolvedModel(
        repo: "sentence-transformers/all-MiniLM-L6-v2",
        name: "all-minilm-l6-v2", family: .bert, params: "23M", quant: "fp32", context: 512),
    "bge-small-en": ResolvedModel(
        repo: "BAAI/bge-small-en-v1.5",
        name: "bge-small-en", family: .bert, params: "33M", quant: "fp32", context: 512),
    "bge-base-en": ResolvedModel(
        repo: "BAAI/bge-base-en-v1.5",
        name: "bge-base-en", family: .bert, params: "109M", quant: "fp32", context: 512),

    // Cross-encoder rerankers (WS7 foundation tier: family
    // detection + clear rejection only; cross-encoder scoring
    // runtime + /v1/rerank endpoint land in follow-up PRs).
    "bge-reranker-base": ResolvedModel(
        repo: "BAAI/bge-reranker-base",
        name: "bge-reranker-base", family: .reranker, params: "278M", quant: "fp32", context: 512),
    "bge-reranker-large": ResolvedModel(
        repo: "BAAI/bge-reranker-large",
        name: "bge-reranker-large", family: .reranker, params: "560M", quant: "fp32", context: 512),
    "bge-reranker-v2-m3": ResolvedModel(
        repo: "BAAI/bge-reranker-v2-m3",
        name: "bge-reranker-v2-m3", family: .reranker, params: "568M", quant: "fp32", context: 8192),
]
