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
]
