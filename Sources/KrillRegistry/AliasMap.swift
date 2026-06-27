import Foundation

/// Curated alias map: short model names -> HuggingFace repos.
///
/// These are the recommended MLX-quantized models from mlx-community
/// for each supported family. krill pull <alias> resolves through
/// this map before falling back to treating the input as a raw HF repo path.
public struct AliasMap: Sendable {
    /// Resolve a model name to a HuggingFace repo path.
    ///
    /// Priority:
    /// 1. Exact match in the built-in alias map
    /// 2. Exact match in the on-disk catalog cache, when a `catalog`
    ///    store is supplied (lets models be added without rebuilding)
    /// 3. Treat as raw HF repo path if it contains "/"
    /// 4. Return nil (not found)
    ///
    /// The built-in map always wins over the catalog, so a catalog can
    /// never silently shadow a curated, tested alias.
    public static func resolve(
        _ name: String, catalog: ModelCatalogStore? = nil
    ) -> ResolvedModel? {
        let normalized = name.lowercased().trimmingCharacters(in: .whitespaces)

        // Direct alias match (built-in, curated map).
        if let entry = aliases[normalized] {
            return entry
        }

        // Catalog fallback: a model added via `krill catalog` without
        // a binary rebuild. Only consulted when a store is supplied.
        if let fromCatalog = catalog?.resolve(normalized) {
            return fromCatalog
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

    // Ornith-9B (qwen3_5): Qwen3-Next-class hybrid (GatedDeltaNet
    // linear-attention + full attention). Native text decoder; the
    // vision tower is deferred to mlx_vlm. `repo` points at the Krill-ready
    // affine-int4 quant we publish under srv-sngh (linked to the upstream
    // `deepreinforce-ai/Ornith-1.0-9B`); see the HF-publish step.
    "ornith-9b": ResolvedModel(
        repo: "srv-sngh/Ornith-1.0-9B-4bit",
        name: "ornith-9b", family: .qwen35, params: "9B", quant: "4bit", context: 262144),

    // Mixture-of-experts (WS6 foundation tier: family detection +
    // clear rejection only; router + expert dispatch lands in
    // follow-up PRs).
    "mixtral-8x7b": ResolvedModel(
        repo: "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        name: "mixtral-8x7b", family: .moe, params: "8x7B", quant: "4bit", context: 32768),
    "qwen3-30b-a3b": ResolvedModel(
        repo: "mlx-community/Qwen3-30B-A3B-4bit",
        name: "qwen3-30b-a3b", family: .moe, params: "30B-A3B", quant: "4bit", context: 40960),
    "olmoe-1b-7b": ResolvedModel(
        repo: "mlx-community/OLMoE-1B-7B-0924-Instruct-4bit",
        name: "olmoe-1b-7b", family: .moe, params: "1B-7B", quant: "4bit", context: 4096),

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
    // Gemma 4 12B "unified" (released 2026-06-03): the encoder-free
    // multimodal SKU. `family: .gemma4Unified` routes it to the
    // `Gemma4UnifiedModel` loader (raw-patch / raw-audio linear front-ends
    // over the shared dense Gemma 4 backbone), NOT the SigLIP/USM path the
    // e2b/e4b/.gemma4 entries use. Trained window 131072 (the checkpoint's
    // max_position_embeddings; an earlier 262144 here was the vocab size).
    "gemma-4-12b": ResolvedModel(
        repo: "mlx-community/gemma-4-12B-it-4bit",
        name: "gemma-4-12b", family: .gemma4Unified, params: "12B", quant: "4bit",
        context: 131072),
    // Community coding/reasoning fine-tune of gemma-4-12B-it, served natively in
    // MLX. Converted GGUF-free from the upstream NVFP4 safetensors
    // (compressed-tensors) via tools/convert_gemma4_compressed_nvfp4_to_bf16.py
    // + tools/requant_gemma4_nvfp4.py (the proven mixed-nvfp4 recipe: 8-bit
    // o_proj + vision/audio projectors). Same `gemma4_unified` backbone as
    // gemma-4-12b - see docs/GEMMA4_12B_CODER_FINETUNE.md.
    "gemma-4-12b-coder": ResolvedModel(
        repo: "srv-sngh/gemma-4-12B-coder-fable5-composer2.5-nvfp4",
        name: "gemma-4-12b-coder", family: .gemma4Unified, params: "12B",
        quant: "nvfp4", context: 131072),
    // Community AGENTIC fine-tune of gemma-4-12B-it (tool-use / tau2), served
    // natively in MLX. Converted from bf16 safetensors -> MLX-layout key remap
    // -> the proven mixed-nvfp4 requant (8-bit o_proj + vision/audio projectors).
    // Same `gemma4_unified` backbone - see docs/GEMMA4_12B_AGENTIC_FINETUNE.md.
    "gemma-4-12b-agentic": ResolvedModel(
        repo: "srv-sngh/gemma-4-12B-agentic-fable5-composer2.5-v2-nvfp4",
        name: "gemma-4-12b-agentic", family: .gemma4Unified, params: "12B",
        quant: "nvfp4", context: 131072),
    // Google's Gemma 4 lineup is E2B / E4B / 12B (unified) / 26B-A4B / 31B.
    // 26B-A4B and 31B are not aliased here yet (the `Gemma4Model` loader is
    // currently e2b-shape-specific) - see docs/FOLLOWUPS_AGENT_DOGFOOD.md §1.

    // GLM-4 (legacy ChatGLM architecture -> `.glm` / `loadGLM`)
    "glm-4-9b": ResolvedModel(
        repo: "mlx-community/glm-4-9b-chat-4bit",
        name: "glm-4-9b", family: .glm, params: "9B", quant: "4bit", context: 131072),

    // GLM-4-0414 / GLM-Z1 generation (arch Glm4ForCausalLM -> `.glm4` /
    // `loadGlm4`, the native Swift+MLX runtime in Glm4Model.swift). mlx-community
    // already hosts these MLX weights, so the aliases point straight at them.
    "glm-4-9b-0414": ResolvedModel(
        repo: "mlx-community/GLM-4-9B-0414-4bit",
        name: "glm-4-9b-0414", family: .glm4, params: "9B", quant: "4bit", context: 32768),
    "glm-z1-9b": ResolvedModel(
        repo: "mlx-community/GLM-Z1-9B-0414-4bit",
        name: "glm-z1-9b", family: .glm4, params: "9B", quant: "4bit", context: 32768),
    "glm-4-32b-0414": ResolvedModel(
        repo: "mlx-community/GLM-4-32B-0414-4bit",
        name: "glm-4-32b-0414", family: .glm4, params: "32B", quant: "4bit", context: 32768),

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

    // MPNet encoder: relative-attention-bias backbone (not vanilla BERT),
    // routed to MPNetEmbeddingModel by model_type. 768-dim, mean pooling.
    "all-mpnet-base-v2": ResolvedModel(
        repo: "sentence-transformers/all-mpnet-base-v2",
        name: "all-mpnet-base-v2", family: .bert, params: "109M", quant: "fp32", context: 514),

    // GTE-v1.5 ("NewModel"): RoPE encoder, GeGLU MLP, CLS-pooled, 8192 ctx.
    // gte-large adds fixed-NTK rope_scaling (theta 160000, factor 2), handled
    // by the GTE encoder's custom-freqs path; gte-base has no scaling.
    "gte-base-en-v1.5": ResolvedModel(
        repo: "Alibaba-NLP/gte-base-en-v1.5",
        name: "gte-base-en-v1.5", family: .bert, params: "137M", quant: "fp32", context: 8192),
    "gte-large-en-v1.5": ResolvedModel(
        repo: "Alibaba-NLP/gte-large-en-v1.5",
        name: "gte-large-en-v1.5", family: .bert, params: "434M", quant: "fp32", context: 8192),

    // ModernBERT encoder: pre-norm RoPE, alternating global/local attention,
    // GeGLU. CLS-pooled, 8192 ctx.
    "gte-modernbert-base": ResolvedModel(
        repo: "Alibaba-NLP/gte-modernbert-base",
        name: "gte-modernbert-base", family: .bert, params: "149M", quant: "fp32", context: 8192),

    // JinaBERT encoder: ALiBi (no positional embeddings) + GLU MLP, mean-pooled.
    "jina-embeddings-v2-base-en": ResolvedModel(
        repo: "jinaai/jina-embeddings-v2-base-en",
        name: "jina-embeddings-v2-base-en", family: .bert, params: "137M", quant: "fp32", context: 8192),

    // nomic-embed-text: a `nomic_bert` RoPE encoder (fused Wqkv + SwiGLU),
    // 768-dim, served fp32. Routed to NomicBertEmbeddingModel by model_type at
    // load. `nomic-embed-text` tracks Ollama's default tag (v1.5).
    "nomic-embed-text": ResolvedModel(
        repo: "nomic-ai/nomic-embed-text-v1.5",
        name: "nomic-embed-text", family: .bert, params: "137M", quant: "fp32", context: 2048),
    "nomic-embed-text-v1.5": ResolvedModel(
        repo: "nomic-ai/nomic-embed-text-v1.5",
        name: "nomic-embed-text-v1.5", family: .bert, params: "137M", quant: "fp32", context: 2048),
    "nomic-embed-text-v1": ResolvedModel(
        repo: "nomic-ai/nomic-embed-text-v1",
        name: "nomic-embed-text-v1", family: .bert, params: "137M", quant: "fp32", context: 2048),

    // nomic-embed-text-v2-moe: a `nomic_bert` MoE encoder (top-2 of 8 experts on
    // every 2nd layer, XLM-R vocab). Routed to NomicBertV2MoEModel by the MoE
    // config fields. Needs the XLM-R Metaspace tokenizer fix. Mean-pooled, fp32.
    "nomic-embed-text-v2-moe": ResolvedModel(
        repo: "nomic-ai/nomic-embed-text-v2-moe",
        name: "nomic-embed-text-v2-moe", family: .bert, params: "475M", quant: "fp32", context: 2048),

    // Decoder-LLM embedders: causal backbones (family .qwen) repurposed as
    // sentence embedders via last-token pooling. Served through the same Qwen
    // loader as chat; the embeddings engine detects the sentence-transformers
    // pooling head on disk and pools the final hidden state. 1536/3584-dim.
    "gte-qwen2-1.5b": ResolvedModel(
        repo: "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        name: "gte-qwen2-1.5b", family: .qwen, params: "1.5B", quant: "fp32", context: 32768),
    "gte-qwen2-7b": ResolvedModel(
        repo: "Alibaba-NLP/gte-Qwen2-7B-instruct",
        name: "gte-qwen2-7b", family: .qwen, params: "7B", quant: "fp32", context: 32768),

    // Mistral-backbone instruction embedders (e5-mistral, SFR). Shipped as base
    // `MistralModel` checkpoints (no `model.` prefix, no lm_head), normalized by
    // the loader's base-model prefix path; fp16 residual overflow is handled by
    // the embeddings engine's embed_tokens fp32 upcast. Queries take the
    // "Instruct: {task}\nQuery: " prefix (via the endpoint's `instruction`
    // field); documents are sent raw. Verified cosine 1.0 vs transformers.
    "e5-mistral-7b-instruct": ResolvedModel(
        repo: "intfloat/e5-mistral-7b-instruct",
        name: "e5-mistral-7b-instruct", family: .mistral, params: "7B", quant: "fp16", context: 4096),
    "sfr-embedding-mistral": ResolvedModel(
        repo: "Salesforce/SFR-Embedding-Mistral",
        name: "sfr-embedding-mistral", family: .mistral, params: "7B", quant: "fp16", context: 4096),

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
