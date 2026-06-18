import Foundation

/// A model manifest describes an installed model in the local registry.
///
/// Manifests live at ~/.krill/models/manifests/<name>.json and reference
/// content-addressed blobs in ~/.krill/models/blobs/.
public struct ModelManifest: Codable, Sendable {
    /// Human-friendly name (e.g., "llama-3.1-8b")
    public let name: String

    /// Model family identifier for architecture dispatch
    public let family: ModelFamily

    /// Parameter count label (e.g., "8B", "3B")
    public let params: String

    /// Quantization identifier (e.g., "4bit", "8bit", "fp16")
    public let quant: String

    /// HuggingFace source repo (e.g., "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    public let source: String

    /// Maximum context length
    public let context: Int

    /// Files in this model (relative paths -> sha256 hashes)
    public let files: [ModelFile]

    /// Optional draft model name for speculative decoding (Phase 3)
    public let draftPair: String?

    /// Chat template identifier
    public let chatTemplate: String

    /// Size on disk in bytes
    public let sizeBytes: Int64

    /// When this model was pulled
    public let pulledAt: Date

    /// Modelfile-derived customizations (WS-C). Optional + decoded with
    /// `decodeIfPresent` (synthesized), so pre-existing manifests without
    /// this key still decode.
    public let overrides: ModelOverrides?

    public init(
        name: String,
        family: ModelFamily,
        params: String,
        quant: String,
        source: String,
        context: Int,
        files: [ModelFile],
        draftPair: String? = nil,
        chatTemplate: String,
        sizeBytes: Int64,
        pulledAt: Date = Date(),
        overrides: ModelOverrides? = nil
    ) {
        self.name = name
        self.family = family
        self.params = params
        self.quant = quant
        self.source = source
        self.context = context
        self.files = files
        self.draftPair = draftPair
        self.chatTemplate = chatTemplate
        self.sizeBytes = sizeBytes
        self.pulledAt = pulledAt
        self.overrides = overrides
    }
}

/// Modelfile-derived overrides layered on top of a base model (WS-C / T1-2).
public struct ModelOverrides: Codable, Sendable, Equatable {
    public var system: String?
    public var template: String?
    public var license: String?
    public var parameters: [String: String]
    public var messages: [[String: String]]

    public init(system: String? = nil, template: String? = nil,
                license: String? = nil,
                parameters: [String: String] = [:],
                messages: [[String: String]] = []) {
        self.system = system
        self.template = template
        self.license = license
        self.parameters = parameters
        self.messages = messages
    }
}

/// A file within a model (weight shard, tokenizer, config, etc.)
public struct ModelFile: Codable, Sendable {
    /// Relative path within the model directory
    public let path: String

    /// SHA256 hash of the file contents
    public let sha256: String

    /// File size in bytes
    public let sizeBytes: Int64

    public init(path: String, sha256: String, sizeBytes: Int64) {
        self.path = path
        self.sha256 = sha256
        self.sizeBytes = sizeBytes
    }
}

/// Supported model families for architecture dispatch.
public enum ModelFamily: String, Codable, Sendable, CaseIterable {
    case llama
    case qwen
    case mistral
    case gemma
    case gemma4
    /// Gemma 4 12B "unified": the encoder-free multimodal Gemma 4 SKU
    /// (`model_type: "gemma4_unified"`). Unlike e2b/e4b/26B-A4B it has no
    /// SigLIP vision tower or USM audio encoder - raw image patches and raw
    /// audio sample-frames project straight into the text embedding space
    /// through thin linear pipelines (`vision_embedder` / `embed_vision` /
    /// `embed_audio`). The text decoder is the same dense Gemma 4 backbone.
    case gemma4Unified = "gemma4_unified"
    case phi
    case glm
    /// GLM-4-0414 / GLM-Z1 generation (arch `Glm4ForCausalLM`, model_type
    /// "glm4"). Distinct native runtime from the legacy ChatGLM `.glm`: separate
    /// q/k/v/o projections, four-RMSNorm sandwich, partial RoPE, fused gate_up.
    case glm4
    case deepseek
    /// Dedicated sentence-embedding encoder (BERT/RoBERTa/MiniLM/BGE/E5).
    /// Not a causal LM - served only via the embeddings endpoints.
    case bert
    /// Qwen2.5-VL vision-language family. Architectural deltas vs the
    /// Qwen 2.5 text family: a SigLIP/ViT-style vision tower with
    /// window attention, a patch merger, and 3D mRoPE (temporal +
    /// height + width axes with the rope_scaling `mrope_section`
    /// split). Foundation only in this build: load + family
    /// detection + capability metadata + clear rejection at the
    /// inference boundary. The native vision tower, patch merger,
    /// mRoPE, and image preprocessing land in follow-up PRs (see
    /// docs/workstreams/WS5_SECOND_NATIVE_VISION_FAMILY.md).
    case qwen25vl = "qwen2_5_vl"
    /// LLaVA-1.5 vision-language family. A CLIP ViT vision tower + a
    /// multi-modal projector (linear -> gelu -> linear) + a Llama text
    /// backbone; the projected CLIP features are spliced into the token
    /// embeddings at the `<image>` placeholder positions and the Llama
    /// stack runs over the merged embeddings (no cross-attention, unlike
    /// mllama). Native Swift+MLX runtime landed in PR #129
    /// (`LlavaForCausalLM`, mlx-vlm logit-parity-verified); the engine
    /// image-serving wiring (preprocessing + prompt construction +
    /// generation) is this family's entry here. Only `model_type` "llava"
    /// (llava-1.5) is supported; llava-next / llava-bunny are not.
    case llava
    /// Llama-3.2-Vision (mllama) vision-language family. A tiled ViT vision
    /// tower + multi-modal projector + a Llama text decoder whose
    /// `cross_attention_layers` attend to the projected vision features (vision
    /// enters via cross-attention, unlike LLaVA's prefix-embed splice). Native
    /// Swift+MLX runtime (`Llama32VisionForCausalLM`), mlx-vlm logit-parity
    /// verified on a synthetic checkpoint. Image-serving wiring (tile /
    /// aspect-ratio preprocessing + a cross-KV decode driver) is a follow-up, so
    /// this declares text generation only for now.
    case llamaVision = "llama_vision"
    /// Mixture-of-experts text LMs. Architectural deltas vs the
    /// dense families: each transformer block's MLP is replaced by
    /// a router + N expert FFNs, where the router picks the top-K
    /// experts per token. Initial members: Mixtral (`mixtral`,
    /// `MixtralForCausalLM`) and Qwen 3 MoE (`qwen3_moe`,
    /// `Qwen3MoeForCausalLM`). Foundation only in this build: the
    /// loader rejects with a clear error before any weight load.
    /// The router + expert dispatch lands in follow-up PRs (see
    /// docs/workstreams/WS6_MOE_RUNTIME_SUPPORT.md).
    case moe
    /// Cross-encoder reranker (e.g. BGE Reranker, Cohere Rerank).
    /// Architecturally close to the `bert` embedding family but
    /// with a sequence-classification head that produces a single
    /// relevance score per (query, document) pair, NOT pooled
    /// embeddings. Foundation only in this build: family detection
    /// + capability metadata + clear rejection at the inference
    /// boundary. The cross-encoder scoring runtime + `/v1/rerank`
    /// endpoint land in follow-up PRs (see
    /// docs/workstreams/WS7_SPECIALIZED_MODEL_TYPES.md).
    case reranker

    /// Detect model family from HuggingFace config.json's `architectures` field.
    public static func detect(from configJSON: [String: Any]) -> ModelFamily? {
        guard let architectures = configJSON["architectures"] as? [String],
              let arch = architectures.first else {
            // Fallback: try model_type
            if let modelType = configJSON["model_type"] as? String {
                return fromModelType(modelType)
            }
            return nil
        }

        let archLower = arch.lowercased()
        // Order matters: check specific before generic. Qwen 2.5-VL
        // (`Qwen2_5_VLForConditionalGeneration`) must be matched
        // BEFORE the generic `qwen` arm so it routes to the
        // multimodal family rather than the dense text loader.
        // Reranker arms are matched BEFORE the bert / roberta arm
        // because they share the same backbone architecture but
        // ship a sequence-classification head that the embedding
        // loader cannot consume. Without this ordering, a
        // `XLMRobertaForSequenceClassification` (BGE Reranker)
        // checkpoint would route to .bert and the loader would
        // either crash on the classifier weights or run with no
        // scoring head at all.
        if archLower.contains("forsequenceclassification") { return .reranker }
        if archLower.contains("crossencoder") { return .reranker }
        if archLower.contains("bert") || archLower.contains("roberta") { return .bert }
        // MPNet (`MPNetForMaskedLM`) is a sentence-embedding encoder but its arch
        // name contains neither "bert" nor "roberta"; the engine loads it via its
        // own relative-attention encoder selected from model_type.
        if archLower.contains("mpnet") { return .bert }
        // GTE-v1.5 ships as `NewModel` (model_type "new"): a RoPE sentence
        // encoder, loaded via the engine's GTE path. Still a `.bert`-family
        // embedder at the registry level so the embeddings gate admits it.
        if archLower.contains("newmodel") { return .bert }
        // Encoder-free unified Gemma 4 (12B) before the generic gemma4 arm:
        // its arch is `Gemma4UnifiedForConditionalGeneration`, which also
        // contains "gemma4", so the specific check must come first.
        if archLower.contains("gemma4unified") { return .gemma4Unified }
        if archLower.contains("gemma4") { return .gemma4 }
        if archLower.contains("gemma") { return .gemma }
        if archLower.contains("chatglm") || archLower.contains("glm") { return .glm }
        if archLower.contains("deepseek") { return .deepseek }
        // LLaVA-1.5 (`LlavaForConditionalGeneration`) must be matched BEFORE
        // the generic `llama` arm: its text backbone is Llama, so the arch
        // name would otherwise need to (and does not) contain "llama". A
        // raw HF llava checkpoint declares `LlavaForConditionalGeneration`.
        if archLower.contains("llavaforconditionalgeneration") { return .llava }
        // mllama (Llama-3.2-Vision) before generic llama: arch is
        // `MllamaForConditionalGeneration`.
        if archLower.contains("mllama") { return .llamaVision }
        if archLower.contains("llama") { return .llama }
        // MoE arms are matched BEFORE the generic qwen / mistral
        // arms so a Qwen 3 MoE or Mixtral checkpoint never silently
        // routes to a dense text loader (which would either crash
        // on the extra router/expert keys or run with garbage MLP
        // weights). Order: specific-then-generic.
        if archLower.contains("mixtral") { return .moe }
        if archLower.contains("qwen3moe") || archLower.contains("qwen2moe") { return .moe }
        if archLower.contains("olmoe") { return .moe }
        if archLower.contains("qwen2_5_vl") || archLower.contains("qwen2vl") { return .qwen25vl }
        if archLower.contains("qwen") { return .qwen }
        if archLower.contains("mistral") { return .mistral }
        if archLower.contains("phi") { return .phi }
        return nil
    }

    private static func fromModelType(_ type: String) -> ModelFamily? {
        switch type.lowercased() {
        case "llama": return .llama
        case "qwen2", "qwen3": return .qwen
        case "mistral": return .mistral
        case "gemma", "gemma2", "gemma3": return .gemma
        case "gemma4", "gemma4_text": return .gemma4
        case "qwen2_5_vl", "qwen2_vl": return .qwen25vl
        case "llava": return .llava
        case "mllama": return .llamaVision
        case "mixtral", "qwen3_moe", "qwen2_moe", "olmoe": return .moe
        case "phi", "phi3": return .phi
        case "glm4": return .glm4
        case "chatglm", "glm", "glm4_moe": return .glm
        case "deepseek_v3": return .deepseek
        case "bert", "roberta", "xlm-roberta", "mpnet", "distilbert": return .bert
        // nomic-embed-text: RoPE encoder, still a sentence-embedding (.bert)
        // family at the registry level; the embedding engine selects the
        // NomicBert encoder from model_type at load time.
        case "nomic_bert", "nomic-bert": return .bert
        // GTE-v1.5 / ModernBERT RoPE encoders (the arch arm already maps the
        // "NewModel"/"ModernBertModel" architectures; this covers config.json
        // that declares only model_type).
        case "new", "modernbert": return .bert
        default: return nil
        }
    }
}
