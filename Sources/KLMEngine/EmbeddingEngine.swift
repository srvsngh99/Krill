import Foundation
import MLX
import MLXNN
import KLMCore
import KLMTokenizer

/// Loads and runs a *dedicated* sentence-embedding model (BERT-style
/// encoder: sentence-transformers / BGE / MiniLM / E5). Separate from
/// `InferenceEngine` so embeddings do not require - or disturb - a loaded
/// chat model, and so a RAG client can embed while chat is unloaded.
public final class EmbeddingEngine: @unchecked Sendable {
    private var model: (any SentenceEmbeddingEncoder)?
    private var tokenizer: KLMTokenizer?
    private var loadedDir: URL?
    private var maxTokens: Int = 512
    private var pooling = EmbeddingPooling.fromEnv()
    /// Decoder-LLM embedders (last-token pooling) append an EOS so the pooled
    /// final position has attended over the whole input; BERT encoders do not.
    private var appendEOS = false
    private var eosTokenId = 0
    private let lock = NSLock()

    public init() {}

    public var loadedModelName: String? {
        withLock { loadedDir?.lastPathComponent }
    }

    public func isLoaded(directory: URL) -> Bool {
        withLock {
            loadedDir?.standardizedFileURL == directory.standardizedFileURL
                && model != nil
        }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock(); defer { lock.unlock() }
        return body()
    }

    private func install(model: any SentenceEmbeddingEncoder, tokenizer: KLMTokenizer,
                         directory: URL, maxTokens: Int,
                         pooling: EmbeddingPooling, appendEOS: Bool, eosTokenId: Int) {
        withLock {
            self.model = model
            self.tokenizer = tokenizer
            self.loadedDir = directory
            self.maxTokens = maxTokens
            self.pooling = pooling
            self.appendEOS = appendEOS
            self.eosTokenId = eosTokenId
        }
    }

    /// Load (or hot-swap to) the embedding model in `directory`.
    public func load(directory: URL) async throws {
        if isLoaded(directory: directory) { return }

        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let mt = Self.modelType(from: data) ?? ""
        let tok = try await KLMTokenizer(from: directory)

        let model: any SentenceEmbeddingEncoder
        let maxTokens: Int
        let pooling: EmbeddingPooling
        var appendEOS = false

        if mt == "nomic_bert",
           let v2 = try? JSONDecoder().decode(NomicBertV2Config.self, from: data), v2.isMoE {
            // nomic-embed-text-v2-moe: same `nomic_bert` model_type as v1.5 but a
            // top-2 mixture of experts on every 2nd layer (XLM-R vocab). Detected
            // by the MoE config fields. Keys match the checkpoint; strict verify
            // guards mismatch. fp32 weights, mean-pooled.
            let m = NomicBertV2MoEModel(v2)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            eval(m)
            model = m
            maxTokens = v2.maxTokens
            pooling = Self.envPooling() ?? .mean
        } else if mt == "nomic_bert" {
            // nomic-embed-text: a RoPE encoder (fused Wqkv + SwiGLU), distinct
            // from vanilla BERT/RoBERTa. Checkpoint keys already match the module
            // keys (no `bert.`/`roberta.` prefix); strict verify guards mismatch.
            let config = try JSONDecoder().decode(NomicBertConfig.self, from: data)
            let m = NomicBertEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling() ?? .mean
        } else if mt == "mpnet" {
            // MPNet: relative-attention-bias encoder, no token-type embeddings,
            // RoBERTa-style offset positions. The checkpoint ships a `pooler` and
            // a `position_ids` buffer this encoder does not use; drop them so a
            // strict-verify update sees an exact key match.
            let config = try JSONDecoder().decode(MPNetConfig.self, from: data)
            let m = MPNetEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil, keyPrefix: nil,
                            keyRewrite: { weights in
                                for key in weights.keys
                                where key.hasPrefix("pooler.") || key == "embeddings.position_ids" {
                                    weights.removeValue(forKey: key)
                                }
                            }, strictVerify: true)
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling() ?? .mean
        } else if mt == "new" {
            // GTE-v1.5 ("NewModel"): RoPE encoder with biased fused qkv, GeGLU
            // MLP, post-norm, no token-type. CLS-pooled. Keys match 1:1.
            let config = try JSONDecoder().decode(GTEConfig.self, from: data)
            let m = GTEEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .cls
        } else if mt == "modernbert" {
            // ModernBERT: pre-norm RoPE encoder with alternating global/local
            // attention (per-layer theta + sliding window), GeGLU, weight-only
            // norms, no biases. Keys map 1:1 (layer 0 has no attn_norm).
            let config = try JSONDecoder().decode(ModernBertConfig.self, from: data)
            let m = ModernBertEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil,
                            keyPrefix: nil, strictVerify: true)
            // ModernBERT ships fp16 weights but its activations overflow fp16
            // (GeGLU intermediates run large); upcast to fp32 for a stable,
            // reference-matching forward.
            m.update(parameters: m.parameters().mapValues { $0.asType(.float32) })
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .cls
        } else if Self.causalEmbedderTypes.contains(mt) {
            // Decoder-LLM embedder (gte-Qwen2, e5-mistral, ...): reuse the
            // already-validated causal backbone via the shared loader, then pool
            // its final hidden state. Last-token pooling appends an EOS upstream
            // so the pooled position has attended over the whole sequence.
            let loaded = try loadModel(from: directory)
            guard let enc = loaded.module as? (any SentenceEmbeddingEncoder) else {
                throw EmbeddingError.unsupported(mt)
            }
            // fp16 decoder backbones (e5-mistral, SFR, ...) carry massive
            // residual-stream activations that overflow fp16 (-> inf -> the
            // final RMSNorm divides to an all-zero vector, so every embedding
            // comes back zero). A full fp32 upcast would double a 7B past this
            // host's RAM, so upcast only embed_tokens: that seeds the residual
            // stream in fp32 and MLX promotes the fp16 weight matmuls to fp32
            // from there, keeping the stream fp32 end to end for one extra
            // embedding table. No-op when the backbone is already fp32.
            let upcast = loaded.module.parameters().flattened()
                .filter { $0.0.contains("embed_tokens") && $0.1.dtype == .float16 }
                .map { ($0.0, $0.1.asType(.float32)) }
            if !upcast.isEmpty {
                loaded.module.update(parameters: ModuleParameters.unflattened(upcast))
            }
            model = enc
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .lastToken
            appendEOS = (pooling == .lastToken)
            // Cap context to keep a single embed forward bounded (these backbones
            // advertise 100k+ positions); deepkrill-scale chunks sit far under it.
            maxTokens = min(Self.maxPositionEmbeddings(from: data) ?? 8192, 8192)
        } else if Self.isJinaBert(data) {
            // jina-embeddings-v2: model_type is "bert" but it uses ALiBi (no
            // positional embeddings) and a GLU MLP, so it must NOT route to the
            // vanilla BERT loader. Drop the unused `pooler.*`; upcast fp16 -> fp32.
            let config = try JSONDecoder().decode(JinaBertConfig.self, from: data)
            let m = JinaBertEmbeddingModel(config)
            try loadWeights(into: m, from: directory, quantization: nil, keyPrefix: nil,
                            keyRewrite: { weights in
                                for key in weights.keys where key.hasPrefix("pooler.") {
                                    weights.removeValue(forKey: key)
                                }
                            }, strictVerify: true)
            m.update(parameters: m.parameters().mapValues { $0.asType(.float32) })
            eval(m)
            model = m
            maxTokens = config.maxTokens
            pooling = Self.envPooling()
                ?? Self.sentenceTransformerPooling(directory: directory) ?? .mean
        } else {
            let config = try JSONDecoder().decode(BertEmbeddingConfig.self, from: data)
            let m = BertEmbeddingModel(config)
            // BertModel checkpoints may prefix keys with `bert.`/`roberta.`.
            let raw = try loadWeightArrays(from: directory)
            let prefix: String? =
                raw.keys.contains { $0.hasPrefix("bert.") } ? "bert."
                : raw.keys.contains { $0.hasPrefix("roberta.") } ? "roberta."
                : nil
            try loadWeights(into: m, from: directory, quantization: nil, keyPrefix: prefix)
            eval(m)
            model = m
            maxTokens = config.maxPositionEmbeddings
            pooling = Self.envPooling() ?? .mean
        }

        install(model: model, tokenizer: tok, directory: directory,
                maxTokens: maxTokens, pooling: pooling,
                appendEOS: appendEOS, eosTokenId: tok.eosTokenId)
    }

    /// Peek the `model_type` field from a raw config.json to select the encoder
    /// architecture without committing to a full config decode.
    private static func modelType(from configData: Data) -> String? {
        struct Peek: Decodable {
            let modelType: String?
            enum CodingKeys: String, CodingKey { case modelType = "model_type" }
        }
        return (try? JSONDecoder().decode(Peek.self, from: configData))?.modelType
    }

    /// jina-embeddings-v2 declares `model_type: "bert"` but uses ALiBi; route it
    /// to the JinaBERT encoder rather than the vanilla BERT loader. Detected by
    /// `position_embedding_type == "alibi"` or a `JinaBert*` architecture.
    private static func isJinaBert(_ configData: Data) -> Bool {
        struct Peek: Decodable {
            let positionEmbeddingType: String?
            let architectures: [String]?
            enum CodingKeys: String, CodingKey {
                case positionEmbeddingType = "position_embedding_type"
                case architectures
            }
        }
        guard let p = try? JSONDecoder().decode(Peek.self, from: configData) else { return false }
        if p.positionEmbeddingType?.lowercased() == "alibi" { return true }
        return p.architectures?.contains { $0.lowercased().contains("jinabert") } ?? false
    }

    public struct EmbedResult: Sendable {
        public let vectors: [[Float]]
        public let promptTokens: Int
    }

    /// Embed a batch of texts. Each text is encoded and run independently
    /// (batch=1) so no padding mask is needed - keeps the forward exact.
    public func embed(_ texts: [String]) throws -> EmbedResult {
        lock.lock()
        let model = self.model
        let tokenizer = self.tokenizer
        let cap = self.maxTokens
        let pooling = self.pooling
        let appendEOS = self.appendEOS
        let eos = self.eosTokenId
        lock.unlock()

        guard let model, let tokenizer else {
            throw EmbeddingError.notLoaded
        }

        var vectors: [[Float]] = []
        vectors.reserveCapacity(texts.count)
        var totalTokens = 0

        for text in texts {
            var ids = tokenizer.encode(text)
            if ids.isEmpty { ids = [tokenizer.bosTokenId] }
            if appendEOS {
                // Last-token decoder embedders pool the EOS position. Normalize
                // to exactly one trailing EOS (some tokenizers, e.g. e5-mistral,
                // already append it; others, e.g. gte-Qwen2, do not), reserving a
                // slot for it when truncating.
                if ids.last == eos { ids.removeLast() }
                if ids.count > cap - 1 { ids = Array(ids.prefix(cap - 1)) }
                ids.append(eos)
            } else if ids.count > cap {
                ids = Array(ids.prefix(cap))
            }
            totalTokens += ids.count

            let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let hidden = model.lastHiddenState(tokens)
            vectors.append(
                poolSentenceEmbedding(hidden, pooling: pooling, normalize: true))
        }

        return EmbedResult(vectors: vectors, promptTokens: totalTokens)
    }

    // MARK: - Decoder-LLM embedder detection

    /// Causal base architectures that can be repurposed as sentence embedders.
    /// Limited to the dense families that conform to `SentenceEmbeddingEncoder`
    /// (see `DecoderEmbedder.swift`), so an unsupported backbone is rejected at
    /// the gate (400) rather than admitted and failing in `load` (500). Add a
    /// family here only once its `*ForCausalLM` conforms (e.g. Gemma for
    /// bge-multilingual-gemma2). Qwen3 MoE has model_type `qwen3_moe`, so it does
    /// not match `qwen3` here.
    private static let causalEmbedderTypes: Set<String> = [
        "qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2",
    ]

    /// True when `directory` holds a decoder-LLM embedder: a causal base arch
    /// plus a sentence-transformers `1_Pooling/config.json`. Used by the server
    /// to admit such models through the embeddings endpoint (their family is
    /// `.qwen`/`.mistral`/... not `.bert`).
    public static func isDecoderEmbedder(directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              let mt = modelType(from: data), causalEmbedderTypes.contains(mt) else {
            return false
        }
        return FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("1_Pooling/config.json").path)
    }

    /// Explicit `KRILL_EMBED_POOLING` override, or nil when unset (so each model
    /// keeps its natural default: mean for BERT, last-token for decoder embedders).
    private static func envPooling() -> EmbeddingPooling? {
        guard let v = ProcessInfo.processInfo.environment["KRILL_EMBED_POOLING"] else {
            return nil
        }
        return EmbeddingPooling.from(v)
    }

    private static func maxPositionEmbeddings(from data: Data) -> Int? {
        struct Peek: Decodable {
            let maxPos: Int?
            enum CodingKeys: String, CodingKey { case maxPos = "max_position_embeddings" }
        }
        return (try? JSONDecoder().decode(Peek.self, from: data))?.maxPos
    }

    /// Read the pooling mode from a sentence-transformers `1_Pooling/config.json`.
    private static func sentenceTransformerPooling(directory: URL) -> EmbeddingPooling? {
        let url = directory.appendingPathComponent("1_Pooling/config.json")
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        if json["pooling_mode_lasttoken"] as? Bool == true { return .lastToken }
        if json["pooling_mode_cls_token"] as? Bool == true { return .cls }
        if json["pooling_mode_mean_tokens"] as? Bool == true { return .mean }
        return nil
    }
}

public enum EmbeddingError: Error, CustomStringConvertible {
    case notLoaded
    case unsupported(String)

    public var description: String {
        switch self {
        case .notLoaded:
            return "No embedding model loaded"
        case .unsupported(let mt):
            return "Model type '\(mt)' is not a supported embedder"
        }
    }
}
