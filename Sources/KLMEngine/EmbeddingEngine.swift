import Foundation
import MLX
import KLMCore
import KLMTokenizer

/// Loads and runs a *dedicated* sentence-embedding model (BERT-style
/// encoder: sentence-transformers / BGE / MiniLM / E5). Separate from
/// `InferenceEngine` so embeddings do not require - or disturb - a loaded
/// chat model, and so a RAG client can embed while chat is unloaded.
public final class EmbeddingEngine: @unchecked Sendable {
    private var model: BertEmbeddingModel?
    private var tokenizer: KLMTokenizer?
    private var loadedDir: URL?
    private var maxTokens: Int = 512
    private let pooling = EmbeddingPooling.fromEnv()
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

    private func install(model: BertEmbeddingModel, tokenizer: KLMTokenizer,
                         directory: URL, maxTokens: Int) {
        withLock {
            self.model = model
            self.tokenizer = tokenizer
            self.loadedDir = directory
            self.maxTokens = maxTokens
        }
    }

    /// Load (or hot-swap to) the embedding model in `directory`.
    public func load(directory: URL) async throws {
        if isLoaded(directory: directory) { return }

        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
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

        let tok = try await KLMTokenizer(from: directory)
        install(model: m, tokenizer: tok, directory: directory,
                maxTokens: config.maxPositionEmbeddings)
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
            if ids.count > cap { ids = Array(ids.prefix(cap)) }
            totalTokens += ids.count

            let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let hidden = model(tokens)
            vectors.append(
                poolSentenceEmbedding(hidden, pooling: pooling, normalize: true))
        }

        return EmbedResult(vectors: vectors, promptTokens: totalTokens)
    }
}

public enum EmbeddingError: Error, CustomStringConvertible {
    case notLoaded

    public var description: String {
        switch self {
        case .notLoaded:
            return "No embedding model loaded"
        }
    }
}
