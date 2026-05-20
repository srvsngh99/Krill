import Foundation
import MLX
import KLMCore
import KLMTokenizer

/// Loads and runs a cross-encoder reranker (BGE Reranker / XLMRoberta
/// or Bert sequence-classification head with a single label).
///
/// Distinct from `EmbeddingEngine` and `InferenceEngine` because:
///   - It is not an autoregressive LM, so `InferenceEngine`'s decode
///     loop is irrelevant.
///   - It is not pooled-embedding, so `EmbeddingEngine`'s pooling +
///     L2-normalize path produces the wrong shape (the reranker
///     reads `[CLS]` and runs a classifier head, not a pooler).
///
/// Usage:
///   `engine.score(query: "...", documents: ["doc1", "doc2", ...])`
/// runs the model once per (query, document) pair and returns the raw
/// classifier logits. Clients sort documents by logit descending.
public final class RerankEngine: @unchecked Sendable {
    private var model: RerankerModel?
    private var tokenizer: KLMTokenizer?
    private var loadedDir: URL?
    private var maxTokens: Int = 512
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

    private func install(model: RerankerModel, tokenizer: KLMTokenizer,
                         directory: URL, maxTokens: Int) {
        withLock {
            self.model = model
            self.tokenizer = tokenizer
            self.loadedDir = directory
            self.maxTokens = maxTokens
        }
    }

    /// Load (or hot-swap to) the reranker model in `directory`.
    public func load(directory: URL) async throws {
        if isLoaded(directory: directory) { return }

        let configURL = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(BertEmbeddingConfig.self, from: data)

        let m = RerankerModel(config, numLabels: 1)
        // BGE Reranker keys are nested under `roberta.` for the
        // backbone and bare `classifier.` for the head. Plain
        // BERT-class rerankers nest under `bert.`. The loader's
        // existing `keyPrefix` argument strips the backbone
        // prefix only; classifier weights already match the
        // RerankerModel module hierarchy.
        let raw = try loadWeightArrays(from: directory)
        let prefix: String? =
            raw.keys.contains { $0.hasPrefix("roberta.") } ? "roberta."
            : raw.keys.contains { $0.hasPrefix("bert.") } ? "bert."
            : nil
        try loadWeights(
            into: m, from: directory, quantization: nil, keyPrefix: prefix)
        eval(m)

        let tok = try await KLMTokenizer(from: directory)
        install(
            model: m, tokenizer: tok, directory: directory,
            maxTokens: config.maxPositionEmbeddings)
    }

    public struct RerankResult: Sendable {
        /// One entry per input document, in the SAME order as the
        /// input. Sort by `score` descending for the ranking.
        public let scores: [Double]
        /// Total tokens forwarded (sum across all (query, doc)
        /// pairs). Useful for cost tracking.
        public let totalTokens: Int
    }

    /// Score every document against the query. Each pair is forwarded
    /// independently (batch = 1) so no padding mask is needed.
    public func score(query: String, documents: [String]) throws -> RerankResult {
        lock.lock()
        let model = self.model
        let tokenizer = self.tokenizer
        let cap = self.maxTokens
        lock.unlock()

        guard let model, let tokenizer else {
            throw RerankError.notLoaded
        }

        var scores: [Double] = []
        scores.reserveCapacity(documents.count)
        var totalTokens = 0

        for doc in documents {
            // Cross-encoder input is the (query, document) pair
            // joined with the model's pair separator. BGE / XLMR
            // expects `</s></s>` between the two; SentenceBERT /
            // Bert use `[SEP]`. We let the tokenizer's pair
            // template handle this by manually constructing the
            // standard `<s> query </s></s> doc </s>` shape.
            //
            // KLMTokenizer.encode(_:) on a single string adds the
            // model-default leading/trailing specials; we cannot
            // assume it supports a pair template, so we encode the
            // two sides separately and stitch with the right
            // specials sourced from tokenizer_config.
            let pairIds = tokenizer.encodePair(query: query, document: doc)
            let ids: [Int]
            if pairIds.count > cap {
                // Truncate the document side, keep the query
                // intact (standard cross-encoder behavior).
                ids = Array(pairIds.prefix(cap))
            } else {
                ids = pairIds
            }
            totalTokens += ids.count

            let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let logits = model(tokens)
            // [1, 1] -> single Float
            MLX.eval(logits)
            let arr = logits.asArray(Float.self)
            scores.append(Double(arr.first ?? 0))
        }

        return RerankResult(scores: scores, totalTokens: totalTokens)
    }
}

public enum RerankError: Error, CustomStringConvertible {
    case notLoaded

    public var description: String {
        switch self {
        case .notLoaded: return "No reranker model loaded"
        }
    }
}
