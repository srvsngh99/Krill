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
        /// input. Sigmoid-normalized to [0, 1] so the values are
        /// directly comparable to Cohere's `/v1/rerank` response
        /// and to thresholds used by LangChain / llama-index
        /// reranker clients. Sort by `score` descending for the
        /// ranking.
        public let scores: [Double]
        /// Raw pre-sigmoid logits, in the same order as `scores`.
        /// Exposed for callers (e.g. parity tests) that need to
        /// compare against an upstream model's raw output.
        public let logits: [Double]
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

        var logits: [Double] = []
        logits.reserveCapacity(documents.count)
        var totalTokens = 0

        for doc in documents {
            // Cross-encoder input is the (query, document) pair
            // joined with the model's pair separator. BGE / XLMR
            // expects `<s> q </s></s> d </s>`; SentenceBERT / Bert
            // use `[CLS] q [SEP] d [SEP]`.
            //
            // Truncation policy: keep the query intact, truncate
            // the document if the joined pair exceeds the model's
            // max_position_embeddings. This matches the standard
            // sentence-transformers / HuggingFace
            // `truncation="only_second"` behavior used by every
            // cross-encoder reranker we ship. A blunt
            // `prefix(cap)` on the joined pair would risk
            // dropping the trailing EOS (so attention sees a
            // sequence the post-processor never produces) or, for
            // a long query, deleting the document entirely.
            let queryIds = tokenizer.encode(query)
            // Reserve budget for the query side and the
            // separators (`</s></s>` for XLMR is 2 tokens; `[SEP]`
            // for Bert is 1; conservatively reserve 4 to cover
            // both plus the trailing EOS).
            let queryReserved = min(queryIds.count, max(0, cap - 4))
            let docBudget = max(8, cap - queryReserved - 4)
            let trimmedDoc: String
            if doc.count > docBudget * 4 {
                // Cheap character-level pre-cap to keep the
                // tokenizer from doing huge work on a doc we will
                // truncate anyway. 4 chars/token is a soft upper
                // bound for SentencePiece.
                trimmedDoc = String(doc.prefix(docBudget * 4))
            } else {
                trimmedDoc = doc
            }
            var pairIds = tokenizer.encodePair(
                query: query, document: trimmedDoc)
            if pairIds.count > cap {
                // Final cap. Preserve the trailing EOS by reading
                // it off the end before truncating, then
                // re-appending after.
                let trailing = pairIds.last
                pairIds = Array(pairIds.prefix(cap - 1))
                if let t = trailing { pairIds.append(t) }
            }
            let ids = pairIds
            totalTokens += ids.count

            let tokens = MLXArray(ids.map { Int32($0) }).reshaped(1, ids.count)
            let out = model(tokens)
            // [1, 1] -> single Float
            MLX.eval(out)
            let arr = out.asArray(Float.self)
            logits.append(Double(arr.first ?? 0))
        }

        // Sigmoid-normalize each logit so callers get a [0, 1]
        // probability directly comparable to Cohere's API and to
        // thresholds in LangChain / llama-index reranker clients.
        // Logits are still exposed via `RerankResult.logits` for
        // parity tests that compare against an upstream raw score.
        let scores = logits.map { 1.0 / (1.0 + exp(-$0)) }
        return RerankResult(
            scores: scores, logits: logits, totalTokens: totalTokens)
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
