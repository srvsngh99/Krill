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
/// tokenizes every (query, document) pair, pads them to a common length,
/// and runs the model once over the whole batch (a key-padding mask keeps
/// padding tokens out of attention). Returns the raw classifier logits;
/// clients sort documents by logit descending.
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

    /// Score every document against the query in a single batched
    /// forward. Pairs are tokenized, padded to the batch's longest
    /// sequence, and run together; a key-padding mask keeps padding
    /// tokens out of attention so the result is identical to scoring
    /// each pair on its own.
    public func score(query: String, documents: [String]) throws -> RerankResult {
        lock.lock()
        let model = self.model
        let tokenizer = self.tokenizer
        let cap = self.maxTokens
        lock.unlock()

        guard let model, let tokenizer else {
            throw RerankError.notLoaded
        }

        if documents.isEmpty {
            return RerankResult(scores: [], logits: [], totalTokens: 0)
        }

        // The query is identical for every pair, so tokenize it once.
        // Reserve budget for the query side and the separators
        // (`</s></s>` for XLMR is 2 tokens; `[SEP]` for Bert is 1;
        // conservatively reserve 4 to cover both plus the trailing EOS).
        let queryIds = tokenizer.encode(query)
        let queryReserved = min(queryIds.count, max(0, cap - 4))
        let docBudget = max(8, cap - queryReserved - 4)

        var perPairIds: [[Int]] = []
        perPairIds.reserveCapacity(documents.count)
        var totalTokens = 0

        for doc in documents {
            // Cross-encoder input is the (query, document) pair joined
            // with the model's pair separator. BGE / XLMR expects
            // `<s> q </s></s> d </s>`; SentenceBERT / Bert use
            // `[CLS] q [SEP] d [SEP]`.
            //
            // Truncation policy: keep the query intact, truncate the
            // document if the joined pair exceeds the model's
            // max_position_embeddings. This matches the standard
            // sentence-transformers / HuggingFace
            // `truncation="only_second"` behavior used by every
            // cross-encoder reranker we ship. A blunt `prefix(cap)` on
            // the joined pair would risk dropping the trailing EOS (so
            // attention sees a sequence the post-processor never
            // produces) or, for a long query, deleting the document
            // entirely.
            let trimmedDoc: String
            if doc.count > docBudget * 4 {
                // Cheap character-level pre-cap to keep the tokenizer
                // from doing huge work on a doc we will truncate
                // anyway. 4 chars/token is a soft upper bound for
                // SentencePiece.
                trimmedDoc = String(doc.prefix(docBudget * 4))
            } else {
                trimmedDoc = doc
            }
            var pairIds = tokenizer.encodePair(
                query: query, document: trimmedDoc)
            if pairIds.count > cap {
                // Final cap. Preserve the trailing EOS by reading it
                // off the end before truncating, then re-appending.
                let trailing = pairIds.last
                pairIds = Array(pairIds.prefix(cap - 1))
                if let t = trailing { pairIds.append(t) }
            }
            totalTokens += pairIds.count
            perPairIds.append(pairIds)
        }

        // Pad every pair to the batch's longest sequence and build an
        // additive key-padding mask ([B, 1, 1, T], broadcast over heads
        // and query rows). Padding ids are arbitrary (0): they are
        // masked out as attention keys, and their query rows are never
        // read - only the CLS row at position 0 is, and CLS is always a
        // real token. The batched result is therefore identical to
        // scoring each pair alone.
        let batch = perPairIds.count
        let maxLen = perPairIds.map(\.count).max() ?? 1
        var tokenBuf = [Int32](repeating: 0, count: batch * maxLen)
        var maskBuf = [Float](repeating: 0, count: batch * maxLen)
        let padMaskValue: Float = -1e9
        for (b, ids) in perPairIds.enumerated() {
            for (t, id) in ids.enumerated() {
                tokenBuf[b * maxLen + t] = Int32(id)
            }
            for t in ids.count ..< maxLen {
                maskBuf[b * maxLen + t] = padMaskValue
            }
        }
        let tokens = MLXArray(tokenBuf).reshaped(batch, maxLen)
        let mask = MLXArray(maskBuf).reshaped(batch, 1, 1, maxLen)

        let out = model(tokens, mask: mask)   // [B, numLabels]
        MLX.eval(out)
        let numLabels = out.dim(1)
        let flat = out.asArray(Float.self)    // row-major [B * numLabels]
        var logits: [Double] = []
        logits.reserveCapacity(batch)
        for b in 0 ..< batch {
            // numLabels == 1 for a reranker: column 0 is the relevance
            // logit. Take it explicitly so a stray multi-label head
            // does not silently read the wrong column.
            logits.append(Double(flat[b * numLabels]))
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
