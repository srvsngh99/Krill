import XCTest
@testable import KLMEngine
@testable import KLMCore
import KLMRuntime

/// Live runtime tests for the cross-encoder rerank path. Gated on
/// `KLM_RERANKER_MODEL_PATH` because the test downloads ~2GB of weights
/// otherwise and we do not want CI doing that implicitly. Run locally
/// with:
///
///     KLM_RERANKER_MODEL_PATH=$HOME/.krillm/models/blobs/bge-reranker-v2-m3 \
///       swift test --filter RerankEngineTests
final class RerankEngineTests: XCTestCase {

    private func liveModelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_RERANKER_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_RERANKER_MODEL_PATH not set; skipping live reranker test")
        }
        let url = URL(fileURLWithPath: path)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_RERANKER_MODEL_PATH does not point to a directory: \(path)")
        }
        return url
    }

    private func requireMLX() throws {
        #if !(os(macOS) && arch(arm64))
        throw XCTSkip("Reranker live runtime tests require macOS arm64")
        #endif
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is not available in this test process")
        }
    }

    func testScoreOrderingMatchesQueryDocumentRelevance() async throws {
        try requireMLX()
        let dir = try liveModelDir()
        let engine = RerankEngine()
        try await engine.load(directory: dir)

        let query = "What is the capital of France?"
        let docs = [
            "Paris is the capital of France.",     // most relevant
            "Apples are red.",                     // irrelevant
            "The Eiffel Tower is in Paris.",       // tangential
            "Hello world.",                        // irrelevant
        ]
        let result = try engine.score(query: query, documents: docs)

        XCTAssertEqual(result.scores.count, docs.count)
        // Strict relevance ordering: doc 0 > doc 2 > {doc 1, doc 3}.
        // Anchors the contract that follow-up changes (e.g. mask
        // tweaks, dtype changes) cannot silently degrade ranking.
        XCTAssertGreaterThan(result.scores[0], result.scores[2],
            "Capital-of-France document must outrank tangential Eiffel doc")
        XCTAssertGreaterThan(result.scores[2], result.scores[1],
            "Tangential Paris doc must outrank unrelated Apples doc")
        XCTAssertGreaterThan(result.scores[2], result.scores[3],
            "Tangential Paris doc must outrank unrelated Hello doc")
    }

    func testScoreIsSigmoidNormalizedInRange() async throws {
        try requireMLX()
        let dir = try liveModelDir()
        let engine = RerankEngine()
        try await engine.load(directory: dir)

        let result = try engine.score(
            query: "What is the capital of France?",
            documents: [
                "Paris is the capital of France.",
                "Apples are red.",
                "The Eiffel Tower is in Paris.",
            ])
        // Sigmoid-normalized scores are in (0, 1) and the
        // capital-of-France doc should be in the high-confidence
        // band (>= 0.95 against the Python reference 0.9998).
        for s in result.scores {
            XCTAssertGreaterThan(s, 0.0)
            XCTAssertLessThan(s, 1.0)
        }
        XCTAssertGreaterThan(result.scores[0], 0.95,
            "High-relevance pair must yield sigmoid >= 0.95")
        XCTAssertLessThan(result.scores[1], 0.05,
            "Irrelevant pair must yield sigmoid <= 0.05")
        // Logits are exposed separately for parity testing; they
        // are NOT in [0, 1].
        XCTAssertEqual(result.logits.count, 3)
        XCTAssertGreaterThan(result.logits[0], 0.0,
            "High-relevance pair logit is positive")
    }

    func testLogitsMatchReferenceWithinTolerance() async throws {
        try requireMLX()
        let dir = try liveModelDir()
        let engine = RerankEngine()
        try await engine.load(directory: dir)

        // Reference logits measured against the same model loaded
        // via sentence-transformers CrossEncoder (Python). These
        // are raw pre-sigmoid scores; the sentence-transformers
        // 0.9998 "probability" corresponds to logit ~= +8.5.
        // Tolerance allows for minor numerical differences
        // between the MLX bf16/fp16 forward and the Python
        // float32 reference; 1.0 logit corresponds to ~0.05
        // sigmoid difference at the saturated end.
        let pairs: [(String, String, Double)] = [
            ("What is the capital of France?",
             "Paris is the capital of France.",
             8.5),
            ("What is the capital of France?",
             "The Eiffel Tower is in Paris.",
             -1.4),
            ("What is the capital of France?",
             "Apples are red.",
             -11.0),
        ]
        for (q, d, ref) in pairs {
            let result = try engine.score(query: q, documents: [d])
            let got = result.logits[0]
            XCTAssertEqual(got, ref, accuracy: 1.0,
                "(q=\(q), d=\(d)) logit got=\(got) ref=\(ref)")
        }
    }
}
