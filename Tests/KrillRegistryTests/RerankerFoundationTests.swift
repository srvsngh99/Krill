import XCTest
@testable import KrillRegistry

/// WS7 foundation: cross-encoder reranker family detection +
/// capability metadata + experimental tier + alias entries. Pins
/// the contract so the follow-up runtime PR (cross-encoder scoring
/// + `/v1/rerank`) cannot silently drop the rejection path or
/// over-claim capabilities.
final class RerankerFoundationTests: XCTestCase {

    // MARK: - Family detection

    func testDetectBGERerankerFromArchitectures() {
        // The BGE Reranker v2-m3 config ships
        // architectures=[XLMRobertaForSequenceClassification] AND
        // model_type=xlm-roberta. The architectures-first arm must
        // catch the reranker before the .bert arm.
        let cfg: [String: Any] = [
            "architectures": ["XLMRobertaForSequenceClassification"],
            "model_type": "xlm-roberta",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .reranker)
    }

    func testDetectBertSequenceClassificationAsReranker() {
        let cfg: [String: Any] = [
            "architectures": ["BertForSequenceClassification"],
            "model_type": "bert",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .reranker,
            "Sequence-classification architectures must NOT route to the embedding loader")
    }

    func testDetectCrossEncoderArchitecture() {
        let cfg: [String: Any] = [
            "architectures": ["CrossEncoderForSequenceClassification"],
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .reranker)
    }

    func testDetectionPrefersRerankerOverBert() {
        // The arch string contains both "roberta" (would route to
        // .bert) and "forsequenceclassification" (reranker). The
        // detection order must match reranker first.
        let cfg: [String: Any] = [
            "architectures": ["XLMRobertaForSequenceClassification"],
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .reranker)
    }

    func testPlainBertEmbeddingStillRoutesToBert() {
        // The reranker detection must NOT swallow plain BERT
        // embedding architectures (no SequenceClassification or
        // CrossEncoder substring).
        let cfg: [String: Any] = [
            "architectures": ["BertModel"],
            "model_type": "bert",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .bert)
    }

    // MARK: - Capability declaration

    func testRerankerDeclaresOnlyRerankerCapability() {
        let caps = ModelCapabilities.capabilities(for: .reranker)
        XCTAssertTrue(caps.contains(.reranker))
        XCTAssertFalse(caps.contains(.textGeneration),
            "Cross-encoder rerankers must NOT advertise textGeneration; they cannot run /api/generate")
        XCTAssertFalse(caps.contains(.embeddings),
            "Cross-encoder rerankers must NOT advertise embeddings; they cannot run /v1/embeddings")
        XCTAssertFalse(caps.contains(.visionInput))
        XCTAssertFalse(caps.contains(.audioInput))
    }

    // MARK: - Support tier

    func testRerankerIsExperimental() {
        XCTAssertEqual(ModelCapabilities.supportTier(for: .reranker), .experimental)
    }

    // MARK: - Stable raw value

    func testRawValueIsStable() {
        XCTAssertEqual(ModelFamily.reranker.rawValue, "reranker")
    }

    // MARK: - Ollama tag

    func testRerankerOllamaTag() {
        // The Capability.reranker.ollamaTag is "reranker" so
        // /api/show capabilities arrays expose the same identifier
        // clients can opt into.
        XCTAssertEqual(Capability.reranker.ollamaTag, "reranker")
    }
}
