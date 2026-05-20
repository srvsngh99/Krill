import XCTest
@testable import KLMCore
import KLMRegistry

/// WS7 foundation: cross-encoder reranker configs are recognized by
/// the loader AND refused (no silent fallback to the embedding
/// loader, which would either crash on the classification head or
/// silently run with no scoring head at all).
final class RerankerLoaderRejectionTests: XCTestCase {

    private func writeConfig(_ json: [String: Any], slug: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-reranker-\(slug)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testBGERerankerV2M3IsRejectedWithDocumentedError() throws {
        // BGE Reranker v2-m3 config: XLM-Roberta backbone +
        // sequence classification head. Exactly the shape that
        // would crash the embedding loader.
        let dir = try writeConfig([
            "architectures": ["XLMRobertaForSequenceClassification"],
            "model_type": "xlm-roberta",
            "hidden_size": 1024,
            "vocab_size": 250002,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
        ], slug: "bge-rerank-v2-m3")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture(let msg) = modelError else {
                XCTFail("Expected unsupportedArchitecture, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("WS7"),
                "Error must point at the WS7 workstream doc")
            XCTAssertTrue(msg.contains("reranker") || msg.contains("Cross-encoder"))
            XCTAssertTrue(msg.contains("bge-small-en"),
                "Error must suggest the embedding-plus-rerank stand-in")
        }
    }

    func testBertSequenceClassificationIsRejected() throws {
        let dir = try writeConfig([
            "architectures": ["BertForSequenceClassification"],
            "model_type": "bert",
            "hidden_size": 768,
            "vocab_size": 30522,
        ], slug: "bert-cls")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture = modelError else {
                XCTFail("Expected unsupportedArchitecture, got \(error)")
                return
            }
        }
    }

    func testPlainBertEmbeddingDoesNotRouteToRerankerArm() throws {
        // The reranker arm uses loose substring matches
        // ("forsequenceclassification", "crossencoder"). Plain
        // BertModel must NOT match those. Discriminator: the
        // family-detection step alone (no weight-loader stages
        // involved) returns .bert. We test detection rather than
        // a full loadModel because:
        //   - loadModel currently has no dedicated .bert dispatch
        //     arm (BertModel falls through to the Llama catch-all
        //     in `ModelLoader.swift`, which then fails on the
        //     missing safetensors); the error class from that
        //     fallthrough is not a stable contract.
        //   - The capability we are pinning here is detection,
        //     not dispatch.
        // Once a dedicated `loadBert` arm lands, the assertion
        // below stays valid and a second assertion can be added
        // that the dispatched arm is the embedding loader.
        let configFromArch: [String: Any] = [
            "architectures": ["BertModel"],
            "model_type": "bert",
        ]
        XCTAssertEqual(
            ModelFamily.detect(from: configFromArch),
            .bert,
            "Plain BertModel must route to .bert, NOT .reranker (no for-sequence-classification substring)")
    }

    func testAmbiguousBackboneWithoutArchitecturesFallsBackToBert() throws {
        // If a config ships model_type=xlm-roberta but no
        // architectures key (rare, but defensive), there is no
        // way to know whether it is an embedding model or a
        // reranker. The detector returns .bert as the safe
        // default; the loader's reranker arm only fires on the
        // architectures substring, so this config would attempt
        // the embedding path. Documenting this as expected.
        let configMissingArch: [String: Any] = [
            "model_type": "xlm-roberta",
        ]
        XCTAssertEqual(
            ModelFamily.detect(from: configMissingArch),
            .bert,
            "When architectures is missing, ambiguous backbones default to .bert (documented behavior)")
    }
}
