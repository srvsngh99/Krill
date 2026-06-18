import XCTest
@testable import KLMCore
import KLMRegistry

/// WS7 runtime: the causal-LM `loadModel(from:)` dispatcher refuses to
/// instantiate a reranker (rerankers are loaded through the dedicated
/// `RerankEngine` instead). The shared family detection still routes
/// `*ForSequenceClassification` / `*CrossEncoder*` architectures to
/// `.reranker`; the rejection error must now point users at
/// `/v1/rerank` rather than at a missing native runtime.
final class RerankerLoaderRejectionTests: XCTestCase {

    private func writeConfig(_ json: [String: Any], slug: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-reranker-\(slug)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testBGERerankerV2M3IsRejectedFromCausalLMDispatcher() throws {
        // The causal-LM dispatcher must NOT load a reranker.
        // Callers that need scoring should hit `/v1/rerank`, which
        // routes through `RerankEngine.load(directory:)` instead.
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
            XCTAssertTrue(msg.contains("/v1/rerank"),
                "Error must redirect users at the rerank endpoint")
            XCTAssertTrue(msg.lowercased().contains("reranker"))
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
        // BertModel must NOT match those.
        let configFromArch: [String: Any] = [
            "architectures": ["BertModel"],
            "model_type": "bert",
        ]
        XCTAssertEqual(
            ModelFamily.detect(from: configFromArch),
            .bert,
            "Plain BertModel must route to .bert, NOT .reranker")
    }

    func testAmbiguousBackboneWithoutArchitecturesFallsBackToBert() throws {
        // If a config ships model_type=xlm-roberta but no
        // architectures key, the safe default is .bert.
        let configMissingArch: [String: Any] = [
            "model_type": "xlm-roberta",
        ]
        XCTAssertEqual(
            ModelFamily.detect(from: configMissingArch),
            .bert)
    }
}
