import XCTest
@testable import KLMCore

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
        // Plain BertModel architecture must reach the embedding
        // family path; the reranker arm must NOT swallow it. We
        // cannot fully instantiate without weights, so the
        // discriminator is the error class.
        let dir = try writeConfig([
            "architectures": ["BertModel"],
            "model_type": "bert",
            "hidden_size": 384,
            "vocab_size": 30522,
        ], slug: "bert-embed")
        defer { try? FileManager.default.removeItem(at: dir) }

        do {
            _ = try loadModel(from: dir)
            XCTFail("Expected failure due to missing weights")
        } catch let error as ModelLoadError {
            if case .unsupportedArchitecture(let msg) = error {
                XCTAssertFalse(msg.contains("WS7"),
                    "Plain BertModel must NOT route through the WS7 reranker rejection arm")
            }
            // Any other ModelLoadError is fine - means the family
            // dispatched and a downstream loader stage failed.
        } catch {
            // Non-ModelLoadError (e.g. WeightLoadError) is the
            // expected path: family routed to .bert, then weight
            // load failed because the temp dir has no safetensors.
        }
    }
}
