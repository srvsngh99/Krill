import XCTest
import MLX
@testable import KLMCore

final class EmbeddingModelTests: XCTestCase {
    func testBertConfigDecodingPlainBert() throws {
        let json = """
        {
          "model_type": "bert",
          "hidden_size": 384,
          "num_hidden_layers": 6,
          "num_attention_heads": 12,
          "intermediate_size": 1536,
          "vocab_size": 30522,
          "max_position_embeddings": 512,
          "type_vocab_size": 2,
          "layer_norm_eps": 1e-12,
          "pad_token_id": 0
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(BertEmbeddingConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 384)
        XCTAssertEqual(cfg.numHiddenLayers, 6)
        XCTAssertEqual(cfg.positionOffset, 0)  // plain BERT: no offset
    }

    func testRobertaConfigAppliesPositionOffset() throws {
        let json = """
        {
          "model_type": "xlm-roberta",
          "hidden_size": 768,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "intermediate_size": 3072,
          "vocab_size": 250002,
          "max_position_embeddings": 514,
          "layer_norm_eps": 1e-5,
          "pad_token_id": 1
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(BertEmbeddingConfig.self, from: json)
        XCTAssertEqual(cfg.positionOffset, 2)  // pad_token_id (1) + 1
        XCTAssertEqual(cfg.typeVocabSize, 2)   // default when absent
    }

    func testMeanPoolingIsL2Normalized() {
        // [1, T=2, H=3]
        let hidden = MLXArray(
            [1.0, 0.0, 0.0,
             3.0, 0.0, 0.0] as [Float]).reshaped(1, 2, 3)
        let v = poolSentenceEmbedding(hidden, pooling: .mean, normalize: true)
        XCTAssertEqual(v.count, 3)
        // mean over T = [2,0,0]; L2-normalized -> [1,0,0]
        XCTAssertEqual(v[0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(v[1], 0.0, accuracy: 1e-5)
        let norm = (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]).squareRoot()
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
    }

    func testClsPoolingTakesFirstToken() {
        let hidden = MLXArray(
            [5.0, 0.0, 0.0,
             9.0, 0.0, 0.0] as [Float]).reshaped(1, 2, 3)
        let v = poolSentenceEmbedding(hidden, pooling: .cls, normalize: false)
        XCTAssertEqual(v, [5.0, 0.0, 0.0])
    }
}
