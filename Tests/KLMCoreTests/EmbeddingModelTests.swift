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

    // MARK: - nomic-bert (RoPE encoder)

    /// nomic configs use GPT-2-style keys (n_embd/n_layer/n_head/n_inner) and
    /// carry rotary params instead of a learned-position table.
    func testNomicConfigDecodingGPT2StyleKeys() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "architectures": ["NomicBertModel"],
          "n_embd": 768,
          "n_layer": 12,
          "n_head": 12,
          "n_inner": 3072,
          "vocab_size": 30528,
          "type_vocab_size": 2,
          "rotary_emb_base": 1000,
          "rotary_emb_fraction": 1.0,
          "max_position_embeddings": 2048,
          "n_positions": 8192
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 768)
        XCTAssertEqual(cfg.numHiddenLayers, 12)
        XCTAssertEqual(cfg.numAttentionHeads, 12)
        XCTAssertEqual(cfg.intermediateSize, 3072)
        XCTAssertEqual(cfg.headDim, 64)
        XCTAssertEqual(cfg.ropeBase, 1000)
        XCTAssertEqual(cfg.rotaryFraction, 1.0)
        XCTAssertEqual(cfg.maxTokens, 2048)  // trained window, not n_positions
    }

    /// Tolerate HF BERT-style aliases (hidden_size/num_hidden_layers/...) and
    /// fall back to sane rotary defaults when the keys are absent.
    func testNomicConfigDecodingBertStyleKeysAndDefaults() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "hidden_size": 256,
          "num_hidden_layers": 4,
          "num_attention_heads": 8,
          "intermediate_size": 512,
          "vocab_size": 1000
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        XCTAssertEqual(cfg.hiddenSize, 256)
        XCTAssertEqual(cfg.numHiddenLayers, 4)
        XCTAssertEqual(cfg.numAttentionHeads, 8)
        XCTAssertEqual(cfg.typeVocabSize, 2)     // default
        XCTAssertEqual(cfg.ropeBase, 1000)       // default
        XCTAssertEqual(cfg.rotaryFraction, 1.0)  // default
        XCTAssertEqual(cfg.maxTokens, 2048)      // default
    }

    /// Exercises the full nomic wiring (embeddings + emb_ln -> fused-Wqkv RoPE
    /// attention -> SwiGLU MLP -> post-norm) on a tiny random-init model: the
    /// last-hidden-state must be `[1, T, H]` and deterministic. Numerical
    /// parity against the reference forward is validated out-of-band against
    /// real weights (a 12-layer numpy reimplementation matches the Swift
    /// output to cosine 1.0000); this guards the architecture plumbing.
    func testNomicForwardShapeAndDeterminism() throws {
        let json = """
        {
          "model_type": "nomic_bert",
          "n_embd": 32, "n_layer": 2, "n_head": 4, "n_inner": 64,
          "vocab_size": 100, "type_vocab_size": 2,
          "rotary_emb_base": 1000, "rotary_emb_fraction": 1.0,
          "max_position_embeddings": 64
        }
        """.data(using: .utf8)!
        let cfg = try JSONDecoder().decode(NomicBertConfig.self, from: json)
        let model = NomicBertEmbeddingModel(cfg)
        let tokens = MLXArray([Int32(1), 2, 3, 4, 5]).reshaped(1, 5)

        let out = model.lastHiddenState(tokens)
        out.eval()
        XCTAssertEqual(out.shape, [1, 5, 32])

        // Deterministic: same input -> identical output (no RNG in forward).
        let out2 = model.lastHiddenState(tokens)
        out2.eval()
        XCTAssertTrue(allClose(out, out2, atol: 0).all().item(Bool.self))

        // Pools to a unit-norm sentence vector of width H.
        let vec = poolSentenceEmbedding(out, pooling: .mean, normalize: true)
        XCTAssertEqual(vec.count, 32)
        let norm = vec.reduce(0) { $0 + $1 * $1 }.squareRoot()
        XCTAssertEqual(norm, 1.0, accuracy: 1e-4)
    }
}
