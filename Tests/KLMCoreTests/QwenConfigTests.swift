import XCTest
@testable import KLMCore

final class QwenConfigTests: XCTestCase {

    private func decode(_ json: String) throws -> QwenConfig {
        try JSONDecoder().decode(QwenConfig.self, from: Data(json.utf8))
    }

    // MARK: - Qwen 2.5 baseline

    func testQwen25ConfigKeepsHistoricalDefaults() throws {
        // Qwen 2.5 omits attention_bias, head_dim, and
        // tie_word_embeddings. WS4 must NOT change their effective
        // values from what loaded successfully before: bias on QKV,
        // derived head_dim, separate lm_head.
        let json = """
        {
          "model_type": "qwen2",
          "hidden_size": 1024,
          "intermediate_size": 4096,
          "num_attention_heads": 16,
          "num_key_value_heads": 16,
          "num_hidden_layers": 24,
          "vocab_size": 151936,
          "rope_theta": 1000000
        }
        """
        let cfg = try decode(json)
        XCTAssertEqual(cfg.modelType, "qwen2")
        XCTAssertTrue(cfg.attentionBias,
            "Qwen 2.5 must default to attention_bias=true so the existing checkpoints still load with biased QKV")
        XCTAssertFalse(cfg.hasQKNorm)
        XCTAssertFalse(cfg.tieWordEmbeddings)
        XCTAssertNil(cfg.explicitHeadDim)
        XCTAssertEqual(cfg.headDim, 64, "Falls back to hidden_size / num_attention_heads")
    }

    // MARK: - Qwen 3

    func testQwen3ConfigEnablesAllArchitecturalDeltas() throws {
        // This is the actual config shipped by
        // mlx-community/Qwen3-0.6B-4bit, trimmed to the keys the
        // decoder reads.
        let json = """
        {
          "model_type": "qwen3",
          "attention_bias": false,
          "head_dim": 128,
          "hidden_size": 1024,
          "intermediate_size": 3072,
          "num_attention_heads": 16,
          "num_key_value_heads": 8,
          "num_hidden_layers": 28,
          "vocab_size": 151936,
          "rms_norm_eps": 1e-6,
          "rope_theta": 1000000,
          "tie_word_embeddings": true
        }
        """
        let cfg = try decode(json)
        XCTAssertEqual(cfg.modelType, "qwen3")
        XCTAssertFalse(cfg.attentionBias,
            "Qwen 3 has no bias on QKV projections")
        XCTAssertTrue(cfg.hasQKNorm,
            "Qwen 3 applies per-head RMSNorm on Q/K before RoPE")
        XCTAssertTrue(cfg.tieWordEmbeddings,
            "Qwen 3 ties embed_tokens with the LM head")
        XCTAssertEqual(cfg.explicitHeadDim, 128)
        XCTAssertEqual(cfg.headDim, 128,
            "Honors explicit head_dim instead of deriving from hidden_size")
    }

    func testQwen3InferenceWhenAttentionBiasOmitted() throws {
        // If a Qwen 3 config is missing `attention_bias` (defensive),
        // the model_type alone is enough to switch to the bias-free
        // configuration.
        let json = """
        {
          "model_type": "qwen3",
          "hidden_size": 1024,
          "intermediate_size": 3072,
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "vocab_size": 151936
        }
        """
        let cfg = try decode(json)
        XCTAssertFalse(cfg.attentionBias)
        XCTAssertTrue(cfg.hasQKNorm)
        XCTAssertTrue(cfg.tieWordEmbeddings)
    }
}
