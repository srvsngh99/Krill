import XCTest
@testable import KrillCore

/// Pins the Phi config deltas that distinguish Phi-4-mini from Phi-3-mini.
/// Getting any of these wrong made Phi-4-mini emit garbage: full RoPE on a
/// partial-rotary head, a randomly-initialized `lm_head` on a tied checkpoint,
/// and plain RoPE where LongRoPE was required.
final class PhiConfigTests: XCTestCase {

    private func decode(_ json: String) throws -> PhiConfig {
        try JSONDecoder().decode(PhiConfig.self, from: Data(json.utf8))
    }

    // MARK: - Phi-3-mini baseline (full rotary, untied, no LongRoPE)

    func testPhi3MiniKeepsHistoricalDefaults() throws {
        let json = """
        {
          "model_type": "phi3",
          "hidden_size": 3072,
          "intermediate_size": 8192,
          "num_attention_heads": 32,
          "num_key_value_heads": 32,
          "num_hidden_layers": 32,
          "vocab_size": 32064,
          "rope_theta": 10000.0
        }
        """
        let cfg = try decode(json)
        XCTAssertEqual(cfg.partialRotaryFactor, 1.0, "Phi-3-mini rotates the full head")
        XCTAssertEqual(cfg.headDim, 96)
        XCTAssertEqual(cfg.ropeDims, 96, "Full rotation: ropeDims == headDim")
        XCTAssertFalse(cfg.tieWordEmbeddings, "Phi-3-mini ships a separate lm_head")
        XCTAssertNil(cfg.ropeScaling, "Phi-3-mini uses plain RoPE")
    }

    // MARK: - Phi-4-mini (partial rotary, tied embeddings, LongRoPE)

    func testPhi4MiniArchitecturalDeltas() throws {
        // Trimmed to the keys the decoder reads, from the actual
        // mlx-community/Phi-4-mini-instruct config.
        let json = """
        {
          "model_type": "phi3",
          "hidden_size": 3072,
          "intermediate_size": 8192,
          "num_attention_heads": 24,
          "num_key_value_heads": 8,
          "num_hidden_layers": 32,
          "vocab_size": 200064,
          "rope_theta": 10000.0,
          "partial_rotary_factor": 0.75,
          "tie_word_embeddings": true,
          "max_position_embeddings": 131072,
          "original_max_position_embeddings": 4096,
          "rope_scaling": {
            "short_factor": [1.0, 1.0, 1.0],
            "long_factor": [1.0, 1.118, 1.25],
            "type": "longrope"
          }
        }
        """
        let cfg = try decode(json)
        XCTAssertEqual(cfg.headDim, 128, "3072 / 24 heads")
        XCTAssertEqual(cfg.ropeDims, 96, "128 * 0.75 partial rotary")
        XCTAssertTrue(cfg.tieWordEmbeddings, "Phi-4-mini ties the LM head to embeddings")
        XCTAssertEqual(cfg.originalMaxPositionEmbeddings, 4096)
        XCTAssertNotNil(cfg.ropeScaling)
        XCTAssertEqual(cfg.ropeScaling?.type, "longrope")
        XCTAssertEqual(cfg.ropeScaling?.shortFactor.count, 3)
    }

    // MARK: - LongRoPE attention factor

    func testSuScaledRoPEAttentionFactor() {
        // The magnitude scale HF derives from the context-extension ratio:
        // sqrt(1 + ln(131072/4096) / ln(4096)) ~= 1.190.
        let scaling = PhiRopeScaling(
            shortFactor: Array(repeating: 1.0, count: 48),
            longFactor: Array(repeating: 1.0, count: 48), type: "longrope")
        let rope = PhiSuScaledRoPE(
            dims: 96, base: 10000, scaling: scaling,
            originalMaxPos: 4096, maxPos: 131072)
        XCTAssertEqual(rope.scale, 1.190, accuracy: 0.01)
        // When the context is not extended the factor collapses to 1.
        let noScale = PhiSuScaledRoPE(
            dims: 96, base: 10000, scaling: scaling,
            originalMaxPos: 4096, maxPos: 4096)
        XCTAssertEqual(noScale.scale, 1.0, accuracy: 1e-6)
    }
}
