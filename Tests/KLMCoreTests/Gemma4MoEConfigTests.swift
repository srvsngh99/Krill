import XCTest
import MLX
@testable import KLMCore

/// Config + loader unpacker tests for the Gemma 4 26B-A4B text MoE
/// path (issue #80). The full MoE forward is exercised end-to-end by
/// `krillm serve --model gemma-4-26b-a4b-it-4bit`; these tests pin
/// the config parsing and the `switch_glu` key-rewrite without
/// needing the 15 GB checkpoint.
final class Gemma4MoEConfigTests: XCTestCase {

    // MARK: - Config parsing

    func testParses26BMoEFields() throws {
        // Slim fixture mirroring 26B-A4B's text_config: MoE on,
        // 128 experts, top-8, dense MLP width 2112, sparse width 704,
        // K-eq-V global attention with `num_global_key_value_heads=2`.
        let cfg = try decodeTextConfig("""
        {
          "hidden_size": 2816, "intermediate_size": 2112,
          "num_attention_heads": 16, "num_key_value_heads": 8,
          "num_hidden_layers": 4, "vocab_size": 1024,
          "head_dim": 256, "global_head_dim": 512,
          "hidden_size_per_layer_input": 0,
          "num_kv_shared_layers": 0,
          "use_double_wide_mlp": false,
          "attention_k_eq_v": true,
          "num_global_key_value_heads": 2,
          "enable_moe_block": true,
          "num_experts": 128,
          "top_k_experts": 8,
          "moe_intermediate_size": 704,
          "layer_types": ["sliding_attention","sliding_attention",
                          "full_attention","sliding_attention"]
        }
        """)

        XCTAssertTrue(cfg.enableMoeBlock)
        XCTAssertEqual(cfg.numExperts, 128)
        XCTAssertEqual(cfg.topKExperts, 8)
        XCTAssertEqual(cfg.moeIntermediateSize, 704)
        XCTAssertTrue(cfg.attentionKEqV)
        XCTAssertEqual(cfg.numGlobalKeyValueHeads, 2)
        XCTAssertEqual(cfg.hiddenSizePerLayerInput, 0)
        XCTAssertFalse(cfg.hasPerLayerInputs)
        XCTAssertEqual(cfg.numKVSharedLayers, 0)
    }

    func testE2BConfigLeavesMoEDisabledByDefault() throws {
        // e2b: no MoE block, no K-eq-V, PLE on.
        let cfg = try decodeTextConfig("""
        {
          "hidden_size": 1536, "intermediate_size": 6144,
          "num_attention_heads": 8, "num_key_value_heads": 1,
          "num_hidden_layers": 35, "vocab_size": 1024,
          "head_dim": 256, "global_head_dim": 512,
          "hidden_size_per_layer_input": 256,
          "num_kv_shared_layers": 20,
          "use_double_wide_mlp": true
        }
        """)

        XCTAssertFalse(cfg.enableMoeBlock)
        XCTAssertNil(cfg.numExperts)
        XCTAssertNil(cfg.topKExperts)
        XCTAssertNil(cfg.moeIntermediateSize)
        XCTAssertFalse(cfg.attentionKEqV)
        XCTAssertNil(cfg.numGlobalKeyValueHeads)
        XCTAssertTrue(cfg.hasPerLayerInputs)
    }

    // MARK: - kvHeads / useKEqV per-layer

    func testKVHeadsAndUseKEqVOnFullLayer() throws {
        let cfg = try decodeTextConfig("""
        {
          "hidden_size": 16, "intermediate_size": 32,
          "num_attention_heads": 4, "num_key_value_heads": 4,
          "num_hidden_layers": 2, "vocab_size": 100,
          "head_dim": 4, "global_head_dim": 8,
          "attention_k_eq_v": true,
          "num_global_key_value_heads": 1,
          "layer_types": ["sliding_attention", "full_attention"]
        }
        """)

        // Sliding layer: standard KV heads, no K-eq-V.
        XCTAssertEqual(cfg.kvHeads(layerIdx: 0), 4)
        XCTAssertFalse(cfg.useKEqV(layerIdx: 0))

        // Full layer: global KV heads + K-eq-V active.
        XCTAssertEqual(cfg.kvHeads(layerIdx: 1), 1)
        XCTAssertTrue(cfg.useKEqV(layerIdx: 1))
    }

    func testKVHeadsIgnoresGlobalCountWhenKEqVDisabled() throws {
        // attention_k_eq_v false (e2b/e4b): global head count is moot.
        let cfg = try decodeTextConfig("""
        {
          "hidden_size": 16, "intermediate_size": 32,
          "num_attention_heads": 4, "num_key_value_heads": 4,
          "num_hidden_layers": 2, "vocab_size": 100,
          "head_dim": 4, "global_head_dim": 8,
          "num_global_key_value_heads": 1,
          "layer_types": ["sliding_attention", "full_attention"]
        }
        """)
        XCTAssertEqual(cfg.kvHeads(layerIdx: 1), 4,
            "K-eq-V off: full layer must keep the standard KV head count")
        XCTAssertFalse(cfg.useKEqV(layerIdx: 1))
    }

    // MARK: - switch_glu unpacker

    /// Qwen3-MoE's `switch_mlp` path must continue to prepend
    /// `experts.` since the parent of the marker there (`mlp`) is NOT
    /// the experts container. The Gemma 4 `switch_glu` path no longer
    /// goes through this unpacker -- the SwitchGLU module hierarchy
    /// matches the in-checkpoint stacked-key layout directly -- so
    /// this pinning is now Qwen3-MoE-only.
    func testUnpackSwitchMLPStillPrependsExperts() throws {
        let numExperts = 2
        let stacked = MLXArray(0 ..< Int32(numExperts * 6)).reshaped(numExperts, 3, 2)
        var weights: [String: MLXArray] = [
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": stacked,
        ]
        unpackStackedMoEWeights(
            &weights, marker: ".switch_mlp.", numExperts: numExperts)

        XCTAssertNil(weights["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
        XCTAssertNotNil(weights["model.layers.0.mlp.experts.0.gate_proj.weight"])
        XCTAssertNotNil(weights["model.layers.0.mlp.experts.1.gate_proj.weight"])
    }

    // MARK: - Helpers

    private func decodeTextConfig(_ json: String) throws -> Gemma4Config {
        let wrapped = """
        { "text_config": \(json) }
        """
        return try JSONDecoder().decode(
            Gemma4Config.self, from: Data(wrapped.utf8))
    }
}
