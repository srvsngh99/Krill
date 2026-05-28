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

    /// Stacked `experts.switch_glu.{proj}.{weight,scales,biases}` keys
    /// must rewrite to `experts.{e}.{proj}.{field}` -- NOT to
    /// `experts.experts.{e}.{proj}.{field}`. The doubled-`experts.`
    /// form was the original failure when the rewrite reused the
    /// Qwen3-MoE template that prepends `experts.` to the prefix; on
    /// Gemma 4 the prefix already ends in `.experts` so the
    /// per-marker branch in `unpackStackedMoEWeights` writes the
    /// shorter target.
    func testUnpackSwitchGLUWritesSingleExpertsLevel() throws {
        let numExperts = 4
        let outDim = 3
        let inDim = 2
        let stacked = MLXArray(0 ..< Int32(numExperts * outDim * inDim))
            .reshaped(numExperts, outDim, inDim)
        var weights: [String: MLXArray] = [
            "language_model.model.layers.0.experts.switch_glu.gate_proj.weight": stacked,
            "language_model.model.layers.0.experts.switch_glu.up_proj.scales": stacked,
            "language_model.model.layers.0.experts.switch_glu.down_proj.biases": stacked,
            "language_model.model.layers.0.self_attn.q_proj.weight":
                MLXArray.zeros([outDim, inDim]),
            "language_model.model.layers.0.mlp.gate_proj.weight":
                MLXArray.zeros([outDim, inDim]),
        ]

        unpackStackedMoEWeights(
            &weights, marker: ".switch_glu.", numExperts: numExperts)

        // Original stacked keys must be gone.
        XCTAssertNil(weights["language_model.model.layers.0.experts.switch_glu.gate_proj.weight"])
        XCTAssertNil(weights["language_model.model.layers.0.experts.switch_glu.up_proj.scales"])
        XCTAssertNil(weights["language_model.model.layers.0.experts.switch_glu.down_proj.biases"])

        // Non-MoE keys untouched.
        XCTAssertNotNil(weights["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(weights["language_model.model.layers.0.mlp.gate_proj.weight"])

        for e in 0 ..< numExperts {
            for (proj, field) in [
                ("gate_proj", "weight"),
                ("up_proj", "scales"),
                ("down_proj", "biases"),
            ] {
                // The "single experts" target. A doubled
                // `experts.experts.\(e)` would mismatch the model's
                // `[Gemma4MoEExpert]` submodule list and trigger the
                // `.incompatibleItems` failure that originally blocked
                // 26B-A4B loading.
                let key = "language_model.model.layers.0.experts.\(e).\(proj).\(field)"
                guard let w = weights[key] else {
                    XCTFail("Missing rewritten key \(key)"); return
                }
                XCTAssertEqual(w.shape, [outDim, inDim])
                XCTAssertNil(
                    weights["language_model.model.layers.0.experts.experts.\(e).\(proj).\(field)"],
                    "switch_glu rewrite must NOT emit a doubled `experts.experts.\(e).*` key")

                // Sanity: the slice content matches the stacked source.
                let diff = abs(w - stacked[e]).max().item(Float.self)
                XCTAssertEqual(diff, 0)
            }
        }
    }

    /// Qwen3-MoE's `switch_mlp` path must continue to prepend
    /// `experts.` since the parent of the marker there (`mlp`) is NOT
    /// the experts container. Regression test for the marker-based
    /// template selection in `unpackStackedMoEWeights`.
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
