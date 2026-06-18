import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import KrillCore

/// Native runtime tests for the Qwen 2 MoE family
/// (`Qwen2MoeForCausalLM`, `model_type: qwen2_moe`). Pins config parsing,
/// module structure (incl. the shared expert), routing counts, and the
/// forward pass on a tiny synthetic instance. End-to-end logit parity vs
/// mlx-lm is covered by the gated `Qwen2MoEParityTests`.
final class Qwen2MoENativeTests: XCTestCase {

    private func tinyConfigJSON(
        hidden: Int = 32,
        intermediate: Int = 64,
        heads: Int = 2,
        kvHeads: Int = 1,
        layers: Int = 2,
        vocab: Int = 64,
        numExperts: Int = 4,
        topK: Int = 2,
        moeIntermediate: Int = 32,
        sharedIntermediate: Int = 48,
        tieEmbeddings: Bool = false
    ) -> [String: Any] {
        return [
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 128,
            "num_experts": numExperts,
            "num_experts_per_tok": topK,
            "moe_intermediate_size": moeIntermediate,
            "shared_expert_intermediate_size": sharedIntermediate,
            "tie_word_embeddings": tieEmbeddings,
            "quantization": ["group_size": 16, "bits": 4],
        ]
    }

    private func decode(_ json: [String: Any]) throws -> Qwen2MoEConfig {
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(Qwen2MoEConfig.self, from: data)
    }

    // MARK: - Config parsing

    func testConfigParsesQwen2MoEFields() throws {
        let cfg = try decode(tinyConfigJSON(
            numExperts: 60, topK: 4, moeIntermediate: 1408, sharedIntermediate: 5632))
        XCTAssertEqual(cfg.numExperts, 60)
        XCTAssertEqual(cfg.numExpertsPerToken, 4)
        XCTAssertEqual(cfg.moeIntermediateSize, 1408)
        XCTAssertEqual(cfg.sharedExpertIntermediateSize, 5632)
    }

    /// The attention projection must request QKV bias and no q/k-norm
    /// (the Qwen 2 deltas vs Qwen 3).
    func testAttentionProjectionIsQwen2Style() throws {
        let cfg = try decode(tinyConfigJSON())
        let attn = cfg.qwenAttentionConfig
        XCTAssertTrue(attn.attentionBias, "Qwen 2 attention has QKV bias")
        XCTAssertFalse(attn.hasQKNorm, "Qwen 2 has no per-head q/k-norm")
    }

    // MARK: - Module construction + forward

    func testModelInstantiatesWithoutWeights() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = Qwen2MoEForCausalLM(cfg)
        XCTAssertEqual(model.config.numExperts, 4)
        XCTAssertNotNil(model.lmHead)
    }

    func testTiedEmbeddingsSkipsLmHead() throws {
        let cfg = try decode(tinyConfigJSON(tieEmbeddings: true))
        let model = Qwen2MoEForCausalLM(cfg)
        XCTAssertNil(model.lmHead)
    }

    func testForwardPassProducesFiniteLogits() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = Qwen2MoEForCausalLM(cfg)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, caches: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64])
        eval(logits)
        let arr = logits.asArray(Float.self)
        XCTAssertNil(arr.firstIndex { !$0.isFinite }, "Forward must not emit NaN/Inf")
    }

    // MARK: - Parameter-cache visibility (router + experts + shared expert)

    func testSparseMoEAndSharedExpertWeightsAreVisible() throws {
        let cfg = try decode(tinyConfigJSON(layers: 1))
        let model = Qwen2MoEForCausalLM(cfg)
        let names = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(names.contains("model.layers.0.mlp.gate.weight"),
            "Router gate weight missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.gate_proj.weight"),
            "SwitchGLU gate_proj weight missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.gate_proj.scales"),
            "SwitchGLU gate_proj scales missing - quantized expert tensor")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.down_proj.weight"),
            "SwitchGLU down_proj weight missing")
        // The shared expert (dense MLP) and its sigmoid gate.
        XCTAssertTrue(names.contains("model.layers.0.mlp.shared_expert.gate_proj.weight"),
            "Shared-expert gate_proj missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.shared_expert.down_proj.weight"),
            "Shared-expert down_proj missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.shared_expert_gate.weight"),
            "Shared-expert sigmoid gate missing")
        XCTAssertTrue(names.contains("lm_head.weight"))
    }

    // MARK: - Routing / utilization

    func testExpertCountsSumToAssignments() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, layers: 1, numExperts: 8, topK: 4, moeIntermediate: 32))
        let mlp = Qwen2MoESparseMLP(cfg)
        let batch = 2, seqLen = 6
        eval(mlp(MLXRandom.normal([batch, seqLen, 64]).asType(.float32)))
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), batch * seqLen * 4,
            "Cumulative expert counts must sum to N * topK")
    }

    func testMoEUtilizationAggregatesEveryLayer() throws {
        let cfg = try decode(tinyConfigJSON(layers: 2, numExperts: 8, topK: 2))
        let model = Qwen2MoEForCausalLM(cfg)
        model.resetMoEUtilizationStats()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped([1, 4])
        eval(model(tokens, caches: nil))

        let util = model.moeUtilization()
        XCTAssertEqual(util.sparseLayers, 2)
        XCTAssertEqual(util.expertsPerLayer, 8)
        XCTAssertEqual(util.totalAssignments, 2 * 4 * 2, "per layer: 4 tokens * topK 2")
        model.resetMoEUtilizationStats()
        XCTAssertEqual(model.moeUtilization().totalAssignments, 0)
    }
}
