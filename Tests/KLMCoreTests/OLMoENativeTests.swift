import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import KLMCore

/// Native runtime tests for the OLMoE family (`OlmoeForCausalLM`,
/// `model_type: olmoe`). Pins config parsing, the whole-projection q/k-norm
/// attention, module structure (no shared expert), routing counts, and the
/// forward pass on a tiny synthetic instance. Logit parity vs mlx-lm is
/// covered by the gated `OLMoEParityTests`.
final class OLMoENativeTests: XCTestCase {

    private func tinyConfigJSON(
        hidden: Int = 32,
        intermediate: Int = 64,
        heads: Int = 2,
        kvHeads: Int = 1,
        layers: Int = 2,
        vocab: Int = 64,
        numExperts: Int = 4,
        topK: Int = 2,
        normTopKProb: Bool = false,
        tieEmbeddings: Bool = false
    ) -> [String: Any] {
        return [
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "max_position_embeddings": 128,
            "num_experts": numExperts,
            "num_experts_per_tok": topK,
            "norm_topk_prob": normTopKProb,
            "tie_word_embeddings": tieEmbeddings,
            "quantization": ["group_size": 16, "bits": 4],
        ]
    }

    private func decode(_ json: [String: Any]) throws -> OLMoEConfig {
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(OLMoEConfig.self, from: data)
    }

    // MARK: - Config parsing

    func testConfigParsesOLMoEFields() throws {
        let cfg = try decode(tinyConfigJSON(
            numExperts: 64, topK: 8, normTopKProb: false))
        XCTAssertEqual(cfg.numExperts, 64)
        XCTAssertEqual(cfg.numExpertsPerToken, 8)
        XCTAssertFalse(cfg.normTopKProb, "OLMoE-1B-7B default is norm_topk_prob: false")
        XCTAssertEqual(cfg.ropeTheta, 10000.0)
    }

    func testTieWordEmbeddingsDefaultsTrue() throws {
        // mlx-lm/HF OLMoE default tie_word_embeddings is true.
        let cfg = try decode([
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 64,
            "num_experts": 4,
            "num_experts_per_tok": 2,
        ])
        XCTAssertTrue(cfg.tieWordEmbeddings)
    }

    // MARK: - Module construction + forward

    func testModelInstantiatesAndTieControlsLmHead() throws {
        let tied = OLMoEForCausalLM(try decode(tinyConfigJSON(tieEmbeddings: true)))
        XCTAssertNil(tied.lmHead, "tie_word_embeddings: true must skip lm_head")
        let untied = OLMoEForCausalLM(try decode(tinyConfigJSON(tieEmbeddings: false)))
        XCTAssertNotNil(untied.lmHead)
    }

    func testForwardPassProducesFiniteLogits() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = OLMoEForCausalLM(cfg)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, caches: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64])
        eval(logits)
        let arr = logits.asArray(Float.self)
        XCTAssertNil(arr.firstIndex { !$0.isFinite }, "Forward must not emit NaN/Inf")
    }

    // MARK: - Parameter-cache visibility

    /// The whole-projection q/k-norm weights and the routed experts must
    /// appear; there must be NO shared-expert keys (OLMoE has none).
    func testAttentionNormAndExpertWeightsAreVisible() throws {
        let cfg = try decode(tinyConfigJSON(layers: 1, tieEmbeddings: false))
        let model = OLMoEForCausalLM(cfg)
        let names = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(names.contains("model.layers.0.self_attn.q_norm.weight"),
            "Whole-projection q_norm weight missing")
        XCTAssertTrue(names.contains("model.layers.0.self_attn.k_norm.weight"),
            "Whole-projection k_norm weight missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.gate.weight"),
            "Router gate weight missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.gate_proj.weight"),
            "SwitchGLU gate_proj weight missing")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.gate_proj.scales"),
            "SwitchGLU gate_proj scales missing - quantized expert tensor")
        // OLMoE has no shared expert.
        XCTAssertFalse(names.contains { $0.contains("shared_expert") },
            "OLMoE must not declare any shared-expert weights")
    }

    /// q_norm normalizes over the full n_heads*head_dim, not per-head.
    func testQKNormDimensionsAreWholeProjection() throws {
        let cfg = try decode(tinyConfigJSON(hidden: 32, heads: 2, kvHeads: 1, layers: 1))
        // head_dim = 32 / 2 = 16; q_norm over 2*16 = 32; k_norm over 1*16 = 16.
        let attn = OLMoEAttention(cfg)
        let flat = attn.parameters().flattened()
        let qNorm = flat.first { $0.0 == "q_norm.weight" }
        let kNorm = flat.first { $0.0 == "k_norm.weight" }
        XCTAssertEqual(qNorm?.1.shape, [32], "q_norm spans n_heads*head_dim")
        XCTAssertEqual(kNorm?.1.shape, [16], "k_norm spans n_kv_heads*head_dim")
    }

    // MARK: - Routing / utilization

    func testExpertCountsSumToAssignments() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, layers: 1, numExperts: 8, topK: 4))
        let mlp = OLMoESparseMLP(cfg)
        let batch = 2, seqLen = 6
        eval(mlp(MLXRandom.normal([batch, seqLen, 64]).asType(.float32)))
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), batch * seqLen * 4,
            "Cumulative expert counts must sum to N * topK")
    }

    func testMoEUtilizationAggregatesEveryLayer() throws {
        let cfg = try decode(tinyConfigJSON(layers: 2, numExperts: 8, topK: 2))
        let model = OLMoEForCausalLM(cfg)
        model.resetMoEUtilizationStats()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped([1, 4])
        eval(model(tokens, caches: nil))
        let util = model.moeUtilization()
        XCTAssertEqual(util.sparseLayers, 2)
        XCTAssertEqual(util.totalAssignments, 2 * 4 * 2)
        model.resetMoEUtilizationStats()
        XCTAssertEqual(model.moeUtilization().totalAssignments, 0)
    }
}
