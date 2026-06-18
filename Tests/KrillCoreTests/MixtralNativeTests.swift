import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import KrillCore

/// Native runtime tests for the Mixtral sparse-MoE family
/// (`MixtralForCausalLM`, `model_type: mixtral`). The router + expert
/// dispatch lives in `MixtralModel.swift`; these tests pin config parsing,
/// module structure, the routing arithmetic, and the forward pass on a
/// tiny synthetic instance (random weights, no real checkpoint), the same
/// way `Qwen3MoENativeTests` covers Qwen 3 MoE. End-to-end numerical
/// correctness is validated separately against a real mlx-community Mixtral
/// checkpoint vs the mlx-lm reference.
final class MixtralNativeTests: XCTestCase {

    private func tinyConfigJSON(
        hidden: Int = 32,
        intermediate: Int = 64,
        heads: Int = 2,
        kvHeads: Int = 1,
        layers: Int = 2,
        vocab: Int = 64,
        numExperts: Int = 4,
        topK: Int = 2
    ) -> [String: Any] {
        return [
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-5,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 128,
            "num_local_experts": numExperts,
            "num_experts_per_tok": topK,
            // The SwitchGLU experts are quantized-only modules
            // (`MixtralQuantizedSwitchedLinear` allocates packed uint32
            // weights + scales/biases at init), so a quantization block is
            // required to assemble. group_size 16 divides every tiny dim
            // here (16/32/64); bits 4 keeps the packed/group shapes
            // integral. Forwards run gather_qmm over placeholder (zero)
            // expert weights -- enough to pin shapes, finiteness, routing
            // counts, and utilization.
            "quantization": ["group_size": 16, "bits": 4],
        ]
    }

    private func decode(_ json: [String: Any]) throws -> MixtralConfig {
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(MixtralConfig.self, from: data)
    }

    // MARK: - Config parsing

    func testConfigParsesMixtralFields() throws {
        let cfg = try decode(tinyConfigJSON(
            intermediate: 14336, numExperts: 8, topK: 2))
        XCTAssertEqual(cfg.numLocalExperts, 8)
        XCTAssertEqual(cfg.numExpertsPerToken, 2)
        XCTAssertEqual(cfg.intermediateSize, 14336)
        XCTAssertEqual(cfg.ropeTheta, 1_000_000.0)
    }

    func testConfigDefaultsApplyWhenFieldsMissing() throws {
        // Omit the optional MoE fields; defaults must match Mixtral-8x7B.
        let cfg = try decode([
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 64,
        ])
        XCTAssertEqual(cfg.numLocalExperts, 8)
        XCTAssertEqual(cfg.numExpertsPerToken, 2)
        // num_key_value_heads defaults to num_attention_heads when absent.
        XCTAssertEqual(cfg.numKeyValueHeads, 2)
    }

    // MARK: - Module construction

    func testModelInstantiatesWithoutWeights() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = MixtralForCausalLM(cfg)
        XCTAssertEqual(model.config.numLocalExperts, 4)
        XCTAssertEqual(model.config.numExpertsPerToken, 2)
    }

    // MARK: - Forward pass (synthetic weights)

    func testForwardPassProducesFiniteLogits() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = MixtralForCausalLM(cfg)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, caches: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64], "Output logits must be [B, L, V]")
        eval(logits)
        let arr = logits.asArray(Float.self)
        XCTAssertEqual(arr.count, 1 * 3 * 64)
        XCTAssertNil(arr.firstIndex { !$0.isFinite }, "Forward must not emit NaN/Inf")
    }

    func testLastTokenOnlySlicesToFinalPosition() throws {
        let cfg = try decode(tinyConfigJSON())
        let model = MixtralForCausalLM(cfg)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped([1, 3])
        let logits = model(tokens, caches: nil, lastTokenOnly: true)
        XCTAssertEqual(logits.shape, [1, 1, 64],
            "lastTokenOnly must slice hidden to the final position before lm_head")
        eval(logits)
    }

    // MARK: - Parameter-cache visibility

    /// The router AND the three stacked SwitchGLU projections must appear in
    /// the parameter cache under the `block_sparse_moe` key (mlx-lm naming,
    /// not `mlp`), so the strict-verify loader can bind the packed expert
    /// tensors. A mis-keyed module would silently drop these on load.
    func testSparseMoEWeightsAreVisibleInParameterCache() throws {
        let cfg = try decode(tinyConfigJSON(layers: 1))
        let model = MixtralForCausalLM(cfg)
        let names = Set(model.parameters().flattened().map { $0.0 })

        XCTAssertTrue(names.contains("model.layers.0.block_sparse_moe.gate.weight"),
            "Router gate weight missing under block_sparse_moe")
        XCTAssertTrue(names.contains("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"),
            "SwitchGLU gate_proj weight missing")
        XCTAssertTrue(names.contains("model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales"),
            "SwitchGLU gate_proj scales missing - quantized expert tensor")
        XCTAssertTrue(names.contains("model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"),
            "SwitchGLU up_proj weight missing")
        XCTAssertTrue(names.contains("model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"),
            "SwitchGLU down_proj weight missing")
        XCTAssertTrue(names.contains("lm_head.weight"),
            "Mixtral has an untied lm_head")
        // The HF per-expert layout must NOT appear: we bind the stacked
        // switch_mlp tensors directly (mlx-community sanitized format).
        XCTAssertFalse(names.contains("model.layers.0.block_sparse_moe.experts.0.w1.weight"),
            "Per-expert keys are gone; SwitchGLU uses stacked switch_mlp tensors")
    }

    // MARK: - Routing / utilization

    /// Every `(token, slot)` assignment lands in exactly one expert, so the
    /// cumulative counts sum to `N * topK`.
    func testExpertCountsSumToAssignments() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, intermediate: 64, layers: 1, numExperts: 8, topK: 2))
        let mlp = MixtralSparseMLP(cfg)
        let batch = 2, seqLen = 6
        eval(mlp(MLXRandom.normal([batch, seqLen, 64]).asType(.float32)))
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), batch * seqLen * 2,
            "Cumulative expert counts must sum to N * topK")
    }

    func testExpertCountsAccumulateAndReset() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, intermediate: 64, layers: 1, numExperts: 8, topK: 2))
        let mlp = MixtralSparseMLP(cfg)
        let x = MLXRandom.normal([1, 5, 64]).asType(.float32)
        eval(mlp(x))
        eval(mlp(x))
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), 2 * 5 * 2)
        mlp.resetUtilizationStats()
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), 0,
            "resetUtilizationStats must zero the tally")
        XCTAssertEqual(mlp.cumulativeExpertCounts.count, 8)
    }

    func testMoEUtilizationAggregatesEveryLayer() throws {
        // Two layers, both sparse (Mixtral has no dense fallback layers).
        let cfg = try decode(tinyConfigJSON(layers: 2, numExperts: 8, topK: 2))
        let model = MixtralForCausalLM(cfg)
        model.resetMoEUtilizationStats()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)]).reshaped([1, 4])
        eval(model(tokens, caches: nil))

        let util = model.moeUtilization()
        XCTAssertEqual(util.sparseLayers, 2)
        XCTAssertEqual(util.expertsPerLayer, 8)
        XCTAssertEqual(util.totalExpertSlots, 16, "2 layers * 8 experts")
        XCTAssertEqual(util.totalAssignments, 2 * 4 * 2, "per layer: 4 tokens * topK 2")
        XCTAssertGreaterThan(util.activeExpertSlots, 0)
        XCTAssertLessThanOrEqual(util.activeExpertSlots, util.totalExpertSlots)

        model.resetMoEUtilizationStats()
        XCTAssertEqual(model.moeUtilization().totalAssignments, 0)
    }
}
