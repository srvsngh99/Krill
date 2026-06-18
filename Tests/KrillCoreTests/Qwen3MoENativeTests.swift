import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// WS6 native runtime: tests for the Qwen 3 MoE
/// (`Qwen3MoeForCausalLM`) native Swift+MLX implementation. The
/// router + expert dispatch lives in `Qwen3MoEModel.swift`; these
/// tests pin the config parsing, module structure, and forward
/// pass on a tiny synthetic instance (random weights, no weight
/// load) so the implementation is exercised without the 16 GB
/// real checkpoint.
final class Qwen3MoENativeTests: XCTestCase {

    private func tinyConfigJSON(
        hidden: Int = 64,
        intermediate: Int = 128,
        heads: Int = 4,
        kvHeads: Int = 2,
        layers: Int = 2,
        vocab: Int = 256,
        headDim: Int = 16,
        numExperts: Int = 4,
        topK: Int = 2,
        moeIntermediate: Int = 32,
        mlpOnlyLayers: [Int] = [],
        sparseStep: Int = 1,
        tieEmbeddings: Bool = false
    ) -> [String: Any] {
        return [
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 128,
            "head_dim": headDim,
            "num_experts": numExperts,
            "num_experts_per_tok": topK,
            "moe_intermediate_size": moeIntermediate,
            "decoder_sparse_step": sparseStep,
            "mlp_only_layers": mlpOnlyLayers,
            "norm_topk_prob": true,
            "tie_word_embeddings": tieEmbeddings,
            "attention_bias": false,
            // The SwitchGLU experts are quantized-only modules
            // (`Qwen3QuantizedSwitchedLinear` allocates packed uint32
            // weights + scales/biases at init). A quantization block is
            // therefore required for the module to assemble. group_size
            // 16 divides every tiny dim used in these tests (16/32/64/
            // 128), and bits 4 keeps the packed/group shapes integral.
            // The synthetic forwards run gather_qmm over placeholder
            // (zero) expert weights, so expert output is zero -- enough
            // to pin shapes, finiteness, and routing/utilization, which
            // is all these no-weight-load tests assert.
            "quantization": ["group_size": 16, "bits": 4],
        ]
    }

    private func decode(_ json: [String: Any]) throws -> Qwen3MoEConfig {
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(Qwen3MoEConfig.self, from: data)
    }

    // MARK: - Config parsing

    func testConfigParsesQwen3MoEFields() throws {
        let cfg = try decode(tinyConfigJSON(
            numExperts: 128, topK: 8, moeIntermediate: 768))
        XCTAssertEqual(cfg.modelType, "qwen3_moe")
        XCTAssertEqual(cfg.numExperts, 128)
        XCTAssertEqual(cfg.numExpertsPerToken, 8)
        XCTAssertEqual(cfg.moeIntermediateSize, 768)
        XCTAssertEqual(cfg.decoderSparseStep, 1)
        XCTAssertTrue(cfg.mlpOnlyLayers.isEmpty)
        XCTAssertTrue(cfg.normTopKProb)
        XCTAssertFalse(cfg.tieWordEmbeddings)
    }

    func testConfigDefaultsApplyWhenFieldsMissing() throws {
        // Minimum-viable Qwen3-MoE config: omits the optional
        // fields. Defaults must match the Qwen3-30B-A3B shape so
        // an upstream config that drops `norm_topk_prob` or
        // `decoder_sparse_step` still produces a usable model.
        let cfg = try decode([
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "vocab_size": 256,
            "num_experts_per_tok": 8,
            "num_experts": 128,
        ])
        XCTAssertEqual(cfg.decoderSparseStep, 1)
        XCTAssertEqual(cfg.mlpOnlyLayers, [])
        XCTAssertTrue(cfg.normTopKProb)
    }

    func testIsSparseLayerHonorsMLPOnlyLayers() throws {
        let cfg = try decode(tinyConfigJSON(
            layers: 4, mlpOnlyLayers: [0, 3], sparseStep: 1))
        XCTAssertFalse(cfg.isSparseLayer(0),
            "Layer 0 listed in mlp_only_layers must fall back to dense")
        XCTAssertTrue(cfg.isSparseLayer(1))
        XCTAssertTrue(cfg.isSparseLayer(2))
        XCTAssertFalse(cfg.isSparseLayer(3))
    }

    func testIsSparseLayerHonorsDecoderSparseStep() throws {
        let cfg = try decode(tinyConfigJSON(
            layers: 4, mlpOnlyLayers: [], sparseStep: 2))
        XCTAssertTrue(cfg.isSparseLayer(0),
            "Layer 0 always sparse when decoder_sparse_step divides it")
        XCTAssertFalse(cfg.isSparseLayer(1),
            "Layer 1 must be dense at step=2")
        XCTAssertTrue(cfg.isSparseLayer(2))
        XCTAssertFalse(cfg.isSparseLayer(3))
    }

    // MARK: - Module construction

    func testModelInstantiatesWithoutWeights() throws {
        // Tiny config; we are NOT loading weights, just verifying
        // that the module hierarchy assembles without throwing.
        let cfg = try decode(tinyConfigJSON())
        let model = Qwen3MoEForCausalLM(cfg)
        XCTAssertEqual(model.config.numExperts, 4)
        XCTAssertEqual(model.config.numExpertsPerToken, 2)
        // `lmHead` is non-nil when tieWordEmbeddings == false.
        XCTAssertNotNil(model.lmHead)
    }

    func testTiedEmbeddingsSkipsLmHead() throws {
        let cfg = try decode(tinyConfigJSON(tieEmbeddings: true))
        let model = Qwen3MoEForCausalLM(cfg)
        XCTAssertNil(model.lmHead,
            "With tie_word_embeddings: true the lm_head module must "
            + "be skipped so the safetensors loader does not need a "
            + "separate lm_head.weight key.")
    }

    // MARK: - Forward pass (synthetic weights)

    func testForwardPassProducesFiniteLogits() throws {
        // A randomly-initialized Qwen3MoE forward should still
        // produce finite logits of the expected shape. This pins
        // the router + expert dispatch arithmetic: shape mismatches
        // or NaN/Inf escapes would surface here.
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, intermediate: 64, heads: 2, kvHeads: 1,
            layers: 2, vocab: 64, headDim: 16,
            numExperts: 4, topK: 2, moeIntermediate: 16))
        let model = Qwen3MoEForCausalLM(cfg)
        // Provide deterministic small input. Tokens in [0, vocab).
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)])
            .reshaped([1, 3])
        let logits = model(tokens, caches: nil)
        XCTAssertEqual(logits.shape, [1, 3, 64],
            "Output logits must be [B, L, V]")
        // Force evaluation and pull a couple of values back to
        // host to confirm finiteness.
        eval(logits)
        let arr = logits.asArray(Float.self)
        XCTAssertEqual(arr.count, 1 * 3 * 64)
        let bad = arr.firstIndex { !$0.isFinite }
        XCTAssertNil(bad, "Forward must not emit NaN/Inf")
    }

    func testForwardWithDenseAndSparseLayersMixed() throws {
        // Two layers, one dense (mlp_only_layers: [0]) and one
        // sparse. Exercises the per-layer dispatch in
        // Qwen3MoETransformerBlock.
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, intermediate: 64, heads: 2, kvHeads: 1,
            layers: 2, vocab: 64, headDim: 16,
            numExperts: 4, topK: 2, moeIntermediate: 16,
            mlpOnlyLayers: [0]))
        let model = Qwen3MoEForCausalLM(cfg)
        let tokens = MLXArray([Int32(0), Int32(1)]).reshaped([1, 2])
        let logits = model(tokens, caches: nil)
        XCTAssertEqual(logits.shape, [1, 2, 64])
        eval(logits)
    }

    // MARK: - Parameter-cache visibility (regression for the dual
    // `@ModuleInfo(key: "mlp")` collision)

    /// Both the sparse router/expert weights AND the dense MLP
    /// weights must appear in `model.parameters().flattened()`. An
    /// earlier draft declared two parallel `@ModuleInfo(key: "mlp")`
    /// properties on the block; Mirror reflection wrote both into
    /// `items["mlp"]` and the second nil overwrote the first
    /// concrete module. The result was that `update(parameters:)`
    /// silently skipped every sparse router + expert weight on
    /// load, leaving them at random init. This test catches that
    /// regression by enumerating the parameter cache directly.
    func testSparseMLPWeightsAreVisibleInParameterCache() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, intermediate: 64, heads: 2, kvHeads: 1,
            layers: 1, vocab: 64, headDim: 16,
            numExperts: 4, topK: 2, moeIntermediate: 16))
        let model = Qwen3MoEForCausalLM(cfg)
        let flat = model.parameters().flattened()
        let names = Set(flat.map { $0.0 })

        // Sparse layer 0 must expose the router weight plus the three
        // stacked SwitchGLU projections (the in-checkpoint key layout:
        // one `[E, O, I_packed]` tensor per projection, not per-expert
        // keys). All three of weight/scales/biases must appear so the
        // strict-verify loader can bind the quantized expert tensors.
        XCTAssertTrue(names.contains("model.layers.0.mlp.gate.weight"),
            "Sparse router weight missing from parameter cache - "
            + "dual @ModuleInfo(key: \"mlp\") collision regression")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.gate_proj.weight"),
            "SwitchGLU gate_proj weight missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.gate_proj.scales"),
            "SwitchGLU gate_proj scales missing - quantized expert tensor")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.up_proj.weight"),
            "SwitchGLU up_proj weight missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.switch_mlp.down_proj.weight"),
            "SwitchGLU down_proj weight missing from parameter cache")
        // The old per-expert key layout must NOT reappear (regression
        // guard: a reverted loader would silently drop the stacked keys).
        XCTAssertFalse(names.contains("model.layers.0.mlp.experts.0.gate_proj.weight"),
            "Per-expert keys are gone; SwitchGLU uses stacked switch_mlp tensors")
    }

    func testDenseFallbackMLPWeightsAreVisibleInParameterCache() throws {
        // Two layers: layer 0 dense (in mlp_only_layers), layer 1
        // sparse. Both MLP shapes must round-trip through the
        // parameter cache or the safetensors loader will silently
        // skip them.
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, intermediate: 64, heads: 2, kvHeads: 1,
            layers: 2, vocab: 64, headDim: 16,
            numExperts: 4, topK: 2, moeIntermediate: 16,
            mlpOnlyLayers: [0]))
        let model = Qwen3MoEForCausalLM(cfg)
        let names = Set(model.parameters().flattened().map { $0.0 })

        // Layer 0: dense QwenMLP keys.
        XCTAssertTrue(names.contains("model.layers.0.mlp.gate_proj.weight"),
            "Dense fallback gate_proj missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.up_proj.weight"),
            "Dense fallback up_proj missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.down_proj.weight"),
            "Dense fallback down_proj missing from parameter cache")
        // Layer 1: sparse router + stacked SwitchGLU experts.
        XCTAssertTrue(names.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(names.contains("model.layers.1.mlp.switch_mlp.gate_proj.weight"))

        // And the two layers must NOT cross-leak: a dense layer
        // must not expose `mlp.gate` (the sparse router key), and
        // a sparse layer must not expose `mlp.gate_proj` (the
        // dense gate-projection key).
        XCTAssertFalse(names.contains("model.layers.0.mlp.gate.weight"),
            "Dense layer 0 must not expose the sparse router key")
        XCTAssertFalse(names.contains("model.layers.1.mlp.gate_proj.weight"),
            "Sparse layer 1 must not expose the dense gate_proj key")
    }

    // Note: the brute-force `referenceForward` scatter-parity and
    // micro-benchmark tests were removed with the scatter dispatch
    // itself. Expert dispatch is now a single `gatherQuantizedMM` per
    // projection (`Qwen3SwitchGLU`); its numerical correctness is
    // verified end-to-end against the real Qwen3-Coder-30B-A3B
    // checkpoint (coherent generation + decode benchmark), the same
    // way Gemma 4's SwitchGLU (PR #82) is validated. A quantized-only
    // switched linear cannot be meaningfully exercised with the random
    // fp weights these no-weight-load unit tests use.

    // MARK: - Expert-utilization instrumentation

    /// Every `(token, slot)` assignment must land in exactly one
    /// expert's tally, so the cumulative counts sum to `N * topK`.
    func testExpertCountsSumToAssignments() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, heads: 2, kvHeads: 1, layers: 1, headDim: 16,
            numExperts: 8, topK: 2, moeIntermediate: 32))
        let mlp = Qwen3MoESparseMLP(cfg)
        let batch = 2, seqLen = 6
        eval(mlp(MLXRandom.normal([batch, seqLen, 64]).asType(.float32)))
        let total = mlp.cumulativeExpertCounts.reduce(0, +)
        XCTAssertEqual(total, batch * seqLen * 2,
            "Cumulative expert counts must sum to N * topK")
    }

    /// Counts accumulate across forwards until reset; reset zeros them.
    func testExpertCountsAccumulateAndReset() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 64, heads: 2, kvHeads: 1, layers: 1, headDim: 16,
            numExperts: 8, topK: 2, moeIntermediate: 32))
        let mlp = Qwen3MoESparseMLP(cfg)
        let x = MLXRandom.normal([1, 5, 64]).asType(.float32)
        eval(mlp(x))
        eval(mlp(x))
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), 2 * 5 * 2,
            "Two forwards must accumulate into the tally")
        mlp.resetUtilizationStats()
        XCTAssertEqual(mlp.cumulativeExpertCounts.reduce(0, +), 0,
            "resetUtilizationStats must zero the tally")
        XCTAssertEqual(mlp.cumulativeExpertCounts.count, 8)
    }

    /// With `topK == numExperts` every token routes to every expert,
    /// so after one forward every expert has a non-zero count.
    func testTopKEqualsExpertsActivatesEveryExpert() throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, heads: 2, kvHeads: 1, layers: 1, headDim: 16,
            numExperts: 4, topK: 4, moeIntermediate: 16))
        let mlp = Qwen3MoESparseMLP(cfg)
        eval(mlp(MLXRandom.normal([1, 3, 32]).asType(.float32)))
        XCTAssertTrue(mlp.cumulativeExpertCounts.allSatisfy { $0 > 0 },
            "topK == numExperts must activate every expert")
    }

    /// `moeUtilization()` aggregates only the sparse layers and
    /// reports a consistent snapshot.
    func testMoEUtilizationAggregatesSparseLayers() throws {
        // Two layers, both sparse (sparseStep 1, no mlp_only_layers).
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, intermediate: 64, heads: 2, kvHeads: 1,
            layers: 2, vocab: 64, headDim: 16,
            numExperts: 8, topK: 2, moeIntermediate: 16))
        let model = Qwen3MoEForCausalLM(cfg)
        model.resetMoEUtilizationStats()
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)])
            .reshaped([1, 4])
        eval(model(tokens, caches: nil))

        let util = model.moeUtilization()
        XCTAssertEqual(util.sparseLayers, 2)
        XCTAssertEqual(util.expertsPerLayer, 8)
        XCTAssertEqual(util.totalExpertSlots, 16, "2 layers * 8 experts")
        // 4 tokens * topK 2 assignments, per sparse layer.
        XCTAssertEqual(util.totalAssignments, 2 * 4 * 2)
        XCTAssertGreaterThan(util.activeExpertSlots, 0)
        XCTAssertLessThanOrEqual(util.activeExpertSlots, util.totalExpertSlots)
        XCTAssertGreaterThan(util.utilizationRatio, 0.0)
        XCTAssertLessThanOrEqual(util.utilizationRatio, 1.0)
        XCTAssertGreaterThan(util.maxExpertLoad, 0)

        model.resetMoEUtilizationStats()
        XCTAssertEqual(model.moeUtilization().totalAssignments, 0,
            "resetMoEUtilizationStats must clear every sparse layer")
    }

    /// Dense (`mlp_only_layers`) layers are excluded from the
    /// utilization snapshot — only true sparse layers are counted.
    func testMoEUtilizationExcludesDenseLayers() throws {
        // Layer 0 dense, layer 1 sparse.
        let cfg = try decode(tinyConfigJSON(
            hidden: 32, intermediate: 64, heads: 2, kvHeads: 1,
            layers: 2, vocab: 64, headDim: 16,
            numExperts: 4, topK: 2, moeIntermediate: 16,
            mlpOnlyLayers: [0]))
        let model = Qwen3MoEForCausalLM(cfg)
        eval(model(MLXArray([Int32(0), Int32(1)]).reshaped([1, 2]), caches: nil))
        let util = model.moeUtilization()
        XCTAssertEqual(util.sparseLayers, 1,
            "Only the one sparse layer must be counted")
        XCTAssertEqual(util.totalExpertSlots, 4)
    }

    // Note: the `unpackSwitchMLPWeights` key-rewrite tests were removed
    // with the helper itself. The mlx-community stacked
    // `mlp.switch_mlp.{proj}.{weight,scales,biases}` tensors now bind
    // directly to the `Qwen3SwitchGLU` module hierarchy (no per-expert
    // unpacking), and the strict-verify loader (.noUnusedKeys,
    // .allModelKeysSet, .shapeMismatch) is the regression guard against
    // a silent key-drop -- a load that does not cover every stacked
    // tensor now crashes at load time rather than decoding garbage.
}
