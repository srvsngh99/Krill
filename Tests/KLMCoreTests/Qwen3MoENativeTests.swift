import XCTest
import MLX
import MLXNN
@testable import KLMCore

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

        // Sparse layer 0 must expose router + expert weights.
        XCTAssertTrue(names.contains("model.layers.0.mlp.gate.weight"),
            "Sparse router weight missing from parameter cache - "
            + "dual @ModuleInfo(key: \"mlp\") collision regression")
        XCTAssertTrue(names.contains("model.layers.0.mlp.experts.0.gate_proj.weight"),
            "Expert 0 gate_proj missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.experts.0.up_proj.weight"),
            "Expert 0 up_proj missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.experts.0.down_proj.weight"),
            "Expert 0 down_proj missing from parameter cache")
        XCTAssertTrue(names.contains("model.layers.0.mlp.experts.3.gate_proj.weight"),
            "Last expert gate_proj missing - all numExperts FFNs must be loadable")
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
        // Layer 1: sparse router + experts.
        XCTAssertTrue(names.contains("model.layers.1.mlp.gate.weight"))
        XCTAssertTrue(names.contains("model.layers.1.mlp.experts.0.gate_proj.weight"))

        // And the two layers must NOT cross-leak: a dense layer
        // must not expose `mlp.gate` (the sparse router key), and
        // a sparse layer must not expose `mlp.gate_proj` (the
        // dense gate-projection key).
        XCTAssertFalse(names.contains("model.layers.0.mlp.gate.weight"),
            "Dense layer 0 must not expose the sparse router key")
        XCTAssertFalse(names.contains("model.layers.1.mlp.gate_proj.weight"),
            "Sparse layer 1 must not expose the dense gate_proj key")
    }

    // MARK: - Scatter dispatch parity + benchmark

    /// Build a sparse MLP, freeze its random weights, run BOTH the
    /// production scatter dispatch (`callAsFunction`) and the
    /// brute-force reference (`referenceForward`) on the same
    /// input, and assert the outputs are numerically equal. The
    /// two differ only in summation order, so the result must
    /// agree within a small fp tolerance. This is the load-bearing
    /// correctness test for the scatter optimization: a wrong
    /// sort / un-sort / index mapping would diverge here.
    private func assertScatterMatchesReference(
        numExperts: Int, topK: Int, hidden: Int, moeIntermediate: Int,
        batch: Int, seqLen: Int, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let cfg = try decode(tinyConfigJSON(
            hidden: hidden, heads: 2, kvHeads: 1, layers: 1, headDim: 16,
            numExperts: numExperts, topK: topK, moeIntermediate: moeIntermediate))
        let mlp = Qwen3MoESparseMLP(cfg)
        // Deterministic-ish input; the random module weights are
        // shared between the two forwards so any divergence is the
        // dispatch logic, not the weights.
        let x = MLXRandom.normal([batch, seqLen, hidden]).asType(.float32)
        let scatter = mlp(x)
        let reference = mlp.referenceForward(x)
        eval(scatter, reference)
        XCTAssertEqual(scatter.shape, reference.shape, file: file, line: line)
        let diff = abs(scatter - reference).max().item(Float.self)
        XCTAssertLessThan(diff, 2e-3,
            "Scatter dispatch must match the brute-force reference "
            + "within fp tolerance; max abs diff was \(diff)",
            file: file, line: line)
    }

    func testScatterMatchesReferenceSmall() throws {
        // 4 experts, top-2, single token.
        try assertScatterMatchesReference(
            numExperts: 4, topK: 2, hidden: 32, moeIntermediate: 16,
            batch: 1, seqLen: 1)
    }

    func testScatterMatchesReferenceMultiToken() throws {
        // 8 experts, top-2, a prefill-sized batch (24 tokens) so
        // multiple tokens route to the same expert.
        try assertScatterMatchesReference(
            numExperts: 8, topK: 2, hidden: 64, moeIntermediate: 32,
            batch: 2, seqLen: 12)
    }

    func testScatterMatchesReferenceManyExperts() throws {
        // 32 experts, top-4 - closer to the Qwen3-30B-A3B
        // sparsity ratio. Some experts will receive zero tokens;
        // the scatter path must skip them and still match.
        try assertScatterMatchesReference(
            numExperts: 32, topK: 4, hidden: 64, moeIntermediate: 32,
            batch: 4, seqLen: 8)
    }

    func testScatterMatchesReferenceTopKEqualsExperts() throws {
        // Degenerate case: top-K == numExperts. Every token routes
        // to every expert; the scatter must still equal the
        // reference (all experts non-empty, full assignment list).
        try assertScatterMatchesReference(
            numExperts: 4, topK: 4, hidden: 32, moeIntermediate: 16,
            batch: 1, seqLen: 6)
    }

    /// Micro-benchmark: time the scatter dispatch against the
    /// brute-force reference on a sparsity ratio that mirrors
    /// Qwen3-30B-A3B (top-K much smaller than numExperts). The
    /// scatter path runs each expert on only its routed tokens,
    /// so total expert-FFN work scales with `topK` instead of
    /// `numExperts`. This is not a hard assertion on wall time
    /// (CI timing is noisy); it prints both timings and asserts
    /// the scatter path is not pathologically slower.
    func testScatterDispatchBenchmark() throws {
        let numExperts = 64
        let topK = 8
        let hidden = 128
        let moeIntermediate = 64
        let cfg = try decode(tinyConfigJSON(
            hidden: hidden, heads: 4, kvHeads: 2, layers: 1, headDim: 16,
            numExperts: numExperts, topK: topK, moeIntermediate: moeIntermediate))
        let mlp = Qwen3MoESparseMLP(cfg)
        // Prefill-sized input: 64 tokens.
        let x = MLXRandom.normal([1, 64, hidden]).asType(.float32)

        // Warm up both paths (first call compiles MLX kernels).
        eval(mlp(x))
        eval(mlp.referenceForward(x))

        let iterations = 20
        let scatterStart = Date()
        for _ in 0 ..< iterations { eval(mlp(x)) }
        let scatterMs = Date().timeIntervalSince(scatterStart) / Double(iterations) * 1000

        let refStart = Date()
        for _ in 0 ..< iterations { eval(mlp.referenceForward(x)) }
        let refMs = Date().timeIntervalSince(refStart) / Double(iterations) * 1000

        print("[WS6 scatter benchmark] experts=\(numExperts) topK=\(topK) "
            + "tokens=64 | scatter=\(String(format: "%.3f", scatterMs))ms "
            + "reference=\(String(format: "%.3f", refMs))ms "
            + "ratio=\(String(format: "%.2fx", refMs / max(scatterMs, 1e-6)))")
        // The scatter path does more index bookkeeping but far
        // fewer expert-FFN flops. On a 64-expert / top-8 shape it
        // should not be more than ~1.5x slower even in the worst
        // case (tiny test FFNs make the fixed overhead dominate);
        // on the real 128-expert checkpoint the FFN flops dominate
        // and the scatter path wins decisively.
        XCTAssertLessThan(scatterMs, refMs * 1.5,
            "Scatter dispatch should not be pathologically slower "
            + "than the brute-force reference")
    }

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
}
