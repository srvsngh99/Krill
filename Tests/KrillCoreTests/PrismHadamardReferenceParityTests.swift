import XCTest
import MLX
import KrillCache
@testable import KrillCore

/// Bisects `loadPrismHadamardQwen35` against a REFERENCE built from
/// `mlx_lm.models.qwen3_5.TextModel`, schema-2-aware (the pack's own bundled
/// `runtime/artifact.py` hard-requires `schema_version == 1` and refuses this
/// pack outright - it is stale for schema 2, which moved the tensor
/// namespace to mlx-vlm's `language_model.` prefix).
///
/// Gated on two environment variables (no absolute paths in source, per the
/// `DeepSeekParityTests` convention - neither the checkpoint nor the
/// generated fixture ship in the repo):
///   - `KRILL_PRISM_BONSAI2_DIR`: a local Prism Hadamard pack directory
///     (e.g. `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, `krill pull
///     bonsai-2-27b`'s destination) - `config.json`, `hadamard.json`,
///     `model.safetensors`, and `runtime/` (the pack's own bundled runtime,
///     needed by the generator below) must all be present.
///   - `KRILL_PRISM_PARITY_FIXTURE`: a `.safetensors` file produced by
///     `tools/verify_prism_hadamard_parity.py` (see its module docstring for
///     the schema-2 workaround and the mask=None wrinkle noted below).
///
/// To regenerate the fixture and run this suite:
/// ```
/// python3 tools/verify_prism_hadamard_parity.py \
///     "$KRILL_PRISM_BONSAI2_DIR" /tmp/prism_ref.safetensors
/// KRILL_PRISM_BONSAI2_DIR="$KRILL_PRISM_BONSAI2_DIR" \
/// KRILL_PRISM_PARITY_FIXTURE=/tmp/prism_ref.safetensors \
///     swift test --filter PrismHadamardReferenceParityTests
/// ```
///
/// Each test isolates a stage of the forward pass, outermost-in: `fwht` ->
/// `PrismPackedEmbedding` -> GDN/MLP layers 0-3 -> the first two
/// full-attention layers (3, 7) -> final logits (`lm_head`) -> cached vs
/// cacheless decode. Every test prints its max-abs/max-rel deviation
/// regardless of pass/fail, so a bisection run reports the actual numbers.
final class PrismHadamardReferenceParityTests: XCTestCase {
    private static let checkpointEnvVar = "KRILL_PRISM_BONSAI2_DIR"
    private static let fixtureEnvVar = "KRILL_PRISM_PARITY_FIXTURE"

    /// The checkpoint directory, or an `XCTSkip` if the env var is unset.
    /// Matches `DeepSeekParityTests.runParity`'s gating structure.
    private func requireCheckpointDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment[Self.checkpointEnvVar] else {
            throw XCTSkip("Set \(Self.checkpointEnvVar) (see tools/verify_prism_hadamard_parity.py)")
        }
        return URL(fileURLWithPath: path)
    }

    /// The generated reference fixture, or an `XCTSkip` if the env var is
    /// unset or the file cannot be loaded.
    private func requireFixture() throws -> [String: MLXArray] {
        guard let path = ProcessInfo.processInfo.environment[Self.fixtureEnvVar] else {
            throw XCTSkip("Set \(Self.fixtureEnvVar) (see tools/verify_prism_hadamard_parity.py)")
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    /// Max abs / max relative deviation between two same-shaped arrays (fp32
    /// upcast for the comparison). Always prints, independent of pass/fail,
    /// so a bisection run reports the actual numbers at every stage.
    @discardableResult
    private func report(_ got: MLXArray, _ ref: MLXArray, _ label: String) -> (maxAbs: Double, maxRel: Double) {
        let a = got.asType(.float32).flattened().asArray(Float.self)
        let b = ref.asType(.float32).flattened().asArray(Float.self)
        guard a.count == b.count else {
            XCTFail("\(label): element count mismatch (\(a.count) vs \(b.count))")
            return (.infinity, .infinity)
        }
        var maxAbs = 0.0
        var maxRel = 0.0
        for i in 0 ..< a.count {
            let x = Double(a[i]), y = Double(b[i])
            let diff = abs(x - y)
            maxAbs = max(maxAbs, diff)
            maxRel = max(maxRel, diff / max(abs(y), 1e-4))
        }
        print("[bisect] \(label): maxAbs=\(maxAbs) maxRel=\(maxRel) n=\(a.count)")
        return (maxAbs, maxRel)
    }

    // MARK: - Stage 1: fwht in isolation

    func testFWHTMatchesReference() throws {
        let checkpointDir = try requireCheckpointDir()
        let fixture = try requireFixture()
        let config = try PrismHadamardConfig(
            configData: Data(contentsOf: checkpointDir.appendingPathComponent("config.json")),
            directory: checkpointDir)
        guard let signs = config.signs[5120] else {
            XCTFail("no width-5120 sign vector"); return
        }
        let x = fixture["fwht_in"]!.asType(.float16)

        let fwd = fwht(x, block: 1024, signs: signs, inverse: false)
        let (fwdAbs, _) = report(fwd, fixture["fwht_fwd"]!, "fwht forward")
        XCTAssertLessThan(fwdAbs, 0.05, "fwht forward deviates from the reference transform")

        let inv = fwht(x, block: 1024, signs: signs, inverse: true)
        let (invAbs, _) = report(inv, fixture["fwht_inv"]!, "fwht inverse")
        XCTAssertLessThan(invAbs, 0.05, "fwht inverse deviates from the reference transform")
    }

    // MARK: - Stage 2: embed_tokens (gather -> dequantize -> INVERSE fwht)

    func testEmbedOutMatchesReference() throws {
        let checkpointDir = try requireCheckpointDir()
        let fixture = try requireFixture()
        let loaded = try loadModel(from: checkpointDir)
        guard let model = loaded.module as? Qwen35ForCausalLM else {
            XCTFail("expected Qwen35ForCausalLM"); return
        }
        let ids = fixture["prompt_ids"]!.asType(.int32).asArray(Int32.self)
        let tokens = MLXArray(ids).reshaped(1, ids.count)

        let h = model.model.embedTokens(tokens)
        MLX.eval(h)
        let (maxAbs, _) = report(h, fixture["embed_out"]!, "embed_out")
        XCTAssertLessThan(maxAbs, 0.1, "embed_tokens output diverges from the reference")
    }

    // MARK: - Stage 3+4: decoder layers 0-3 (GDN + MLP) and layer 3 (full-attn)

    /// Layers 0/1/2 (GatedDeltaNet - linear attention, no mask dependence
    /// beyond zeroing invalid positions) are asserted against the reference
    /// dump directly. Layers 3 and 7 (the first two full-attention layers)
    /// are only PRINTED, not asserted: the reference dump script
    /// (`tools/verify_prism_hadamard_parity.py`) captures per-layer
    /// intermediates by calling `layer(h, mask=None, cache=None)` directly -
    /// bypassing `Qwen3_5TextModel.__call__`'s own `create_attention_mask` -
    /// so its captured full-attention layers are UNMASKED (every position
    /// attends to every other, including future ones), while Krill's
    /// `Qwen35Attention` always builds a proper causal mask internally
    /// (`qwen35CausalMask`, whenever `L > 1`). That is why layer3/7 diverge
    /// wildly here (confirmed a reference-fixture artifact, not a Krill bug,
    /// by `testLogitsMatchReference` below and `testCachedDecodeMatchesCacheless`
    /// - the FULL model forward, which DOES apply masking on both sides,
    /// matches the reference almost exactly with the correct argmax).
    func testLayerOutputsMatchReference() throws {
        let checkpointDir = try requireCheckpointDir()
        let fixture = try requireFixture()
        let loaded = try loadModel(from: checkpointDir)
        guard let model = loaded.module as? Qwen35ForCausalLM else {
            XCTFail("expected Qwen35ForCausalLM"); return
        }
        let ids = fixture["prompt_ids"]!.asType(.int32).asArray(Int32.self)
        let tokens = MLXArray(ids).reshaped(1, ids.count)
        let unmaskedFullAttentionLayers: Set<Int> = [3, 7]

        var h = model.model.embedTokens(tokens)
        for i in 0 ..< 8 {
            h = model.model.layers[i](h, cache: nil, mropeCosSin: nil)
            MLX.eval(h)
            if let ref = fixture["layer\(i)_out"] {
                let (maxAbs, _) = report(h, ref, "layer\(i)_out")
                if !unmaskedFullAttentionLayers.contains(i) {
                    XCTAssertLessThan(maxAbs, 0.5, "layer \(i) output diverges from the reference")
                }
            }
        }
    }

    // MARK: - Stage 5: full forward -> lm_head logits

    func testLogitsMatchReference() throws {
        let checkpointDir = try requireCheckpointDir()
        let fixture = try requireFixture()
        let loaded = try loadModel(from: checkpointDir)
        let ids = fixture["prompt_ids"]!.asType(.int32).asArray(Int32.self)
        let tokens = MLXArray(ids).reshaped(1, ids.count)

        let logits = loaded.forward(tokens, nil)
        MLX.eval(logits)
        let last = logits[0, ids.count - 1]
        let (maxAbs, _) = report(last, fixture["logits"]!, "logits")
        XCTAssertLessThan(maxAbs, 2.0, "final logits diverge from the reference")

        // Same top-5 sanity the reference script prints: ' Paris' should lead.
        let top = argMax(last)
        print("[bisect] argmax token id = \(top.item(Int32.self))")
    }

    // MARK: - Stage 6: incremental (cached) decode vs cacheless, self-consistency

    /// The reference only exercises a cacheless prefill. `krill run` always
    /// decodes incrementally through `GatedDeltaCache` (linear layers) /
    /// `KVCache` (full-attention layers) - a state-threading bug there would
    /// still pass every stage above (single cacheless forward) while
    /// producing garbage at generation time. Self-consistency check: prefill
    /// + N cached single-token steps must equal a cacheless forward over the
    /// growing prefix at every step, exactly like `Qwen35RealCheckpointTests`
    /// pins for Ornith.
    func testCachedDecodeMatchesCacheless() throws {
        let checkpointDir = try requireCheckpointDir()
        let fixture = try requireFixture()
        let loaded = try loadModel(from: checkpointDir)
        var seq = fixture["prompt_ids"]!.asType(.int32).asArray(Int32.self).map { Int($0) }

        let caches = makeKVCaches(spec: loaded.cacheSpec, numLayers: loaded.numLayers) as [KVCacheProtocol]
        let pre = loaded.forward(MLXArray(seq.map { Int32($0) }).reshaped(1, seq.count), caches)
        MLX.eval(pre)
        var next = Int(argMax(pre[0, seq.count - 1]).item(Int32.self))
        for step in 0 ..< 5 {
            seq.append(next)
            let stepLogits = loaded.forward(MLXArray([Int32(next)]).reshaped(1, 1), caches)
            MLX.eval(stepLogits)
            let cachedNext = Int(argMax(stepLogits[0, 0]).item(Int32.self))

            let full = loaded.forward(MLXArray(seq.map { Int32($0) }).reshaped(1, seq.count), nil)
            MLX.eval(full)
            let cachelessNext = Int(argMax(full[0, seq.count - 1]).item(Int32.self))

            print("[bisect] decode step \(step): cached=\(cachedNext) cacheless=\(cachelessNext)")
            XCTAssertEqual(cachedNext, cachelessNext, "cached decode diverged from cacheless at step \(step)")
            next = cachelessNext
        }
    }
}
