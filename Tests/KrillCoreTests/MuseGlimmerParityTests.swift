import XCTest
import MLX
import KrillCache
@testable import KrillCore

/// Logit-parity check for the native Muse Glimmer runtime
/// (`MuseGlimmerForConditionalGeneration`, model_type "muse_glimmer") against
/// the transformers reference.
///
/// Gated on `KRILL_MUSE_GLIMMER_PARITY_DIR`, a directory produced by
/// `tools/verify_muse_glimmer_parity.py` holding a tiny random Muse Glimmer
/// checkpoint (config.json + model.safetensors) plus `reference_logits.json`.
/// Skipped when the env var is unset — the fixture is generated on demand, not
/// committed, mirroring the other parity tests.
///
/// Green as of 2026-08-11 against transformers 5.16.0.dev0. It caught two real
/// bugs in the position-embedding resample on its first run (wrong
/// align_corners convention, and a missing zero-padding mask on out-of-range
/// taps) — neither of which changes a single tensor SHAPE, so nothing else in
/// the suite would have noticed. SYNTHETIC ONLY: the real 30B checkpoint has
/// never been loaded here, so this gates semantics, not throughput.
///
/// It pins the five things most likely to be wrong in this port:
///
///  1. **The attention output gate.** `attn_out * sigmoid(gate_proj(x))` before
///     `o_proj`, fed the layer's NORMED input.
///  2. **NoPE on the full-attention layers.** The fixture puts a
///     `layer_rope_theta == 0` layer mid-stack and uses a multi-token prompt,
///     so rotating it diverges.
///  3. **The sliding window actually biting.** The fixture's window (6) is
///     smaller than the prompt (12), so a runtime that hands the sliding layers
///     the full context diverges.
///  4. **The two RMSNorm formulas.** `(1 + w)` per layer, plain `w` at the end.
///     The oracle initialises norm weights away from zero so the two cannot
///     coincide.
///  5. **Cache kinds.** Sliding layers need `RotatingKVCache`, the NoPE layer a
///     full `KVCache`; the incremental-decode case is what catches a mix-up.
final class MuseGlimmerParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let num_hidden_layers: Int
        let sliding_window: Int
        let layer_types: [String]
        let layer_rope_theta: [Double]
        let last_token_logits: [Float]
        let argmax: Int
        let step_logits: [Float]
        let softcap: Float
        let max_abs_tolerance: Double?
        // Optional so this suite can also be pointed at a REAL checkpoint,
        // where the reference is another runtime's logits and there is no
        // synthetic vision fixture. The vision tests skip when these are absent.
        let vision_grid: [Int]?
        let vision_patch_dim: Int?
        let vision_pixel_values: [Float]?
        let vision_tower_out: [Float]?
        let vision_tower_shape: [Int]?
        let vision_image_features: [Float]?
        let vision_image_features_shape: [Int]?
    }

    private func fixture() throws -> (URL, Reference) {
        guard let dirPath = ProcessInfo.processInfo
            .environment["KRILL_MUSE_GLIMMER_PARITY_DIR"] else {
            throw XCTSkip(
                "Set KRILL_MUSE_GLIMMER_PARITY_DIR "
                + "(see tools/verify_muse_glimmer_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(
            contentsOf: dir.appendingPathComponent("reference_logits.json"))
        return (dir, try JSONDecoder().decode(Reference.self, from: refData))
    }

    /// Compare two logit vectors: argmax agreement, cosine ~1, small max-abs.
    private func assertMatches(
        _ got: [Float], _ want: [Float], label: String, maxAbsTolerance: Double = 2e-3,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(got.count, want.count, "\(label): vocab size mismatch",
                       file: file, line: line)
        guard got.count == want.count, !got.isEmpty else { return }

        var gotMax = 0, wantMax = 0
        for i in 1 ..< got.count {
            if got[i] > got[gotMax] { gotMax = i }
            if want[i] > want[wantMax] { wantMax = i }
        }
        XCTAssertEqual(gotMax, wantMax,
            "\(label): native argmax \(gotMax) != reference argmax \(wantMax)",
            file: file, line: line)

        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(want[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999,
            "\(label): cosine \(cosine) too low", file: file, line: line)
        XCTAssertLessThan(maxAbs, maxAbsTolerance,
            "\(label): max abs logit diff \(maxAbs) too large", file: file, line: line)
    }

    func testFixtureExercisesTheRiskySurfaces() throws {
        let (_, ref) = try fixture()
        XCTAssertTrue(ref.layer_rope_theta.contains(0),
            "fixture must contain a NoPE layer or it does not gate the NoPE split")
        XCTAssertTrue(ref.layer_types.contains("sliding_attention"))
        XCTAssertTrue(ref.layer_types.contains("full_attention"))
        // Synthetic fixtures deliberately use a window shorter than the prompt so
        // the window actually bites. A REAL checkpoint has window 2048 against a
        // short prompt, where not biting is the correct behaviour — so only
        // assert this when the fixture is small enough to be synthetic.
        if ref.sliding_window < 64 {
            XCTAssertLessThan(ref.sliding_window, ref.tokens.count,
                "synthetic fixture must be longer than the window")
        }
    }

    func testNativeMuseGlimmerMatchesReferenceLogits() throws {
        let (dir, ref) = try fixture()
        let loaded = try loadModel(from: dir)

        XCTAssertEqual(loaded.family, "muse_glimmer",
            "a muse_glimmer checkpoint must route to the muse_glimmer rule, "
            + "not a dense fallback")
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)
        XCTAssertEqual(loaded.numLayers, ref.num_hidden_layers)

        // Cache kinds must follow layer_types, not be uniform.
        let spec = try XCTUnwrap(loaded.cacheSpec)
        XCTAssertEqual(spec.count, ref.layer_types.count,
            "reference layer_types must describe every layer")
        for (i, kind) in spec.enumerated() where i < ref.layer_types.count {
            if ref.layer_types[i] == "sliding_attention" {
                XCTAssertEqual(kind, .rotating(window: ref.sliding_window),
                    "layer \(i) is sliding and needs a windowed cache")
            } else {
                XCTAssertEqual(kind, .standard, "layer \(i) is full-attention")
            }
        }

        let tokens = MLXArray(ref.tokens.map { Int32($0) })
            .reshaped([1, ref.tokens.count])
        let logits = loaded.forward(tokens, nil)          // [1, L, V]
        let last = logits[0, ref.tokens.count - 1, 0...]  // [V]
        eval(last)
        let got = last.asArray(Float.self)

        XCTAssertLessThan(got.map { abs($0) }.max()!, ref.softcap,
            "logits must be tanh-softcapped to +/- \(ref.softcap)")
        assertMatches(got, ref.last_token_logits, label: "prefill",
                      maxAbsTolerance: ref.max_abs_tolerance ?? 2e-3)
    }

    func testNativeMuseGlimmerCachedDecodeMatchesReference() throws {
        let (dir, ref) = try fixture()
        let loaded = try loadModel(from: dir)

        let caches = makeKVCaches(spec: loaded.cacheSpec, numLayers: loaded.numLayers)
        let prefixIds = ref.tokens.dropLast().map { Int32($0) }
        let prefix = MLXArray(prefixIds).reshaped([1, prefixIds.count])
        eval(loaded.forward(prefix, caches))

        let stepId = MLXArray([Int32(ref.tokens[ref.tokens.count - 1])]).reshaped([1, 1])
        let stepLogits = loaded.forward(stepId, caches)   // [1, 1, V]
        let step = stepLogits[0, 0, 0...]
        eval(step)
        assertMatches(step.asArray(Float.self), ref.step_logits, label: "cached decode",
                      maxAbsTolerance: ref.max_abs_tolerance ?? 2e-3)
    }

    /// The perception encoder, gated independently of the text stack. This is
    /// where the host-side geometry lives — the window permutation and its
    /// un-permute, the interleaved `[w, h, w, h]` 2-axis RoPE with 1-based
    /// positions, and the channel-major `pixel_shuffle` — none of which
    /// produce a shape error when they are subtly wrong.
    func testVisionTowerMatchesReference() throws {
        let (dir, ref) = try fixture()
        guard let vg = ref.vision_grid, let vpd = ref.vision_patch_dim,
              let vpv = ref.vision_pixel_values, let vto = ref.vision_tower_out,
              let vts = ref.vision_tower_shape else {
            throw XCTSkip("fixture carries no vision reference (real-checkpoint mode)")
        }
        let loaded = try loadModel(from: dir)
        let model = try XCTUnwrap(
            loaded.module as? MuseGlimmerForConditionalGeneration)
        let tower = try XCTUnwrap(model.model.visionTower)

        let grid = (t: vg[0], h: vg[1], w: vg[2])
        let n = grid.t * grid.h * grid.w
        let pixels = MLXArray(vpv, [n, vpd])

        let out = tower(pixels, grids: [grid])
        eval(out)
        XCTAssertEqual(out.shape, vts)
        assertMatches(out.asArray(Float.self), vto,
                      label: "vision tower", maxAbsTolerance: 2e-3)
    }

    /// The full `get_image_features` chain: tower -> adapter -> projection ->
    /// weightless `perception_emb_norm`, landing in the text hidden size.
    func testImageFeaturesMatchReference() throws {
        let (dir, ref) = try fixture()
        guard let vg = ref.vision_grid, let vpd = ref.vision_patch_dim,
              let vpv = ref.vision_pixel_values, let vif = ref.vision_image_features,
              let vifs = ref.vision_image_features_shape else {
            throw XCTSkip("fixture carries no vision reference (real-checkpoint mode)")
        }
        let loaded = try loadModel(from: dir)
        let model = try XCTUnwrap(
            loaded.module as? MuseGlimmerForConditionalGeneration)

        let grid = (t: vg[0], h: vg[1], w: vg[2])
        let n = grid.t * grid.h * grid.w
        let pixels = MLXArray(vpv, [n, vpd])

        let feats = try XCTUnwrap(
            model.model.imageFeatures(pixelValues: pixels, grids: [grid]))
        eval(feats)
        XCTAssertEqual(feats.shape, vifs)
        assertMatches(feats.asArray(Float.self), vif,
                      label: "image features", maxAbsTolerance: 2e-3)
    }

    /// The prefill-specialized closure must agree with slicing the full forward.
    func testPrefillForwardMatchesFullForward() throws {
        let (dir, ref) = try fixture()
        let loaded = try loadModel(from: dir)
        let prefill = try XCTUnwrap(loaded.prefillForward)

        let tokens = MLXArray(ref.tokens.map { Int32($0) })
            .reshaped([1, ref.tokens.count])
        let sliced = loaded.forward(tokens, nil)[0, ref.tokens.count - 1, 0...]
        let direct = prefill(tokens, nil)[0, 0, 0...]
        eval(sliced, direct)
        assertMatches(direct.asArray(Float.self), sliced.asArray(Float.self),
                      label: "prefillForward", maxAbsTolerance: 1e-4)
    }
}
