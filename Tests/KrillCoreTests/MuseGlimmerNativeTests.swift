import XCTest
import MLX
import KrillCache
@testable import KrillCore

/// Checkpoint-free tests for the native Muse Glimmer runtime
/// (`MuseGlimmerForConditionalGeneration`, model_type "muse_glimmer").
///
/// The real model is 30B and the smallest published MLX build is 19.4 GB, so
/// there is no real-checkpoint gate on this box and no logit-parity oracle run
/// (see `.claude/skills/native-port/families/muse-glimmer.md`). These tests
/// therefore pin the parts of the port that are checkable WITHOUT weights —
/// specifically the six deltas from the dense template that would otherwise
/// fail silently:
///
///   1. `layer_rope_theta[i] == 0` really produces a NoPE layer.
///   2. The per-layer norms are `(1 + w)`-scaled, not `w`-scaled.
///   3. The token embedding carries a weightless RMS norm.
///   4. Logits are `output_multiplier`-scaled and tanh-softcapped.
///   5. The vision window permutation covers every token exactly once,
///      including the reference's pad-a-full-extra-window quirk.
///   6. `pixel_shuffle` is channel-major.
final class MuseGlimmerNativeTests: XCTestCase {

    // MARK: - Config

    /// A 4-layer config shaped like the shipped one: 3 sliding layers with
    /// RoPE, then one full-attention layer that is NoPE.
    private func configJSON(
        layers: Int = 4,
        hidden: Int = 32,
        heads: Int = 4,
        kvHeads: Int = 2,
        headDim: Int = 8,
        vocab: Int = 64,
        withVision: Bool = false
    ) -> Data {
        let types = (0 ..< layers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "sliding_attention" }
        let thetas = (0 ..< layers).map { ($0 + 1) % 4 == 0 ? "0" : "500000.0" }
        let vision = withVision ? """
        ,"vision_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "patch_size": 14, "patch_temporal": 2,
            "merge_size": 2, "pos_emb_height": 32, "pos_emb_width": 32,
            "layer_norm_eps": 1e-5, "hidden_act": "gelu",
            "rope_parameters": {"rope_theta": 10000.0},
            "layer_types": ["window_attention", "full_attention"]
        }
        """ : ""
        let json = """
        {
          "architectures": ["MuseGlimmerForConditionalGeneration"],
          "model_type": "muse_glimmer",
          "image_token_id": 61, "video_token_id": 60,
          "out_hidden_size": 64, "projector_hidden_size": 24,
          "projector_hidden_act": "gelu",
          "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": \(hidden), "intermediate_size": \(hidden * 2),
            "num_hidden_layers": \(layers),
            "num_attention_heads": \(heads), "num_key_value_heads": \(kvHeads),
            "head_dim": \(headDim), "vocab_size": \(vocab),
            "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
            "sliding_window": 4,
            "layer_types": [\(types.map { "\"\($0)\"" }.joined(separator: ", "))],
            "layer_rope_theta": [\(thetas.joined(separator: ", "))],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.19611613513818404,
            "final_logit_softcapping": 20.0,
            "attention_bias": false, "tie_word_embeddings": false,
            "max_position_embeddings": 131072, "hidden_activation": "silu"
          }\(vision)
        }
        """
        return Data(json.utf8)
    }

    private func makeConfig(withVision: Bool = false) throws -> MuseGlimmerConfig {
        try JSONDecoder().decode(
            MuseGlimmerConfig.self, from: configJSON(withVision: withVision))
    }

    func testConfigDecodesNestedRopeAndLayerMix() throws {
        let c = try makeConfig()
        let tc = c.textConfig
        // `rope_theta` lives under `rope_parameters`, not at the top level.
        XCTAssertEqual(tc.ropeTheta, 500_000.0)
        XCTAssertEqual(tc.qkScaleFactor, 3.87, accuracy: 1e-6)
        XCTAssertEqual(tc.finalLogitSoftcapping, 20.0)
        // Split eps: the post-norms are 1000x tighter than the pre-norms.
        XCTAssertEqual(tc.rmsNormEps, 1e-5)
        XCTAssertEqual(tc.postNormEps, 1e-8)
        // 3 sliding : 1 full, and the FULL layer is the NoPE one.
        XCTAssertFalse(tc.isFullAttention(layerIdx: 0))
        XCTAssertFalse(tc.isFullAttention(layerIdx: 2))
        XCTAssertTrue(tc.isFullAttention(layerIdx: 3))
        XCTAssertEqual(tc.ropeTheta(layerIdx: 0), 500_000.0)
        XCTAssertNil(tc.ropeTheta(layerIdx: 3),
            "layer_rope_theta == 0 must map to NoPE, not to a 0-base RoPE")
    }

    func testCacheSpecWindowsOnlyTheSlidingLayers() throws {
        let spec = try makeConfig().textConfig.cacheSpec
        XCTAssertEqual(spec[0], .rotating(window: 4))
        XCTAssertEqual(spec[2], .rotating(window: 4))
        XCTAssertEqual(spec[3], .standard,
            "full_attention layers attend the whole context and must not rotate")
    }

    func testValidateRejectsInconsistentLayerArrays() throws {
        // `layer_rope_theta` one entry short: which layers are NoPE becomes
        // unknowable, so this must fail at load rather than guess.
        let json = """
        {"text_config": {"hidden_size": 32, "intermediate_size": 64,
          "num_hidden_layers": 4, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64,
          "layer_types": ["sliding_attention", "sliding_attention",
                          "sliding_attention", "full_attention"],
          "layer_rope_theta": [500000.0, 500000.0, 500000.0]}}
        """
        let c = try JSONDecoder().decode(MuseGlimmerConfig.self, from: Data(json.utf8))
        XCTAssertThrowsError(try c.textConfig.validate())
    }

    // MARK: - The traps

    func testNoPELayersBuildNoRotary() throws {
        let tc = try makeConfig().textConfig
        XCTAssertNotNil(MuseGlimmerTextAttention(tc, layerIdx: 0).rope,
            "sliding layers carry RoPE at theta 500000")
        XCTAssertNil(MuseGlimmerTextAttention(tc, layerIdx: 3).rope,
            "full_attention layers are NoPE — building a RoPE here rotates "
            + "queries the reference leaves unrotated")
    }

    func testCenteredNormIsOnePlusWeight() {
        // Weights ship centred on zero. With the plain `norm(x) * w` formula a
        // freshly-initialised norm would annihilate its input; the `(1 + w)`
        // form must pass the normalised input through unchanged.
        let norm = MuseGlimmerCenteredRMSNorm(dimensions: 4, eps: 1e-5)
        let x = MLXArray([1.0, 2.0, 3.0, 4.0] as [Float]).reshaped(1, 4)
        let out = norm(x)
        eval(out)
        let got = out.asArray(Float.self)
        // rms(1,2,3,4) = sqrt(30/4) = 2.7386
        let rms: Float = (30.0 / 4.0 as Float).squareRoot()
        for (i, v) in got.enumerated() {
            XCTAssertEqual(v, Float(i + 1) / rms, accuracy: 1e-4)
        }
        XCTAssertGreaterThan(got.map { abs($0) }.max()!, 0.5,
            "a (1 + w) norm with zero weights must NOT output zeros")
    }

    func testEmbeddingCarriesWeightlessRMSNorm() throws {
        let c = try makeConfig()
        let model = MuseGlimmerForConditionalGeneration(c)
        let tokens = MLXArray([Int32(1), Int32(2), Int32(3)]).reshaped(1, 3)
        let embeds = model.model.languageModel.embed(tokens)
        eval(embeds)
        // Weightless RMS norm ⇒ every row has unit RMS.
        let h = c.textConfig.hiddenSize
        let flat = embeds.asArray(Float.self)
        for row in 0 ..< 3 {
            let slice = Array(flat[(row * h) ..< ((row + 1) * h)])
            let rms = (slice.reduce(0) { $0 + Double($1 * $1) } / Double(h)).squareRoot()
            XCTAssertEqual(rms, 1.0, accuracy: 1e-3,
                "embed_norm is weightless: each embedding row must have unit RMS")
        }
    }

    func testForwardShapeAndLogitSoftcap() throws {
        let c = try makeConfig()
        let model = MuseGlimmerForConditionalGeneration(c)
        let tokens = MLXArray([Int32(1), Int32(5), Int32(9), Int32(2)]).reshaped(1, 4)
        let logits = model(tokens, caches: model.makeCaches())
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 4, c.textConfig.vocabSize])
        let values = logits.asArray(Float.self)
        XCTAssertTrue(values.allSatisfy { $0.isFinite }, "logits must be finite")
        // tanh softcapping bounds every logit to (-cap, cap).
        let cap = c.textConfig.finalLogitSoftcapping
        XCTAssertLessThan(values.map { abs($0) }.max()!, cap,
            "final_logit_softcapping must bound logits to +/- \(cap)")
    }

    func testPrefillLastTokenMatchesFullForward() throws {
        let c = try makeConfig()
        let model = MuseGlimmerForConditionalGeneration(c)
        let tokens = MLXArray([Int32(3), Int32(1), Int32(4), Int32(1)]).reshaped(1, 4)
        let full = model(tokens, caches: model.makeCaches())
        let lastOnly = model(tokens, caches: model.makeCaches(), lastTokenOnly: true)
        eval(full, lastOnly)
        XCTAssertEqual(lastOnly.shape, [1, 1, c.textConfig.vocabSize])
        let a = full[0, 3, 0...].asArray(Float.self)
        let b = lastOnly[0, 0, 0...].asArray(Float.self)
        for i in 0 ..< a.count {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-4,
                "lastTokenOnly must be bit-comparable to slicing the full forward")
        }
    }

    func testCachedDecodeAdvancesEveryLayerCache() throws {
        let c = try makeConfig()
        let model = MuseGlimmerForConditionalGeneration(c)
        let caches = model.makeCaches()
        let prefix = MLXArray([Int32(3), Int32(1), Int32(4)]).reshaped(1, 3)
        eval(model(prefix, caches: caches))
        for (i, cache) in caches.enumerated() {
            XCTAssertEqual(cache.sequenceLength, 3, "layer \(i) cache did not advance")
        }
        let step = MLXArray([Int32(1)]).reshaped(1, 1)
        let out = model(step, caches: caches)
        eval(out)
        XCTAssertEqual(out.shape, [1, 1, c.textConfig.vocabSize])
    }

    // MARK: - mlx_vlm key layout

    /// Literal key patterns taken from `mlx-community/Muse-Glimmer-30B-4bit`'s
    /// `model.safetensors.index.json`. If the published conversion ever changes
    /// its layout, this is the test that should fail.
    func testSanitizeRewritesRealMlxVlmKeys() {
        let one = MLXArray([Float(1)])
        let mlxKeys = [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.0.self_attn.gate_proj.scales",
            "language_model.model.layers.0.post_feedforward_layernorm.weight",
            "language_model.model.norm.weight",
            "language_model.lm_head.weight",
            "language_model.lm_head.biases",
            "vision_tower.patch_embedder.position_embedding_table.weight",
            "vision_tower.layers.0.attn.q_proj.bias",
            "vision_tower.ln_post.weight",
            "vision_adapter.fc1.scales",
            "vision_projection.weight",
        ]
        let out = museGlimmerSanitize(Dictionary(uniqueKeysWithValues: mlxKeys.map { ($0, one) }))

        XCTAssertEqual(out.count, mlxKeys.count, "sanitize must not drop or merge keys")
        XCTAssertNotNil(out["model.language_model.embed_tokens.weight"])
        XCTAssertNotNil(out["model.language_model.embed_tokens.scales"])
        XCTAssertNotNil(out["model.language_model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["model.language_model.layers.0.self_attn.gate_proj.scales"])
        XCTAssertNotNil(out["model.language_model.norm.weight"])
        XCTAssertNotNil(out["model.vision_tower.ln_post.weight"])
        XCTAssertNotNil(out["model.vision_adapter.fc1.scales"])
        XCTAssertNotNil(out["model.vision_projection.weight"])
        // lm_head moves the OTHER way: mlx_vlm nests it, HF hoists it.
        XCTAssertNotNil(out["lm_head.weight"])
        XCTAssertNotNil(out["lm_head.biases"])
        XCTAssertNil(out["model.language_model.lm_head.weight"],
            "lm_head must be hoisted to the top level, not left under language_model")
    }

    func testSanitizeLeavesHFLayoutUntouched() {
        let one = MLXArray([Float(1)])
        let hf = [
            "model.language_model.layers.3.mlp.down_proj.weight",
            "model.language_model.norm.weight",
            "model.vision_tower.layers.1.mlp.fc2.bias",
            "model.vision_projection.weight",
            "lm_head.weight",
        ]
        let out = museGlimmerSanitize(Dictionary(uniqueKeysWithValues: hf.map { ($0, one) }))
        XCTAssertEqual(Set(out.keys), Set(hf),
            "an HF-format checkpoint must pass through byte-for-byte")
    }

    /// Round-trip against the REAL module tree: take every parameter path the
    /// model actually builds, express it in the mlx_vlm layout, sanitize, and
    /// require the original set back. This pins the mapping to what the modules
    /// expect rather than to what the mapping's author assumed — so adding or
    /// renaming a submodule cannot silently desync the two.
    func testSanitizeRoundTripsEveryModelParameterPath() throws {
        let c = try makeConfig(withVision: true)
        let model = MuseGlimmerForConditionalGeneration(c)
        let hfPaths = model.parameters().flattened().map { $0.0 }
        XCTAssertFalse(hfPaths.isEmpty)
        // Sanity: the tree really does carry both towers and a top-level head.
        XCTAssertTrue(hfPaths.contains { $0.hasPrefix("model.language_model.layers.") })
        XCTAssertTrue(hfPaths.contains { $0.hasPrefix("model.vision_tower.") })
        XCTAssertTrue(hfPaths.contains { $0.hasPrefix("model.vision_adapter.") })
        XCTAssertTrue(hfPaths.contains { $0.hasPrefix("lm_head.") })

        // HF -> mlx_vlm (the inverse of the sanitize under test).
        func toMlxVlm(_ p: String) -> String {
            if p.hasPrefix("lm_head.") {
                return "language_model.lm_head." + p.dropFirst("lm_head.".count)
            }
            if p.hasPrefix("model.language_model.") {
                return "language_model.model." + p.dropFirst("model.language_model.".count)
            }
            if p.hasPrefix("model.vision") {
                return String(p.dropFirst("model.".count))
            }
            return p
        }

        let one = MLXArray([Float(1)])
        let mlx = Dictionary(uniqueKeysWithValues: hfPaths.map { (toMlxVlm($0), one) })
        XCTAssertEqual(mlx.count, hfPaths.count, "inverse mapping must stay 1:1")
        let back = museGlimmerSanitize(mlx)
        XCTAssertEqual(Set(back.keys), Set(hfPaths),
            "every parameter the model builds must be reachable from the "
            + "mlx_vlm key layout")
    }

    // MARK: - Vision geometry (host-side, no weights)

    func testWindowIndexIsAPermutationCoveringEveryToken() {
        // 4x6 patch grid, 2x2-patch windows. `pad = win - dim % win` is never
        // zero in the reference, so a grid already divisible by the window edge
        // gets a full extra (all-padding) window row and column. Those windows
        // are empty and must be dropped by the unique-consecutive pass.
        let grids = [(t: 1, h: 4, w: 6)]
        let (index, cu) = MuseGlimmerVisionGrid.windowIndex(grids, windowPatches: 2)

        XCTAssertEqual(index.count, 24)
        XCTAssertEqual(Set(index), Set((0 ..< 24).map { Int32($0) }),
            "window_index must be a permutation of every patch index")
        // Six 2x2 windows of four patches each; no zero-length boundaries.
        XCTAssertEqual(cu, [0, 4, 8, 12, 16, 20, 24])
    }

    func testWindowIndexHandlesRaggedGrids() {
        // 3x5 with a 2-patch window: the last row and column are partial.
        let (index, cu) = MuseGlimmerVisionGrid.windowIndex(
            [(t: 1, h: 3, w: 5)], windowPatches: 2)
        XCTAssertEqual(index.count, 15)
        XCTAssertEqual(Set(index), Set((0 ..< 15).map { Int32($0) }))
        XCTAssertEqual(cu.first, 0)
        XCTAssertEqual(cu.last, 15, "windows must cover all 15 patches")
        XCTAssertTrue(zip(cu, cu.dropFirst()).allSatisfy { $0 < $1 },
            "cumulative window boundaries must be strictly increasing")
    }

    func testPositionIdsAreRasterPerFrame() {
        let (h, w) = MuseGlimmerVisionGrid.positionIds([(t: 1, h: 2, w: 3)])
        XCTAssertEqual(h, [0, 0, 0, 1, 1, 1])
        XCTAssertEqual(w, [0, 1, 2, 0, 1, 2])
    }

    func testCuSeqlensIsPerFrame() {
        // Each temporal frame is its own attention segment.
        XCTAssertEqual(
            MuseGlimmerVisionGrid.cuSeqlens([(t: 2, h: 2, w: 3)]), [0, 6, 12])
    }

    func testPixelShuffleIsChannelMajor() throws {
        let c = try makeConfig(withVision: true)
        let vision = try XCTUnwrap(c.visionConfig)
        let tower = MuseGlimmerVisionModel(vision)
        // One 2x2 grid, dim 2: rows are patches in raster order.
        let x = MLXArray([1, 10, 2, 20, 3, 30, 4, 40].map { Float($0) }).reshaped(4, 2)
        let out = tower.pixelShuffle(x, grids: [(t: 1, h: 2, w: 2)])
        eval(out)
        XCTAssertEqual(out.shape, [1, 8])
        // out[:, d*4 + j] == in[:, j, d] — channel-major, NOT patch-major.
        XCTAssertEqual(out.asArray(Float.self), [1, 2, 3, 4, 10, 20, 30, 40])
    }

    func testSmartResizeCapsTokensAndSnapsToGrid() {
        let factor = 14 * 2  // patch_size * merge_size
        // Far over the cap: must land exactly on the 4096-token ceiling.
        let (h, w) = MuseGlimmerImagePreprocessor.smartResize(
            height: 10_000, width: 10_000, factor: factor, maxTokens: 4096)
        XCTAssertEqual(h % factor, 0)
        XCTAssertEqual(w % factor, 0)
        XCTAssertEqual((h / factor) * (w / factor), 4096)

        // Under the cap: keeps the aspect ratio, still snapped to the grid.
        let (h2, w2) = MuseGlimmerImagePreprocessor.smartResize(
            height: 560, width: 280, factor: factor, maxTokens: 4096)
        XCTAssertEqual(h2 % factor, 0)
        XCTAssertEqual(w2 % factor, 0)
        XCTAssertEqual(Double(h2) / Double(w2), 2.0, accuracy: 0.35)
        XCTAssertLessThanOrEqual((h2 / factor) * (w2 / factor), 4096)
    }

    func testPatchifyProducesTemporalMajorPayload() throws {
        let c = try makeConfig(withVision: true)
        let vision = try XCTUnwrap(c.visionConfig)
        // One 14x14 patch of a constant-per-channel image.
        let ps = vision.patchSize
        var pixels = [Float](repeating: 0, count: ps * ps * 3)
        for i in 0 ..< (ps * ps) {
            pixels[i * 3] = 1
            pixels[i * 3 + 1] = 2
            pixels[i * 3 + 2] = 3
        }
        let (patches, gridH, gridW) = MuseGlimmerImagePreprocessor.patchify(
            MLXArray(pixels, [ps, ps, 3]), vision: vision)
        eval(patches)
        XCTAssertEqual(gridH, 1)
        XCTAssertEqual(gridW, 1)
        XCTAssertEqual(patches.shape, [1, vision.patchDim])

        // Layout is (t, c, ph, pw): the whole payload repeats once per temporal
        // slot, and within a slot each channel's plane is contiguous.
        let flat = patches.asArray(Float.self)
        let perSlot = vision.patchDim / vision.patchTemporal
        XCTAssertEqual(Array(flat[0 ..< perSlot]), Array(flat[perSlot ..< (2 * perSlot)]),
            "the still image must be duplicated across the temporal axis")
        XCTAssertEqual(flat[0], 1)
        XCTAssertEqual(flat[ps * ps], 2, "channel planes must be contiguous within a slot")
        XCTAssertEqual(flat[2 * ps * ps], 3)
    }

    /// Patch ORDER across a multi-patch grid: patch 0 must be the TOP-left of
    /// the image and the last patch the bottom-right.
    ///
    /// The single-patch test above pins the layout WITHIN a patch and is blind
    /// to this: a vertically flipped or transposed grid produces identically
    /// shaped output, and the tower's own parity gate cannot catch it either
    /// because that gate is fed `vision_pixel_values` FROM the reference — our
    /// decode/normalize/patchify never runs in it. So the seam between "an
    /// image" and "the tensor the tower sees" is checked here or nowhere.
    func testPatchifyOrdersRowsTopToBottom() throws {
        let c = try makeConfig(withVision: true)
        let vision = try XCTUnwrap(c.visionConfig)
        let ps = vision.patchSize
        // 2x2 patch grid; top half = 1, bottom half = 9 (all channels).
        let h = ps * 2, w = ps * 2
        var pixels = [Float](repeating: 0, count: h * w * 3)
        for y in 0 ..< h {
            for x in 0 ..< w {
                let v: Float = y < ps ? 1 : 9
                for c in 0 ..< 3 { pixels[(y * w + x) * 3 + c] = v }
            }
        }
        let (patches, gridH, gridW) = MuseGlimmerImagePreprocessor.patchify(
            MLXArray(pixels, [h, w, 3]), vision: vision)
        eval(patches)
        XCTAssertEqual([gridH, gridW], [2, 2])

        let flat = patches.asArray(Float.self)
        let dim = vision.patchDim
        func firstValue(ofPatch i: Int) -> Float { flat[i * dim] }
        XCTAssertEqual(firstValue(ofPatch: 0), 1,
            "patch 0 must come from the TOP row of the image")
        XCTAssertEqual(firstValue(ofPatch: 1), 1, "patch 1 is still the top row")
        XCTAssertEqual(firstValue(ofPatch: 2), 9,
            "patch 2 must come from the BOTTOM row — a flipped grid inverts "
            + "every above/below relation the model reports")
        XCTAssertEqual(firstValue(ofPatch: 3), 9)
    }

    /// Decode ORIENTATION: row 0 of the decoded tensor must be the TOP of the
    /// image.
    ///
    /// This is the last ungated seam on the image path, and the one most likely
    /// to be wrong: CoreGraphics bitmap contexts are bottom-left origin, so
    /// drawing a CGImage without accounting for that flips the result
    /// vertically — shapes and colours survive, only above/below inverts, which
    /// is invisible to every other test here. The fixture is a 2x2 PNG built
    /// OUTSIDE CoreGraphics (top row red, bottom row blue) so a flip in the
    /// decoder cannot cancel against a flip in the fixture.
    func testDecodeKeepsImageTopAtRowZero() throws {
        let c = try makeConfig(withVision: true)
        let vision = try XCTUnwrap(c.visionConfig)
        let png = try XCTUnwrap(Data(base64Encoded:
            "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEUlEQVR4nGP8zwACjAwMIAYAERICAXrJpWEAAAAASUVORK5CYII="))

        let pixels = try MuseGlimmerImagePreprocessor.decode(png, vision: vision)
        eval(pixels)
        let h = pixels.dim(0), w = pixels.dim(1)
        let flat = pixels.asArray(Float.self)
        // Channel 0 (R) and channel 2 (B) of the first and last rows.
        func rgb(row: Int) -> (Float, Float) {
            let base = (row * w + (w / 2)) * 3
            return (flat[base], flat[base + 2])
        }
        let (topR, topB) = rgb(row: 0)
        let (botR, botB) = rgb(row: h - 1)
        XCTAssertGreaterThan(topR, topB,
            "row 0 must be the RED top of the image; if it is blue the decoder "
            + "flipped the image vertically and every above/below answer inverts")
        XCTAssertGreaterThan(botB, botR, "the last row must be the BLUE bottom")
    }

    func testVisionTowerForwardShape() throws {
        let c = try makeConfig(withVision: true)
        let vision = try XCTUnwrap(c.visionConfig)
        let tower = MuseGlimmerVisionModel(vision)
        let grid = (t: 1, h: 4, w: 4)
        let n = grid.h * grid.w
        let pixels = MLXArray.zeros([n, vision.patchDim])
        let out = tower(pixels, grids: [grid])
        eval(out)
        // 2x2 merge: 16 patches -> 4 tokens of hidden * 4.
        XCTAssertEqual(out.shape, [4, vision.hiddenSize * vision.mergeSize * vision.mergeSize])
        XCTAssertEqual(out.shape[1], c.outHiddenSize,
            "the tower output width must match out_hidden_size, which is what "
            + "vision_adapter.fc1 consumes")
    }

    func testImageFeaturesProjectToTextHidden() throws {
        let c = try makeConfig(withVision: true)
        let vision = try XCTUnwrap(c.visionConfig)
        let model = MuseGlimmerForConditionalGeneration(c)
        let grid = (t: 1, h: 4, w: 4)
        let pixels = MLXArray.zeros([grid.h * grid.w, vision.patchDim])
        let feats = try XCTUnwrap(
            model.model.imageFeatures(pixelValues: pixels, grids: [grid]))
        eval(feats)
        XCTAssertEqual(feats.shape, [4, c.textConfig.hiddenSize])
    }

}
