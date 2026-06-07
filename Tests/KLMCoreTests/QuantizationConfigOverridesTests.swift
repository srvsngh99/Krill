import XCTest
@testable import KLMCore

/// Covers the per-module override path on `QuantizationConfig`. The
/// motivating crash: `mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit`
/// declares 4-bit base + 8-bit MoE gate overrides; loading the gates
/// as 4-bit (uniform) hits a `quantized_matmul` scales-shape mismatch
/// at first inference. The fix carries the overrides through to MLX
/// at quantize time, so each layer instantiates at its own bits.
final class QuantizationConfigOverridesTests: XCTestCase {

    // MARK: - Decoder

    func testUniformQuantDecodesWithNoOverrides() throws {
        // The most common shape across mlx-community 4-bit checkpoints.
        let q = try decode("""
        { "group_size": 64, "bits": 4 }
        """)
        XCTAssertEqual(q.groupSize, 64)
        XCTAssertEqual(q.bits, 4)
        XCTAssertTrue(q.moduleOverrides.isEmpty)
    }

    func testMixedPrecisionCollectsPerModuleOverrides() throws {
        // Mirrors the Qwen3-Coder layout: scalar defaults plus nested
        // objects under per-layer module names.
        let q = try decode("""
        {
          "group_size": 64,
          "bits": 4,
          "model.layers.0.mlp.gate": { "group_size": 64, "bits": 8 },
          "model.layers.1.mlp.gate": { "group_size": 64, "bits": 8 },
          "model.layers.2.mlp.gate": { "group_size": 32, "bits": 6 }
        }
        """)
        XCTAssertEqual(q.groupSize, 64)
        XCTAssertEqual(q.bits, 4)
        XCTAssertEqual(q.moduleOverrides.count, 3)
        XCTAssertEqual(
            q.moduleOverrides["model.layers.0.mlp.gate"],
            .init(groupSize: 64, bits: 8))
        XCTAssertEqual(
            q.moduleOverrides["model.layers.2.mlp.gate"],
            .init(groupSize: 32, bits: 6))
    }

    func testDecoderIgnoresUnknownTopLevelStringFields() throws {
        // Some quants ship informational `mode` / `method` strings at the
        // top level. They are not module overrides; ignoring them keeps
        // the override count honest and prevents the decoder from
        // partially-decoding a string and aborting the whole config.
        let q = try decode("""
        {
          "group_size": 64,
          "bits": 4,
          "mode": "affine",
          "method": "absmax",
          "model.layers.0.mlp.gate": { "group_size": 64, "bits": 8 }
        }
        """)
        XCTAssertEqual(q.groupSize, 64)
        XCTAssertEqual(q.bits, 4)
        XCTAssertEqual(q.moduleOverrides.count, 1)
        XCTAssertNotNil(q.moduleOverrides["model.layers.0.mlp.gate"])
    }

    func testDecoderSkipsNonObjectExtraKeysForwardsCompat() throws {
        // A future field that is neither a scalar default nor a
        // `ModuleQuant` object (e.g. an array, a number, a string the
        // decoder does not yet know about) must not blow up an
        // otherwise-valid config.
        let q = try decode("""
        {
          "group_size": 64,
          "bits": 4,
          "scheme_version": 2,
          "calibration_set": ["wikitext", "c4"],
          "model.layers.0.mlp.gate": { "group_size": 64, "bits": 8 }
        }
        """)
        XCTAssertEqual(q.moduleOverrides.count, 1)
        XCTAssertNotNil(q.moduleOverrides["model.layers.0.mlp.gate"])
    }

    // MARK: - effective(for:)

    func testEffectiveReturnsOverrideWhenModulePathMatches() {
        let q = QuantizationConfig(
            groupSize: 64, bits: 4,
            moduleOverrides: [
                "model.layers.0.mlp.gate": .init(groupSize: 64, bits: 8),
                "model.layers.7.mlp.gate": .init(groupSize: 32, bits: 6),
            ])
        let r0 = q.effective(for: "model.layers.0.mlp.gate")
        XCTAssertEqual(r0.groupSize, 64)
        XCTAssertEqual(r0.bits, 8)
        let r7 = q.effective(for: "model.layers.7.mlp.gate")
        XCTAssertEqual(r7.groupSize, 32)
        XCTAssertEqual(r7.bits, 6)
    }

    func testEffectiveFallsBackToTopLevelWhenNoOverride() {
        // Any module path the override map does not name uses the
        // top-level defaults. This is the path every non-gate layer in
        // Qwen3-Coder takes (q_proj, k_proj, v_proj, o_proj, experts.*).
        let q = QuantizationConfig(
            groupSize: 64, bits: 4,
            moduleOverrides: [
                "model.layers.0.mlp.gate": .init(groupSize: 64, bits: 8),
            ])
        let r = q.effective(for: "model.layers.0.self_attn.q_proj")
        XCTAssertEqual(r.groupSize, 64)
        XCTAssertEqual(r.bits, 4)
    }

    func testEffectiveOnUniformConfigAlwaysReturnsTopLevel() {
        // The no-override case: every lookup returns the defaults.
        let q = QuantizationConfig(groupSize: 32, bits: 4)
        let r = q.effective(for: "anything.at.all")
        XCTAssertEqual(r.groupSize, 32)
        XCTAssertEqual(r.bits, 4)
    }

    // MARK: - Strict decoding

    func testDecoderThrowsWhenGroupSizeMissing() {
        // The mlx-community contract is that a `"quantization"` block
        // declares both scalars. A block missing `group_size` is
        // malformed; surface it instead of silently using a default
        // that may not match what the weight files were packed with.
        XCTAssertThrowsError(
            try decode("""
            { "bits": 4 }
            """)
        ) { err in
            guard case DecodingError.keyNotFound(let key, _) = err else {
                return XCTFail("expected keyNotFound, got \(err)")
            }
            XCTAssertEqual(key.stringValue, "group_size")
        }
    }

    func testDecoderThrowsWhenBitsMissing() {
        XCTAssertThrowsError(
            try decode("""
            { "group_size": 64 }
            """)
        ) { err in
            guard case DecodingError.keyNotFound(let key, _) = err else {
                return XCTFail("expected keyNotFound, got \(err)")
            }
            XCTAssertEqual(key.stringValue, "bits")
        }
    }

    func testDecoderThrowsOnEmptyObject() {
        // The empty-object footgun: prior to this guard the decoder
        // silently returned (groupSize: 64, bits: 4) for `{}`, which
        // could mask a malformed checkpoint. Make it loud.
        XCTAssertThrowsError(try decode("{}"))
    }

    // MARK: - Round-trip

    func testEncodeDecodeRoundTrips() throws {
        let original = QuantizationConfig(
            groupSize: 64, bits: 4,
            moduleOverrides: [
                "model.layers.0.mlp.gate": .init(groupSize: 64, bits: 8),
                "model.layers.1.mlp.gate": .init(groupSize: 64, bits: 8),
            ])
        let data = try JSONEncoder().encode(original)
        let round = try JSONDecoder().decode(QuantizationConfig.self, from: data)
        XCTAssertEqual(round.groupSize, original.groupSize)
        XCTAssertEqual(round.bits, original.bits)
        XCTAssertEqual(round.moduleOverrides, original.moduleOverrides)
    }

    // MARK: - Quantization mode (affine / nvfp4 / mxfp4)

    func testModeDefaultsToAffineWhenAbsent() throws {
        // Every historical mlx-community 4-bit checkpoint omits `mode`;
        // it must default to "affine" so behavior is unchanged.
        let q = try decode("""
        { "group_size": 64, "bits": 4 }
        """)
        XCTAssertEqual(q.mode, "affine")
        XCTAssertEqual(q.effective(for: "any.module").mode, "affine")
    }

    func testNvfp4ModeParsesAndPropagatesThroughEffective() throws {
        // nvfp4 (4-bit float) checkpoints carry `"mode": "nvfp4"` and use
        // group_size 16. The mode must reach `effective(for:)` so the
        // loader can select the right MLX QuantizationMode.
        let q = try decode("""
        { "group_size": 16, "bits": 4, "mode": "nvfp4" }
        """)
        XCTAssertEqual(q.mode, "nvfp4")
        let eff = q.effective(for: "model.layers.0.self_attn.q_proj")
        XCTAssertEqual(eff.groupSize, 16)
        XCTAssertEqual(eff.bits, 4)
        XCTAssertEqual(eff.mode, "nvfp4")
    }

    func testPerModuleModeOverridesTopLevel() throws {
        // A per-module override may carry its own mode; one without a
        // mode inherits the top-level mode.
        let q = try decode("""
        {
          "group_size": 16, "bits": 4, "mode": "nvfp4",
          "model.layers.0.mlp.down_proj": { "group_size": 32, "bits": 4, "mode": "mxfp4" },
          "model.layers.0.self_attn.q_proj": { "group_size": 16, "bits": 4 }
        }
        """)
        XCTAssertEqual(q.effective(for: "model.layers.0.mlp.down_proj").mode, "mxfp4")
        // override without `mode` inherits the top-level nvfp4
        XCTAssertEqual(q.effective(for: "model.layers.0.self_attn.q_proj").mode, "nvfp4")
        // unlisted module also inherits top-level
        XCTAssertEqual(q.effective(for: "model.layers.5.mlp.up_proj").mode, "nvfp4")
    }

    func testNonAffineModeRoundTrips() throws {
        let original = QuantizationConfig(groupSize: 16, bits: 4, mode: "nvfp4")
        let data = try JSONEncoder().encode(original)
        let round = try JSONDecoder().decode(QuantizationConfig.self, from: data)
        XCTAssertEqual(round.mode, "nvfp4")
        XCTAssertEqual(round.groupSize, 16)
        XCTAssertEqual(round.bits, 4)
    }

    func testAffineModeOmittedFromEncoding() throws {
        // Affine configs must encode byte-identically to the historical
        // (mode-less) form, so `mode` is only emitted for non-affine.
        let q = QuantizationConfig(groupSize: 64, bits: 4)
        let json = String(data: try JSONEncoder().encode(q), encoding: .utf8)!
        XCTAssertFalse(json.contains("mode"))
    }

    // MARK: - Helpers

    private func decode(_ json: String) throws -> QuantizationConfig {
        try JSONDecoder().decode(
            QuantizationConfig.self, from: Data(json.utf8))
    }
}
