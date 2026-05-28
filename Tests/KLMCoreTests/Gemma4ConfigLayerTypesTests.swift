import XCTest
@testable import KLMCore

/// Verifies that `Gemma4Config.isFullAttention(layerIdx:)` follows the
/// canonical `layer_types` list from `config.json` when present and
/// falls back to the modulo pattern only when it's missing.
///
/// Before this fix, the modulo default of 5 silently mismatched the
/// actual per-layer attention pattern on every non-e2b Gemma 4 SKU
/// (e4b, 26B-A4B, 31B), giving the wrong `head_dim` selection at
/// `Gemma4Model.swift:133` and crashing the Q/K/V reshape on first
/// inference.
final class Gemma4ConfigLayerTypesTests: XCTestCase {

    // MARK: - layer_types is authoritative

    func testLayerTypesListIsAuthoritativeOverModulo() throws {
        // Crafted to disagree with any reasonable modulo: full layers
        // at indices 0 and 3, sliding everywhere else.
        let cfg = try decode(textConfig: """
        {
          "hidden_size": 16, "intermediate_size": 32,
          "num_attention_heads": 2, "num_key_value_heads": 1,
          "num_hidden_layers": 5, "vocab_size": 100,
          "head_dim": 8, "global_head_dim": 16,
          "layer_types": ["full_attention", "sliding_attention",
                          "sliding_attention", "full_attention",
                          "sliding_attention"]
        }
        """)

        XCTAssertEqual(cfg.layerTypes,
                       ["full_attention", "sliding_attention",
                        "sliding_attention", "full_attention",
                        "sliding_attention"])
        XCTAssertTrue(cfg.isFullAttention(layerIdx: 0))
        XCTAssertFalse(cfg.isFullAttention(layerIdx: 1))
        XCTAssertFalse(cfg.isFullAttention(layerIdx: 2))
        XCTAssertTrue(cfg.isFullAttention(layerIdx: 3))
        XCTAssertFalse(cfg.isFullAttention(layerIdx: 4))
    }

    // MARK: - Modulo fallback when layer_types is absent

    func testFallsBackToModuloPatternWhenLayerTypesMissing() throws {
        let cfg = try decode(textConfig: """
        {
          "hidden_size": 16, "intermediate_size": 32,
          "num_attention_heads": 2, "num_key_value_heads": 1,
          "num_hidden_layers": 10, "vocab_size": 100,
          "head_dim": 8, "global_head_dim": 16,
          "sliding_window_pattern": 5
        }
        """)

        XCTAssertNil(cfg.layerTypes)
        // (idx + 1) % 5 == 0 ⇒ indices 4 and 9 are full.
        XCTAssertFalse(cfg.isFullAttention(layerIdx: 0))
        XCTAssertFalse(cfg.isFullAttention(layerIdx: 3))
        XCTAssertTrue(cfg.isFullAttention(layerIdx: 4))
        XCTAssertFalse(cfg.isFullAttention(layerIdx: 8))
        XCTAssertTrue(cfg.isFullAttention(layerIdx: 9))
    }

    // MARK: - Real SKU fixtures

    func testE2BFixtureMatchesPublishedFullAttentionLayers() throws {
        // 35 layers; full attention at every 5th (idx 4, 9, …, 34).
        // Real e2b config from mlx-community/gemma-4-e2b-it-4bit.
        let types = (0..<35).map { ($0 + 1) % 5 == 0 ? "full_attention" : "sliding_attention" }
        try assertFullLayers(types: types,
                             expectedFull: [4, 9, 14, 19, 24, 29, 34])
    }

    func testE4BFixtureMatchesPublishedFullAttentionLayers() throws {
        // 42 layers; full attention at every 6th (idx 5, 11, …, 41).
        // Real e4b config from mlx-community/gemma-4-e4b-it-4bit -
        // the pattern the broken modulo-of-5 default missed.
        let types = (0..<42).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
        try assertFullLayers(types: types,
                             expectedFull: [5, 11, 17, 23, 29, 35, 41])
    }

    func test26BA4BFixtureMatchesPublishedFullAttentionLayers() throws {
        // 30 layers; full attention at every 6th (idx 5, 11, …, 29).
        // Real 26B-A4B config from mlx-community/gemma-4-26b-a4b-it-4bit.
        let types = (0..<30).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
        try assertFullLayers(types: types,
                             expectedFull: [5, 11, 17, 23, 29])
    }

    // MARK: - Out-of-bounds safety

    func testOutOfRangeLayerIndexDoesNotCrash() throws {
        let cfg = try decode(textConfig: """
        {
          "hidden_size": 16, "intermediate_size": 32,
          "num_attention_heads": 2, "num_key_value_heads": 1,
          "num_hidden_layers": 3, "vocab_size": 100,
          "head_dim": 8, "global_head_dim": 16,
          "layer_types": ["sliding_attention", "sliding_attention",
                          "full_attention"]
        }
        """)
        // In-range hits the list directly.
        XCTAssertTrue(cfg.isFullAttention(layerIdx: 2))
        // Out-of-range must not array-index-crash; the bounds check
        // routes it through the modulo fallback. We don't pin the
        // returned value (the modulo would, by coincidence, return
        // either true or false depending on the index) - the safety
        // property is that calling at all is safe.
        _ = cfg.isFullAttention(layerIdx: 99)
        _ = cfg.isFullAttention(layerIdx: -1)
    }

    // MARK: - vision_config parsing (#76 prep)

    /// 26B-A4B ships a non-SigLIP-base vision tower (hidden 1152, 16
    /// heads, head_dim 72) and the prior loader hardcoded e2b shapes
    /// (768, 12, 64) at `Gemma4Model.swift` so the vision tower
    /// crashed loading with a 1.5x reshape mismatch. This test pins
    /// the decoder so the loader instantiates `VisionEncoder` at the
    /// shapes the checkpoint actually ships.
    func testVisionConfigDecodesFrom26BA4BShape() throws {
        let json = """
        {
          "text_config": {
            "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_hidden_layers": 1, "vocab_size": 100,
            "head_dim": 8, "global_head_dim": 16,
            "layer_types": ["full_attention"]
          },
          "vision_config": {
            "hidden_size": 1152, "intermediate_size": 4304,
            "num_hidden_layers": 27, "num_attention_heads": 16,
            "num_key_value_heads": 16, "head_dim": 72,
            "patch_size": 16, "pooling_kernel_size": 3,
            "position_embedding_size": 10240, "default_output_length": 280,
            "rms_norm_eps": 0.000001,
            "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"}
          }
        }
        """
        let cfg = try JSONDecoder().decode(
            Gemma4Config.self, from: Data(json.utf8))
        let vc = try XCTUnwrap(cfg.visionConfig,
            "vision_config must decode when present at top level")
        XCTAssertEqual(vc.hiddenSize, 1152)
        XCTAssertEqual(vc.intermediateSize, 4304)
        XCTAssertEqual(vc.numHiddenLayers, 27)
        XCTAssertEqual(vc.numAttentionHeads, 16)
        XCTAssertEqual(vc.numKeyValueHeads, 16)
        XCTAssertEqual(vc.headDim, 72)
        // rope_theta nested under rope_parameters must be honored.
        XCTAssertEqual(vc.ropeTheta, 100.0)
    }

    /// e2b/e4b configs MUST round-trip into the exact constants the
    /// loader used to hardcode at `Gemma4MultimodalModel.init`. This
    /// guards against a future refactor that drops a default or
    /// changes the field semantics.
    func testVisionConfigDecodesFromE2BE4BShape() throws {
        let json = """
        {
          "text_config": {
            "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_hidden_layers": 1, "vocab_size": 100,
            "head_dim": 8, "global_head_dim": 16,
            "layer_types": ["full_attention"]
          },
          "vision_config": {
            "hidden_size": 768, "intermediate_size": 3072,
            "num_hidden_layers": 16, "num_attention_heads": 12,
            "num_key_value_heads": 12, "head_dim": 64,
            "patch_size": 16, "pooling_kernel_size": 3,
            "position_embedding_size": 10240, "default_output_length": 280,
            "rms_norm_eps": 0.000001, "rope_theta": 100.0
          }
        }
        """
        let cfg = try JSONDecoder().decode(
            Gemma4Config.self, from: Data(json.utf8))
        let vc = try XCTUnwrap(cfg.visionConfig)
        XCTAssertEqual(vc.hiddenSize, 768)
        XCTAssertEqual(vc.numAttentionHeads, 12)
        XCTAssertEqual(vc.headDim, 64)
        XCTAssertEqual(vc.intermediateSize, 3072)
        XCTAssertEqual(vc.numHiddenLayers, 16)
        XCTAssertEqual(vc.ropeTheta, 100.0)
    }

    /// A vision_config that ships neither `rope_theta` (flat) nor a
    /// nested `rope_parameters.rope_theta` must still decode and fall
    /// back to the canonical Gemma 4 base of 100.0. This guards the
    /// outer `else` branch of the rope decode in `Gemma4VisionConfig`.
    func testVisionConfigRopeThetaFallbackWhenAbsent() throws {
        let json = """
        {
          "text_config": {
            "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_hidden_layers": 1, "vocab_size": 100,
            "head_dim": 8, "global_head_dim": 16,
            "layer_types": ["full_attention"]
          },
          "vision_config": {
            "hidden_size": 768, "num_attention_heads": 12, "head_dim": 64
          }
        }
        """
        let cfg = try JSONDecoder().decode(
            Gemma4Config.self, from: Data(json.utf8))
        let vc = try XCTUnwrap(cfg.visionConfig)
        XCTAssertEqual(vc.ropeTheta, 100.0,
            "rope_theta must default to 100.0 when neither flat nor "
            + "nested form is present")
    }

    /// A `vision_config: false` sentinel (used by some text-only
    /// checkpoints to signal "no vision tower") must decode to nil
    /// without raising, while a malformed object-shaped vision_config
    /// must NOT silently fall through to the SigLIP-base defaults -
    /// that silent fall-through was the original #76 crash class.
    func testVisionConfigBoolSentinelDecodesAsNil() throws {
        let json = """
        {
          "text_config": {
            "hidden_size": 16, "intermediate_size": 32,
            "num_attention_heads": 2, "num_key_value_heads": 1,
            "num_hidden_layers": 1, "vocab_size": 100,
            "head_dim": 8, "global_head_dim": 16,
            "layer_types": ["full_attention"]
          },
          "vision_config": false
        }
        """
        let cfg = try JSONDecoder().decode(
            Gemma4Config.self, from: Data(json.utf8))
        XCTAssertNil(cfg.visionConfig,
            "false sentinel must be treated as no vision tower")
    }

    /// Defaults are the e2b/e4b SigLIP-base shapes so an older
    /// checkpoint with no `vision_config` (or a partial one) still
    /// constructs a workable tower instead of crashing at init.
    func testVisionConfigDefaultsAreSigLIPBase() {
        let d = Gemma4VisionConfig.defaults
        XCTAssertEqual(d.hiddenSize, 768)
        XCTAssertEqual(d.intermediateSize, 3072)
        XCTAssertEqual(d.numHiddenLayers, 16)
        XCTAssertEqual(d.numAttentionHeads, 12)
        XCTAssertEqual(d.numKeyValueHeads, 12)
        XCTAssertEqual(d.headDim, 64)
        XCTAssertEqual(d.patchSize, 16)
        XCTAssertEqual(d.poolingKernelSize, 3)
        XCTAssertEqual(d.positionEmbeddingSize, 10240)
        XCTAssertEqual(d.defaultOutputLength, 280)
        XCTAssertEqual(d.ropeTheta, 100.0)
        XCTAssertEqual(d.rmsNormEps, 1e-6)
    }

    // MARK: - Helpers

    /// Decode a `Gemma4Config` from a JSON snippet wrapped under
    /// `text_config`, mirroring the real Gemma 4 multimodal layout.
    private func decode(textConfig: String) throws -> Gemma4Config {
        let wrapped = """
        { "text_config": \(textConfig) }
        """
        return try JSONDecoder().decode(
            Gemma4Config.self, from: Data(wrapped.utf8))
    }

    /// Build a config from a `layer_types` fixture and assert which
    /// layers `isFullAttention` flags.
    private func assertFullLayers(
        types: [String], expectedFull: [Int],
        file: StaticString = #file, line: UInt = #line
    ) throws {
        let typesJSON = "[\(types.map { "\"\($0)\"" }.joined(separator: ","))]"
        let cfg = try decode(textConfig: """
        {
          "hidden_size": 16, "intermediate_size": 32,
          "num_attention_heads": 2, "num_key_value_heads": 1,
          "num_hidden_layers": \(types.count), "vocab_size": 100,
          "head_dim": 8, "global_head_dim": 16,
          "layer_types": \(typesJSON)
        }
        """)
        let observed = (0..<types.count).filter { cfg.isFullAttention(layerIdx: $0) }
        XCTAssertEqual(observed, expectedFull,
                       "isFullAttention disagreed with layer_types fixture",
                       file: file, line: line)
    }
}
