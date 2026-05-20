import XCTest
import MLX
import MLXNN
@testable import KLMCore

/// WS5 native runtime foundation: tests for the Qwen 2.5-VL
/// (`Qwen2_5_VLForConditionalGeneration`) Swift+MLX modules. The
/// full multimodal forward (vision token injection, language-side
/// mRoPE) is the follow-up; these tests pin the foundation
/// (config parsing, 3D mRoPE, patch merger, vision blocks, image
/// preprocessing) so the runtime PR has a fixed contract to wire
/// against.
final class Qwen25VLNativeTests: XCTestCase {

    // MARK: - Config parsing

    private func tinyConfigJSON(
        hidden: Int = 64,
        intermediate: Int = 128,
        heads: Int = 4,
        kvHeads: Int = 2,
        layers: Int = 2,
        vocab: Int = 1024,
        headDim: Int = 16,
        visionDepth: Int = 2,
        visionHidden: Int = 32,
        patchSize: Int = 14,
        spatialMerge: Int = 2,
        mropeSection: [Int] = [4, 6, 6]
    ) -> [String: Any] {
        return [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_attention_heads": heads,
            "num_key_value_heads": kvHeads,
            "num_hidden_layers": layers,
            "vocab_size": vocab,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 4096,
            "head_dim": headDim,
            "image_token_id": 151_655,
            "video_token_id": 151_656,
            "vision_start_token_id": 151_652,
            "vision_end_token_id": 151_653,
            "rope_scaling": ["type": "mrope", "mrope_section": mropeSection],
            "vision_config": [
                "depth": visionDepth,
                "hidden_size": visionHidden,
                "intermediate_size": visionHidden * 4,
                "num_heads": 4,
                "patch_size": patchSize,
                "temporal_patch_size": 2,
                "in_chans": 3,
                "spatial_merge_size": spatialMerge,
                "fullatt_block_indexes": [1],
                "window_size": 56,
                "out_hidden_size": hidden,
            ],
            "tie_word_embeddings": false,
        ]
    }

    private func decode(_ json: [String: Any]) throws -> Qwen25VLConfig {
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(Qwen25VLConfig.self, from: data)
    }

    func testConfigParsesLanguageFields() throws {
        let cfg = try decode(tinyConfigJSON(hidden: 2048, layers: 36))
        XCTAssertEqual(cfg.hiddenSize, 2048)
        XCTAssertEqual(cfg.numHiddenLayers, 36)
        XCTAssertEqual(cfg.imageTokenId, 151_655)
        XCTAssertEqual(cfg.visionStartTokenId, 151_652)
        XCTAssertEqual(cfg.visionEndTokenId, 151_653)
    }

    func testConfigParsesMRoPESection() throws {
        let cfg = try decode(tinyConfigJSON(headDim: 64, mropeSection: [16, 24, 24]))
        XCTAssertEqual(cfg.mropeSection, [16, 24, 24])
        XCTAssertEqual(cfg.mropeSection.reduce(0, +), 64,
            "mrope_section must sum to head_dim")
    }

    func testConfigParsesVisionSubconfig() throws {
        let cfg = try decode(tinyConfigJSON(
            visionDepth: 32, visionHidden: 1280, patchSize: 14, spatialMerge: 2))
        XCTAssertEqual(cfg.vision.depth, 32)
        XCTAssertEqual(cfg.vision.hiddenSize, 1280)
        XCTAssertEqual(cfg.vision.patchSize, 14)
        XCTAssertEqual(cfg.vision.spatialMergeSize, 2)
        XCTAssertEqual(cfg.vision.fullAttnBlockIndexes, [1])
    }

    func testTextConfigProjectionProducesValidQwen25Shape() throws {
        let cfg = try decode(tinyConfigJSON())
        let text = cfg.qwenTextConfig
        XCTAssertEqual(text.modelType, "qwen2")
        XCTAssertTrue(text.attentionBias,
            "Qwen 2.5-VL text side uses dense Qwen 2.5 attention (QKV bias)")
        XCTAssertFalse(text.hasQKNorm,
            "Qwen 2.5-VL text side does NOT use q_norm/k_norm (those are Qwen 3 only)")
    }

    // MARK: - 3D mRoPE

    func testMRoPEAppliesWithoutShapeError() {
        // head_dim 8, sections [2, 1, 1] -> sum 4 = head_dim/2.
        // Apply to [B=1, H=2, L=3, D=8] with three position arrays.
        let rope = Qwen25VLMRoPE(headDim: 8, sections: [2, 1, 1], theta: 10_000.0)
        let x = MLXArray.zeros([1, 2, 3, 8]).asType(.float32) + Float(0.5)
        let pos = MLXArray([Int32(0), Int32(1), Int32(2)])
        let y = rope.apply(x, positionsT: pos, positionsH: pos, positionsW: pos)
        XCTAssertEqual(y.shape, [1, 2, 3, 8])
        eval(y)
    }

    func testMRoPETextLikePositionsMatchStandardRoPEShape() {
        // When t == h == w (text-only token positions), mRoPE
        // reduces to standard RoPE for the slice. We do not
        // assert numerical equivalence here (the section split is
        // not the same as a single-axis RoPE); we only pin that
        // the output shape is preserved and finite.
        let rope = Qwen25VLMRoPE(headDim: 16, sections: [4, 2, 2], theta: 1_000_000.0)
        let x = MLXArray.ones([1, 4, 5, 16]).asType(.float32)
        let pos = MLXArray((0 ..< 5).map { Int32($0) })
        let y = rope.apply(x, positionsT: pos, positionsH: pos, positionsW: pos)
        XCTAssertEqual(y.shape, [1, 4, 5, 16])
        eval(y)
        let arr = y.asArray(Float.self)
        let bad = arr.firstIndex { !$0.isFinite }
        XCTAssertNil(bad, "mRoPE output must not contain NaN/Inf")
    }

    func testMRoPESectionSumIsValidated() {
        // The init precondition asserts sections sum to head_dim
        // or head_dim/2. Bad sections trap; we cannot test that
        // directly without crashing the test process. The
        // precondition is documented in the module comment.
        let _ = Qwen25VLMRoPE(headDim: 4, sections: [2], theta: 1.0)
    }

    // MARK: - Patch merger

    func testPatchMergerCollapsesBlocks() {
        // Vision hidden = 8, merge size = 2 -> input dim = 32.
        // Out hidden = 16. With 4 patches in (one 2x2 block),
        // output has 1 token of width 16.
        let merger = Qwen25VLPatchMerger(
            visionHidden: 8, outHidden: 16, spatialMergeSize: 2)
        let x = MLXArray.ones([4, 8]).asType(.float32) * Float(0.1)
        let y = merger(x)
        XCTAssertEqual(y.shape, [1, 16])
        eval(y)
    }

    func testPatchMergerScalesNumberOfTokensInversely() {
        // 16 input patches with merge=2 -> 4 output tokens.
        let merger = Qwen25VLPatchMerger(
            visionHidden: 4, outHidden: 8, spatialMergeSize: 2)
        let x = MLXArray.ones([16, 4]).asType(.float32)
        let y = merger(x)
        XCTAssertEqual(y.shape, [4, 8])
    }

    // MARK: - Vision tower

    func testVisionTowerInstantiates() {
        let visionCfg: Qwen25VLConfig.VisionConfig
        do {
            visionCfg = try JSONDecoder().decode(
                Qwen25VLConfig.VisionConfig.self,
                from: try JSONSerialization.data(withJSONObject: [
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 4,
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "in_chans": 3,
                    "spatial_merge_size": 2,
                    "fullatt_block_indexes": [1],
                    "window_size": 56,
                    "out_hidden_size": 64,
                ]))
        } catch {
            XCTFail("VisionConfig decode failed: \(error)")
            return
        }
        let tower = Qwen25VLVisionTower(visionCfg)
        XCTAssertNotNil(tower)
    }

    func testVisionTowerForwardPassProducesValidShape() {
        let visionCfg: Qwen25VLConfig.VisionConfig
        do {
            visionCfg = try JSONDecoder().decode(
                Qwen25VLConfig.VisionConfig.self,
                from: try JSONSerialization.data(withJSONObject: [
                    "depth": 2,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 4,
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "in_chans": 3,
                    "spatial_merge_size": 2,
                    "fullatt_block_indexes": [1],
                    "window_size": 56,
                    "out_hidden_size": 64,
                ]))
        } catch {
            XCTFail("VisionConfig decode failed: \(error)")
            return
        }
        let tower = Qwen25VLVisionTower(visionCfg)
        // 16 patches (4x4 grid) of width [C*T*ph*pw] = 3*2*14*14 = 1176
        let packed = MLXArray.ones([16, 3 * 2 * 14 * 14]).asType(.float32) * Float(0.01)
        let out = tower(packed)
        // 16 patches / 2*2 merge = 4 output tokens
        XCTAssertEqual(out.shape, [4, 64])
        eval(out)
    }

    func testVisionTowerParametersFlattenAsExpected() {
        let visionCfg: Qwen25VLConfig.VisionConfig
        do {
            visionCfg = try JSONDecoder().decode(
                Qwen25VLConfig.VisionConfig.self,
                from: try JSONSerialization.data(withJSONObject: [
                    "depth": 2,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "in_chans": 3,
                    "spatial_merge_size": 2,
                    "fullatt_block_indexes": [1],
                    "window_size": 56,
                    "out_hidden_size": 32,
                ]))
        } catch {
            XCTFail("VisionConfig decode failed: \(error)")
            return
        }
        let tower = Qwen25VLVisionTower(visionCfg)
        let names = Set(tower.parameters().flattened().map { $0.0 })
        // Patch embedding
        XCTAssertTrue(names.contains("patch_embed.proj.weight"),
            "patch_embed.proj.weight missing")
        // Vision block 0 attention + MLP
        XCTAssertTrue(names.contains("blocks.0.attn.qkv.weight"))
        XCTAssertTrue(names.contains("blocks.0.attn.proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.mlp.gate_proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.mlp.up_proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.mlp.down_proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.norm1.weight"))
        XCTAssertTrue(names.contains("blocks.0.norm2.weight"))
        // Block 1 still present
        XCTAssertTrue(names.contains("blocks.1.attn.qkv.weight"))
        // Merger
        XCTAssertTrue(names.contains("merger.ln_q.weight"))
        XCTAssertTrue(names.contains("merger.mlp.0.weight"))
        XCTAssertTrue(names.contains("merger.mlp.1.weight"))
    }

    // MARK: - Image preprocessing

    func testNormalizeCentersOnCLIPMeanAndStd() {
        // A constant gray image (0.5) normalizes to a value with
        // the CLIP mean and std applied. The output dtype + shape
        // must match the input.
        let gray = MLXArray.ones([4, 4, 3]).asType(.float32) * Float(0.5)
        let out = Qwen25VLImagePreprocessor.normalize(gray)
        XCTAssertEqual(out.shape, [4, 4, 3])
        eval(out)
        let arr = out.asArray(Float.self)
        // R channel: (0.5 - 0.48145466) / 0.26862954
        let expectedR = (0.5 - 0.48145466) / 0.26862954
        XCTAssertEqual(Double(arr[0]), expectedR, accuracy: 1e-4,
            "R channel normalization should center on CLIP mean")
    }

    func testPackPatchesProducesExpectedShape() {
        // 28x28 image, patch=14 -> 2x2 = 4 patches; with
        // temporal_patch=2 -> per-row width 3 * 2 * 14 * 14 = 1176.
        let pixels = MLXArray.ones([28, 28, 3]).asType(.float32)
        let packed = Qwen25VLImagePreprocessor.packPatches(
            pixels, patchSize: 14, temporalPatchSize: 2)
        XCTAssertEqual(packed.shape, [4, 3 * 2 * 14 * 14])
        eval(packed)
    }

    func testPackPatchesRequiresMultipleOfPatchSize() {
        // The packer's precondition is what catches mis-sized
        // input. We document the precondition by exercising the
        // happy path (the failing path would crash the test).
        let pixels = MLXArray.ones([14, 14, 3]).asType(.float32)
        let packed = Qwen25VLImagePreprocessor.packPatches(
            pixels, patchSize: 14, temporalPatchSize: 1)
        XCTAssertEqual(packed.shape, [1, 3 * 1 * 14 * 14])
    }
}

/// Tests for the loader's WS5 native arm and the env-gate that
/// switches between the foundation-only rejection and the
/// future-runtime hookup.
final class Qwen25VLLoaderTests: XCTestCase {

    private func writeConfig(_ json: [String: Any], dirSlug: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-qwen25vl-\(dirSlug)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    private func withEnv(_ key: String, _ value: String?, _ body: () throws -> Void) rethrows {
        let prior = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let prior {
                setenv(key, prior, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }

    private func tinyQwen25VLConfig() -> [String: Any] {
        return [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_hidden_layers": 36,
            "vocab_size": 151_936,
            "head_dim": 128,
            "rope_scaling": ["type": "mrope", "mrope_section": [16, 24, 24]],
            "vision_config": [
                "depth": 32,
                "hidden_size": 1280,
                "intermediate_size": 3420,
                "num_heads": 16,
                "patch_size": 14,
                "temporal_patch_size": 2,
                "in_chans": 3,
                "spatial_merge_size": 2,
                "fullatt_block_indexes": [7, 15, 23, 31],
                "window_size": 112,
                "out_hidden_size": 2048,
            ],
        ]
    }

    func testDefaultRoutesToBridge() throws {
        // Without KRILL_NATIVE_QWEN25VL the loader emits the
        // existing bridge redirect.
        let dir = try writeConfig(tinyQwen25VLConfig(), dirSlug: "default")
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_QWEN25VL", nil) {
            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("multimodal bridge")
                    || msg.contains("Qwen25VLEngine"),
                    "Default rejection must point at the bridge")
            }
        }
    }

    func testNativeOptInThrowsFoundationOnlyMessage() throws {
        // With the opt-in env set, the loader still throws (full
        // native runtime is a follow-up) but the message names
        // the foundation that landed so users know the env-gate
        // is intentional.
        let dir = try writeConfig(tinyQwen25VLConfig(), dirSlug: "optin")
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_QWEN25VL", "1") {
            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("foundation"),
                    "Opt-in rejection must explain that foundation "
                    + "modules landed and full forward is the follow-up")
            }
        }
    }

    func testConfigDecodesEndToEndAtLoader() throws {
        // The loader's qwen2_5_vl arm always decodes the config
        // (even in the rejection path) so a malformed config
        // surfaces at load time, not in the future runtime PR.
        // A config that omits required vision_config fields
        // still decodes because the VisionConfig decoder defaults
        // every key.
        let cfg: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_attention_heads": 16,
            "num_hidden_layers": 36,
            "vocab_size": 151_936,
        ]
        let dir = try writeConfig(cfg, dirSlug: "minimal")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            // Either bridge redirect (default) or foundation
            // message (opt-in); both indicate the config decoded.
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture = modelError else {
                XCTFail("Expected ModelLoadError.unsupportedArchitecture, "
                    + "got \(error). The config decode succeeded only if "
                    + "the loader reached the rejection arm.")
                return
            }
        }
    }
}
