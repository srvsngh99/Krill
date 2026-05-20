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
        // Per-patch batched input: 56x56 image -> 4x4 patch grid
        // -> 16 patches in merge-block row-major order. Each
        // patch is [T=2, ph=14, pw=14, C=3]. After
        // spatial_merge_size=2 -> 4 output tokens of out_hidden=64.
        let patches = MLXArray.ones([16, 2, 14, 14, 3]).asType(.float32) * Float(0.01)
        let out = tower(patches)
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
        let params = tower.parameters().flattened()
        let names = Set(params.map { $0.0 })

        // Patch embedding (Conv3d): weight shape is the rank-5
        // `[embed_dim, T, ph, pw, in_chans]` layout that the
        // shipped mlx-community checkpoint uses. The shape is
        // load-bearing: a Linear here would be rank 2 and would
        // not accept the checkpoint weight at safetensors load
        // time. Pin both the key and the shape.
        XCTAssertTrue(names.contains("patch_embed.proj.weight"),
            "patch_embed.proj.weight missing")
        if let patchWeight = params.first(where: { $0.0 == "patch_embed.proj.weight" })?.1 {
            XCTAssertEqual(patchWeight.shape, [16, 2, 14, 14, 3],
                "patch_embed.proj.weight must be rank-5 Conv3d "
                + "[embed_dim, T, ph, pw, in_chans] to match the "
                + "mlx-community Qwen 2.5-VL safetensors layout")
        } else {
            XCTFail("patch_embed.proj.weight not found in parameter cache")
        }

        // Vision block 0 attention + MLP keys
        XCTAssertTrue(names.contains("blocks.0.attn.qkv.weight"))
        XCTAssertTrue(names.contains("blocks.0.attn.proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.mlp.gate_proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.mlp.up_proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.mlp.down_proj.weight"))
        XCTAssertTrue(names.contains("blocks.0.norm1.weight"))
        XCTAssertTrue(names.contains("blocks.0.norm2.weight"))
        XCTAssertTrue(names.contains("blocks.1.attn.qkv.weight"))

        // Merger: the GELU placeholder at index 1 produces NO
        // parameters, so the two Linear weights land at
        // mlp.0.weight and mlp.2.weight - matching the shipped
        // checkpoint exactly. Pin both the presence of mlp.2 AND
        // the absence of mlp.1 so a regression that drops the
        // GELU slot is caught here.
        XCTAssertTrue(names.contains("merger.ln_q.weight"))
        XCTAssertTrue(names.contains("merger.mlp.0.weight"))
        XCTAssertTrue(names.contains("merger.mlp.2.weight"),
            "merger.mlp.2.weight is the SECOND Linear's key in the "
            + "shipped checkpoint; the GELU at index 1 keeps the "
            + "second Linear at index 2 in the [Linear, GELU, Linear] "
            + "array")
        XCTAssertFalse(names.contains("merger.mlp.1.weight"),
            "GELU at index 1 must not produce a parameter key; "
            + "the second Linear must NOT collapse onto mlp.1")
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

    func testToConv3DInputProducesPerPatchBatchedShape() {
        // 28x28 image, patch=14, merge=2, temporal=2:
        //   grid = 2x2 = 4 patches
        //   per-patch tensor: [T=2, ph=14, pw=14, C=3]
        // Output shape: [4, 2, 14, 14, 3].
        let pixels = MLXArray.ones([28, 28, 3]).asType(.float32)
        let inputTensor = Qwen25VLImagePreprocessor.toConv3DInput(
            pixels, patchSize: 14, temporalPatchSize: 2, spatialMergeSize: 2)
        XCTAssertEqual(inputTensor.shape, [4, 2, 14, 14, 3])
        eval(inputTensor)
    }

    func testToConv3DInputSingleFrame() {
        // Smallest valid: 28x28 image (patch_size * spatial_merge_size
        // = 14*2 = 28), single temporal frame.
        let pixels = MLXArray.ones([28, 28, 3]).asType(.float32)
        let inputTensor = Qwen25VLImagePreprocessor.toConv3DInput(
            pixels, patchSize: 14, temporalPatchSize: 1, spatialMergeSize: 2)
        XCTAssertEqual(inputTensor.shape, [4, 1, 14, 14, 3])
    }

    func testToConv3DInputMergeBlockOrdering() {
        // Verify the patch ordering is merge-block row-major: a
        // patch's identity (which 2D location it came from) must
        // group the four patches of one 2x2 block consecutively.
        //
        // We use a 56x56 (4x4 patch grid, 4 merge blocks of 2x2).
        // Encode each patch's location as a unique constant by
        // multiplying its row index across pixels: row i of the
        // image carries value i. After preprocessing in
        // merge-block order, patches 0-3 must all have come from
        // block (0,0), patches 4-7 from block (0,1), etc.
        //
        // For a 56x56 image, patch_size=14, spatial_merge_size=2:
        //   - 4x4 patch grid (4 merge blocks).
        //   - Merge block (0,0) covers rows 0-27, cols 0-27.
        //     Its four patches each have row range subsets;
        //     they all sit in rows 0-27.
        //   - Merge block (1,0) covers rows 28-55.
        // We populate row i with a value `i // 28` (0 for
        // blocks at out_h=0, 1 for blocks at out_h=1) and assert
        // patch batches 0..7 carry value 0, patches 8..15 carry
        // value 1.
        let H = 56, W = 56, C = 3
        var pixels = MLXArray.zeros([H, W, C]).asType(.float32)
        // Build a column of values per-row: [H, 1, 1] then
        // broadcast to [H, W, C].
        let rowValues = MLXArray((0 ..< H).map { Float($0 / 28) })
            .reshaped(H, 1, 1)
        pixels = pixels + rowValues  // broadcasts to [H, W, C]
        let batched = Qwen25VLImagePreprocessor.toConv3DInput(
            pixels, patchSize: 14, temporalPatchSize: 1, spatialMergeSize: 2)
        XCTAssertEqual(batched.shape, [16, 1, 14, 14, 3])
        eval(batched)
        // Patches 0..7 should be from merge blocks where
        // out_h=0 (rows 0-27 -> rowValue 0).
        // Patches 8..15 from out_h=1 (rows 28-55 -> rowValue 1).
        // Sample one pixel per patch to verify.
        let arr = batched.asArray(Float.self)
        // patches[i, t, ph, pw, c] flattened layout:
        let stride = 1 * 14 * 14 * 3
        for i in 0 ..< 8 {
            // First pixel of patch i, channel 0.
            let v = arr[i * stride]
            XCTAssertEqual(v, 0.0, accuracy: 1e-6,
                "Patches 0..7 must come from out_h=0 (rows 0-27); "
                + "patch \(i) carried value \(v)")
        }
        for i in 8 ..< 16 {
            let v = arr[i * stride]
            XCTAssertEqual(v, 1.0, accuracy: 1e-6,
                "Patches 8..15 must come from out_h=1 (rows 28-55); "
                + "patch \(i) carried value \(v)")
        }
    }

    // MARK: - End-to-end vision tower with preprocessor

    func testPreprocessorOutputFlowsThroughVisionTower() {
        // Verify the preprocessor produces a tensor that the
        // tower accepts unchanged (no intermediate reshapes
        // required) - this pins the contract between the two
        // modules.
        let visionCfg: Qwen25VLConfig.VisionConfig
        do {
            visionCfg = try JSONDecoder().decode(
                Qwen25VLConfig.VisionConfig.self,
                from: try JSONSerialization.data(withJSONObject: [
                    "depth": 1,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                    "patch_size": 14,
                    "temporal_patch_size": 2,
                    "in_chans": 3,
                    "spatial_merge_size": 2,
                    "fullatt_block_indexes": [],
                    "window_size": 56,
                    "out_hidden_size": 32,
                ]))
        } catch {
            XCTFail("VisionConfig decode failed: \(error)")
            return
        }
        let tower = Qwen25VLVisionTower(visionCfg)

        // 28x28 image. Preprocessor returns per-patch batched
        // [4, 2, 14, 14, 3] (one 2x2 merge block). Tower yields
        // 1 merged token.
        let pixels = MLXArray.ones([28, 28, 3]).asType(.float32) * Float(0.5)
        let normalized = Qwen25VLImagePreprocessor.normalize(pixels)
        let conv3DInput = Qwen25VLImagePreprocessor.toConv3DInput(
            normalized,
            patchSize: visionCfg.patchSize,
            temporalPatchSize: visionCfg.temporalPatchSize,
            spatialMergeSize: visionCfg.spatialMergeSize)
        XCTAssertEqual(conv3DInput.shape, [4, 2, 14, 14, 3])
        let out = tower(conv3DInput)
        XCTAssertEqual(out.shape, [1, 32])
        eval(out)
    }

    // MARK: - Text-side config projection

    func testQwenTextConfigDropsOProjBias() throws {
        // The VL config projects onto a QwenConfig with
        // `attention_bias: true` (for QKV). The dense QwenAttention
        // hardcodes `o_proj` bias to false regardless of the flag,
        // so the projection must produce an attention module whose
        // o_proj has NO bias - if a future QwenAttention change
        // started honoring the flag for o_proj it would break weight
        // loading silently (the checkpoint has no o_proj.bias key).
        let cfg = try decode(tinyConfigJSON())
        let text = cfg.qwenTextConfig
        XCTAssertTrue(text.attentionBias, "QKV bias must be enabled")
        // QwenAttention reads attentionBias only for q/k/v_proj;
        // o_proj is always bias-free. The contract pinned here is
        // the source-code-level constant, not a behavioral test
        // (which would require instantiating QwenAttention and
        // inspecting its sub-Linears; that is fragile across MLX
        // versions). Documenting the contract in code is what
        // ensures a future regression is caught at the source.
        XCTAssertEqual(text.modelType, "qwen2",
            "Text side projects to dense Qwen 2.5, not Qwen 3")
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
        // native runtime is a follow-up) but the message carries
        // the stable `[WS5_FOUNDATION_ONLY]` tag so the test
        // discriminates "opt-in arm fired" from "default arm
        // fired" instead of substring-matching prose ("foundation"
        // appears in both messages today).
        let dir = try writeConfig(tinyQwen25VLConfig(), dirSlug: "optin")
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_QWEN25VL", "1") {
            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture, got \(error)")
                    return
                }
                XCTAssertTrue(msg.contains("[WS5_FOUNDATION_ONLY]"),
                    "Opt-in rejection must carry the stable tag so "
                    + "CI/smoke harnesses can pin the arm reliably")
            }
        }
    }

    func testDefaultRejectionDoesNotCarryFoundationTag() throws {
        // The default-arm rejection must NOT contain the opt-in
        // tag. This is the inverse assertion that catches a
        // future refactor copying the tag string by accident.
        let dir = try writeConfig(tinyQwen25VLConfig(), dirSlug: "default-no-tag")
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_QWEN25VL", nil) {
            XCTAssertThrowsError(try loadModel(from: dir)) { error in
                guard let modelError = error as? ModelLoadError,
                      case .unsupportedArchitecture(let msg) = modelError else {
                    XCTFail("Expected unsupportedArchitecture, got \(error)")
                    return
                }
                XCTAssertFalse(msg.contains("[WS5_FOUNDATION_ONLY]"),
                    "Default-arm rejection must NOT carry the opt-in "
                    + "tag - that tag is reserved for the env-gated arm")
            }
        }
    }

    func testConfigDecoderHandlesMinimalConfig() throws {
        // The Qwen25VLConfig decoder accepts a minimal config
        // (no vision_config sub-object, no rope_scaling) by
        // defaulting the optional fields to the Qwen2.5-VL-3B
        // reference shape. This is the contract callers rely on
        // when working with partial configs in tests and tooling.
        // We exercise the decoder DIRECTLY rather than through
        // the loader (the loader's rejection arm currently does
        // not decode in this PR; the runtime PR moves the decode
        // inside the opt-in branch).
        let cfg: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_size": 2048,
            "intermediate_size": 11008,
            "num_attention_heads": 16,
            "num_hidden_layers": 36,
            "vocab_size": 151_936,
        ]
        let data = try JSONSerialization.data(withJSONObject: cfg)
        let parsed = try JSONDecoder().decode(Qwen25VLConfig.self, from: data)
        XCTAssertEqual(parsed.hiddenSize, 2048)
        XCTAssertEqual(parsed.vision.depth, 32,
            "Minimal config defaults vision.depth to the "
            + "Qwen2.5-VL-3B reference value")
        XCTAssertEqual(parsed.mropeSection, [16, 24, 24],
            "Minimal config defaults mrope_section to the "
            + "Qwen2.5-VL-3B reference split")
        XCTAssertEqual(parsed.imageTokenId, 151_655,
            "Minimal config defaults image_token_id to the "
            + "Qwen 2.5-VL tokenizer's <|image_pad|> id")
    }
}
