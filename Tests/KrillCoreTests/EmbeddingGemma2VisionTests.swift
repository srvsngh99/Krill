import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// EmbeddingGemma 2 image path, without the real checkpoint: preprocessor
/// geometry (values pinned to the HF `get_aspect_ratio_preserving_size`),
/// resize / patch layout, the unpadded tower forward vs the reference's padded
/// one, scatter, and STRICT weight binding on a tiny synthetic tower.
final class EmbeddingGemma2VisionTests: XCTestCase {

    // MARK: Preprocessor geometry

    func testTargetSizeMatchesHF() throws {
        // (height, width, budget) -> (height, width), computed with transformers 5.19.
        let cases: [(Int, Int, Int, Int, Int)] = [
            (480, 640, 280, 672, 912), (800, 480, 280, 1008, 576), (300, 1200, 280, 384, 1584),
            (224, 224, 280, 768, 768), (1080, 1920, 280, 576, 1056), (37, 100, 280, 480, 1296),
            (240, 320, 280, 672, 912), (200, 300, 280, 624, 960),
            (480, 640, 140, 480, 624), (1080, 1920, 140, 384, 720), (16, 16, 140, 528, 528),
            (480, 640, 70, 336, 432), (480, 640, 1120, 1344, 1824),
            // one side rounds to zero: clamp to one block, other side capped
            (10, 4000, 280, 48, 13440), (4000, 10, 140, 6720, 48), (2000, 30, 280, 6528, 96),
        ]
        for (h, w, budget, eh, ew) in cases {
            let t = try EG2ImagePreprocessor.targetSize(height: h, width: w, maxPatches: budget * 9)
            XCTAssertEqual(t.height, eh, "\(h)x\(w) budget \(budget)")
            XCTAssertEqual(t.width, ew, "\(h)x\(w) budget \(budget)")
            XCTAssertLessThanOrEqual((t.height / 16) * (t.width / 16), budget * 9)
            XCTAssertEqual(t.height % 48, 0)
            XCTAssertEqual(t.width % 48, 0)
        }
    }

    func testTargetSizeRejectsEmpty() {
        XCTAssertThrowsError(try EG2ImagePreprocessor.targetSize(height: 0, width: 10, maxPatches: 2520))
    }

    func testUnsupportedBudgetRejected() {
        let img = EG2RGBImage(pixels: [UInt8](repeating: 9, count: 64 * 64 * 3), width: 64, height: 64)
        XCTAssertThrowsError(try EG2ImagePreprocessor.prepare(img, maxSoftTokens: 100)) {
            XCTAssertEqual($0 as? EG2ImageError, .unsupportedSoftTokenBudget(100))
        }
    }

    func testPatchCountsAndSoftTokensPerAspectRatio() throws {
        // patches = (H/16)*(W/16), soft tokens = patches / 9, never above the budget.
        for (w, h) in [(640, 480), (480, 800), (1200, 300), (224, 224), (100, 37)] {
            let img = EG2RGBImage(pixels: [UInt8](repeating: 100, count: w * h * 3), width: w, height: h)
            let p = try EG2ImagePreprocessor.prepare(img)
            XCTAssertEqual(p.patches.count, p.numPatches * 768)
            XCTAssertEqual(p.softTokens * 9, p.numPatches)
            XCTAssertLessThanOrEqual(p.softTokens, 280)
            XCTAssertGreaterThan(p.softTokens, 240, "\(w)x\(h) should use nearly the whole budget")
            XCTAssertEqual(p.resizedWidth % 48, 0)
            XCTAssertEqual(p.resizedHeight % 48, 0)
        }
        // video-frame budget
        let img = EG2RGBImage(pixels: [UInt8](repeating: 1, count: 320 * 240 * 3), width: 320, height: 240)
        let f = try EG2ImagePreprocessor.prepare(img, maxSoftTokens: 140)
        XCTAssertLessThanOrEqual(f.softTokens, 140)
    }

    // MARK: Resize + patch layout

    func testResizeKeepsConstantColourAndShape() {
        var flat = [UInt8]()
        for _ in 0 ..< 50 * 30 { flat += [10, 200, 77] }
        let img = EG2RGBImage(pixels: flat, width: 50, height: 30)
        for (w, h) in [(48, 48), (300, 120), (7, 5)] {
            let r = EG2ImagePreprocessor.resize(img, toWidth: w, height: h)
            XCTAssertEqual(r.width, w); XCTAssertEqual(r.height, h)
            XCTAssertEqual(r.pixels.count, w * h * 3)
            for i in 0 ..< w * h {
                XCTAssertEqual(r.pixels[i * 3], 10); XCTAssertEqual(r.pixels[i * 3 + 1], 200)
                XCTAssertEqual(r.pixels[i * 3 + 2], 77)
            }
        }
    }

    func testResizeIdentityAndBlockAverage() {
        // identity is a no-op
        let px: [UInt8] = (0 ..< 4 * 4 * 3).map { UInt8($0 * 5) }
        let img = EG2RGBImage(pixels: px, width: 4, height: 4)
        XCTAssertEqual(EG2ImagePreprocessor.resize(img, toWidth: 4, height: 4).pixels, px)
        // a horizontal ramp stays (weakly) monotone when upscaled, rows stay identical
        var rampPx = [UInt8]()
        for x in 0 ..< 8 { let v = UInt8(x * 30); rampPx += [v, v, v] }
        let ramp = EG2RGBImage(pixels: rampPx, width: 8, height: 1)
        let up = EG2ImagePreprocessor.resize(ramp, toWidth: 32, height: 3)
        for y in 0 ..< 3 {
            for x in 1 ..< 32 {
                let cur = Int(up.pixels[(y * 32 + x) * 3])
                let prev = Int(up.pixels[(y * 32 + x - 1) * 3])
                XCTAssertGreaterThanOrEqual(cur, prev - 1, "row \(y) col \(x)")
            }
            XCTAssertEqual(Array(up.pixels[(y * 32 * 3) ..< ((y + 1) * 32 * 3)]), Array(up.pixels[0 ..< 32 * 3]))
        }
    }

    func testPatchLayoutIsRowMajorPatchesThenPyPxC() throws {
        // 768x768 is its own target at the 280 budget, so no resize happens and
        // every patch value can be checked exactly. pixel(x, y, c) = (3x + 7y + 11c) mod 256.
        let w = 768, h = 768
        var px = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0 ..< h { for x in 0 ..< w { for c in 0 ..< 3 {
            px[(y * w + x) * 3 + c] = UInt8((3 * x + 7 * y + 11 * c) % 256)
        } } }
        let p = try EG2ImagePreprocessor.prepare(EG2RGBImage(pixels: px, width: w, height: h))
        XCTAssertEqual(p.gridW, 48); XCTAssertEqual(p.gridH, 48)
        XCTAssertEqual(p.softTokens, 256)
        // (patch gy, gx) -> element (py, px, c): row-major patches, then py, px, c fastest.
        for (gy, gx, py, qx, c) in [(0, 0, 0, 0, 0), (0, 1, 0, 0, 2), (1, 0, 3, 5, 1), (47, 47, 15, 15, 2), (13, 30, 7, 9, 0)] {
            let got = p.patches[(gy * 48 + gx) * 768 + (py * 16 + qx) * 3 + c]
            let want = Float((3 * (gx * 16 + qx) + 7 * (gy * 16 + py) + 11 * c) % 256) / 255
            XCTAssertEqual(got, want, accuracy: 1e-6, "patch (\(gy),\(gx)) px (\(py),\(qx)) c\(c)")
        }
    }

    // MARK: Tiny tower

    private func tinyVisionConfigJSON(clipped: Bool = false) -> String {
        """
        {"model_type": "embedding_gemma2", "boi_token_id": 255999, "eoi_token_id": 258882, "image_token_id": 258880,
         "text_config": {"embedding_dim": 12, "head_dim": 8, "hidden_size": 16, "hidden_size_per_layer_input": 8,
           "intermediate_size": 32, "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
           "per_layer_config": {"1": {"head_dim": 16, "num_key_value_heads": 1}}, "sliding_window": 4, "vocab_size": 50},
         "vision_config": {"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
           "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16, "patch_size": 4,
           "pooling_kernel_size": 3, "default_output_length": 8, "position_embedding_size": 40,
           "rope_parameters": {"rope_theta": 100.0, "rope_type": "axial"}, "rms_norm_eps": 1e-6,
           "use_clipped_linears": \(clipped), "standardize": false}}
        """
    }

    private func makeTinyTower() throws -> EG2VisionTower {
        let cfg = try JSONDecoder().decode(EG2VisionConfig.self, from: Data(tinyVisionConfigJSON().utf8))
        let t = EG2VisionTower(cfg, textHidden: 16)
        MLXRandom.seed(11)
        let p = t.parameters().flattened().map { (k, v) -> (String, MLXArray) in
            if k.hasSuffix("_min") || k.hasSuffix("_max") { return (k, v) }
            if k.hasSuffix("norm.weight") { return (k, MLXArray.ones(v.shape) + MLXRandom.normal(v.shape) * 0.05) }
            return (k, MLXRandom.normal(v.shape) * 0.15)
        }
        t.update(parameters: ModuleParameters.unflattened(p))
        t.setComputeDtype(.float32)
        eval(t)
        return t
    }

    /// Writes the tower's tensors (minus clip scalars, like the real checkpoint)
    /// plus config.json into a temp dir; returns the dir.
    private func writeTinyCheckpoint(
        mutate: (inout [String: MLXArray]) -> Void = { _ in }, clipped: Bool = false
    ) throws -> URL {
        let tower = try makeTinyTower()
        var arrays: [String: MLXArray] = [:]
        for (k, v) in tower.parameters().flattened()
        where !(k.hasSuffix("_min") || k.hasSuffix("_max")) { arrays[k] = v }
        mutate(&arrays)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("eg2-vision-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(tinyVisionConfigJSON(clipped: clipped).utf8).write(to: dir.appendingPathComponent("config.json"))
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    func testStrictBindingAcceptsExactCheckpoint() throws {
        let dir = try writeTinyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (tower, report) = try loadEG2VisionTower(directory: dir, dtype: .float32)
        // 2 layers x 7 clippable linears x 4 scalars
        XCTAssertEqual(report.defaultedClipScalars, 2 * 7 * 4)
        XCTAssertGreaterThan(report.bound, 0)
        let img = EG2PreparedImage(patches: [Float](repeating: 0.3, count: 36 * 48), gridW: 6, gridH: 6,
                                   patchSize: 4, poolingKernel: 3)
        let s = tower.softTokens(img)
        XCTAssertEqual(s.shape, [4, 16])
        XCTAssertTrue(s.asArray(Float.self).allSatisfy { $0.isFinite })
    }

    func testStrictBindingRejectsUnknownKey() throws {
        let dir = try writeTinyCheckpoint { $0["vision_tower.encoder.layers.0.bogus.weight"] = MLXArray.zeros([3]) }
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2VisionTower(directory: dir))
    }

    func testStrictBindingRejectsMissingTensor() throws {
        let dir = try writeTinyCheckpoint { $0.removeValue(forKey: "embed_vision.embedding_projection.weight") }
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2VisionTower(directory: dir))
        let dir2 = try writeTinyCheckpoint {
            $0.removeValue(forKey: "vision_tower.encoder.layers.1.self_attn.q_norm.weight") }
        defer { try? FileManager.default.removeItem(at: dir2) }
        XCTAssertThrowsError(try loadEG2VisionTower(directory: dir2))
    }

    func testStrictBindingRejectsWrongShape() throws {
        let dir = try writeTinyCheckpoint {
            $0["vision_tower.patch_embedder.input_proj.weight"] = MLXArray.zeros([32, 47]) }
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2VisionTower(directory: dir))
    }

    func testMissingClipScalarsOnlyForgivenWhenClippingIsOff() throws {
        // config says clipped linears are ON but the checkpoint has no scalars: must fail.
        let dir = try writeTinyCheckpoint(clipped: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2VisionTower(directory: dir))
    }

    func testNoVisionTowerIsReported() throws {
        let dir = try writeTinyCheckpoint { arrays in
            for k in arrays.keys { arrays.removeValue(forKey: k) }
            arrays["language_model.embed_tokens.weight"] = MLXArray.zeros([2, 2])
        }
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEG2VisionTower(directory: dir)) {
            guard case EG2VisionLoadError.noVisionTower = $0 else { return XCTFail("\($0)") }
        }
    }

    // MARK: Unpadded forward == the reference's padded forward

    func testUnpaddedTowerMatchesPaddedReferenceSemantics() throws {
        let tower = try makeTinyTower()
        // 24x24 image, patch 4 -> 6x6 = 36 patches -> 4 soft tokens. The padded
        // path (existing VisionEncoder) uses maxPatches = 8 * 9 = 72, so 36 real +
        // 36 masked padding patches: the reference's situation.
        let H = 24, W = 24
        MLXRandom.seed(3)
        let pix = MLXRandom.uniform(low: 0, high: 1, [1, 3, H, W])
        eval(pix)
        let padded = tower.embedVision(tower.visionTower(pix))  // [1, 4, 16]
        // Same pixels, as patches (py, px, c) in row-major patch order.
        let a = pix.asArray(Float.self)
        var patches = [Float](); patches.reserveCapacity(36 * 48)
        for gy in 0 ..< 6 { for gx in 0 ..< 6 { for py in 0 ..< 4 { for px in 0 ..< 4 { for c in 0 ..< 3 {
            patches.append(a[(c * H + gy * 4 + py) * W + gx * 4 + px])
        } } } } }
        let mine = tower.softTokens(patches: patches, gridW: 6, gridH: 6)
        XCTAssertEqual(mine.shape, [4, 16])
        let d = (mine - padded.reshaped(4, 16)).abs().max().item(Float.self)
        XCTAssertLessThan(d, 1e-4, "unpadded forward diverged from the padded reference semantics")
    }
}
