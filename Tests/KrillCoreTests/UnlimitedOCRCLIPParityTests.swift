import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Feature parity for the native DeepEncoder CLIP-L tower (`vision_model`) vs the
/// HF reference. Feeds a fixed `patch_embeds` grid and compares the full
/// `[1, 257, 1024]` output. Gated on KRILL_UNLIMITED_OCR_DIR (model snapshot,
/// for the vision weights) + KRILL_UNLIMITED_OCR_CLIP_REF (the fixture from
/// tools/unlimited_ocr_clip_reference.py). Both sides run fp32, so parity is
/// tight (cosine > 0.9999, small max-abs).
final class UnlimitedOCRCLIPParityTests: XCTestCase {
    func testCLIPTowerMatchesHFReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dirPath = env["KRILL_UNLIMITED_OCR_DIR"],
              let refPath = env["KRILL_UNLIMITED_OCR_CLIP_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_DIR + KRILL_UNLIMITED_OCR_CLIP_REF "
                + "(tools/unlimited_ocr_clip_reference.py)")
        }

        // Load the CLIP module and bind the `model.vision_model.*` weights.
        let model = DeepEncoderCLIP()
        let dir = URL(fileURLWithPath: dirPath)
        let shards = try FileManager.default.contentsOfDirectory(atPath: dirPath)
            .filter { $0.hasSuffix(".safetensors") }
        var vis: [String: MLXArray] = [:]
        let prefix = "model.vision_model."
        for shard in shards {
            for (k, v) in try MLX.loadArrays(url: dir.appendingPathComponent(shard))
            where k.hasPrefix(prefix) && !k.contains("patch_embedding") {
                vis[String(k.dropFirst(prefix.count))] = v.asType(.float32)
            }
        }
        XCTAssertFalse(vis.isEmpty, "no model.vision_model.* weights found in \(dirPath)")
        try model.update(parameters: ModuleParameters.unflattened(vis.map { ($0.key, $0.value) }),
                         verify: [.shapeMismatch])
        eval(model)

        // Fixed input + reference output from the fixture.
        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let patchEmbeds = fx["patch_embeds"]!.asType(.float32)
        let refOut = fx["output"]!.asType(.float32)

        let out = model(patchEmbeds: patchEmbeds).asType(.float32)
        eval(out)
        XCTAssertEqual(out.shape, refOut.shape, "output shape")

        let got = out.reshaped([-1]).asArray(Float.self)
        let ref = refOut.reshaped([-1]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
        for i in 0 ..< got.count {
            let x = Double(got[i]), y = Double(ref[i])
            dot += x * y; na += x * x; nb += y * y
            maxAbs = max(maxAbs, abs(x - y))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        print("[unlimited-ocr CLIP parity] cosine=\(cosine) maxAbs=\(maxAbs)")
        XCTAssertGreaterThan(cosine, 0.9999, "CLIP feature cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 0.05, "CLIP feature max-abs diff \(maxAbs) too large")
    }
}
