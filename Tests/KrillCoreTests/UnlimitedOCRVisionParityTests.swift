import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// End-to-end vision parity: the full DeepEncoder (SAM -> CLIP -> concat ->
/// projector) vs the HF reference, on a fixed image. Validates the SAM/CLIP
/// composition + the linear projector that yields the [1,256,1280] LM-space
/// features. Gated on KRILL_UNLIMITED_OCR_DIR + KRILL_UNLIMITED_OCR_VIS_REF
/// (tools/unlimited_ocr_vision_reference.py).
final class UnlimitedOCRVisionParityTests: XCTestCase {
    func testDeepEncoderMatchesHFReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dirPath = env["KRILL_UNLIMITED_OCR_DIR"],
              let refPath = env["KRILL_UNLIMITED_OCR_VIS_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_DIR + KRILL_UNLIMITED_OCR_VIS_REF "
                + "(tools/unlimited_ocr_vision_reference.py)")
        }

        let model = DeepEncoder()
        let dir = URL(fileURLWithPath: dirPath)
        let shards = try FileManager.default.contentsOfDirectory(atPath: dirPath)
            .filter { $0.hasSuffix(".safetensors") }
        // Bind sam_model.* / vision_model.* / projector.* (strip leading `model.`).
        let keep = ["model.sam_model.", "model.vision_model.", "model.projector."]
        var w: [String: MLXArray] = [:]
        for shard in shards {
            for (k, v) in try MLX.loadArrays(url: dir.appendingPathComponent(shard))
            where keep.contains(where: k.hasPrefix) {
                let key = String(k.dropFirst("model.".count))
                if key.contains("vision_model.embeddings.patch_embedding") { continue }  // bypassed
                var arr = v.asType(.float32)
                // SAM conv weights: PyTorch [out,in,kH,kW] -> MLX [out,kH,kW,in]
                // (pos_embed is 4-D but must not be transposed).
                if arr.ndim == 4 && !key.hasSuffix("pos_embed") {
                    arr = arr.transposed(0, 2, 3, 1)
                }
                w[key] = arr
            }
        }
        XCTAssertFalse(w.isEmpty, "no vision weights found")
        try model.update(parameters: ModuleParameters.unflattened(w.map { ($0.key, $0.value) }),
                         verify: [.shapeMismatch])
        eval(model)

        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let image = fx["image"]!.asType(.float32).transposed(0, 2, 3, 1)  // channels-last
        let refOut = fx["features"]!.asType(.float32)                     // [1,256,1280]

        let out = model(image: image).asType(.float32)
        eval(out)
        XCTAssertEqual(out.shape, refOut.shape, "feature shape")

        let got = out.reshaped([-1]).asArray(Float.self)
        let ref = refOut.reshaped([-1]).asArray(Float.self)
        var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
        for i in 0 ..< got.count {
            let x = Double(got[i]), y = Double(ref[i])
            dot += x * y; na += x * x; nb += y * y
            maxAbs = max(maxAbs, abs(x - y))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        print("[unlimited-ocr VISION parity] cosine=\(cosine) maxAbs=\(maxAbs)")
        XCTAssertGreaterThan(cosine, 0.9999, "vision feature cosine \(cosine) too low")
    }
}
