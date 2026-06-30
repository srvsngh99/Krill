import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Feature parity for the native DeepEncoder SAM-ViT-B tower (`sam_model`) vs the
/// HF reference: feeds a fixed [1,3,1024,1024] image and compares the
/// [1,1024,16,16] patch-embeds output. Exercises windowed + global attention,
/// decomposed relative-position bias, and the neck/downsample convs.
///
/// Gated on KRILL_UNLIMITED_OCR_DIR (model snapshot) + KRILL_UNLIMITED_OCR_SAM_REF
/// (tools/unlimited_ocr_sam_reference.py). 4-D conv weights are transposed from
/// PyTorch [out,in,kH,kW] to MLX channels-last [out,kH,kW,in]; pos_embed is not.
final class UnlimitedOCRSAMParityTests: XCTestCase {
    func testSAMTowerMatchesHFReference() throws {
        let env = ProcessInfo.processInfo.environment
        guard let dirPath = env["KRILL_UNLIMITED_OCR_DIR"],
              let refPath = env["KRILL_UNLIMITED_OCR_SAM_REF"] else {
            throw XCTSkip("Set KRILL_UNLIMITED_OCR_DIR + KRILL_UNLIMITED_OCR_SAM_REF "
                + "(tools/unlimited_ocr_sam_reference.py)")
        }

        let model = DeepEncoderSAM()
        let dir = URL(fileURLWithPath: dirPath)
        let shards = try FileManager.default.contentsOfDirectory(atPath: dirPath)
            .filter { $0.hasSuffix(".safetensors") }
        var sam: [String: MLXArray] = [:]
        let prefix = "model.sam_model."
        for shard in shards {
            for (k, v) in try MLX.loadArrays(url: dir.appendingPathComponent(shard))
            where k.hasPrefix(prefix) {
                let key = String(k.dropFirst(prefix.count))
                var arr = v.asType(.float32)
                // PyTorch conv weight [out,in,kH,kW] -> MLX channels-last [out,kH,kW,in].
                // pos_embed is also 4-D but must NOT be transposed.
                if arr.ndim == 4 && key != "pos_embed" {
                    arr = arr.transposed(0, 2, 3, 1)
                }
                sam[key] = arr
            }
        }
        XCTAssertFalse(sam.isEmpty, "no model.sam_model.* weights found in \(dirPath)")
        try model.update(parameters: ModuleParameters.unflattened(sam.map { ($0.key, $0.value) }),
                         verify: [.shapeMismatch])
        eval(model)

        let fx = try MLX.loadArrays(url: URL(fileURLWithPath: refPath))
        let imageCF = fx["image"]!.asType(.float32)            // [1,3,1024,1024]
        let image = imageCF.transposed(0, 2, 3, 1)             // -> channels-last
        let refOut = fx["output"]!.asType(.float32)            // [1,1024,16,16]

        let out = model(image: image).asType(.float32)
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
        print("[unlimited-ocr SAM parity] cosine=\(cosine) maxAbs=\(maxAbs)")
        XCTAssertGreaterThan(cosine, 0.9999, "SAM feature cosine \(cosine) too low")
    }
}
