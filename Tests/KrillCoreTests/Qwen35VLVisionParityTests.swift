import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// Parity gate for the native Qwen3.5-VL vision tower against mlx_vlm's real
/// `qwen3_5.vision.VisionModel` (which subclasses the shared Qwen3-VL tower).
/// Fixture (tiny synthetic config + random weights + a random patch batch and
/// the oracle's output) is produced by `tools/verify_qwen3_5_vl_parity.py` and
/// committed under `Fixtures/qwen3_5_vl`. Green = the Conv3d patch embed, the
/// bilinearly-interpolated learnable position embedding, the vision-rotary
/// full-attention blocks, and the PatchMerger all match the oracle.
final class Qwen35VLVisionParityTests: XCTestCase {
    private var fixtureDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/qwen3_5_vl")
    }

    func testVisionTowerMatchesMLXVLM() throws {
        let dir = fixtureDir
        guard FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("weights.safetensors").path) else {
            throw XCTSkip("qwen3_5_vl fixture missing — run tools/verify_qwen3_5_vl_parity.py")
        }

        // Config from meta.json (`config` sub-object).
        let metaData = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let configData = try JSONSerialization.data(withJSONObject: meta["config"]!)
        let config = try JSONDecoder().decode(Qwen35VLVisionConfig.self, from: configData)
        let grid = (meta["grid_thw"] as! [[Int]]).map { (t: $0[0], h: $0[1], w: $0[2]) }

        // Build + load weights.
        let model = Qwen35VLVisionModel(config)
        let weights = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let nested = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
        try model.update(parameters: nested, verify: [.all])

        // Run + compare.
        let io = try MLX.loadArrays(url: dir.appendingPathComponent("io.safetensors"))
        let pixelValues = io["pixel_values"]!
        let expected = io["output"]!
        let out = model(pixelValues, grids: grid)

        XCTAssertEqual(out.shape, expected.shape, "output shape")
        compare(out, expected, "vision_output")
    }

    private func compare(_ got: MLXArray, _ ref: MLXArray, _ name: String) {
        let gv = got.asType(.float32).flattened().asArray(Float.self)
        let rv = ref.asType(.float32).flattened().asArray(Float.self)
        XCTAssertEqual(gv.count, rv.count, "\(name) element count")
        var dot = 0.0, na = 0.0, nb = 0.0, maxAbs = 0.0
        for i in 0 ..< gv.count {
            let x = Double(gv[i]), y = Double(rv[i])
            dot += x * y; na += x * x; nb += y * y
            maxAbs = max(maxAbs, abs(x - y))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999, "\(name) cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-3, "\(name) max-abs \(maxAbs) too large")
    }
}
