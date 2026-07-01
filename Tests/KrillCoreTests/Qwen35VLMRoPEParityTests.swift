import XCTest
import Foundation
import MLX
@testable import KrillCore

/// Parity gate for the interleaved sectioned 3D mRoPE against mlx_vlm's real
/// `Qwen3_5RotaryEmbedding.apply_rotary`. The fixture uses DISTINCT t/h/w
/// positions so the frequency-axis selector is genuinely exercised (a wrong
/// selector or pairing fails here even though text-only t==h==w would not).
final class Qwen35VLMRoPEParityTests: XCTestCase {
    private var fixtureDir: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/qwen3_5_mrope")
    }

    func testInterleavedMRoPEMatchesMLXVLM() throws {
        let dir = fixtureDir
        guard FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("mrope.safetensors").path) else {
            throw XCTSkip("mrope fixture missing — run tools/verify_qwen3_5_mrope_parity.py")
        }
        let metaData = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        let meta = try JSONSerialization.jsonObject(with: metaData) as! [String: Any]
        let headDim = meta["head_dim"] as! Int
        let partial = (meta["partial_rotary_factor"] as! NSNumber).floatValue
        let section = meta["mrope_section"] as! [Int]
        let theta = (meta["theta"] as! NSNumber).floatValue

        let a = try MLX.loadArrays(url: dir.appendingPathComponent("mrope.safetensors"))
        let q = a["q"]!, k = a["k"]!, positions = a["positions"]!
        let qRef = a["q_out"]!, kRef = a["k_out"]!

        let mrope = Qwen35VLMRoPE(
            headDim: headDim, partialRotaryFactor: partial, mropeSection: section, theta: theta)
        let (cos, sin) = mrope.buildCosSin(positions)
        let qOut = applyPartialMRoPE(q, cos: cos, sin: sin, rotaryDim: mrope.rotaryDim)
        let kOut = applyPartialMRoPE(k, cos: cos, sin: sin, rotaryDim: mrope.rotaryDim)

        compare(qOut, qRef, "q")
        compare(kOut, kRef, "k")
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
