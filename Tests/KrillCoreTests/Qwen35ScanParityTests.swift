import XCTest
import MLX
@testable import KrillCore

/// Isolated parity gate for the GatedDeltaNet delta-rule scan — the novel SSM
/// core of the qwen3_5 port — against mlx-lm's `gated_delta_ops` reference.
/// Fixture produced by the python dumper into /tmp/q35-scan (see the session
/// notes / families/qwen3_5.md). Tests the scan math in isolation BEFORE the
/// full decoder is wired up, per the native-port plan.
final class Qwen35ScanParityTests: XCTestCase {
    func testGatedDeltaScanMatchesMLXLM() throws {
        let path = "/tmp/q35-scan/arrays.safetensors"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("scan fixture missing — run the python dumper first")
        }
        let a = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let q = a["q"]!, k = a["k"]!, v = a["v"]!, g = a["g"]!, beta = a["beta"]!
        let yRef = a["y_ref"]!, stateRef = a["state_ref"]!

        let (y, state) = gatedDeltaScan(q: q, k: k, v: v, g: g, beta: beta)

        compare(y, yRef, "y")
        compare(state, stateRef, "state")
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
