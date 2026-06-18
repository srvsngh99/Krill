import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Stage 2 gate: the native `WhisperEncoder` must match a NumPy reference run
/// on the SAME real `whisper-small.en` weights and a bit-identical synthetic
/// mel (so the gate isolates the encoder math from any mel tolerance). Golden
/// values come from `tools/whisper_ref.py` (the encoder reference loop).
///
/// Live test: skips when the checkpoint is absent (CI). Point it at any
/// `weights.npz` via `KRILL_WHISPER_NPZ`, else it auto-finds the mlx-community
/// `whisper-small.en-mlx` blob in the HF cache.
final class WhisperEncoderParityTests: XCTestCase {

    /// A converted Krill whisper model dir (`tools/convert_whisper.py`
    /// output): `model.safetensors` + `config.json`. Override with
    /// `KRILL_WHISPER_DIR`, else the default `~/.krill/whisper-small.en`.
    private func modelDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        let dir: URL = env["KRILL_WHISPER_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".krill/whisper-small.en")
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path) ? dir : nil
    }

    /// Deterministic synthetic mel `[1, 3000, 80]`, reproduced exactly in the
    /// Python reference.
    private func syntheticMel() -> MLXArray {
        var mel = [Float](repeating: 0, count: 3000 * 80)
        for t in 0 ..< 3000 {
            for m in 0 ..< 80 {
                mel[t * 80 + m] = Float(0.1 * sin(0.05 * Double(t) + 0.3 * Double(m)))
            }
        }
        return MLXArray(mel, [1, 3000, 80])
    }

    func testEncoderParity() throws {
        guard let dir = modelDir() else {
            throw XCTSkip("whisper model dir not found (set KRILL_WHISPER_DIR)")
        }
        let all = try loadWeightArrays(from: dir)
        var enc = [String: MLXArray]()
        for (k, v) in all where k.hasPrefix("encoder.") {
            enc[String(k.dropFirst("encoder.".count))] = v.asType(.float32)
        }
        let model = WhisperEncoder(WhisperConfig())
        let nested = ModuleParameters.unflattened(enc.map { ($0.key, $0.value) })
        try model.update(parameters: nested, verify: [.all])
        eval(model)

        let out = model(syntheticMel())
        XCTAssertEqual(out.shape, [1, 1500, 768])

        // Global stats from the NumPy reference (fp64).
        XCTAssertEqual(MLX.mean(out).item(Float.self), -0.005018, accuracy: 5e-3)
        XCTAssertEqual(MLX.std(out).item(Float.self), 1.690891, accuracy: 5e-2)

        // Pointwise golden, enc[t, c]. Tolerance covers fp32-vs-fp64 drift
        // accumulated across 12 layers.
        let golden: [(Int, Int, Float)] = [
            (0, 0, -1.023836), (0, 1, -0.892371), (0, 100, 0.024859),
            (1, 0, -0.401704), (750, 384, 0.227545), (1499, 767, 0.833863),
            (100, 50, -0.832457), (500, 500, -0.230967),
        ]
        for (t, c, want) in golden {
            XCTAssertEqual(out[0, t, c].item(Float.self), want, accuracy: 5e-2, "enc[\(t),\(c)]")
        }
    }
}
