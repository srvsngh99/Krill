import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Stage 3 gate: the native `WhisperDecoder` must (a) match a NumPy reference
/// on the real `whisper-small.en` weights (cross-attending to the encoder
/// output of a bit-identical synthetic mel) and (b) be KV-cache
/// self-consistent (token-by-token decode == full prefill). Golden values from
/// `tools/whisper_encdec_ref.py`.
///
/// Live test: skips when the converted model dir is absent (CI). See
/// `WhisperEncoderParityTests` for the `KRILL_WHISPER_DIR` override.
final class WhisperDecoderParityTests: XCTestCase {

    private func modelDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        let dir: URL = env["KRILL_WHISPER_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".krill/whisper-small.en")
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path) ? dir : nil
    }

    private func syntheticMel() -> MLXArray {
        var mel = [Float](repeating: 0, count: 3000 * 80)
        for t in 0 ..< 3000 {
            for m in 0 ..< 80 {
                mel[t * 80 + m] = Float(0.1 * sin(0.05 * Double(t) + 0.3 * Double(m)))
            }
        }
        return MLXArray(mel, [1, 3000, 80])
    }

    private func loadModel(_ dir: URL) throws -> WhisperModel {
        var weights = try loadWeightArrays(from: dir)
        weights = weights.mapValues { $0.asType(.float32) }
        let model = WhisperModel(WhisperConfig())
        let nested = ModuleParameters.unflattened(weights.map { ($0.key, $0.value) })
        try model.update(parameters: nested, verify: [.all])
        eval(model)
        return model
    }

    func testDecoderParityAndCacheConsistency() throws {
        guard let dir = modelDir() else {
            throw XCTSkip("whisper model dir not found (set KRILL_WHISPER_DIR)")
        }
        let model = try loadModel(dir)
        let audio = model.encoder(syntheticMel())          // [1, 1500, 768]

        // (a) Prefill parity against the NumPy golden.
        let tokens = MLXArray([1, 2, 3, 4, 5].map { Int32($0) }, [1, 5])
        let cache = model.newCache()
        let logits = model.decoder(tokens, audioFeatures: audio, cache: cache)
        XCTAssertEqual(logits.shape, [1, 5, 51864])
        XCTAssertEqual(MLX.argMax(logits[0, 4]).item(Int32.self), 3)

        let golden: [(Int, Int, Float)] = [
            (0, 0, -16.9853), (0, 100, -22.0169), (4, 50256, 7.2212),
            (4, 1, 4.2484), (2, 2000, -1.0073), (4, 13, 4.6849),
        ]
        for (t, vi, want) in golden {
            XCTAssertEqual(logits[0, t, vi].item(Float.self), want, accuracy: 0.2, "logit[\(t),\(vi)]")
        }

        // (b) KV-cache self-consistency: decode the same tokens one at a time;
        // the final step must match the prefill's last-position logits.
        let stepCache = model.newCache()
        var lastStep: MLXArray? = nil
        for tid in [1, 2, 3, 4, 5] {
            let tok = MLXArray([Int32(tid)], [1, 1])
            lastStep = model.decoder(tok, audioFeatures: audio, cache: stepCache)
        }
        let prefillLast = logits[0, 4].asType(.float32)
        let stepLast = lastStep![0, 0].asType(.float32)
        let maxDiff = MLX.max(MLX.abs(prefillLast - stepLast)).item(Float.self)
        XCTAssertLessThan(maxDiff, 1e-2, "incremental decode diverged from prefill")
    }
}
