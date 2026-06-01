import XCTest
import MLX
@testable import KLMCore

/// Logit-parity check for the native Mixtral runtime against mlx-lm.
///
/// Gated on `KLM_MIXTRAL_PARITY_DIR`, a directory produced by
/// `tools/verify_mixtral_parity.py` containing a tiny *quantized* Mixtral
/// checkpoint (config.json + model.safetensors) and `reference_logits.json`
/// (mlx-lm's last-token logits for a fixed token sequence). This test loads
/// the same packed weights through `loadModel` and asserts the native
/// `MixtralForCausalLM` forward matches mlx-lm.
///
/// It validates the two surfaces most likely to be wrong in the port: the
/// `block_sparse_moe` router (softmax-over-all -> top-k -> renormalize) and
/// the `gatherQuantizedMM` SwitchGLU expert dispatch. Both runtimes are MLX,
/// so parity should be ~bit-exact. Skipped when the env var is unset (the
/// fixture is generated on demand, not committed), mirroring the other
/// real-checkpoint live tests.
final class MixtralParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
    }

    func testNativeMixtralMatchesMLXLMLogits() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KLM_MIXTRAL_PARITY_DIR"] else {
            throw XCTSkip("Set KLM_MIXTRAL_PARITY_DIR (see tools/verify_mixtral_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)

        let loaded = try loadModel(from: dir)
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)

        let tokens = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])
        let logits = loaded.forward(tokens, nil)          // [1, L, V]
        let last = logits[0, ref.tokens.count - 1, 0...]  // [V]
        eval(last)
        let got = last.asArray(Float.self)

        XCTAssertEqual(got.count, ref.last_token_logits.count, "vocab size mismatch")

        // Argmax must agree (the decode-relevant invariant).
        var maxIdx = 0
        for i in 1 ..< got.count where got[i] > got[maxIdx] { maxIdx = i }
        XCTAssertEqual(maxIdx, ref.argmax,
            "Native argmax \(maxIdx) != mlx-lm argmax \(ref.argmax)")

        // Cosine similarity ~1 and small max-abs diff: same MLX kernels on the
        // same packed weights should match to fp tolerance.
        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(ref.last_token_logits[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999,
            "Native vs mlx-lm last-token logits cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-2,
            "Native vs mlx-lm max abs logit diff \(maxAbs) too large")
    }
}
