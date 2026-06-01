import XCTest
import MLX
@testable import KLMCore

/// Logit-parity check for the native OLMoE runtime against mlx-lm.
///
/// Gated on `KLM_OLMOE_PARITY_DIR`, a directory produced by
/// `tools/verify_olmoe_parity.py` (tiny quantized OLMoE +
/// `reference_logits.json`). Loads the same packed weights through
/// `loadModel` and asserts the native forward matches mlx-lm. Validates the
/// whole-projection q/k-norm attention, the router (no shared expert), and
/// the `gatherQuantizedMM` SwitchGLU. Skipped when unset.
final class OLMoEParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
    }

    func testNativeOLMoEMatchesMLXLMLogits() throws {
        guard let dirPath = ProcessInfo.processInfo.environment["KLM_OLMOE_PARITY_DIR"] else {
            throw XCTSkip("Set KLM_OLMOE_PARITY_DIR (see tools/verify_olmoe_parity.py)")
        }
        let dir = URL(fileURLWithPath: dirPath)
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)

        let loaded = try loadModel(from: dir)
        XCTAssertEqual(loaded.vocabSize, ref.vocab_size)

        let tokens = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])
        let logits = loaded.forward(tokens, nil)
        let last = logits[0, ref.tokens.count - 1, 0...]
        eval(last)
        let got = last.asArray(Float.self)
        XCTAssertEqual(got.count, ref.last_token_logits.count)

        var maxIdx = 0
        for i in 1 ..< got.count where got[i] > got[maxIdx] { maxIdx = i }
        XCTAssertEqual(maxIdx, ref.argmax,
            "Native argmax \(maxIdx) != mlx-lm argmax \(ref.argmax)")

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
