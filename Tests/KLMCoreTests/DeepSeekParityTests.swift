import XCTest
import MLX
@testable import KLMCore

/// Logit-parity check for the native DeepSeek-V2 / V2-Lite runtime against
/// mlx-lm. Gated on `KLM_DEEPSEEK_V2_PARITY_DIR`, a directory produced by
/// `tools/verify_deepseek_parity.py <dir> v2`. Validates MLA attention (low-rank
/// KV bottleneck, split rope/nope head dims), YaRN RoPE, the shared expert, the
/// `first_k_dense_replace` dense-layer prefix, the softmax/greedy router, and
/// the `gatherQuantizedMM` SwitchGLU on identical packed weights. Skipped when
/// unset.
///
/// DeepSeek-V3 uses an absorbed MLA representation (embed_q / unembed_out) the
/// native runtime does not load yet (docs/BACKLOG.md); its `noaux_tc` sigmoid
/// gating is covered structurally by `DeepSeekNativeTests`.
final class DeepSeekParityTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let last_token_logits: [Float]
        let argmax: Int
    }

    private func runParity(_ envVar: String) throws {
        guard let dirPath = ProcessInfo.processInfo.environment[envVar] else {
            throw XCTSkip("Set \(envVar) (see tools/verify_deepseek_parity.py)")
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
            "[\(envVar)] native argmax \(maxIdx) != mlx-lm argmax \(ref.argmax)")

        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(ref.last_token_logits[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999, "[\(envVar)] logits cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-2, "[\(envVar)] max abs logit diff \(maxAbs) too large")
    }

    func testNativeDeepSeekV2MatchesMLXLMLogits() throws {
        try runParity("KLM_DEEPSEEK_V2_PARITY_DIR")
    }
}
