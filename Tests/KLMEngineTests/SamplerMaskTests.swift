import XCTest
import MLX
import KLMSampler

/// Verifies the optional grammar logit `mask:` on `Sampler` composes
/// correctly with every sampling mode: a masked-out token must never be
/// chosen, regardless of greedy / temperature / top-p / penalties — and
/// the unmasked path (mask == nil) must be unaffected.
final class SamplerMaskTests: XCTestCase {

    /// Additive mask that allows only the ids in `allow` (bias 0) and
    /// forbids the rest (bias -1e9), matching `JSONTokenMask`'s convention.
    private func mask(vocab: Int, allow: Set<Int>) -> MLXArray {
        var bias = [Float](repeating: -1e9, count: vocab)
        for id in allow { bias[id] = 0 }
        return MLXArray(bias)
    }

    func testGreedyRespectsMask() {
        // Token 5 has the highest logit but is masked out; the only allowed
        // token is 2, so greedy must pick 2.
        var logits = [Float](repeating: 0, count: 8)
        logits[5] = 100
        logits[2] = 1
        let s = Sampler(params: .greedy)
        let m = mask(vocab: 8, allow: [2])
        XCTAssertEqual(s.sample(MLXArray(logits), mask: m), 2)
    }

    func testGreedyUnmaskedUnaffected() {
        var logits = [Float](repeating: 0, count: 8)
        logits[5] = 100
        let s = Sampler(params: .greedy)
        // No mask → byte-identical to before: argmax wins.
        XCTAssertEqual(s.sample(MLXArray(logits)), 5)
    }

    func testTemperatureNeverPicksMasked() {
        // Highest-logit token (7) is masked; sampling many times under a
        // permissive temperature (top-p off) must never return it. Allowed
        // tokens use distinct logits to keep the nucleus math unambiguous.
        var logits = [Float](repeating: 0, count: 10)
        logits[7] = 50
        logits[1] = 6; logits[3] = 5; logits[5] = 4
        let allow: Set<Int> = [1, 3, 5]
        let m = mask(vocab: 10, allow: allow)
        let s = Sampler(params: SamplingParams(temperature: 1.2, seed: 42))
        for _ in 0 ..< 200 {
            let tok = s.sample(MLXArray(logits), mask: m)
            XCTAssertTrue(allow.contains(tok), "picked forbidden token \(tok)")
        }
    }

    func testTopPNeverPicksMasked() {
        // Same invariant under nucleus sampling, with distinct allowed
        // logits so top-p selects a well-defined nucleus.
        var logits = [Float](repeating: 0, count: 10)
        logits[7] = 50              // highest overall, but forbidden
        logits[2] = 8; logits[4] = 7; logits[6] = 3
        let allow: Set<Int> = [2, 4, 6]
        let m = mask(vocab: 10, allow: allow)
        let s = Sampler(params: SamplingParams(temperature: 1.0, topP: 0.9, seed: 7))
        for _ in 0 ..< 200 {
            let tok = s.sample(MLXArray(logits), mask: m)
            XCTAssertTrue(allow.contains(tok), "picked forbidden token \(tok)")
        }
    }

    func testPenaltiesComposeWithMask() {
        // Penalty-aware path (recent: non-empty + repetitionPenalty != 1)
        // must still honor the mask.
        var logits = [Float](repeating: 0, count: 8)
        logits[6] = 100   // highest, but forbidden
        logits[1] = 2
        logits[4] = 1
        let allow: Set<Int> = [1, 4]
        let m = mask(vocab: 8, allow: allow)
        let s = Sampler(params: SamplingParams(repetitionPenalty: 1.3))
        for _ in 0 ..< 50 {
            let tok = s.sample(MLXArray(logits), recent: [1, 1, 4], mask: m)
            XCTAssertTrue(allow.contains(tok), "penalty path picked forbidden \(tok)")
        }
    }

    func testSingleAllowedTokenAlwaysWins() {
        // When the grammar leaves exactly one legal token, every mode must
        // converge on it.
        var logits = [Float](repeating: 10, count: 16)
        logits[0] = -5  // even the only-allowed token having a low logit
        let m = mask(vocab: 16, allow: [0])
        for params in [SamplingParams.greedy,
                       SamplingParams(temperature: 1.0, topP: 0.9, seed: 7),
                       SamplingParams(temperature: 0.8, topK: 5, seed: 7)] {
            let s = Sampler(params: params)
            XCTAssertEqual(s.sample(MLXArray(logits), mask: m), 0)
        }
    }
}
