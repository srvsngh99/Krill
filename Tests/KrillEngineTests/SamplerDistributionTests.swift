import XCTest
import MLX
import KrillSampler

/// Regression guard: temperature sampling must actually follow the model's
/// distribution.
///
/// `MLXRandom.categorical` takes LOGITS and softmaxes internally. The sampler
/// used to hand it `softmax(scaled)` - probabilities - so it computed
/// `softmax(softmax(logits))`. Probabilities live in [0, 1], so every token
/// collapsed into `exp([0, 1]) = [1, e]` and the draw went nearly UNIFORM over
/// the vocabulary. Worse, the top-k/top-p/min-p filters set rejected logits to
/// -1e9, which softmaxes to exactly 0 and then re-weights to `exp(0) = 1` -
/// identical weight to a token the model gave 0.9 probability (`exp(0.9)`
/// = 2.46). Every model emitted fluent-looking garbage at any temperature > 0
/// while greedy stayed correct, and the effect grew with vocabulary size.
///
/// These tests fail loudly on that shape: with a sharply peaked distribution a
/// correct sampler picks the peak nearly always, and a filtered-out token must
/// never appear.
final class SamplerDistributionTests: XCTestCase {

    /// A large vocab is what made the old bug obvious: the uniform mass scales
    /// with vocab size while the intended peak does not.
    private let vocab = 4096

    /// Logits with one dominant token (~99.9% of the mass under softmax).
    private func peaked(at id: Int) -> [Float] {
        var logits = [Float](repeating: 0, count: vocab)
        logits[id] = 20
        return logits
    }

    func testTemperatureSamplingFollowsPeak() {
        let peak = 1234
        let sampler = Sampler(params: SamplingParams(temperature: 0.6))
        let logits = MLXArray(peaked(at: peak))

        var hits = 0
        for _ in 0 ..< 100 where sampler.sample(logits) == peak { hits += 1 }
        // The peak holds ~99.9% of the probability mass at temperature 0.6.
        // The double-softmax bug scored ~0 here (uniform over 4096 tokens).
        XCTAssertGreaterThan(hits, 90,
            "temperature sampling picked the dominant token only \(hits)/100 times "
            + "- the distribution is being flattened")
    }

    func testTopKExcludesFilteredTokens() {
        // Two plausible tokens; everything else must be unreachable under top-k 2.
        var logits = [Float](repeating: 0, count: vocab)
        logits[10] = 5
        logits[20] = 4.5
        let sampler = Sampler(params: SamplingParams(temperature: 1.0, topK: 2))
        let arr = MLXArray(logits)

        for _ in 0 ..< 100 {
            let picked = sampler.sample(arr)
            XCTAssertTrue(picked == 10 || picked == 20,
                "top-k 2 drew \(picked), which is outside the top-2 set")
        }
    }

    func testTopPExcludesTail() {
        // Token 7 alone exceeds p=0.9, so nucleus sampling must always pick it.
        var logits = [Float](repeating: 0, count: vocab)
        logits[7] = 15
        let sampler = Sampler(params: SamplingParams(temperature: 1.0, topP: 0.9))
        let arr = MLXArray(logits)

        for _ in 0 ..< 50 {
            XCTAssertEqual(sampler.sample(arr), 7,
                "top-p 0.9 drew a tail token the nucleus excludes")
        }
    }

    /// Sampling must remain sensitive to relative logit differences rather than
    /// collapsing toward uniform: a clearly-better token should dominate a
    /// clearly-worse one over many draws.
    func testRelativeOrderingIsPreserved() {
        var logits = [Float](repeating: -30, count: vocab)
        logits[100] = 3.0
        logits[200] = 0.0   // ~e^3 = 20x less likely than token 100
        let sampler = Sampler(params: SamplingParams(temperature: 1.0))
        let arr = MLXArray(logits)

        var a = 0, b = 0
        for _ in 0 ..< 300 {
            let t = sampler.sample(arr)
            if t == 100 { a += 1 } else if t == 200 { b += 1 }
        }
        XCTAssertGreaterThan(a, b * 3,
            "token 100 should dominate token 200 (~20:1); got \(a):\(b)")
    }
}
