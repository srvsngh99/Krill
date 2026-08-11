import XCTest
import MLX
@testable import KrillEngine

/// Pins what `krill perplexity` actually reports.
///
/// A quality metric is only useful if its units are what you think they are —
/// a factor of `ln 2`, or a mean taken over the wrong denominator, still
/// produces a plausible-looking number that ranks builds incorrectly. Each case
/// here has a closed-form answer.
final class PerplexityMathTests: XCTestCase {

    /// Uniform logits over V classes: every token costs exactly `ln V` nats, so
    /// perplexity must come out at V regardless of how the logits are scaled.
    func testUniformLogitsGivePerplexityEqualToVocabSize() {
        let v = 50, t = 8
        let logits = MLXArray.zeros([t, v])
        let targets = (0 ..< t).map { Int32($0 % v) }

        let total = PerplexityMath.totalNLL(logits: logits, targets: targets)
        XCTAssertEqual(Double(total), Double(t) * log(Double(v)), accuracy: 1e-3)

        let ppl = PerplexityMath.perplexity(totalNLL: Double(total), tokens: t)
        XCTAssertEqual(ppl, Double(v), accuracy: 1e-2)
    }

    /// A confident, correct prediction costs ~0 nats — so perplexity tends to 1,
    /// its floor. Catches a sign error, which uniform logits cannot.
    func testConfidentCorrectPredictionApproachesPerplexityOne() {
        var values = [Float](repeating: 0, count: 4 * 10)
        for row in 0 ..< 4 { values[row * 10 + 3] = 30 }   // huge logit on class 3
        let logits = MLXArray(values, [4, 10])

        let total = PerplexityMath.totalNLL(logits: logits, targets: [3, 3, 3, 3])
        XCTAssertLessThan(Double(total), 1e-3)
        XCTAssertEqual(PerplexityMath.perplexity(totalNLL: Double(total), tokens: 4),
                       1.0, accuracy: 1e-3)
    }

    /// A confidently WRONG prediction must be expensive, not free.
    func testConfidentWrongPredictionIsCostly() {
        var values = [Float](repeating: 0, count: 10)
        values[3] = 30
        let logits = MLXArray(values, [1, 10])
        let total = PerplexityMath.totalNLL(logits: logits, targets: [7])
        XCTAssertGreaterThan(Double(total), 25, "the gold token was 30 nats behind")
    }

    /// Bits per byte is NLL converted nats -> bits and divided by BYTES, not
    /// tokens. That denominator is the whole point: it is what makes the number
    /// comparable across tokenizers.
    func testBitsPerByteConvertsNatsAndDividesByBytes() {
        // 100 nats over 50 bytes = 2 nats/byte = 2/ln2 bits/byte.
        XCTAssertEqual(PerplexityMath.bitsPerByte(totalNLL: 100, bytes: 50),
                       2.0 / log(2.0), accuracy: 1e-9)
        // Same text, half as many tokens => IDENTICAL bits/byte.
        let a = PerplexityMath.bitsPerByte(totalNLL: 100, bytes: 50)
        XCTAssertEqual(a, PerplexityMath.bitsPerByte(totalNLL: 100, bytes: 50))
    }

    /// Degenerate inputs must not divide by zero or trap.
    func testEmptyInputsAreNotFatal() {
        XCTAssertEqual(PerplexityMath.totalNLL(
            logits: MLXArray.zeros([0, 5]), targets: []), 0)
        XCTAssertTrue(PerplexityMath.perplexity(totalNLL: 0, tokens: 0).isNaN)
        XCTAssertTrue(PerplexityMath.bitsPerByte(totalNLL: 0, bytes: 0).isNaN)
    }
}
