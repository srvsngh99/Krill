import XCTest
import MLX
import KrillSampler

/// `Sampler.sampleWithLogprobs` backs OpenAI/Ollama `logprobs` (Phase 1,
/// docs/LOGPROBS_PLAN.md). The single most consequential decision it
/// implements (§4.1): the reported logprob is a log-softmax of the RAW
/// forward-pass logits - computed BEFORE penalties, temperature scaling, and
/// top-k/top-p/min-p truncation, and identically defined at temperature 0
/// (greedy). These tests fail if that ever regresses to a post-filter
/// distribution.
final class SamplerLogprobsTests: XCTestCase {
    private let vocab = 32

    /// Manual log-softmax reference, independent of the Sampler / MLX path.
    private func manualLogSoftmax(_ logits: [Float]) -> [Float] {
        let m = logits.max() ?? 0
        let sumExp = logits.reduce(Float(0)) { $0 + Foundation.expf($1 - m) }
        let logSumExp = m + Foundation.logf(sumExp)
        return logits.map { $0 - logSumExp }
    }

    func testRawLogprobMatchesManualLogSoftmaxAtGreedy() {
        var logits = [Float](repeating: 0, count: vocab)
        logits[3] = 5.0
        logits[7] = 4.0
        logits[15] = -2.0
        let expected = manualLogSoftmax(logits)

        // Greedy: temperature 0, no filters.
        let sampler = Sampler(params: .greedy)
        let (token, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), topLogprobs: 5)

        XCTAssertEqual(token, 3, "greedy must still pick the argmax token")
        XCTAssertEqual(info.logprob, expected[3], accuracy: 1e-4)
    }

    func testRawLogprobIsUnaffectedByTemperatureTopKTopP() {
        var logits = [Float](repeating: 0, count: vocab)
        logits[3] = 5.0
        logits[7] = 4.0
        logits[15] = -2.0
        let expected = manualLogSoftmax(logits)

        // A request with temperature/top-k/top-p active still reports the
        // SAME raw logprob for whichever token gets sampled - the
        // post-filter distribution must never leak into the reported value.
        let sampler = Sampler(params: SamplingParams(temperature: 1.0, topP: 1.0, topK: 0))
        let (token, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), topLogprobs: 0)
        XCTAssertEqual(info.logprob, expected[token], accuracy: 1e-4)
    }

    func testRawLogprobIgnoresActivePenalties() {
        // A heavily repetition-penalized token's REPORTED logprob must still
        // reflect the untouched raw distribution (docs/LOGPROBS_PLAN.md §4.1:
        // "penalties are decoding heuristics, not the model's belief").
        var logits = [Float](repeating: 0, count: vocab)
        logits[3] = 5.0
        let expected = manualLogSoftmax(logits)

        let sampler = Sampler(params: SamplingParams(temperature: 0.0, repetitionPenalty: 4.0))
        // Token 3 is "recent", so applyPenalties would divide its logit by 4
        // before sampling - but greedy still argmaxes the RAW logits here
        // since nothing else is anywhere near as large, and the REPORTED
        // logprob for token 3 must equal the raw log-softmax regardless.
        let (_, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), recent: [3], topLogprobs: 0)
        XCTAssertEqual(info.logprob, expected[3], accuracy: 1e-4)
    }

    func testTopLogprobsOrderedDescendingAndCorrectCount() {
        var logits = [Float](repeating: -10, count: vocab)
        logits[1] = 5.0   // rank 0
        logits[2] = 4.0   // rank 1
        logits[3] = 3.0   // rank 2
        let expected = manualLogSoftmax(logits)

        let sampler = Sampler(params: .greedy)
        let (_, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), topLogprobs: 3)

        XCTAssertEqual(info.topAlternates.count, 3)
        XCTAssertEqual(info.topAlternates.map(\.tokenId), [1, 2, 3])
        for alt in info.topAlternates {
            XCTAssertEqual(alt.logprob, expected[alt.tokenId], accuracy: 1e-4)
        }
        // Strictly descending.
        for i in 1 ..< info.topAlternates.count {
            XCTAssertGreaterThan(info.topAlternates[i - 1].logprob, info.topAlternates[i].logprob)
        }
    }

    func testTopLogprobsZeroReturnsEmptyArray() {
        var logits = [Float](repeating: 0, count: vocab)
        logits[0] = 1.0
        let sampler = Sampler(params: .greedy)
        let (_, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), topLogprobs: 0)
        XCTAssertEqual(info.topAlternates.count, 0)
    }

    func testSampleWithLogprobsChosenTokenMatchesPlainSampleForSameSeed() {
        // The ACTUAL token drawn must be identical to the existing `sample`
        // path for the same inputs - `sampleWithLogprobs` must not change
        // what gets sampled, only add reporting.
        var logits = [Float](repeating: 0, count: vocab)
        logits[9] = 10.0
        let plain = Sampler(params: .greedy)
        let withLP = Sampler(params: .greedy)
        let plainToken = plain.sample(MLXArray(logits))
        let (lpToken, _, _) = withLP.sampleWithLogprobs(MLXArray(logits), topLogprobs: 1)
        XCTAssertEqual(plainToken, lpToken)
        XCTAssertEqual(plainToken, 9)
    }

    func testRawLogprobUnaffectedWhenLogitsAlreadyFloat32AndPenaltiesMutateSharedObject() {
        // `sampleWithLogprobs` no longer round-trips through the host to get
        // an independent copy of the raw distribution (finding #2): it now
        // relies on the log-softmax graph being BUILT before
        // `applyPenalties` runs. `MLXArray(logits)` from a `[Float]` is
        // ALREADY `.float32`, so `to1D(logits).asType(.float32)` is a
        // double pass-through (`asType` short-circuits `return self` at
        // matching dtype - MLXArray.swift) and really is the SAME Swift
        // object `applyPenalties` mutates in place. This test drives that
        // exact case with penalties on MULTIPLE recent tokens (including
        // ones NOT chosen) and checks the raw logprob everywhere in the
        // top-N, not just at the sampled token, so a partial/positional
        // corruption from the scatter would be caught too.
        var logits = [Float](repeating: -10, count: vocab)
        logits[1] = 5.0
        logits[2] = 4.0
        logits[3] = 3.0
        let mlxLogits = MLXArray(logits)
        XCTAssertEqual(mlxLogits.dtype, .float32)
        let expected = manualLogSoftmax(logits)

        let sampler = Sampler(params: SamplingParams(temperature: 0.0, repetitionPenalty: 1.5))
        // Penalize tokens 1 and 2 (mild enough that token 1 - 5.0/1.5=3.33 -
        // still beats penalized token 2 - 4.0/1.5=2.67 - and unpenalized
        // token 3 at 3.0, so the SAMPLED token is unaffected) while reading
        // the raw logprob at every position in the top-3.
        let (token, _, info) = sampler.sampleWithLogprobs(
            mlxLogits, recent: [1, 2], topLogprobs: 3)
        XCTAssertEqual(token, 1, "greedy still picks the raw argmax despite the penalty")
        XCTAssertEqual(info.logprob, expected[1], accuracy: 1e-4)
        XCTAssertEqual(info.topAlternates.map(\.tokenId), [1, 2, 3])
        for alt in info.topAlternates {
            XCTAssertEqual(alt.logprob, expected[alt.tokenId], accuracy: 1e-4)
        }
    }

    func testTopNPartialSelectionMatchesFullSortReferenceOnRandomLogits() {
        // Task A (2026-09-30): `sampleWithLogprobs` switched its top-N
        // selection from a full O(V log V) argSort to an O(V) argPartition
        // + a small O(N log N) sort over just the N candidates. This test
        // pins that the new path picks the SAME set of tokens, in the same
        // descending order, as an independent full-sort reference - across
        // several random vocabularies and several N values (including N
        // equal to the whole vocab, the boundary case for argPartition's
        // `kth`).
        var rng = SystemRandomNumberGenerator()
        for trial in 0 ..< 20 {
            let v = 50 + (trial % 3) * 40 // vary vocab size a bit: 50, 90, 130
            var logits = [Float](repeating: 0, count: v)
            for i in 0 ..< v {
                // Random, effectively-unique floats - ties are not the point
                // of this test (argPartition's own docs say tie order among
                // an exact tie is undefined), just that the CHOSEN top-N set
                // and its descending order match a reference sort.
                logits[i] = Float.random(in: -20 ... 20, using: &rng)
            }
            let expectedLogSoftmax = manualLogSoftmax(logits)
            // Reference: full descending sort of (tokenId, logprob) pairs.
            let referenceOrder = (0 ..< v).sorted { expectedLogSoftmax[$0] > expectedLogSoftmax[$1] }

            for n in [0, 1, 5, 20, v] where n <= v {
                let sampler = Sampler(params: .greedy)
                let (_, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), topLogprobs: n)
                XCTAssertEqual(info.topAlternates.count, n, "trial \(trial) n=\(n) v=\(v)")
                guard n > 0 else { continue }
                let expectedTop = Array(referenceOrder.prefix(n))
                XCTAssertEqual(
                    Set(info.topAlternates.map(\.tokenId)), Set(expectedTop),
                    "trial \(trial) n=\(n) v=\(v): top-N set must match the full-sort reference")
                XCTAssertEqual(
                    info.topAlternates.map(\.tokenId), expectedTop,
                    "trial \(trial) n=\(n) v=\(v): top-N order must match the full-sort reference")
                for alt in info.topAlternates {
                    XCTAssertEqual(
                        alt.logprob, expectedLogSoftmax[alt.tokenId], accuracy: 1e-4,
                        "trial \(trial) n=\(n) v=\(v) token \(alt.tokenId)")
                }
                // Strictly descending (values are effectively-unique random
                // floats, so no tie plateau is expected here).
                for i in 1 ..< info.topAlternates.count {
                    XCTAssertGreaterThan(
                        info.topAlternates[i - 1].logprob, info.topAlternates[i].logprob,
                        "trial \(trial) n=\(n) v=\(v)")
                }
            }
        }
    }

    func testGrammarMaskDoesNotAffectReportedRawLogprob() {
        // A grammar mask forbids a token for SAMPLING purposes only; the
        // reported logprob must reflect the unmasked raw distribution
        // (docs/LOGPROBS_PLAN.md §4.1 - "before... grammar mask").
        var logits = [Float](repeating: 0, count: vocab)
        logits[3] = 5.0
        logits[7] = 4.0
        let expected = manualLogSoftmax(logits)

        var maskValues = [Float](repeating: 0, count: vocab)
        maskValues[3] = -2e9   // forbid token 3 in the grammar mask
        let mask = MLXArray(maskValues)

        let sampler = Sampler(params: .greedy)
        let (token, _, info) = sampler.sampleWithLogprobs(MLXArray(logits), mask: mask, topLogprobs: 2)
        XCTAssertEqual(token, 7, "masked token 3 must never be sampled")
        // The top alternate (rank 0 of the RAW distribution) is still token 3,
        // even though it could never be sampled under the mask.
        XCTAssertEqual(info.topAlternates.first?.tokenId, 3)
        XCTAssertEqual(info.topAlternates.first?.logprob ?? .nan, expected[3], accuracy: 1e-4)
    }
}
