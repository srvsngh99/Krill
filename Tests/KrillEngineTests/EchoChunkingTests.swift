import XCTest
import MLX
import KrillCache
import KrillSampler
import KrillRuntime
@testable import KrillEngine

/// Unit tests for `chunkedPromptLogprobs` (factored out of
/// `InferenceEngine.echoPromptLogprobs`, docs/LOGPROBS_PLAN.md Phase 3,
/// §5.4) — specifically the chunk-BOUNDARY arithmetic, which a real-model
/// run with a short prompt (everything fitting in one 512-token chunk)
/// cannot exercise at all: the running KV offset across chunks, a chunk's
/// LAST position scoring the FIRST token of the NEXT chunk, and the final
/// chunk's one-fewer `scoreCount`.
///
/// The synthetic "model" here is deliberately CONTEXT-DEPENDENT (unlike
/// `SpeculativeLogprobsTests`' context-free lookup table): its forward
/// closure reads back the FULL accumulated key sequence from the real
/// `KVCache` passed to it (`cache.update(keys:values:)` returns everything
/// ever appended to that cache, across every previous call), and predicts
/// each position's next token from the RUNNING SUM of every token id seen
/// so far — not just the current chunk's own tokens. This means the model
/// can only produce the right answer if `chunkedPromptLogprobs` (a) reuses
/// the SAME cache object across every chunk in one call (not a fresh one
/// per chunk) and (b) slices/positions each chunk correctly — exactly the
/// two properties a context-free model cannot distinguish a bug in.
final class EchoChunkingTests: XCTestCase {
    private func requireMLX() throws {
        #if os(macOS) && arch(arm64) && canImport(MLX)
        if ProcessInfo.processInfo.environment["KRILL_SKIP_MLX_TESTS"] == "1" {
            throw XCTSkip("MLX tests skipped by KRILL_SKIP_MLX_TESTS")
        }
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is not available to this test process")
        }
        #else
        throw XCTSkip("MLX-backed tests require macOS arm64 with MLX/Metal")
        #endif
    }

    private func withMLX<T>(_ body: () throws -> T) throws -> T {
        try requireMLX()
        return try Device.withDefaultDevice(.cpu) { try body() }
    }

    private static let vocab = 13

    /// A context-dependent synthetic forward: position i's logits are a
    /// clear spike at `(runningSum(tokenIds[0...i])) % vocab`, where
    /// `runningSum` is read back from the REAL KVCache's accumulated keys —
    /// so this closure alone cannot compute the right answer without the
    /// full prefix, which only arrives correctly if the caller threads the
    /// SAME cache across every chunk.
    private func syntheticForward(_ tokens: MLXArray, _ caches: [KVCacheProtocol]?) -> MLXArray {
        guard let cache = caches?.first as? KVCache else {
            XCTFail("expected a real KVCache")
            return MLXArray.zeros([1, 1, Self.vocab])
        }
        let ids = tokens.asArray(Int32.self)
        let chunkLen = ids.count
        let newK = MLXArray(ids.map { Float($0) }, [1, 1, chunkLen, 1])
        let (fullK, _) = cache.update(keys: newK, values: newK)
        let fullIds = fullK.reshaped(-1).asArray(Float.self).map { Int($0.rounded()) }
        let totalSoFar = fullIds.count
        let startGlobal = totalSoFar - chunkLen
        var runningSum = fullIds[0 ..< startGlobal].reduce(0, +)
        var rows: [Float] = []
        rows.reserveCapacity(chunkLen * Self.vocab)
        for i in 0 ..< chunkLen {
            runningSum += fullIds[startGlobal + i]
            var row = (0 ..< Self.vocab).map { Float($0) * 0.07 }  // distinct, small
            row[((runningSum % Self.vocab) + Self.vocab) % Self.vocab] = 6.0
            rows.append(contentsOf: row)
        }
        return MLXArray(rows, [1, chunkLen, Self.vocab])
    }

    /// 17 token ids (prime length, deliberately not a multiple of any tested
    /// chunk size), every value a valid index into `vocab` (the synthetic
    /// forward's chosen-token gather requires that).
    private static let tokenIds: [Int] = [2, 5, 1, 9, 0, 7, 3, 11, 4, 8, 6, 12, 1, 5, 9, 2, 10]

    private func run(chunkSize: Int, topLogprobs: Int = 3) throws -> [TokenLogprobInfo?] {
        try withMLX {
            let caches: [KVCacheProtocol] = [KVCache()]
            return chunkedPromptLogprobs(
                tokenIds: Self.tokenIds, caches: caches, chunkSize: chunkSize,
                topLogprobs: topLogprobs, forward: syntheticForward)
        }
    }

    /// Independent (no MLX, no `chunkedPromptLogprobs`) ground truth for
    /// position `k` (1-indexed the same way `infos[k]` is): predicts
    /// `tokenIds[k]` from a model row built at GLOBAL sequence position
    /// `k - 1` (the row produced right after consuming `tokenIds[k-1]`),
    /// exactly mirroring `syntheticForward`'s `runningSum = sum(tokenIds[0...k-1])`
    /// (inclusive) / `winner = runningSum % vocab` definition, but computed
    /// here in plain Swift `Double` arithmetic from `Self.tokenIds` directly -
    /// a genuine ground truth, not just "whatever `chunkedPromptLogprobs`
    /// happened to compute at another chunk size" (cross-chunk-size
    /// agreement alone cannot catch a bug that shifts EVERY chunk size's
    /// "next token" the same systematic way, e.g. scoring a position against
    /// its own token instead of the next one).
    private func referenceLogSoftmaxRow(atGlobalPosition j: Int) -> [Double] {
        let runningSum = Self.tokenIds[0 ... j].reduce(0, +)
        let winner = ((runningSum % Self.vocab) + Self.vocab) % Self.vocab
        var row = (0 ..< Self.vocab).map { Double($0) * 0.07 }
        row[winner] = 6.0
        let m = row.max()!
        let sumExp = row.reduce(0.0) { $0 + Foundation.exp($1 - m) }
        let logSumExp = m + Foundation.log(sumExp)
        return row.map { $0 - logSumExp }
    }

    func testChunkSizesAgreeWithSingleChunkReference() throws {
        let reference = try run(chunkSize: Self.tokenIds.count)  // >= length: one chunk
        XCTAssertEqual(reference.count, Self.tokenIds.count)
        XCTAssertNil(reference[0])
        for i in 1 ..< reference.count {
            XCTAssertNotNil(reference[i], "position \(i) should have a real logprob")
        }
        // "the right count (prompt length minus 1, before any BOS handling)"
        let realCount = reference.dropFirst().compactMap { $0 }.count
        XCTAssertEqual(realCount, Self.tokenIds.count - 1)

        for chunkSize in [3, 4, 7, Self.tokenIds.count + 50] {
            let got = try run(chunkSize: chunkSize)
            XCTAssertEqual(got.count, reference.count, "chunkSize=\(chunkSize): entry count")
            XCTAssertNil(got[0], "chunkSize=\(chunkSize): first entry must stay nil")
            for i in 1 ..< reference.count {
                let refInfo = try XCTUnwrap(reference[i], "reference[\(i)]")
                let gotInfo = try XCTUnwrap(got[i], "chunkSize=\(chunkSize) got[\(i)]")
                XCTAssertEqual(
                    Double(gotInfo.logprob), Double(refInfo.logprob), accuracy: 1e-5,
                    "chunkSize=\(chunkSize): sampled logprob mismatch at position \(i)")
                XCTAssertEqual(
                    gotInfo.topAlternates.count, refInfo.topAlternates.count,
                    "chunkSize=\(chunkSize): alternate count mismatch at position \(i)")
                for j in 0 ..< refInfo.topAlternates.count {
                    XCTAssertEqual(
                        gotInfo.topAlternates[j].tokenId, refInfo.topAlternates[j].tokenId,
                        "chunkSize=\(chunkSize): alternate[\(j)] token id mismatch at position \(i)")
                    XCTAssertEqual(
                        Double(gotInfo.topAlternates[j].logprob),
                        Double(refInfo.topAlternates[j].logprob), accuracy: 1e-5,
                        "chunkSize=\(chunkSize): alternate[\(j)] logprob mismatch at position \(i)")
                }
            }
        }
    }

    /// A chunk boundary that lands exactly between two tokens (chunkSize=4
    /// against a 17-token sequence: chunks are [0..4)[4..8)[8..12)[12..16)[16..17))
    /// specifically stresses "the last position of chunk k scores the FIRST
    /// token of chunk k+1" (e.g. position 3, the last of chunk 0, must score
    /// `tokenIds[4]`) and "the final chunk's scoreCount is one fewer than its
    /// length" (the last chunk here is a single token, contributing zero
    /// scored positions). Verified indirectly above via full agreement with
    /// the reference at every position; this test additionally pins the
    /// exact position count so a future change can't silently shrink or
    /// duplicate the boundary-adjacent positions.
    /// Catches bugs cross-chunk-size agreement alone cannot: a SYSTEMATIC
    /// off-by-one in which token gets scored (e.g. a position's own token
    /// instead of the next one) would still agree with itself across every
    /// chunk size, since the same wrong formula applies uniformly regardless
    /// of chunking. This test instead checks every chunk size's reported
    /// value against `referenceLogSoftmaxRow`, computed independently in
    /// plain Swift with no dependency on `chunkedPromptLogprobs` at all.
    func testEveryChunkSizeMatchesIndependentGroundTruth() throws {
        for chunkSize in [3, 4, 7, Self.tokenIds.count] {
            let got = try run(chunkSize: chunkSize)
            for k in 1 ..< Self.tokenIds.count {
                let info = try XCTUnwrap(got[k], "chunkSize=\(chunkSize) got[\(k)]")
                let refRow = referenceLogSoftmaxRow(atGlobalPosition: k - 1)
                let chosen = Self.tokenIds[k]
                XCTAssertEqual(
                    Double(info.logprob), refRow[chosen], accuracy: 1e-5,
                    "chunkSize=\(chunkSize) position \(k): chosen-token logprob vs independent ground truth")
                let sortedRef = refRow.enumerated().sorted { $0.element > $1.element }
                XCTAssertEqual(info.topAlternates.count, 3)
                for j in 0 ..< 3 {
                    XCTAssertEqual(
                        info.topAlternates[j].tokenId, sortedRef[j].offset,
                        "chunkSize=\(chunkSize) position \(k): alternate[\(j)] id vs independent ground truth")
                    XCTAssertEqual(
                        Double(info.topAlternates[j].logprob), sortedRef[j].element, accuracy: 1e-5,
                        "chunkSize=\(chunkSize) position \(k): alternate[\(j)] logprob vs independent ground truth")
                }
            }
        }
    }

    func testChunkSizeFourProducesExactlySixteenRealEntries() throws {
        let got = try run(chunkSize: 4)
        XCTAssertEqual(got.count, 17)
        let realPositions = (1 ..< got.count).filter { got[$0] != nil }
        XCTAssertEqual(realPositions, Array(1 ..< 17), "every position 1..<17 must have a real entry, none skipped or duplicated")
    }

    func testChunkSizeZeroOrNegativeMeansOneChunk() throws {
        let reference = try run(chunkSize: Self.tokenIds.count)
        let got = try run(chunkSize: 0)
        XCTAssertEqual(got.count, reference.count)
        for i in 1 ..< reference.count {
            XCTAssertEqual(
                Double(try XCTUnwrap(got[i]).logprob),
                Double(try XCTUnwrap(reference[i]).logprob), accuracy: 1e-5)
        }
    }

    func testEmptyAndSingleTokenSequences() throws {
        try withMLX {
            let empty = chunkedPromptLogprobs(
                tokenIds: [], caches: [KVCache()], chunkSize: 4,
                topLogprobs: 3, forward: syntheticForward)
            XCTAssertEqual(empty.count, 0)

            let single = chunkedPromptLogprobs(
                tokenIds: [5], caches: [KVCache()], chunkSize: 4,
                topLogprobs: 3, forward: syntheticForward)
            XCTAssertEqual(single.count, 1)
            XCTAssertNil(single[0])
        }
    }
}
