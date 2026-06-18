import XCTest
import MLX
@testable import KrillEngine
@testable import KrillCache
import KrillRuntime

/// Tests for n-gram (prompt-lookup) speculative decoding: the `NgramProposer`
/// matching logic (pure, no MLX) and the target-only cache rollback math.
///
/// The end-to-end byte-identical greedy parity gate (n-gram on == off) runs in
/// the benchmark / release gate against a real model; these are the deterministic
/// unit contracts.
final class NgramSpeculativeDecodingTests: XCTestCase {

    // MARK: - NgramProposer: longest-suffix matching

    func testProposesContinuationOfMatchedNgram() {
        let p = NgramProposer(config: .init(maxN: 3, minN: 1, maxDraft: 10, searchWindow: 0))
        p.reset(prompt: [5, 6, 7, 5, 6])      // suffix [5,6] matched at index 0
        XCTAssertEqual(p.propose(), [7, 5, 6])
    }

    func testNoMatchReturnsEmpty() {
        let p = NgramProposer(config: .init(maxN: 3, minN: 1, maxDraft: 10, searchWindow: 0))
        p.reset(prompt: [1, 2, 3, 4, 5])      // strictly increasing, no repeat
        XCTAssertEqual(p.propose(), [])
    }

    func testMaxDraftCapsProposalLength() {
        let p = NgramProposer(config: .init(maxN: 3, minN: 1, maxDraft: 2, searchWindow: 0))
        p.reset(prompt: [5, 6, 7, 8, 9, 5, 6])  // suffix [5,6] at index 0, continuation [7,8,9,...]
        XCTAssertEqual(p.propose(), [7, 8])      // capped to 2
    }

    func testPrefersLongestMatch() {
        // [9,1,2] occurs once at index 0; [1,2] also matches the shorter window,
        // but the longest (n=3) match should win.
        let p = NgramProposer(config: .init(maxN: 3, minN: 1, maxDraft: 10, searchWindow: 0))
        p.reset(prompt: [9, 1, 2, 7, 0, 9, 1, 2])  // suffix [9,1,2] matched at index 0
        XCTAssertEqual(p.propose(), [7, 0, 9, 1, 2])
    }

    func testChoosesMostRecentOccurrence() {
        // Single-token suffix [1]; the rightmost earlier [1] is at index 2, so the
        // continuation must start from index 3, not from the index-0 occurrence.
        let p = NgramProposer(config: .init(maxN: 1, minN: 1, maxDraft: 10, searchWindow: 0))
        p.reset(prompt: [1, 9, 1, 8, 1])
        XCTAssertEqual(p.propose(), [8, 1])
    }

    func testTruncatesProposalAtEOS() {
        let p = NgramProposer(
            config: .init(maxN: 3, minN: 1, maxDraft: 10, searchWindow: 0), eosIds: [8])
        p.reset(prompt: [5, 6, 7, 8, 5, 6])    // continuation [7,8,...] -> stop after EOS 8
        XCTAssertEqual(p.propose(), [7, 8])
    }

    func testSearchWindowBoundsMatchRegion() {
        // The only earlier [5,6] is at the very start; a small window excludes it.
        var prompt = [5, 6]
        prompt.append(contentsOf: Array(repeating: 0, count: 10))
        prompt.append(contentsOf: [5, 6])      // L = 14, suffix [5,6] at 12..13
        let windowed = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 4, searchWindow: 4))
        windowed.reset(prompt: prompt)
        XCTAssertEqual(windowed.propose(), [], "match at index 0 is outside the 4-token window")

        let whole = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 4, searchWindow: 0))
        whole.reset(prompt: prompt)
        // suffix [5,6] matches index 0; continuation = history[2..6] = four 0s.
        XCTAssertEqual(whole.propose(), [0, 0, 0, 0],
                       "whole-history search finds the index-0 match")
    }

    func testAppendExtendsHistoryAndChangesProposal() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 10, searchWindow: 0))
        p.reset(prompt: [5, 6, 7])
        XCTAssertEqual(p.propose(), [])         // no repeat yet
        p.append([5, 6])                        // now suffix [5,6] matches index 0
        XCTAssertEqual(p.history, [5, 6, 7, 5, 6])
        XCTAssertEqual(p.propose(), [7, 5, 6])
    }

    func testAdaptiveCapShrinksOnRejectionAndGrowsOnAcceptance() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 16, searchWindow: 0))
        p.reset(prompt: [1, 2, 3])
        XCTAssertEqual(p.effectiveCap, 16, "starts optimistic at maxDraft")

        // Repeated zero-acceptance rounds decay the cap toward 1.
        for _ in 0 ..< 20 { p.recordOutcome(acceptedDraft: 0, proposed: 8) }
        XCTAssertEqual(p.effectiveCap, 1, "collapsed acceptance shrinks cap to 1 (floor)")

        // Full-acceptance rounds grow it back by one each.
        p.recordOutcome(acceptedDraft: 4, proposed: 4)
        p.recordOutcome(acceptedDraft: 5, proposed: 5)
        XCTAssertEqual(p.effectiveCap, 3, "grows by one per fully-accepted round")

        // A partial-accept clamps to just past where it broke.
        p.recordOutcome(acceptedDraft: 1, proposed: 3)
        XCTAssertEqual(p.effectiveCap, 2)
    }

    func testAdaptiveCapBoundsProposalLength() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 16, searchWindow: 0))
        p.reset(prompt: [5, 6, 7, 8, 9, 10, 11, 5, 6])  // suffix [5,6] -> long continuation
        for _ in 0 ..< 20 { p.recordOutcome(acceptedDraft: 0, proposed: 4) }  // collapse to cap=1
        XCTAssertEqual(p.propose(), [7], "shrunk cap limits the proposal to one token")
    }

    func testResetRestoresOptimisticCap() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0))
        p.reset(prompt: [1, 2])
        for _ in 0 ..< 10 { p.recordOutcome(acceptedDraft: 0, proposed: 4) }
        XCTAssertEqual(p.effectiveCap, 1)
        p.reset(prompt: [1, 2])
        XCTAssertEqual(p.effectiveCap, 8, "reset restores the optimistic cap")
    }

    // MARK: - Stall monitor (auto-disable / handoff trigger)

    func testStallMonitorLatchesOnSustainedNonEcho() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0,
                                            monitorWindow: 4, stallThreshold: 0.30))
        p.reset(prompt: [1, 2])
        XCTAssertFalse(p.stalled, "starts unstalled")
        // Three no-match rounds: window not yet full, no verdict.
        p.recordRound(extraTokens: 0)
        p.recordRound(extraTokens: 0)
        p.recordRound(extraTokens: 0)
        XCTAssertFalse(p.stalled, "partial window does not trip the monitor")
        // Fourth fills the window at avg 0.0 < 0.30 -> latched.
        p.recordRound(extraTokens: 0)
        XCTAssertTrue(p.stalled, "a full window averaging below threshold latches stalled")
    }

    func testStallMonitorStaysClearWhenLookupPaysOff() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0,
                                            monitorWindow: 4, stallThreshold: 0.30))
        p.reset(prompt: [1, 2])
        // Mixed echo: a couple of no-match rounds amid productive ones keeps the
        // windowed average above threshold, so the monitor never trips.
        for r in [3, 0, 2, 0, 4, 0, 3, 0] { p.recordRound(extraTokens: r) }
        XCTAssertFalse(p.stalled, "productive lookup keeps the average above threshold")
    }

    func testStallMonitorIsSticky() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0,
                                            monitorWindow: 2, stallThreshold: 0.30))
        p.reset(prompt: [1, 2])
        p.recordRound(extraTokens: 0)
        p.recordRound(extraTokens: 0)
        XCTAssertTrue(p.stalled)
        // A later burst of perfect rounds does NOT un-stall it within a generation.
        for _ in 0 ..< 10 { p.recordRound(extraTokens: 8) }
        XCTAssertTrue(p.stalled, "stalled is sticky until reset()")
        // reset() clears it for the next generation.
        p.reset(prompt: [1, 2])
        XCTAssertFalse(p.stalled)
    }

    func testFastBailOnConsecutiveMisses() {
        // maxConsecutiveMisses trips well before the (larger) window would.
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0,
                                            monitorWindow: 48, maxConsecutiveMisses: 6))
        p.reset(prompt: [1, 2])
        for _ in 0 ..< 5 { p.recordRound(extraTokens: 0) }
        XCTAssertFalse(p.stalled, "5 misses < threshold of 6")
        p.recordRound(extraTokens: 0)
        XCTAssertTrue(p.stalled, "6 consecutive misses fast-bails before the 48 window")
    }

    func testFastBailResetByProductiveRound() {
        // A productive round resets the consecutive-miss run, so dense echo with
        // occasional gaps never fast-bails.
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0,
                                            monitorWindow: 48, maxConsecutiveMisses: 6))
        p.reset(prompt: [1, 2])
        for _ in 0 ..< 5 { p.recordRound(extraTokens: 0) }
        p.recordRound(extraTokens: 4)            // echo match resets the run
        for _ in 0 ..< 5 { p.recordRound(extraTokens: 0) }
        XCTAssertFalse(p.stalled, "the productive round reset the miss counter")
    }

    func testStallMonitorDisabledWhenWindowZero() {
        let p = NgramProposer(config: .init(maxN: 2, minN: 1, maxDraft: 8, searchWindow: 0,
                                            monitorWindow: 0, stallThreshold: 0.30))
        p.reset(prompt: [1, 2])
        for _ in 0 ..< 100 { p.recordRound(extraTokens: 0) }
        XCTAssertFalse(p.stalled, "monitorWindow == 0 disables the monitor entirely")
    }

    func testEmptyAndShortHistory() {
        let p = NgramProposer(config: .init(maxN: 3, minN: 1, maxDraft: 10, searchWindow: 0))
        p.reset(prompt: [])
        XCTAssertEqual(p.propose(), [])
        p.reset(prompt: [42])                   // single token, no earlier occurrence
        XCTAssertEqual(p.propose(), [])
    }

    // MARK: - Target-only cache rollback (n-gram has no draft cache)

    func testNgramRejectionTruncatesOnlyTargetCacheToAcceptedPrefix() throws {
        try withMLX {
            let cache = KVCache()
            _ = cache.update(
                keys: kvTensor(start: 0, count: 8),
                values: kvTensor(start: 1_000, count: 8))
            let previousLength = cache.sequenceLength    // 8

            // Verify wrote k=5 rows; reject after accepting 2 (1 matched + 1 replacement).
            _ = cache.update(
                keys: kvTensor(start: 100, count: 5),
                values: kvTensor(start: 1_100, count: 5))
            XCTAssertEqual(cache.sequenceLength, 13)

            let acceptedCount = 2
            cache.truncate(to: previousLength + acceptedCount)

            let snapshot = try XCTUnwrap(cache.snapshot())
            MLX.eval(snapshot.keys, snapshot.values)
            XCTAssertEqual(cache.sequenceLength, 10)
            XCTAssertEqual(
                snapshot.keys.asArray(Float.self),
                (0 ..< 8).map { Float($0) } + [100, 101])
        }
    }

    // MARK: - Helpers

    private func kvTensor(start: Int, count: Int) -> MLXArray {
        MLXArray((start ..< start + count).map { Float($0) }, [1, 1, count, 1])
    }

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
        if ProcessInfo.processInfo.environment["KRILL_MLX_TEST_DEVICE"] == "gpu" {
            return try body()
        }
        return try Device.withDefaultDevice(.cpu) { try body() }
    }
}
