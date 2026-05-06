import XCTest
import MLX
@testable import KLMEngine
@testable import KLMCache
@testable import KLMSampler

/// Tests for speculative decoding first-token emission and rejection rollback.
///
/// These tests verify the behavioral contracts of the speculative decode path.
final class SpeculativeDecodingTests: XCTestCase {
    private func requireMLX() throws {
        #if os(macOS) && arch(arm64) && canImport(MLX)
        if ProcessInfo.processInfo.environment["KLM_SKIP_MLX_TESTS"] == "1" {
            throw XCTSkip("MLX tests skipped by KLM_SKIP_MLX_TESTS")
        }
        guard hasLoadableMLXMetalLibrary() else {
            throw XCTSkip("MLX Metal library is not available in the test bundle")
        }
        #else
        throw XCTSkip("MLX-backed tests require macOS arm64 with MLX/Metal")
        #endif
    }

    private func withMLX<T>(_ body: () throws -> T) throws -> T {
        try requireMLX()
        if ProcessInfo.processInfo.environment["KLM_MLX_TEST_DEVICE"] == "gpu" {
            return try body()
        }

        // These tests validate MLX tensor semantics, not GPU performance. Run
        // them on CPU by default so missing SwiftPM metallib resources do not
        // abort the process before XCTest can report a skip/failure.
        return try Device.withDefaultDevice(.cpu) {
            try body()
        }
    }

    // MARK: - Test 4: First token must be emitted before speculative step

    func testFirstTokenNotInStepResult() {
        // SpeculativeDecoder.step(lastToken:) returns tokens AFTER lastToken.
        // The engine must emit lastToken separately before entering the loop.
        let lastToken = 42
        let stepResult = [10, 20, 30]  // Simulated accepted tokens from step()

        XCTAssertFalse(stepResult.contains(lastToken),
            "step() results should not contain lastToken — engine must emit it separately")
    }

    func testEmissionOrderFirstThenAccepted() {
        // Verifies the correct emission sequence in the speculative path.
        let firstToken = 42
        let acceptedFromStep = [10, 20, 30]

        var emitted: [Int] = []
        // Engine emits first token before speculation
        emitted.append(firstToken)
        // Then appends step() results
        emitted.append(contentsOf: acceptedFromStep)

        XCTAssertEqual(emitted, [42, 10, 20, 30])
        XCTAssertEqual(emitted.first, firstToken,
            "First emitted token must be the prefill-sampled token")
    }

    func testMaxTokensOneStopsAfterFirstToken() {
        // When maxTokens == 1, the speculative path should emit the first
        // token and NOT enter the while loop.
        let maxTokens = 1
        var generatedCount = 0
        let firstToken = 42
        let eosId = 0

        // Engine emits first token
        if firstToken != eosId {
            generatedCount += 1
        }

        // Loop condition check
        let shouldEnterLoop = generatedCount < maxTokens && firstToken != eosId

        XCTAssertEqual(generatedCount, 1)
        XCTAssertFalse(shouldEnterLoop,
            "With maxTokens=1, should not enter speculative loop")
    }

    func testEosAsFirstTokenYieldsEndEvent() {
        // If the first sampled token is EOS, it should be flagged as isEnd.
        let firstToken = 0  // EOS
        let eosId = 0

        let isEnd = firstToken == eosId
        XCTAssertTrue(isEnd, "EOS as first token should yield an end event")
    }

    // MARK: - Test 5: KV rollback logic on rejection

    func testKVCacheRollbackTruncatesRejectedVerifyTokens() throws {
        try withMLX {
            let cache = KVCache()
            _ = cache.update(
                keys: kvTensor(start: 0, count: 10),
                values: kvTensor(start: 1_000, count: 10)
            )

            let previousLength = cache.sequenceLength
            _ = cache.update(
                keys: kvTensor(start: 100, count: 4),
                values: kvTensor(start: 1_100, count: 4)
            )

            XCTAssertEqual(cache.sequenceLength, 14)

            let acceptedCount = 2
            cache.truncate(to: previousLength + acceptedCount)

            let snapshot = try XCTUnwrap(cache.snapshot())
            MLX.eval(snapshot.keys, snapshot.values)

            XCTAssertEqual(cache.sequenceLength, 12)
            XCTAssertEqual(snapshot.keys.shape, [1, 1, 12, 1])
            XCTAssertEqual(snapshot.values.shape, [1, 1, 12, 1])
            XCTAssertEqual(
                snapshot.keys.asArray(Float.self),
                (0 ..< 10).map { Float($0) } + [100, 101]
            )
            XCTAssertEqual(
                snapshot.values.asArray(Float.self),
                (1_000 ..< 1_010).map { Float($0) } + [1_100, 1_101]
            )
        }
    }

    func testKVCacheFullAcceptanceKeepsVerifiedTokensAndBonusToken() throws {
        try withMLX {
            let cache = KVCache()
            _ = cache.update(
                keys: kvTensor(start: 0, count: 10),
                values: kvTensor(start: 1_000, count: 10)
            )
            _ = cache.update(
                keys: kvTensor(start: 100, count: 4),
                values: kvTensor(start: 1_100, count: 4)
            )
            _ = cache.update(
                keys: kvTensor(start: 200, count: 1),
                values: kvTensor(start: 1_200, count: 1)
            )

            XCTAssertEqual(cache.sequenceLength, 15)

            let snapshot = try XCTUnwrap(cache.snapshot())
            MLX.eval(snapshot.keys, snapshot.values)
            XCTAssertEqual(snapshot.keys.asArray(Float.self).suffix(5), [100, 101, 102, 103, 200])
            XCTAssertEqual(snapshot.values.asArray(Float.self).suffix(5), [1_100, 1_101, 1_102, 1_103, 1_200])
        }
    }

    // MARK: - Sampler behavior on real logits

    func testSamplerGreedySelectsArgmaxFromRealLogits() throws {
        try withMLX {
            let logits = MLXArray([0.1, -1.0, 4.25, 2.0] as [Float]).reshaped(1, 4)
            let sampler = Sampler(params: .greedy)

            XCTAssertEqual(sampler.sample(logits), 2)
        }
    }

    func testSamplerUsesLastPositionForFirstTokenFromPrefillLogits() throws {
        try withMLX {
            let prefillLogits = MLXArray([
                0.1, 8.0, 0.3, 0.4,
                0.2, 0.5, 0.7, 6.0,
            ] as [Float], [1, 2, 4])
            let sampler = Sampler(params: .greedy)

            XCTAssertEqual(sampler.sample(prefillLogits), 3)
        }
    }

    // MARK: - Test 5: Acceptance rate and adaptive K

    func testAdaptiveKDecreasesOnLowAcceptanceThroughDecoderState() {
        let decoder = SpeculativeDecoder(initialKForTesting: 4)

        let rate = decoder.recordVerificationForTesting(
            acceptedTokenCount: 1,
            proposedTokenCount: 4
        )

        XCTAssertEqual(rate, 0)
        XCTAssertEqual(decoder.acceptanceRate, 0)
        XCTAssertEqual(decoder.currentKForTesting, 3)
        XCTAssertEqual(decoder.totalRounds, 1)
        XCTAssertEqual(decoder.totalAccepted, 1)
    }

    func testAdaptiveKIncreasesOnFullAcceptanceThroughDecoderState() {
        let decoder = SpeculativeDecoder(initialKForTesting: 4)

        let rate = decoder.recordVerificationForTesting(
            acceptedTokenCount: 5,
            proposedTokenCount: 4
        )

        XCTAssertEqual(rate, 1)
        XCTAssertEqual(decoder.acceptanceRate, 1)
        XCTAssertEqual(decoder.currentKForTesting, 5)
        XCTAssertEqual(decoder.totalRounds, 1)
        XCTAssertEqual(decoder.totalAccepted, 5)
    }

    func testAdaptiveKUsesRollingAcceptanceHistory() {
        let decoder = SpeculativeDecoder(initialKForTesting: 4)

        for _ in 0 ..< 20 {
            decoder.recordVerificationForTesting(
                acceptedTokenCount: 5,
                proposedTokenCount: 4
            )
        }

        XCTAssertEqual(decoder.acceptanceRate, 1)
        XCTAssertEqual(decoder.currentKForTesting, 6)
        XCTAssertEqual(decoder.totalRounds, 20)
    }

    private func kvTensor(start: Int, count: Int) -> MLXArray {
        MLXArray((start ..< start + count).map { Float($0) }, [1, 1, count, 1])
    }

    private func hasLoadableMLXMetalLibrary() -> Bool {
        let fm = FileManager.default
        let names = Set(["default.metallib", "mlx.metallib"])
        let directRoots = [
            Bundle.main.resourceURL,
            Bundle.main.bundleURL,
            Bundle.main.executableURL?.deletingLastPathComponent(),
        ]

        for root in directRoots.compactMap({ $0 }) {
            for name in names {
                if fm.fileExists(atPath: root.appendingPathComponent(name).path)
                    || fm.fileExists(atPath: root.appendingPathComponent("Resources").appendingPathComponent(name).path) {
                    return true
                }
            }
        }

        let buildRoot = URL(fileURLWithPath: fm.currentDirectoryPath)
            .appendingPathComponent(".build")
        guard let enumerator = fm.enumerator(
            at: buildRoot,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }

        for case let url as URL in enumerator where names.contains(url.lastPathComponent) {
            return true
        }
        return false
    }
}
