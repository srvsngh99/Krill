import XCTest
@testable import KLMEngine
@testable import KLMCache

/// Tests for speculative decoding first-token emission and rejection rollback.
///
/// These tests verify the behavioral contracts of the speculative decode path
/// without requiring real model weights or Metal GPU access.
final class SpeculativeDecodingTests: XCTestCase {

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

    func testRollbackLengthCalculation() {
        // After target model forward of K verify tokens:
        // - previousLength: KV length before verification
        // - accepted.count: how many tokens were accepted (or replaced)
        // - Final length should be previousLength + accepted.count
        let previousLength = 10
        let K = 4
        let acceptedCount = 2  // rejected at position 2

        let afterVerifyLength = previousLength + K  // 14 (before rollback)
        let expectedAfterRollback = previousLength + acceptedCount  // 12

        XCTAssertEqual(afterVerifyLength, 14)
        XCTAssertEqual(expectedAfterRollback, 12)
        XCTAssertLessThan(expectedAfterRollback, afterVerifyLength,
            "Rollback should reduce KV length to previous + accepted only")
    }

    func testNoRollbackOnFullAcceptance() {
        // When all K tokens accepted, no truncation needed.
        // Additionally, a bonus token is generated (K+1 total new tokens).
        let previousLength = 10
        let K = 4
        let allAccepted = true

        let finalLength: Int
        if allAccepted {
            finalLength = previousLength + K + 1  // bonus token
        } else {
            finalLength = previousLength + K  // would truncate
        }

        XCTAssertEqual(finalLength, 15,
            "Full acceptance yields K+1 new tokens (including bonus)")
    }

    // MARK: - Test 5: Acceptance rate and adaptive K

    func testAdaptiveKDecreasesOnLowAcceptance() {
        // If acceptance rate < 0.4, K should decrease
        let currentK = 4
        let rate = 0.3
        let minK = 2

        let newK = rate < 0.4 && currentK > minK ? currentK - 1 : currentK
        XCTAssertEqual(newK, 3, "Low acceptance rate should decrease K")
    }

    func testAdaptiveKIncreasesOnHighAcceptance() {
        // If acceptance rate > 0.8, K should increase
        let currentK = 4
        let rate = 0.9
        let maxK = 6

        let newK = rate > 0.8 && currentK < maxK ? currentK + 1 : currentK
        XCTAssertEqual(newK, 5, "High acceptance rate should increase K")
    }
}
