import XCTest
@testable import KrillEngine

/// The policy that replaced seven unrelated per-dialect constants. These pin the
/// two properties that matter: an explicit request is never second-guessed, and
/// an absent one is derived from the context rather than from a magic number.
final class TokenBudgetTests: XCTestCase {

    // MARK: - resolve

    func testExplicitRequestIsHonouredExactly() {
        XCTAssertEqual(
            TokenBudget.resolve(requested: 2048, contextWindow: 131_072), 2048,
            "an explicit ceiling is a deliberate cost/latency decision")
    }

    func testATinyExplicitRequestIsNotRaisedToTheFloor() {
        // The floor bounds DERIVED budgets only. Silently turning `40` into
        // `256` would be the same class of bug this type removes: overriding
        // what the caller actually asked for.
        XCTAssertEqual(TokenBudget.resolve(requested: 40, contextWindow: 131_072), 40)
        XCTAssertEqual(TokenBudget.resolve(requested: 1, contextWindow: 0), 1)
    }

    func testExplicitRequestWinsOverALargerContext() {
        XCTAssertEqual(TokenBudget.resolve(requested: 512, contextWindow: 1_000_000), 512)
    }

    func testDerivesFromContextWhenUnset() {
        // 8192 window, no prompt yet: everything but the reserve is available,
        // which is under the derived ceiling.
        XCTAssertEqual(
            TokenBudget.resolve(requested: nil, contextWindow: 8192),
            8192 - TokenBudget.contextReserve)
    }

    func testDerivedBudgetSubtractsThePrompt() {
        XCTAssertEqual(
            TokenBudget.resolve(requested: nil, contextWindow: 8192, promptTokens: 6000),
            8192 - 6000 - TokenBudget.contextReserve)
    }

    func testDerivedBudgetIsCappedForAHugeContext() {
        // Gemma 4 declares 131072. Deriving "all remaining context" would be
        // ~130k tokens: hours of decode, a huge KV cache, and a batched-decoder
        // step bound big enough to trip the Metal watchdog and kill the server.
        XCTAssertEqual(
            TokenBudget.resolve(requested: nil, contextWindow: 131_072),
            TokenBudget.derivedCeiling)
        XCTAssertEqual(
            TokenBudget.resolve(requested: nil, contextWindow: 1_048_576, promptTokens: 5_000),
            TokenBudget.derivedCeiling)
    }

    func testBatchedRowsGetATighterDerivedCeiling() {
        // Batched decode multiplies GPU work by live rows; a long-running row
        // raises the chance of tripping Metal's command-buffer watchdog, which
        // surfaces as an uncaught MLX exception that kills the server.
        XCTAssertLessThan(TokenBudget.batchedDerivedCeiling, TokenBudget.derivedCeiling)
        XCTAssertEqual(
            TokenBudget.resolve(
                requested: nil, contextWindow: 131_072,
                ceiling: TokenBudget.batchedDerivedCeiling),
            TokenBudget.batchedDerivedCeiling)
        XCTAssertGreaterThan(
            TokenBudget.batchedDerivedCeiling, 512,
            "still comfortably above the 512 default it replaced")
    }

    func testAnExplicitRequestIgnoresEvenTheBatchedCeiling() {
        XCTAssertEqual(
            TokenBudget.resolve(
                requested: 20_000, contextWindow: 131_072,
                ceiling: TokenBudget.batchedDerivedCeiling),
            20_000, "an explicit budget is the user's choice on every path")
    }

    func testAnExplicitRequestIsNeverCappedByTheDerivedCeiling() {
        // Asking for more than the derived default is allowed; it is simply not
        // what you get by default.
        XCTAssertEqual(
            TokenBudget.resolve(requested: 60_000, contextWindow: 131_072), 60_000)
    }

    func testUnlimitedSentinelIsTreatedAsUnset() {
        XCTAssertEqual(
            TokenBudget.resolve(requested: TokenBudget.unlimited, contextWindow: 8192),
            TokenBudget.resolve(requested: nil, contextWindow: 8192),
            "Ollama's num_predict:-1 must mean the same as omitting the field")
    }

    func testUnknownContextFallsBackWellAboveTheOldDefault() {
        let budget = TokenBudget.resolve(requested: nil, contextWindow: 0)
        XCTAssertEqual(budget, TokenBudget.unknownContextFallback)
        XCTAssertGreaterThan(
            budget, 1024,
            "1024 was the old default; a reasoning model can spend that thinking")
    }

    func testNeverReturnsLessThanTheFloorEvenWhenContextIsExhausted() {
        // Prompt larger than the window: remaining is negative.
        let budget = TokenBudget.resolve(
            requested: nil, contextWindow: 4096, promptTokens: 9000, floor: 256)
        XCTAssertEqual(budget, 256)
        XCTAssertGreaterThan(budget, 0, "a non-positive ceiling would generate nothing")
    }

    // MARK: - isDerived

    func testIsDerivedRecognizesEveryNonPositiveForm() {
        XCTAssertTrue(TokenBudget.isDerived(nil))
        XCTAssertTrue(TokenBudget.isDerived(TokenBudget.unlimited))
        XCTAssertTrue(TokenBudget.isDerived(0))
        XCTAssertFalse(TokenBudget.isDerived(1))
        XCTAssertFalse(TokenBudget.isDerived(4096))
    }

    // MARK: - the sentinel must never reach a raw comparison

    /// The batcher decides a row is finished with `generated >= maxTokens`.
    /// An unresolved sentinel makes that true on the FIRST token (`0 >= -1`),
    /// so every batched reply comes back empty. Batch entry points must resolve
    /// before admitting a row; this pins the arithmetic that makes it fatal.
    func testUnresolvedSentinelWouldTerminateARowImmediately() {
        let generated = 0
        XCTAssertTrue(
            generated >= TokenBudget.unlimited,
            "0 >= -1 is why an unresolved sentinel must never reach the batcher")
        XCTAssertFalse(
            generated >= TokenBudget.resolve(requested: TokenBudget.unlimited, contextWindow: 8192),
            "a resolved budget must leave room to generate")
    }

    func testResolveAlwaysYieldsAPositiveCeiling() {
        // Whatever goes in - sentinel, nil, zero, an exhausted context - what
        // comes out must be safe to compare against a generated-token count.
        let cases: [(Int?, Int, Int)] = [
            (nil, 0, 0), (TokenBudget.unlimited, 0, 0), (0, 8192, 0),
            (nil, 8192, 999_999), (TokenBudget.unlimited, 4096, 4096),
            (nil, 0, 500), (-99, 32768, 100),
        ]
        for (requested, window, prompt) in cases {
            let budget = TokenBudget.resolve(
                requested: requested, contextWindow: window, promptTokens: prompt)
            XCTAssertGreaterThan(
                budget, 0,
                "resolve(\(String(describing: requested)), \(window), \(prompt)) must be positive")
        }
    }

    // MARK: - CLI value parsing

    func testParseAcceptsEveryDeriveSpelling() {
        for raw in ["auto", "AUTO", " auto ", "-1", "unlimited", "model", "context"] {
            XCTAssertEqual(
                TokenBudget.parse(raw), TokenBudget.unlimited,
                "'\(raw)' should mean derive-from-context")
        }
    }

    func testParseAcceptsAPositiveCount() {
        XCTAssertEqual(TokenBudget.parse("4096"), 4096)
        XCTAssertEqual(TokenBudget.parse("1"), 1)
    }

    func testParseRejectsNonsenseAndNonPositiveNumbers() {
        // Anything not understood must fail loudly at the flag rather than
        // silently becoming a default the user did not ask for.
        for raw in ["", "abc", "0", "-5", "3.5", "1e6"] {
            XCTAssertNil(TokenBudget.parse(raw), "'\(raw)' should be rejected")
        }
    }

    func testParsedDeriveSpellingResolvesLikeAnAbsentFlag() {
        let viaAuto = TokenBudget.resolve(
            requested: TokenBudget.parse("auto"), contextWindow: 32768)
        let viaAbsent = TokenBudget.resolve(requested: nil, contextWindow: 32768)
        XCTAssertEqual(viaAuto, viaAbsent)
    }

    // MARK: - truncation detection

    func testTruncatedWhenTheCeilingIsReachedWithoutAStopToken() {
        XCTAssertTrue(GenerationStats.truncated(generated: 1024, limit: 1024, sawStop: false))
    }

    func testNotTruncatedWhenTheModelEmittedAStopToken() {
        XCTAssertFalse(
            GenerationStats.truncated(generated: 1024, limit: 1024, sawStop: true),
            "a model that stopped cleanly on its final allowed step is not cut off")
    }

    func testNotTruncatedWhenGenerationEndedEarly() {
        XCTAssertFalse(GenerationStats.truncated(generated: 300, limit: 1024, sawStop: true))
        XCTAssertFalse(GenerationStats.truncated(generated: 300, limit: 1024, sawStop: false))
    }

    func testStatsDefaultToNotTruncated() {
        let stats = GenerationStats(
            promptTokens: 10, generatedTokens: 20, prefillTime: 0.1, decodeTime: 0.2)
        XCTAssertFalse(stats.hitTokenLimit)
    }

    func testStatsCarryTheFlagWhenSet() {
        let stats = GenerationStats(
            promptTokens: 10, generatedTokens: 1024, prefillTime: 0.1, decodeTime: 0.2,
            hitTokenLimit: true)
        XCTAssertTrue(stats.hitTokenLimit)
    }
}
