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
        // 8192 window, no prompt yet: everything but the reserve is available.
        XCTAssertEqual(
            TokenBudget.resolve(requested: nil, contextWindow: 8192),
            8192 - TokenBudget.contextReserve)
    }

    func testDerivedBudgetSubtractsThePrompt() {
        XCTAssertEqual(
            TokenBudget.resolve(requested: nil, contextWindow: 8192, promptTokens: 6000),
            8192 - 6000 - TokenBudget.contextReserve)
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
