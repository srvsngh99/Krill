import XCTest
@testable import KrillEngine

/// Precedence for turning the reasoning ("thinking") channel on:
/// explicit per-call flag > KRILL_ENABLE_THINKING env > off.
final class ResolveThinkingTests: XCTestCase {
    func testExplicitFlagWins() {
        // An explicit flag (TUI toggle / interactive config) overrides the env.
        XCTAssertTrue(InferenceEngine.resolveThinking(explicit: true, env: "0"))
        XCTAssertFalse(InferenceEngine.resolveThinking(explicit: false, env: "1"))
    }

    func testEnvUsedWhenNoExplicitFlag() {
        for on in ["1", "true", "yes", "on", "TRUE", "On"] {
            XCTAssertTrue(InferenceEngine.resolveThinking(explicit: nil, env: on), "env \(on)")
        }
        for off in ["0", "false", "no", "off", ""] {
            XCTAssertFalse(InferenceEngine.resolveThinking(explicit: nil, env: off), "env \(off)")
        }
    }

    func testDefaultOffWhenNothingSet() {
        // Server / single-shot pass nil and set no env -> off (prior behavior; no
        // silent API change). Interactive callers pass their config value instead.
        XCTAssertFalse(InferenceEngine.resolveThinking(explicit: nil, env: nil))
    }
}

/// Precedence for the reasoning depth:
/// explicit per-request > KRILL_REASONING_EFFORT env > config key > nil.
final class ResolveReasoningEffortTests: XCTestCase {
    private func resolve(_ explicit: String?, _ env: String?, _ config: String?) -> String? {
        InferenceEngine.resolveReasoningEffort(explicit: explicit, env: env, config: config)
    }

    func testExplicitBeatsEnvBeatsConfig() {
        XCTAssertEqual(resolve("low", "medium", "xhigh"), "low")
        XCTAssertEqual(resolve(nil, "medium", "xhigh"), "medium")
        XCTAssertEqual(resolve(nil, nil, "xhigh"), "xhigh")
        XCTAssertNil(resolve(nil, nil, nil))
    }

    func testInvalidLevelFallsThroughAtEachLevel() {
        XCTAssertEqual(resolve("high", "medium", "xhigh"), "medium")
        XCTAssertEqual(resolve(nil, "minimal", "low"), "low")
        XCTAssertNil(resolve("high", "", "bogus"))
    }

    func testLevelsAreCaseAndSpaceInsensitive() {
        XCTAssertEqual(resolve(nil, nil, " XHigh "), "xhigh")
    }
}
