import XCTest
@testable import KrillHarness

final class AgentEnvironmentPromptTests: XCTestCase {
    func testAskDirectiveIsPresentForEveryPosture() {
        for mode in PermissionMode.allCases {
            XCTAssertTrue(
                AgentEnvironment.permissionDirectives(for: mode)
                    .contains(AgentEnvironment.askUserDirective),
                "ask_user guidance missing for \(mode.rawValue)")
        }
    }

    func testPlanSteerMatchesInitialEffectivePosture() {
        for mode in PermissionMode.allCases {
            let directives = AgentEnvironment.permissionDirectives(for: mode)
            XCTAssertEqual(
                directives.contains(AgentEnvironment.planSystemSteer),
                mode.initialEffective == .plan,
                "plan steer mismatch for \(mode.rawValue)")
        }
        XCTAssertTrue(
            AgentEnvironment.permissionDirectives(for: .adaptive)
                .contains(AgentEnvironment.adaptivePlanTail))
        XCTAssertFalse(
            AgentEnvironment.permissionDirectives(for: .plan)
                .contains(AgentEnvironment.adaptivePlanTail))
    }

    func testPlanningPromptsNameBothInteractiveTools() {
        XCTAssertTrue(AgentEnvironment.planSystemSteer.contains("ask_user"))
        XCTAssertTrue(AgentEnvironment.planSystemSteer.contains("request_execute"))
        XCTAssertTrue(AgentEnvironment.planTurnPrefix.contains("ask_user"))
        XCTAssertTrue(AgentEnvironment.planTurnPrefix.contains("request_execute"))
        XCTAssertTrue(AgentEnvironment.toolDirective.contains("ask_user"))
    }

    /// The stop-after-results directive must exempt BOTH interactive tools. With
    /// only the `ask_user` carve-out, a planning run is told to stop the moment
    /// its read-only investigation returns - so it never reaches
    /// `request_execute` and adaptive mode silently accomplishes nothing.
    func testToolDirectiveExemptsRequestExecuteAsWellAsAskUser() {
        XCTAssertTrue(
            AgentEnvironment.toolDirective.contains("request_execute"),
            "toolDirective must exempt request_execute, or planning runs stop before promoting")
        XCTAssertTrue(AgentEnvironment.toolDirective.contains("ask_user"))
    }

    /// "Stop once you have the tool results" is wrong while planning, where the
    /// turn's correct terminal action is a `request_execute` CALL. The directive
    /// is appended last, so recency made models obey it over the plan steer and
    /// end the turn narrating "I'll ask for permission" - changing nothing.
    func testPlanningPosturesGetTheDirectiveThatEndsInAToolCall() {
        for mode in PermissionMode.allCases {
            let directive = AgentEnvironment.toolDirective(for: mode)
            if mode.initialEffective == .plan {
                XCTAssertEqual(
                    directive, AgentEnvironment.planningToolDirective,
                    "\(mode.rawValue) plans, so it must not be told to stop after tool results")
                XCTAssertTrue(directive.contains("request_execute"))
                XCTAssertFalse(
                    directive.contains("do not call any more tools"),
                    "the stop-after-results clause must not reach a planning posture")
            } else {
                XCTAssertEqual(directive, AgentEnvironment.toolDirective, mode.rawValue)
            }
        }
    }

    /// The TUI's frozen seed cannot track Shift+Tab, so the per-turn prefix has
    /// to carry the "emit the call" instruction independently.
    func testPlanTurnPrefixDemandsTheCallNotAnAnnouncement() {
        XCTAssertTrue(AgentEnvironment.planTurnPrefix.contains("emit the call itself"))
    }
}
