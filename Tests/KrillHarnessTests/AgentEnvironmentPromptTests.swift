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
}
