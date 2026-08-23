import XCTest
@testable import KrillHarness

final class PermissionBoxTests: XCTestCase {
    func testAcceptAllPromotionIsRefusedWithoutMutation() {
        let box = PermissionBox(mode: .adaptive, allow: ["read_file"], deny: ["bash"])
        let before = box.policy
        XCTAssertFalse(box.promote(to: .acceptAll))
        XCTAssertEqual(box.effective, before.mode)
        XCTAssertEqual(box.policy.allow, before.allow)
        XCTAssertEqual(box.policy.deny, before.deny)
    }

    func testPromotionOnlyLeavesPlanningPosturesAndOnlyOnce() {
        for mode in [PermissionMode.ask, .acceptEdits, .acceptAll] {
            let box = PermissionBox(mode: mode)
            XCTAssertFalse(box.promote(to: .acceptEdits))
            XCTAssertEqual(box.effective, mode)
        }
        let box = PermissionBox(mode: .plan)
        XCTAssertTrue(box.promote(to: .acceptEdits))
        XCTAssertFalse(box.promote(to: .ask))
        XCTAssertEqual(box.effective, .acceptEdits)
    }

    func testHumanDemotionRearmsPromotion() {
        let box = PermissionBox(mode: .plan)
        XCTAssertTrue(box.promote(to: .acceptEdits))
        box.setPolicy(mode: .plan)
        XCTAssertTrue(box.promote(to: .ask))
        XCTAssertEqual(box.effective, .ask)
    }

    func testAdaptiveIdentityChipAndDenyListSurvivePromotion() {
        let box = PermissionBox(mode: .adaptive, deny: ["bash"])
        XCTAssertEqual(box.origin, .adaptive)
        XCTAssertEqual(box.chipLabel, "adaptive (planning)")
        XCTAssertTrue(box.promote(to: .acceptEdits))
        XCTAssertEqual(box.origin, .adaptive)
        XCTAssertEqual(box.chipLabel, "adaptive (executing)")
        XCTAssertTrue(box.policy.deny.contains("bash"))
        guard case .deny = box.policy.decision(toolName: "bash", isReadOnly: false) else {
            return XCTFail("deny list must remain authoritative after promotion")
        }
    }

    func testConcurrentPromotionAndReadsAreSafe() {
        let box = PermissionBox(mode: .adaptive)
        DispatchQueue.concurrentPerform(iterations: 1_000) { i in
            if i % 20 == 0 {
                box.setPolicy(mode: .adaptive)
                _ = box.promote(to: .acceptEdits)
            } else {
                _ = box.chipLabel
                _ = box.policy
            }
        }
        XCTAssertTrue([PermissionMode.plan, .acceptEdits].contains(box.effective))
    }
}
