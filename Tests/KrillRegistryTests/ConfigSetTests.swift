import XCTest
@testable import KrillRegistry

final class ConfigSetTests: XCTestCase {

    func testUpsertAppendsToEmpty() {
        let out = KrillConfig.upsertTOML("", key: "default_mode", value: "agent")
        XCTAssertEqual(out, "default_mode = \"agent\"\n")
    }

    func testUpsertReplacesExistingPreservingOtherLines() {
        let existing = """
        # my config
        default_model = "gemma-4-12b"
        default_mode = "chat"
        thinking = "true"
        """
        let out = KrillConfig.upsertTOML(existing, key: "default_mode", value: "agent")
        XCTAssertTrue(out.contains("default_mode = \"agent\""))
        XCTAssertFalse(out.contains("default_mode = \"chat\""), "old value replaced in place")
        XCTAssertTrue(out.contains("# my config"), "comments preserved")
        XCTAssertTrue(out.contains("default_model = \"gemma-4-12b\""), "other keys untouched")
        XCTAssertTrue(out.contains("thinking = \"true\""))
        // Exactly one assignment for the key.
        let count = out.components(separatedBy: "default_mode = ").count - 1
        XCTAssertEqual(count, 1)
    }

    func testUpsertAppendsNewKey() {
        let existing = "default_model = \"x\"\n"
        let out = KrillConfig.upsertTOML(existing, key: "default_agent_posture", value: "auto")
        XCTAssertTrue(out.contains("default_model = \"x\""))
        XCTAssertTrue(out.contains("default_agent_posture = \"auto\""))
    }

    func testUpsertIgnoresCommentedKey() {
        // A commented-out key is not a real assignment; the new one is appended.
        let existing = "# default_mode = \"chat\"\n"
        let out = KrillConfig.upsertTOML(existing, key: "default_mode", value: "agent")
        XCTAssertTrue(out.contains("# default_mode = \"chat\""), "comment left intact")
        XCTAssertTrue(out.contains("default_mode = \"agent\""))
    }

    func testSetRejectsUnknownKey() {
        XCTAssertThrowsError(try KrillConfig.set(key: "not_a_real_key", value: "x"))
    }

    func testWritableKeysRoundTripThroughParser() {
        // A value written for default_agent_posture is read back by load's parser.
        let out = KrillConfig.upsertTOML("", key: "default_agent_posture", value: "accept-edits")
        var cfg = KrillConfig()
        cfg.mergeFromTOML(out)
        XCTAssertEqual(cfg.defaultAgentPosture, "accept-edits")
    }
}
