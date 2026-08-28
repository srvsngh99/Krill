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

    func testSetRejectsUnsafeValue() {
        // A quote / backslash / newline can't round-trip through the quote-trimming
        // parser, so it is rejected rather than written (would corrupt the file).
        XCTAssertThrowsError(try KrillConfig.set(key: "default_model", value: "a\"b"))
        XCTAssertThrowsError(try KrillConfig.set(key: "default_model", value: "a\\b"))
        XCTAssertThrowsError(try KrillConfig.set(key: "default_model", value: "a\nb"))
    }

    func testUpsertIsSectionScoped() {
        // A bare key inside a [section] table must NOT be replaced; the global
        // key is inserted before the first section header instead.
        let existing = """
        default_model = "x"

        [server]
        default_model = "should-not-touch"
        """
        let out = KrillConfig.upsertTOML(existing, key: "default_mode", value: "agent")
        XCTAssertTrue(out.contains("default_model = \"should-not-touch\""), "section key untouched")
        // The new global key lands before the [server] header.
        let idxNew = out.range(of: "default_mode = \"agent\"")!.lowerBound
        let idxSection = out.range(of: "[server]")!.lowerBound
        XCTAssertLessThan(idxNew, idxSection, "new key inserted in the global region")
    }

    func testUpsertDoesNotReplaceKeyInsideSection() {
        let existing = """
        [server]
        default_mode = "chat"
        """
        let out = KrillConfig.upsertTOML(existing, key: "default_mode", value: "agent")
        XCTAssertTrue(out.contains("default_mode = \"chat\""), "table key left intact")
        XCTAssertTrue(out.contains("default_mode = \"agent\""), "global key added")
        // The global one is before the section header.
        XCTAssertLessThan(
            out.range(of: "default_mode = \"agent\"")!.lowerBound,
            out.range(of: "[server]")!.lowerBound)
    }

    func testWritableKeysRoundTripThroughParser() {
        // Both the new key and the legacy alias are read back by load's parser.
        let out = KrillConfig.upsertTOML("", key: "default_agent_posture", value: "accept-edits")
        var cfg = KrillConfig()
        cfg.mergeFromTOML(out)
        XCTAssertEqual(cfg.defaultAgentPermissions, "accept-edits",
                       "legacy default_agent_posture key still parses")
        var cfg2 = KrillConfig()
        cfg2.mergeFromTOML("default_agent_permissions = \"ask\"\n")
        XCTAssertEqual(cfg2.defaultAgentPermissions, "ask")
    }

    func testServerAPIKeyParsesAndIsRedacted() {
        var cfg = KrillConfig()
        cfg.mergeFromTOML("server_api_key = \"top-secret\"\n")
        XCTAssertEqual(cfg.serverAPIKey, "top-secret")
        let displayed = Dictionary(uniqueKeysWithValues: cfg.displayPairs())
        XCTAssertEqual(displayed["server_api_key"], "(set)")
        XCTAssertFalse(cfg.displayPairs().description.contains("top-secret"))
    }

    func testVoiceSettingsDefaultsAndTOMLRoundTrip() {
        var cfg = KrillConfig()
        XCTAssertEqual(cfg.voiceEngine, "apple")
        XCTAssertEqual(cfg.voiceLanguage, "auto")
        XCTAssertEqual(cfg.voiceIdentifier, "")
        XCTAssertEqual(cfg.voiceRate, 0)
        XCTAssertEqual(cfg.voiceWhisperModel, "base.en")
        XCTAssertEqual(cfg.voiceOrb, "balanced")

        let keys = [
            ("voice_engine", "whisper"), ("voice_language", "en-GB"),
            ("voice_identifier", "com.example.voice"), ("voice_rate", "0.45"),
            ("voice_whisper_model", "small.en"),
            ("voice_orb", "lively"),
        ]
        let toml = keys.reduce("") { KrillConfig.upsertTOML($0, key: $1.0, value: $1.1) }
        cfg.mergeFromTOML(toml)
        XCTAssertEqual(cfg.voiceEngine, "whisper")
        XCTAssertEqual(cfg.voiceLanguage, "en-GB")
        XCTAssertEqual(cfg.voiceIdentifier, "com.example.voice")
        XCTAssertEqual(cfg.voiceRate, 0.45)
        XCTAssertEqual(cfg.voiceWhisperModel, "small.en")
        XCTAssertEqual(cfg.voiceOrb, "lively")
        let display = Dictionary(uniqueKeysWithValues: cfg.displayPairs())
        XCTAssertEqual(display["voice_engine"], "whisper")
        XCTAssertEqual(display["voice_orb"], "lively")
        XCTAssertTrue(KrillConfig.writableKeys.contains("voice_rate"))
        XCTAssertTrue(KrillConfig.writableKeys.contains("voice_orb"))
    }

    func testShellOutputToModelDefaultsOnAndRoundTrips() {
        var cfg = KrillConfig()
        XCTAssertTrue(cfg.shellOutputToModel, "a `!` run feeds the model by default")

        cfg.mergeFromTOML(KrillConfig.upsertTOML("", key: "shell_output_to_model", value: "false"))
        XCTAssertFalse(cfg.shellOutputToModel)

        // The parser accepts the same truthy spellings as `thinking`.
        for truthy in ["true", "1", "on", "yes"] {
            var on = KrillConfig()
            on.shellOutputToModel = false
            on.mergeFromTOML("shell_output_to_model = \"\(truthy)\"\n")
            XCTAssertTrue(on.shellOutputToModel, "\(truthy) reads as on")
        }

        // Case must not matter, and must not matter DIFFERENTLY in the file
        // parser than in the TUI's live toggle - both go through parseBool.
        for spelling in ["TRUE", "True", "On", "YES", " true "] {
            var mixed = KrillConfig()
            mixed.shellOutputToModel = false
            mixed.mergeFromTOML("shell_output_to_model = \"\(spelling)\"\n")
            XCTAssertTrue(mixed.shellOutputToModel, "\(spelling) reads as on")
            XCTAssertTrue(KrillConfig.parseBool(spelling), "live toggle agrees on \(spelling)")
        }
        XCTAssertFalse(KrillConfig.parseBool("FALSE"))
        XCTAssertFalse(KrillConfig.parseBool("nonsense"))

        let display = Dictionary(uniqueKeysWithValues: cfg.displayPairs())
        XCTAssertEqual(display["shell_output_to_model"], "false")
        XCTAssertTrue(KrillConfig.writableKeys.contains("shell_output_to_model"))
    }

}
