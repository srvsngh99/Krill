import XCTest
@testable import KrillRegistry
import KrillObs

final class ObservabilityConfigTests: XCTestCase {
    func testDefaultsAreOnFullThirtyDaysOneGB() {
        let c = KrillConfig()
        XCTAssertTrue(c.eventLog)
        XCTAssertEqual(c.sessionLog, "full")
        XCTAssertEqual(c.logRetentionDays, 30)
        XCTAssertEqual(c.logMaxMB, 1024)
        XCTAssertEqual(c.obsSettings.sessionLog, .full)
    }

    func testTOMLKeys() {
        var c = KrillConfig()
        c.mergeFromTOML("""
        event_log = "off"
        session_log = "meta"
        log_retention_days = "7"
        log_max_mb = "256"
        """)
        XCTAssertFalse(c.eventLog)
        XCTAssertEqual(c.obsSettings.sessionLog, .meta)
        XCTAssertEqual(c.logRetentionDays, 7)
        XCTAssertEqual(c.logMaxMB, 256)
    }

    func testBadValuesAreIgnored() {
        var c = KrillConfig()
        c.mergeFromTOML("session_log = \"loud\"\nlog_retention_days = \"-3\"\nlog_max_mb = \"abc\"")
        XCTAssertEqual(c.sessionLog, "full")
        XCTAssertEqual(c.logRetentionDays, 30)
        XCTAssertEqual(c.logMaxMB, 1024)
    }

    func testEnvironmentWinsOverConfig() {
        var c = KrillConfig()
        c.mergeFromTOML("session_log = \"full\"\nevent_log = \"on\"")
        c.mergeFromEnvironment(["KRILL_SESSION_LOG": "off", "KRILL_EVENT_LOG": "off",
                                "KRILL_LOG_RETENTION_DAYS": "2", "KRILL_LOG_MAX_MB": "10"])
        XCTAssertEqual(c.obsSettings.sessionLog, .off)
        XCTAssertFalse(c.eventLog)
        XCTAssertEqual(c.logRetentionDays, 2)
        XCTAssertEqual(c.logMaxMB, 10)
    }

    func testKnobsAreSettableAndShown() {
        for k in ["event_log", "session_log", "log_retention_days", "log_max_mb"] {
            XCTAssertTrue(KrillConfig.writableKeys.contains(k), k)
        }
        let shown = Dictionary(uniqueKeysWithValues: KrillConfig().displayPairs().map { ($0.key, $0.value) })
        XCTAssertEqual(shown["session_log"], "full")
        XCTAssertEqual(shown["log_max_mb"], "1024")
    }

    func testObsHomeOverridesOnlyTheObservabilityRoot() {
        XCTAssertEqual(ObsPaths.root(environment: ["KRILL_OBS_HOME": "/tmp/x"]).path, "/tmp/x")
        XCTAssertTrue(ObsPaths.root(environment: [:]).path.hasSuffix("/.krill"))
    }
}
