import XCTest
@testable import KrillObs

struct FakeProbe: SystemProbe {
    var installList: [KrillInstall] = []
    var jobs: [LaunchdJob] = []
    var ports: [Int: PortStatus] = [:]
    var disk: (free: Int64, total: Int64)? = (200 * 1_073_741_824, 460 * 1_073_741_824)
    var crashes: [URL] = []
    func installs() -> [KrillInstall] { installList }
    func launchdJobs() -> [LaunchdJob] { jobs }
    func portStatus(_ port: Int) -> PortStatus { ports[port] ?? PortStatus(port: port, listening: false) }
    func diskFree(at path: String) -> (free: Int64, total: Int64)? { disk }
    func crashReports(limit: Int) -> [URL] { crashes }
    func platform() -> String { "TestOS 1.0 arm64" }
}

final class DoctorTests: ObsTestCase {
    private func run(_ p: FakeProbe) -> DoctorReport {
        Doctor.run(version: "0.26.0", runningBinary: "/usr/local/bin/krill", settings: settings, configPort: 57455, probe: p)
    }
    private func msgs(_ r: DoctorReport, _ s: Severity) -> [String] { r.findings.filter { $0.severity == s }.map(\.message) }

    func testFlagsAnUnloadedLaunchdJobAndItsDeadPort() {
        var p = FakeProbe()
        p.installList = [KrillInstall(path: "/usr/local/bin/krill", resolved: "/usr/local/libexec/krill/krill", version: "0.26.0", codesign: "valid (adhoc)")]
        p.jobs = [
            LaunchdJob(label: "com.example.krill-llm", plist: "/L/a.plist", program: "/usr/local/bin/krill",
                       arguments: ["/usr/local/bin/krill", "serve", "--port", "57456"], loaded: false),
            LaunchdJob(label: "dev.example.krill-weekly", plist: "/L/b.plist", program: "/usr/local/bin/krill",
                       arguments: ["/usr/local/bin/krill", "serve", "--port", "57457"], loaded: true, state: "active", pid: 123, runs: 1, lastExit: "(never exited)"),
        ]
        p.ports = [57457: PortStatus(port: 57457, listening: true, healthz: "ok, model m", owner: "krill")]
        let r = run(p)
        let fails = msgs(r, .fail)
        XCTAssertTrue(fails.contains { $0.contains("com.example.krill-llm") && $0.contains("NOT LOADED") }, "\(fails)")
        XCTAssertTrue(fails.contains { $0.contains(":57456") && $0.contains("nothing is listening") }, "\(fails)")
        XCTAssertFalse(fails.contains { $0.contains("krill-weekly") })
        XCTAssertTrue(msgs(r, .ok).contains { $0.contains(":57457") && $0.contains("answers /healthz") })
        XCTAssertEqual(r.worst, .fail)
        XCTAssertTrue(r.render().contains("[FAIL]"))
    }

    func testDetectsMultipleInstallsAndVersionSkew() {
        var p = FakeProbe()
        p.installList = [
            KrillInstall(path: "/usr/local/bin/krill", resolved: "/usr/local/libexec/krill/krill", version: "0.26.0"),
            KrillInstall(path: "/opt/homebrew/bin/krill", resolved: "/opt/homebrew/Cellar/krill/0.25.9/libexec/krill", version: "0.25.9"),
        ]
        let w = msgs(run(p), .warn)
        XCTAssertTrue(w.contains { $0.contains("2 different krill installs") && $0.contains("/opt/homebrew") && $0.contains("/usr/local") }, "\(w)")
        XCTAssertTrue(w.contains { $0.contains("disagree on version") })
    }

    func testTwoSymlinksToOneBinaryAreOneInstall() {
        var p = FakeProbe()
        p.installList = [
            KrillInstall(path: "/usr/local/bin/krill", resolved: "/x/krill", version: "0.26.0"),
            KrillInstall(path: "/home/u/bin/krill", resolved: "/x/krill", version: "0.26.0"),
        ]
        XCTAssertFalse(msgs(run(p), .warn).contains { $0.contains("different krill installs") })
    }

    func testWhichInstallEachJobUses() {
        var p = FakeProbe()
        p.installList = [
            KrillInstall(path: "/usr/local/bin/krill", resolved: "/usr/local/libexec/krill/krill", version: "0.26.0"),
            KrillInstall(path: "/opt/homebrew/bin/krill", resolved: "/opt/homebrew/Cellar/krill/0.26.0/libexec/krill", version: "0.26.1"),
        ]
        p.jobs = [LaunchdJob(label: "j", plist: "/p", program: "/opt/homebrew/bin/krill", arguments: [], loaded: true, state: "active", pid: 1)]
        let text = run(p).render()
        XCTAssertTrue(text.contains("runs /opt/homebrew/bin/krill (v0.26.1)"), text)
    }

    func testCodesignKillIsReportedAndSeverityDependsOnWhetherItRecovered() {
        var p = FakeProbe()
        p.jobs = [LaunchdJob(label: "j", plist: "/p", program: "/bin/sh", arguments: [], loaded: true, state: "active", pid: 9, runs: 3, lastExit: "OS_REASON_CODESIGNING")]
        XCTAssertTrue(msgs(run(p), .warn).contains { $0.contains("code-signing kill") && $0.contains("running again") })
        p.jobs[0].pid = nil; p.jobs[0].state = "not running"
        XCTAssertTrue(msgs(run(p), .fail).contains { $0.contains("code-signing kill") && $0.contains("NOT running") })
    }

    func testInvalidCodesignOnABinaryIsAFailure() {
        var p = FakeProbe()
        p.installList = [KrillInstall(path: "/usr/local/bin/krill", resolved: "/r", version: "1", codesign: "invalid: a sealed resource is missing or invalid")]
        XCTAssertTrue(msgs(run(p), .fail).contains { $0.contains("code signature problem") })
    }

    func testLowDiskWarnsAndVeryLowFails() {
        var p = FakeProbe()
        p.disk = (14 * 1_073_741_824, 460 * 1_073_741_824)
        XCTAssertTrue(msgs(run(p), .warn).contains { $0.contains("14.0 GB free") && $0.contains("low") })
        p.disk = (3 * 1_073_741_824, 460 * 1_073_741_824)
        XCTAssertTrue(msgs(run(p), .fail).contains { $0.contains("3.0 GB free") })
        p.disk = (300 * 1_073_741_824, 460 * 1_073_741_824)
        XCTAssertTrue(msgs(run(p), .ok).contains { $0.contains("300.0 GB free") })
    }

    func testReportsLastCrashAndRecentErrors() {
        var p = FakeProbe()
        p.crashes = [URL(fileURLWithPath: "/home/u/Library/Logs/DiagnosticReports/krill-2026-10-22-014228.ips")]
        let day = ObsPaths.dayString(Date())
        write(settings.logsDir.appendingPathComponent("serve-\(day).jsonl"), """
        {"ts":"\(JSONLine.isoUTC(Date().addingTimeInterval(-60)))","event":"request","method":"POST","endpoint":"/v1/embeddings","status":500,"error":"boom"}
        {"ts":"\(JSONLine.isoUTC(Date().addingTimeInterval(-30)))","event":"unclean_exit","dead_pid":42,"version":"0.26.0","crash_reports":""}

        """)
        let r = run(p)
        let w = msgs(r, .warn)
        XCTAssertTrue(w.contains { $0.contains("unclean exit seen") && $0.contains("pid 42") })
        XCTAssertTrue(w.contains { $0.contains("krill-2026-10-22-014228.ips") })
        XCTAssertTrue(w.contains { $0.contains("1 in the last 24 hours") })
        XCTAssertTrue(r.render().contains("error=\"boom\""))
    }

    func testRenderFitsOneScreenAndIsValidJSON() throws {
        let r = run(FakeProbe())
        XCTAssertLessThan(r.render().split(separator: "\n").count, 45)
        let d = try JSONSerialization.data(withJSONObject: r.jsonObject())
        XCTAssertNotNil(try JSONSerialization.jsonObject(with: d) as? [String: Any])
    }

    func testDoctorWritesNothing() {
        _ = run(FakeProbe())
        XCTAssertFalse(FileManager.default.fileExists(atPath: settings.root.appendingPathComponent("run").path))
        XCTAssertTrue(files(tmp).isEmpty)
    }

    func testLaunchctlPrintParsing() {
        var j = LaunchdJob(label: "x", plist: "", program: "", loaded: true)
        LiveProbe.parseLaunchctl("""
        gui/501/x = {
            state = running
            pid = 56844
            runs = 3
            last exit reason = OS_REASON_CODESIGNING
        }
        """, into: &j)
        XCTAssertEqual(j.state, "running"); XCTAssertEqual(j.pid, 56844)
        XCTAssertEqual(j.runs, 3); XCTAssertEqual(j.lastExit, "OS_REASON_CODESIGNING")
    }

    func testLiveProbeNeverStartsAnything() {
        // Reads only: a port nobody listens on, and the real disk.
        let probe = LiveProbe()
        XCTAssertFalse(probe.portStatus(57499).listening)
        XCTAssertNotNil(probe.diskFree(at: tmp.path))
    }
}
