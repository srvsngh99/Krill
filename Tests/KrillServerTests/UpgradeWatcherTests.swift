import XCTest
import Foundation
@testable import KrillServer

/// Self-restart on upgrade: change detection, stability/probe gates, deferral while busy, and the
/// launchd-vs-terminal decision. All world access is faked except the stat/symlink tests.
final class UpgradeWatcherTests: XCTestCase {
    typealias FP = UpgradeWatcher.Fingerprint

    final class Fake: @unchecked Sendable {
        let l = NSLock()
        var fp: FP? = FP(resolved: "/c/0.26.1/bin/krill", inode: 1, mtime: 100, size: 10)
        var probe: UpgradeWatcher.Probe = .ok(version: "0.27.0")
        var probes = 0
        var pending = 0
        var clock = Date(timeIntervalSince1970: 1_000)
        var activity = Date(timeIntervalSince1970: 0)
    }

    func make(_ f: Fake, mayRestart: Bool = true, current: String = "0.26.1") -> UpgradeWatcher {
        UpgradeWatcher(
            currentVersion: current,
            settings: .init(interval: 60, quiet: 10, mayRestart: mayRestart),
            fingerprint: { f.fp },
            probe: { _ in f.probes += 1; return f.probe },
            pending: { f.pending },
            now: { f.clock },
            lastActivity: { f.activity })
    }

    let upgraded = FP(resolved: "/c/0.27.0/bin/krill", inode: 7, mtime: 200, size: 11)

    func testUnchangedBinaryNeverProbes() async {
        let f = Fake(), w = make(f)
        for _ in 0..<3 { let o = await w.tick(); XCTAssertEqual(o, .unchanged) }
        XCTAssertEqual(f.probes, 0)
    }

    func testBrewStyleSymlinkRepointNeedsOneStableIntervalThenRestarts() async {
        let f = Fake(), w = make(f)
        f.fp = upgraded
        var o = await w.tick(); XCTAssertEqual(o, .settling)
        XCTAssertEqual(f.probes, 0)
        o = await w.tick(); XCTAssertEqual(o, .restart(from: "0.26.1", to: "0.27.0"))
    }

    func testInstallShStyleInPlaceReplaceIsDetectedByInodeAndMtime() async {
        let f = Fake(), w = make(f)
        f.fp = FP(resolved: "/usr/local/libexec/krill/krill", inode: 1, mtime: 100, size: 10)
        let w2 = make(f)                                  // baseline = this file
        f.fp = FP(resolved: "/usr/local/libexec/krill/krill", inode: 2, mtime: 300, size: 10)   // rm + cp
        var o = await w2.tick(); XCTAssertEqual(o, .settling)
        o = await w2.tick(); XCTAssertEqual(o, .restart(from: "0.26.1", to: "0.27.0"))
        _ = w
    }

    func testFileStillBeingWrittenKeepsSettling() async {
        let f = Fake(), w = make(f)
        f.fp = upgraded
        var o = await w.tick(); XCTAssertEqual(o, .settling)
        f.fp = FP(resolved: upgraded.resolved, inode: 7, mtime: 201, size: 5_000)   // grew since last look
        o = await w.tick(); XCTAssertEqual(o, .settling)
        XCTAssertEqual(f.probes, 0)
    }

    func testMissingBinaryMidSwapIsIgnored() async {
        let f = Fake(), w = make(f)
        f.fp = nil
        let o = await w.tick(); XCTAssertEqual(o, .unchanged)
    }

    func testBrokenBinaryIsRejectedOnceAndNotReprobed() async {
        let f = Fake(), w = make(f)
        f.fp = upgraded
        f.probe = .rejected(reason: "codesign --verify failed (exit 1)")
        _ = await w.tick()
        var o = await w.tick(); XCTAssertEqual(o, .ignored(reason: "codesign --verify failed (exit 1)"))
        o = await w.tick(); XCTAssertEqual(o, .unchanged)
        XCTAssertEqual(f.probes, 1)
        // A later, good file at a new fingerprint is considered afresh.
        f.fp = FP(resolved: upgraded.resolved, inode: 8, mtime: 300, size: 12)
        f.probe = .ok(version: "0.27.0")
        _ = await w.tick()
        o = await w.tick(); XCTAssertEqual(o, .restart(from: "0.26.1", to: "0.27.0"))
    }

    func testTimeoutIsRetriedNotRejected() async {
        let f = Fake(), w = make(f)
        f.fp = upgraded
        f.probe = .retry(reason: "codesign --verify timed out")
        _ = await w.tick()
        var o = await w.tick(); XCTAssertEqual(o, .inconclusive(reason: "codesign --verify timed out"))
        o = await w.tick(); XCTAssertEqual(o, .inconclusive(reason: "codesign --verify timed out"))
        f.probe = .ok(version: "0.27.0")
        o = await w.tick(); XCTAssertEqual(o, .restart(from: "0.26.1", to: "0.27.0"))
        XCTAssertEqual(f.probes, 3)
    }

    func testSameVersionReinstallIsNotAnUpgrade() async {
        let f = Fake(), w = make(f)
        f.fp = upgraded
        f.probe = .ok(version: "0.26.1")
        _ = await w.tick()
        var o = await w.tick(); XCTAssertEqual(o, .sameVersion)
        o = await w.tick(); XCTAssertEqual(o, .unchanged)       // adopted as the new baseline
    }

    func testDefersWhileRequestsInFlightThenRestartsWhenIdleAndQuiet() async {
        let f = Fake(), w = make(f)
        f.fp = upgraded
        _ = await w.tick()
        f.pending = 2
        var o = await w.tick(); XCTAssertEqual(o, .deferred(version: "0.27.0", pending: 2, quietFor: 1_000))
        f.pending = 0
        f.activity = f.clock.addingTimeInterval(-3)            // request 3 s ago: inside the 10 s quiet period
        o = await w.tick()
        guard case .deferred(_, let n, let q) = o else { return XCTFail("\(o)") }
        XCTAssertEqual(n, 0); XCTAssertEqual(q, 3, accuracy: 0.001)
        f.clock = f.clock.addingTimeInterval(8)                // now 11 s quiet
        o = await w.tick(); XCTAssertEqual(o, .restart(from: "0.26.1", to: "0.27.0"))
    }

    func testTerminalServerOnlyGivesNoticeOnce() async {
        let f = Fake(), w = make(f, mayRestart: false)
        f.fp = upgraded
        _ = await w.tick()
        var o = await w.tick(); XCTAssertEqual(o, .notice(version: "0.27.0"))
        o = await w.tick(); XCTAssertEqual(o, .notice(version: "0.27.0"))
        XCTAssertEqual(f.probes, 1)
    }

    // MARK: launchd vs terminal, knobs

    func testLaunchdLabelNeedsJobLabelAndPid1Parent() {
        let job = ["XPC_SERVICE_NAME": "dev.example.krill-weekly"]
        XCTAssertEqual(UpgradeWatcher.launchdLabel(environment: job, parentPID: 1), "dev.example.krill-weekly")
        XCTAssertNil(UpgradeWatcher.launchdLabel(environment: job, parentPID: 4242))          // spawned from a shell
        XCTAssertNil(UpgradeWatcher.launchdLabel(environment: ["XPC_SERVICE_NAME": "0"], parentPID: 1))
        XCTAssertNil(UpgradeWatcher.launchdLabel(environment: ["XPC_SERVICE_NAME": "application.com.apple.Terminal.1"], parentPID: 1))
        XCTAssertNil(UpgradeWatcher.launchdLabel(environment: [:], parentPID: 1))
    }

    func testKnobEnvBeatsConfig() {
        XCTAssertTrue(UpgradeWatcher.restartEnabled(config: true, environment: [:]))
        XCTAssertFalse(UpgradeWatcher.restartEnabled(config: false, environment: [:]))
        XCTAssertFalse(UpgradeWatcher.restartEnabled(config: true, environment: ["KRILL_AUTO_RESTART_ON_UPGRADE": "0"]))
        XCTAssertTrue(UpgradeWatcher.restartEnabled(config: false, environment: ["KRILL_AUTO_RESTART_ON_UPGRADE": "true"]))
        XCTAssertEqual(UpgradeWatcher.doubleEnv("K", default: 60, environment: ["K": "0.5"]), 0.5)
        XCTAssertEqual(UpgradeWatcher.doubleEnv("K", default: 60, environment: ["K": "x"]), 60)
    }

    func testParseVersion() {
        XCTAssertEqual(UpgradeWatcher.parseVersion("Krill v0.27.0\nBy the Krill authors"), "0.27.0")
        XCTAssertNil(UpgradeWatcher.parseVersion("garbage"))
        XCTAssertNil(UpgradeWatcher.parseVersion(""))
    }

    func testInvocationPathIsAbsoluteNotResolved() {
        XCTAssertEqual(UpgradeWatcher.invocationPath(argv0: "/opt/homebrew/bin/krill", cwd: "/x", path: ""), "/opt/homebrew/bin/krill")
        XCTAssertEqual(UpgradeWatcher.invocationPath(argv0: "./b/krill", cwd: "/x", path: ""), "/x/b/krill")
        XCTAssertEqual(UpgradeWatcher.invocationPath(argv0: "krill", cwd: "/x", path: "/a:/b", isExecutable: { $0 == "/b/krill" }), "/b/krill")
        XCTAssertNil(UpgradeWatcher.invocationPath(argv0: "krill", cwd: "/x", path: "/a", isExecutable: { _ in false }))
    }

    // MARK: real stat on fake files and symlinks

    func testStatFingerprintFollowsSymlinkRepointAndInPlaceReplace() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("upg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let a = dir.appendingPathComponent("a"), b = dir.appendingPathComponent("b"), link = dir.appendingPathComponent("krill").path
        try "one".write(to: a, atomically: true, encoding: .utf8)
        try "two-longer".write(to: b, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: a.path)
        let f1 = try XCTUnwrap(UpgradeWatcher.statFingerprint(link))
        XCTAssertTrue(f1.resolved.hasSuffix("/a") && !f1.resolved.hasSuffix("krill"))   // resolved target, not the link
        XCTAssertEqual(UpgradeWatcher.statFingerprint(link), f1)                       // stable
        try FileManager.default.removeItem(atPath: link)
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: b.path)
        let f2 = try XCTUnwrap(UpgradeWatcher.statFingerprint(link))
        XCTAssertNotEqual(f1, f2)                                                      // brew: symlink repointed
        try FileManager.default.removeItem(at: b)                                      // install.sh: rm + cp
        try "three".write(to: b, atomically: true, encoding: .utf8)
        XCTAssertNotEqual(UpgradeWatcher.statFingerprint(link), f2)
        try FileManager.default.removeItem(at: b)
        XCTAssertNil(UpgradeWatcher.statFingerprint(link))                             // dangling
    }

    func testRealProbeRejectsTruncatedAndNonExecutableFiles() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("upg-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let p = dir.appendingPathComponent("krill")
        try Data([0xcf, 0xfa, 0xed, 0xfe, 0x00]).write(to: p)
        let fp = FP(resolved: p.path, inode: 1, mtime: 1, size: 5)
        var r = await UpgradeWatcher.realProbe(fp, timeout: 5)
        guard case .rejected = r else { return XCTFail("non-executable accepted: \(r)") }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: p.path)
        r = await UpgradeWatcher.realProbe(fp, timeout: 5)
        guard case .rejected = r else { return XCTFail("truncated binary accepted: \(r)") }
    }
}
