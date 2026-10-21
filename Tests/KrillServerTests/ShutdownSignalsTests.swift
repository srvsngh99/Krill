import XCTest
import Foundation
@testable import KrillServer

/// SIGTERM / SIGINT handling for `krill serve` (v0.26.0 fix: v0.25.2 ignored both when the launcher
/// had left them ignored or blocked, so only SIGKILL stopped it).
final class ShutdownSignalsTests: XCTestCase {
    override func tearDown() { ShutdownSignals.exitHook = nil; ShutdownSignals.resetForTesting() }

    final class Box: @unchecked Sendable {
        let l = NSLock(); private var v: [Int32] = []
        func add(_ x: Int32) { l.lock(); v.append(x); l.unlock() }
        var all: [Int32] { l.lock(); defer { l.unlock() }; return v }
    }

    func testFirstSignalRunsHandlerSecondForcesExit() {
        let handled = Box(), exits = Box()
        ShutdownSignals.exitHook = { exits.add($0) }
        ShutdownSignals.setHandler { handled.add($0) }
        XCTAssertFalse(ShutdownSignals.isShuttingDown)
        ShutdownSignals.dispatch(SIGTERM)
        XCTAssertTrue(ShutdownSignals.isShuttingDown)
        XCTAssertEqual(handled.all, [SIGTERM])
        XCTAssertEqual(exits.all, [])
        ShutdownSignals.dispatch(SIGINT)
        XCTAssertEqual(handled.all, [SIGTERM], "handler runs once")
        XCTAssertEqual(exits.all, [128 + SIGINT], "a second signal forces an immediate exit")
    }

    func testSignalBeforeServerIsReadyExitsAtOnce() {
        let exits = Box()
        ShutdownSignals.exitHook = { exits.add($0) }
        ShutdownSignals.dispatch(SIGINT)
        XCTAssertEqual(exits.all, [128 + SIGINT])
    }

    final class Counter: @unchecked Sendable {
        let l = NSLock(); var n: Int
        init(_ n: Int) { self.n = n }
        func dec() -> Int { l.lock(); defer { l.unlock() }; let v = n; n = max(0, n - 1); return v }
    }

    func testDrainReturnsWhenRequestsFinish() async {
        let start = Date()
        let remaining = Counter(3)
        let ok = await ShutdownSignals.waitForDrain(grace: 5) { remaining.dec() }
        XCTAssertTrue(ok)
        XCTAssertLessThan(Date().timeIntervalSince(start), 1)
    }

    func testDrainGivesUpAfterTheGraceTimeout() async {
        let start = Date()
        let ok = await ShutdownSignals.waitForDrain(grace: 0.3) { 1 }
        XCTAssertFalse(ok, "a stuck request must not hold shutdown forever")
        let t = Date().timeIntervalSince(start)
        XCTAssertGreaterThanOrEqual(t, 0.3)
        XCTAssertLessThan(t, 1.5)
    }

    func testGraceDefaultsAndEnv() {
        XCTAssertEqual(ShutdownSignals.graceSeconds(environment: [:]), 3)
        XCTAssertEqual(ShutdownSignals.graceSeconds(environment: ["KRILL_SHUTDOWN_GRACE_SECONDS": "1.5"]), 1.5)
        XCTAssertEqual(ShutdownSignals.graceSeconds(environment: ["KRILL_SHUTDOWN_GRACE_SECONDS": "junk"]), 3)
    }

    // MARK: real process (macOS: posix_spawn with an inherited ignore / mask)

    #if os(macOS)

    enum Inherit { case plain, ignored, blocked }

    /// Spawn `krill serve`, optionally with SIGTERM/SIGINT ignored or blocked on exec (what a
    /// launcher can hand down), then signal it. Returns the wait status and the time it took.
    func runServe(signal sig: Int32, inherit: Inherit) throws -> (status: Int32, seconds: Double) {
        let bin: URL
        if let env = ProcessInfo.processInfo.environment["KRILL_BIN"] { bin = URL(fileURLWithPath: env) }
        else { bin = Bundle(for: Self.self).bundleURL.deletingLastPathComponent().appendingPathComponent("krill") }
        try XCTSkipUnless(FileManager.default.isExecutableFile(atPath: bin.path), "krill binary not built at \(bin.path)")

        let home = FileManager.default.temporaryDirectory.appendingPathComponent("sig-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: home.appendingPathComponent(".krill"), withIntermediateDirectories: true)
        let port = try freePort()
        var env = ProcessInfo.processInfo.environment
        env["HOME"] = home.path
        env["KRILL_API_KEY"] = ""
        env["KRILL_MODELS_DIR"] = home.appendingPathComponent("models").path
        let args = [bin.path, "serve", "--port", "\(port)", "--host", "127.0.0.1"]

        var attr: posix_spawnattr_t? = nil
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        var mask = sigset_t()
        sigemptyset(&mask)
        if inherit == .blocked { sigaddset(&mask, SIGTERM); sigaddset(&mask, SIGINT) }
        posix_spawnattr_setsigmask(&attr, &mask)
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETSIGMASK))
        let oldTerm = signal(SIGTERM, inherit == .ignored ? SIG_IGN : SIG_DFL)
        let oldInt = signal(SIGINT, inherit == .ignored ? SIG_IGN : SIG_DFL)
        var pid: pid_t = 0
        let cArgs = args.map { strdup($0) } + [nil]
        let cEnv = env.map { strdup("\($0.key)=\($0.value)") } + [nil]
        var actions: posix_spawn_file_actions_t? = nil
        posix_spawn_file_actions_init(&actions)
        defer { posix_spawn_file_actions_destroy(&actions) }
        posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0)
        posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0)
        let rc = posix_spawn(&pid, bin.path, &actions, &attr, cArgs, cEnv)
        signal(SIGTERM, oldTerm)
        signal(SIGINT, oldInt)
        cArgs.forEach { free($0) }
        cEnv.forEach { free($0) }
        XCTAssertEqual(rc, 0)
        defer { kill(pid, SIGKILL); var s: Int32 = 0; waitpid(pid, &s, 0) }

        var up = false
        for _ in 0 ..< 100 {
            if let (_, code) = try? URLSession.shared.syncGet("http://127.0.0.1:\(port)/healthz"), code == 200 { up = true; break }
            Thread.sleep(forTimeInterval: 0.1)
        }
        XCTAssertTrue(up, "server did not start")
        let t0 = Date()
        kill(pid, sig)
        var status: Int32 = -1
        var done = false
        while Date().timeIntervalSince(t0) < 10 {
            var st: Int32 = 0
            if waitpid(pid, &st, WNOHANG) == pid { status = st; done = true; break }
            Thread.sleep(forTimeInterval: 0.05)
        }
        XCTAssertTrue(done, "krill serve ignored signal \(sig) (inherit \(inherit))")
        return (status, Date().timeIntervalSince(t0))
    }

    func freePort() throws -> Int { Int.random(in: 41000 ..< 49000) }

    func assertCleanExit(_ r: (status: Int32, seconds: Double), file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(r.status & 0x7f, 0, "killed by a signal instead of exiting (status \(r.status))", file: file, line: line)
        XCTAssertEqual((r.status >> 8) & 0xff, 0, "exit code", file: file, line: line)
        XCTAssertLessThan(r.seconds, 5, "took too long", file: file, line: line)
    }

    func testSIGTERMExitsCleanlyWhenInheritedIgnored() throws { assertCleanExit(try runServe(signal: SIGTERM, inherit: .ignored)) }
    func testSIGINTExitsCleanlyWhenInheritedIgnored() throws { assertCleanExit(try runServe(signal: SIGINT, inherit: .ignored)) }
    func testSIGTERMExitsCleanlyWhenInheritedBlocked() throws { assertCleanExit(try runServe(signal: SIGTERM, inherit: .blocked)) }
    func testSIGINTExitsCleanlyWhenInheritedBlocked() throws { assertCleanExit(try runServe(signal: SIGINT, inherit: .blocked)) }
    func testSIGTERMExitsCleanlyWithDefaultDisposition() throws { assertCleanExit(try runServe(signal: SIGTERM, inherit: .plain)) }
    #endif
}

extension URLSession {
    /// Blocking GET for tests: (body, status).
    func syncGet(_ url: String) throws -> (Data, Int) {
        let sem = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var out: (Data, Int)?
        let t = dataTask(with: URL(string: url)!) { d, r, _ in
            if let d, let r = r as? HTTPURLResponse { out = (d, r.statusCode) }
            sem.signal()
        }
        t.resume()
        if sem.wait(timeout: .now() + 2) == .timedOut { t.cancel(); throw URLError(.timedOut) }
        guard let out else { throw URLError(.badServerResponse) }
        return out
    }
}
