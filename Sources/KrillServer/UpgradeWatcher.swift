import Foundation
import KrillRegistry
import NIOCore

/// Self-restart onto a newly installed `krill` (`brew upgrade krill` / `install.sh`).
///
/// A running server keeps executing the old image after the file on disk is replaced.
/// `UpgradeWatcher` polls (default every 60 s, one `stat`) for the binary the process was
/// launched as, decides whether a *usable, different* version was installed, and - only under
/// launchd and only when idle - hands over to a graceful drain and relaunch. See
/// docs/SELF_RESTART_ON_UPGRADE.md. Everything that touches the world (stat, codesign, running
/// `version`, the clock, the in-flight count) is injected, so the state machine is unit-testable.
/// Quiet-period clock: the HTTP handler stamps it when a request head arrives.
public enum RequestActivity {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var stamp = Date()
    public static func note() { lock.lock(); stamp = Date(); lock.unlock() }
    public static var last: Date { lock.lock(); defer { lock.unlock() }; return stamp }
}

public final class UpgradeWatcher: @unchecked Sendable {

    // MARK: - Types

    /// What identifies "the installed binary": the fully resolved path plus inode, mtime, size.
    /// `brew upgrade` changes `resolved` (new Cellar dir); `install.sh` (rm + cp) changes inode/mtime.
    public struct Fingerprint: Equatable, Sendable {
        public var resolved: String
        public var inode: UInt64
        public var mtime: Double
        public var size: Int64
        public init(resolved: String, inode: UInt64, mtime: Double, size: Int64) {
            self.resolved = resolved; self.inode = inode; self.mtime = mtime; self.size = size
        }
    }

    /// Result of vetting a candidate binary.
    public enum Probe: Equatable, Sendable {
        case ok(version: String)
        case rejected(reason: String)       // unusable: ignored until the file changes again
        case retry(reason: String)          // inconclusive (timeout under load): look again next tick
    }

    /// What one `tick()` did, for logging and tests.
    public enum Outcome: Equatable, Sendable {
        case unchanged
        case settling                       // changed; waiting one interval for the file to be stable
        case ignored(reason: String)        // changed but unusable (logged once per fingerprint)
        case inconclusive(reason: String)   // vetting timed out (busy machine); retried next tick
        case sameVersion                    // changed but reports the running version
        case notice(version: String)        // upgrade found, restart not allowed here (terminal / opted out)
        case deferred(version: String, pending: Int, quietFor: Double)
        case restart(from: String, to: String)
    }

    public struct Settings: Sendable {
        public var interval: Double = 60
        public var quiet: Double = 10
        /// Restart allowed (launchd job + knob on). False => notice only.
        public var mayRestart: Bool
        public init(interval: Double = 60, quiet: Double = 10, mayRestart: Bool) {
            self.interval = interval; self.quiet = quiet; self.mayRestart = mayRestart
        }
    }

    // MARK: - Injected world

    private let currentVersion: String
    private let settings: Settings
    private let fingerprint: @Sendable () -> Fingerprint?
    private let probe: @Sendable (Fingerprint) async -> Probe
    private let pending: @Sendable () async -> Int
    private let now: @Sendable () -> Date
    private let lastActivity: @Sendable () -> Date
    private let logger = KrillLogging.makeLogger("krill.upgrade")

    // MARK: - State

    private let lock = NSLock()
    private var baseline: Fingerprint?
    private var candidate: Fingerprint?     // last changed fingerprint seen (awaiting a stable second look)
    private var rejected: Fingerprint?      // already vetted and refused: stay silent about it
    private var validated: (fp: Fingerprint, version: String)?
    private var lastDeferredPending: Int?
    private var noticed = false
    private var retryLogged = false

    public init(currentVersion: String = KrillVersion,
                settings: Settings,
                fingerprint: @escaping @Sendable () -> Fingerprint?,
                probe: @escaping @Sendable (Fingerprint) async -> Probe,
                pending: @escaping @Sendable () async -> Int,
                now: @escaping @Sendable () -> Date = { Date() },
                lastActivity: @escaping @Sendable () -> Date = { RequestActivity.last }) {
        self.currentVersion = currentVersion
        self.settings = settings
        self.fingerprint = fingerprint
        self.probe = probe
        self.pending = pending
        self.now = now
        self.lastActivity = lastActivity
        self.baseline = fingerprint()
    }

    // MARK: - One check

    public func tick() async -> Outcome {
        guard let fp = fingerprint() else { return .unchanged }   // mid-swap / briefly missing: look again next time
        let known: Fingerprint? = lock.withLock { baseline }
        if fp == known { lock.withLock { candidate = nil; validated = nil }; return .unchanged }
        if lock.withLock({ rejected }) == fp { return .unchanged }

        // Stability: a changed fingerprint must be seen unchanged on a second check.
        if lock.withLock({ candidate }) != fp {
            lock.withLock { candidate = fp; validated = nil; noticed = false; retryLogged = false; lastDeferredPending = nil }
            return .settling
        }

        if lock.withLock({ validated })?.fp != fp {
            switch await probe(fp) {
            case .rejected(let reason):
                lock.withLock { rejected = fp; candidate = nil }
                logger.notice("[upgrade] ignoring changed binary \(fp.resolved): \(reason)")
                return .ignored(reason: reason)
            case .retry(let reason):
                if lock.withLock({ () -> Bool in let f = !retryLogged; retryLogged = true; return f }) {
                    logger.notice("[upgrade] could not vet changed binary \(fp.resolved) yet (\(reason)); will retry")
                }
                return .inconclusive(reason: reason)
            case .ok(let v) where v == currentVersion:
                lock.withLock { baseline = fp; candidate = nil }
                return .sameVersion
            case .ok(let v):
                lock.withLock { validated = (fp, v) }
                logger.notice("[upgrade] Krill \(v) is installed (running \(currentVersion)) at \(fp.resolved)")
            }
        }
        let target = lock.withLock { validated!.version }

        guard settings.mayRestart else {
            let first = lock.withLock { () -> Bool in let f = !noticed; noticed = true; return f }
            if first {
                let msg = "Krill \(target) is installed; restart to use it (running \(currentVersion))."
                print(msg); logger.notice("[upgrade] \(msg)")
            }
            return .notice(version: target)
        }

        let n = await pending()
        let quietFor = now().timeIntervalSince(lastActivity())
        if n > 0 || quietFor < settings.quiet {
            let changed = lock.withLock { () -> Bool in
                let c = lastDeferredPending != n; lastDeferredPending = n; return c
            }
            if changed {
                logger.notice("[upgrade] restart to \(target) deferred: waiting for \(n) in-flight request(s), quiet period \(Int(settings.quiet)) s")
            }
            return .deferred(version: target, pending: n, quietFor: quietFor)
        }
        return .restart(from: currentVersion, to: target)
    }

    // MARK: - Decisions (pure, unit-tested)

    /// `XPC_SERVICE_NAME` is the job label under launchd. Terminals launched from the GUI carry
    /// "0" or "application.*" there, so those are not jobs; a launchd job's parent is pid 1.
    public static func launchdLabel(environment: [String: String], parentPID: Int32) -> String? {
        guard parentPID == 1,
              let raw = environment["XPC_SERVICE_NAME"]?.trimmingCharacters(in: .whitespaces),
              !raw.isEmpty, raw != "0", !raw.hasPrefix("application.") else { return nil }
        return raw
    }

    /// Knob resolution: env `KRILL_AUTO_RESTART_ON_UPGRADE` beats config `auto_restart_on_upgrade`; default on.
    public static func restartEnabled(config: Bool, environment: [String: String]) -> Bool {
        guard let raw = environment["KRILL_AUTO_RESTART_ON_UPGRADE"]?.lowercased()
            .trimmingCharacters(in: .whitespaces), !raw.isEmpty else { return config }
        return !["0", "false", "no", "off"].contains(raw)
    }

    public static func doubleEnv(_ key: String, default d: Double, environment: [String: String]) -> Double {
        if let raw = environment[key], let v = Double(raw.trimmingCharacters(in: .whitespaces)), v >= 0 { return v }
        return d
    }

    /// Absolute path the process was launched as (not its resolved target): `argv[0]` made
    /// absolute, searched on PATH when it has no slash.
    public static func invocationPath(
        argv0: String = CommandLine.arguments.first ?? "",
        cwd: String = FileManager.default.currentDirectoryPath,
        path: String = ProcessInfo.processInfo.environment["PATH"] ?? "",
        isExecutable: (String) -> Bool = { FileManager.default.isExecutableFile(atPath: $0) }
    ) -> String? {
        if argv0.contains("/") {
            let abs = argv0.hasPrefix("/") ? argv0 : cwd + "/" + argv0
            return (abs as NSString).standardizingPath
        }
        for dir in path.split(separator: ":") {
            let c = String(dir) + "/" + argv0
            if isExecutable(c) { return c }
        }
        return nil
    }

    // MARK: - Real world

    public static func statFingerprint(_ path: String) -> Fingerprint? {
        guard let resolved = realpath(path, nil) else { return nil }
        defer { free(resolved) }
        var st = stat()
        guard stat(resolved, &st) == 0 else { return nil }
        #if os(macOS)
        let ts = st.st_mtimespec
        #else
        let ts = st.st_mtim
        #endif
        let m = Double(ts.tv_sec) + Double(ts.tv_nsec) / 1e9
        return Fingerprint(resolved: String(cString: resolved), inode: UInt64(st.st_ino), mtime: m, size: Int64(st.st_size))
    }

    /// Exists, executable, (macOS) `codesign --verify` clean, and `version` prints a version, all within `timeout`.
    public static func realProbe(_ fp: Fingerprint, timeout: Double = 90) async -> Probe {
        guard FileManager.default.isExecutableFile(atPath: fp.resolved) else {
            return .rejected(reason: "not an executable file")
        }
        #if os(macOS)
        let cs = await run("/usr/bin/codesign", ["--verify", fp.resolved], timeout: timeout)
        if cs.status == -1 { return .retry(reason: "codesign --verify timed out") }
        guard cs.status == 0 else { return .rejected(reason: "codesign --verify failed (exit \(cs.status))") }
        #endif
        let v = await run(fp.resolved, ["version"], timeout: timeout)
        if v.status == -1 { return .retry(reason: "`version` timed out") }
        guard v.status == 0, let ver = parseVersion(v.output) else {
            return .rejected(reason: "`version` failed (exit \(v.status))")
        }
        return .ok(version: ver)
    }

    /// "Krill v0.26.1" -> "0.26.1".
    public static func parseVersion(_ output: String) -> String? {
        guard let line = output.split(separator: "\n").first else { return nil }
        let words = line.split(separator: " ")
        guard words.count >= 2, words[0] == "Krill" else { return nil }
        let v = words[1].hasPrefix("v") ? String(words[1].dropFirst()) : String(words[1])
        return v.first?.isNumber == true ? v : nil
    }

    static func run(_ exe: String, _ args: [String], timeout: Double) async -> (status: Int32, output: String) {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        let out = Pipe()
        p.standardOutput = out; p.standardError = FileHandle.nullDevice
        do { try p.run() } catch { return (127, "") }
        let deadline = Date().addingTimeInterval(timeout)
        while p.isRunning {
            if Date() >= deadline { p.terminate(); return (-1, "") }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        let data = out.fileHandleForReading.availableData
        return (p.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    #if os(macOS)
    /// Make sure the job comes back onto the on-disk binary whatever its KeepAlive says.
    ///
    /// An exit code alone is not enough (exit 0 does not relaunch under `SuccessfulExit = false`;
    /// nothing relaunches a job with no KeepAlive), and `launchctl kickstart -k` run from inside the
    /// job is not enough either: launchd kills the job and its process group (including that very
    /// `launchctl`) before it starts the replacement, so the job just stops (verified live). So:
    /// spawn a helper in its own session (`POSIX_SPAWN_SETSID`, outside the job's process group) that
    /// waits a moment and runs a plain `launchctl kickstart gui/<uid>/<label>`, then exit non-zero.
    /// KeepAlive (true or SuccessfulExit=false) relaunches on the non-zero exit; for jobs without
    /// KeepAlive the helper's kickstart starts it; if both fire the second is a harmless no-op.
    /// Returns false if the helper could not be spawned.
    public static func spawnDetachedKickstart(label: String, delay: Int = 2) -> Bool {
        var attr: posix_spawnattr_t?
        posix_spawnattr_init(&attr)
        defer { posix_spawnattr_destroy(&attr) }
        posix_spawnattr_setflags(&attr, Int16(POSIX_SPAWN_SETSID))
        let script = "sleep \(delay); /bin/launchctl kickstart gui/\(getuid())/\(label)"
        let argv = ["/bin/sh", "-c", script]
        var cargs: [UnsafeMutablePointer<CChar>?] = argv.map { strdup($0) } + [nil]
        defer { cargs.forEach { free($0) } }
        var pid: pid_t = 0
        return posix_spawn(&pid, "/bin/sh", nil, &attr, cargs, environ) == 0
    }
    #endif
}

// MARK: - Server wiring

extension KrillServer {
    /// Start the ~60 s upgrade poll (see ``UpgradeWatcher``). Returns the task, or nil when there
    /// is nothing to watch.
    func startUpgradeWatch(channel: Channel, evictor: Task<Void, Never>) -> Task<Void, Never>? {
        let env = ProcessInfo.processInfo.environment
        guard let path = UpgradeWatcher.invocationPath() else { return nil }
        var label: String?
        #if os(macOS)
        label = UpgradeWatcher.launchdLabel(environment: env, parentPID: getppid())
        #endif
        let mayRestart = label != nil
            && UpgradeWatcher.restartEnabled(config: autoRestartOnUpgrade, environment: env)
        let interval = max(0.05, UpgradeWatcher.doubleEnv("KRILL_UPGRADE_CHECK_SECONDS", default: 60, environment: env))
        let quiet = UpgradeWatcher.doubleEnv("KRILL_UPGRADE_QUIET_SECONDS", default: 10, environment: env)
        let queue = genQueue
        let watcher = UpgradeWatcher(
            settings: .init(interval: interval, quiet: quiet, mayRestart: mayRestart),
            fingerprint: { UpgradeWatcher.statFingerprint(path) },
            probe: { await UpgradeWatcher.realProbe($0) },
            pending: { await queue.pending })
        let log = KrillLogging.makeLogger("krill.upgrade")
        return Task { [self] in
            while !Task.isCancelled, !ShutdownSignals.isShuttingDown {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                guard case .restart(let from, let to) = await watcher.tick() else { continue }
                log.notice("[upgrade] restarting Krill \(from) -> \(to) (idle; draining then relaunching via launchd)")
                print("Krill \(to) installed; restarting from \(from)...")
                // TODO(observability): emit a serve event-log record here once the daily
                // serve-YYYY-MM-DD.jsonl writer lands (feat/observability): {event: "upgrade_restart", from, to}.
                #if os(macOS)
                let job = label
                ShutdownSignals.beginShutdown()   // 503 new work; keeps `main` parked until we exit
                await gracefulShutdown(channel: channel, evictor: evictor,
                                       grace: ShutdownSignals.graceSeconds()) {
                    // Detached kickstart helper + non-zero exit: relaunches under any KeepAlive setting
                    // (see UpgradeWatcher.spawnDetachedKickstart).
                    if let job { _ = UpgradeWatcher.spawnDetachedKickstart(label: job) }
                    ShutdownSignals.exitNow(75)
                }
                #endif
                return
            }
        }
    }
}
