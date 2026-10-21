import Foundation

/// Process-wide observability runtime: the structured event log, the run marker and the
/// settings every other writer reads. Everything here is a no-op until `configure` is called, so
/// libraries and tests that never configure it write nothing.
public enum Obs {
    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var settings = ObsSettings(eventLog: false, sessionLog: .off, root: ObsPaths.root())
        var configured = false
        var kind = "cli"
        var writer: LineWriter?
        var marker: URL?
        var port: Int?
    }
    private static let state = State()

    /// Current settings (defaults, all writers off, until `configure`).
    public static var settings: ObsSettings {
        state.lock.lock(); defer { state.lock.unlock() }
        return state.settings
    }

    public static var isConfigured: Bool {
        state.lock.lock(); defer { state.lock.unlock() }
        return state.configured
    }

    /// Wire the event log for this process. `kind` is the file prefix: `serve` for `krill serve`,
    /// `cli` for everything interactive. Also starts the crash breadcrumb, sweeps markers left by
    /// dead processes into `unclean_exit` events, and prunes old logs in the background.
    /// Idempotent for the same process (a second call replaces the settings).
    public static func configure(kind: String, settings: ObsSettings, port: Int? = nil,
                                 version: String = "", prune: Bool = true) {
        state.lock.lock()
        state.settings = settings
        state.kind = kind
        state.port = port
        state.configured = true
        if settings.eventLog {
            let dir = settings.logsDir
            let prefix = kind
            state.writer = LineWriter(label: "krill.obs.events") { d in
                dir.appendingPathComponent("\(prefix)-\(ObsPaths.dayString(d)).jsonl")
            }
        } else {
            state.writer = nil
        }
        state.lock.unlock()

        guard settings.eventLog else { return }
        // Crash breadcrumb: sweep first (so a stale marker of a dead sibling is reported), then arm ours.
        for stale in RunMarkers.sweepStale(dir: settings.runDir) {
            emit("unclean_exit", CrashReport.fields(for: stale, settings: settings))
        }
        let marker = RunMarkers.write(dir: settings.runDir, kind: kind, port: port, version: version)
        state.lock.lock(); state.marker = marker; state.lock.unlock()
        if prune {
            DispatchQueue.global(qos: .utility).async {
                Retention.pruneIfDue(settings: settings)
            }
        }
    }

    /// Queue one lifecycle/request event. Adds `pid` (and `port` for the server) so lines from
    /// two servers sharing a file can be told apart. No-op when the event log is off.
    public static func emit(_ event: String, _ fields: [(String, LogValue)] = [], ts: Date = Date()) {
        state.lock.lock()
        let w = state.writer
        let port = state.port
        state.lock.unlock()
        guard let w else { return }
        var f: [(String, LogValue)] = [("pid", .int(Int(getpid())))]
        if let port { f.append(("port", .int(port))) }
        f.append(contentsOf: fields)
        w.append(LogEvent(event, ts: ts, f))
    }

    /// True when `emit` would write.
    public static var eventsEnabled: Bool {
        state.lock.lock(); defer { state.lock.unlock() }
        return state.writer != nil
    }

    /// Flush queued events to disk (shutdown path, tests).
    public static func flush() {
        state.lock.lock(); let w = state.writer; state.lock.unlock()
        w?.flush()
    }

    /// Clean shutdown: flush everything, then clear the unclean-exit marker. Safe to call twice.
    public static func shutdown() {
        flush()
        state.lock.lock()
        let m = state.marker
        state.marker = nil
        state.lock.unlock()
        if let m { RunMarkers.clear(m) }
    }

    /// Test hook: forget configuration.
    public static func resetForTesting() {
        state.lock.lock()
        state.writer = nil; state.marker = nil; state.configured = false; state.port = nil
        state.settings = ObsSettings(eventLog: false, sessionLog: .off, root: ObsPaths.root())
        state.lock.unlock()
    }
}
