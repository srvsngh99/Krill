import Foundation

/// SIGINT / SIGTERM handling for `krill serve`.
///
/// Root cause of "the server ignores SIGTERM and SIGINT" (v0.25.2): the server relied on the
/// *default* disposition of both signals. A process inherits its ignored-signal set and its
/// signal mask across `exec` from whatever launched it. A non-interactive shell's `cmd &`
/// ignores SIGINT; supervisors, test harnesses and agent runners hand down an ignored or
/// blocked SIGTERM. In either case the default action (terminate) never runs, so only SIGKILL
/// could stop the server (reproduced: a parent that sets SIGTERM/SIGINT to SIG_IGN, or blocks
/// them, before exec - `server.pid` survives `kill -TERM` indefinitely). Where the default did
/// run it killed the process mid-request with nothing unloaded.
///
/// `installDefaults()` makes the server own both signals: it unblocks them, sets them to
/// ignore (so a dispatch signal source sees an inherited-ignored signal) and observes them via
/// kqueue-backed dispatch sources on a private queue, which fire whatever the inherited
/// disposition was and do not depend on the Swift concurrency pool being responsive.
///
/// Call `installDefaults()` first thing in `serve` (a Ctrl-C during model load then exits
/// immediately), and `setHandler` once the listener is up to switch to the graceful drain.
public enum ShutdownSignals {
    /// Signals that request a graceful shutdown.
    public static let signals: [Int32] = [SIGINT, SIGTERM]

    /// Grace period (seconds) for in-flight requests: `KRILL_SHUTDOWN_GRACE_SECONDS`, default 3.
    public static func graceSeconds(
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> Double {
        if let raw = environment["KRILL_SHUTDOWN_GRACE_SECONDS"],
           let v = Double(raw.trimmingCharacters(in: .whitespaces)), v >= 0 { return v }
        return 3
    }

    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var count = 0
        var handler: (@Sendable (Int32) -> Void)?
        var sources: [DispatchSourceSignal] = []
        var shuttingDown = false
    }
    private static let state = State()
    private static let queue = DispatchQueue(label: "krill.shutdown-signals")

    /// True once the first signal arrived; new generation/embedding requests get HTTP 503.
    public static var isShuttingDown: Bool {
        state.lock.lock(); defer { state.lock.unlock() }
        return state.shuttingDown
    }

    /// Install the sources (idempotent). Until `setHandler` is called a signal exits at once with 128+signal.
    public static func installDefaults() {
        state.lock.lock(); defer { state.lock.unlock() }
        guard state.sources.isEmpty else { return }
        var mask = sigset_t()
        sigemptyset(&mask)
        for s in signals { sigaddset(&mask, s) }
        pthread_sigmask(SIG_UNBLOCK, &mask, nil)
        state.sources = signals.map { sig in
            signal(sig, SIG_IGN)
            let src = DispatchSource.makeSignalSource(signal: sig, queue: queue)
            src.setEventHandler { dispatch(sig) }
            src.resume()
            return src
        }
    }

    /// The graceful handler; runs once, on the first signal. A second signal exits immediately.
    public static func setHandler(_ h: @escaping @Sendable (Int32) -> Void) {
        state.lock.lock(); state.handler = h; state.lock.unlock()
    }

    /// Enter the shutting-down state without a signal (self-restart on upgrade): new generation
    /// requests get 503 and `parkWhileShuttingDown()` holds `main` until the drain path exits.
    public static func beginShutdown() {
        state.lock.lock(); state.shuttingDown = true; state.lock.unlock()
    }

    static func resetForTesting() {
        state.lock.lock(); state.count = 0; state.handler = nil; state.shuttingDown = false; state.lock.unlock()
    }

    /// Hold the caller (the server's main task) while a graceful shutdown runs; `exitNow` ends the process.
    public static func parkWhileShuttingDown() async {
        guard isShuttingDown else { return }
        try? await Task.sleep(nanoseconds: .max / 2)
    }

    /// Deliver `sig` as if it had arrived (the dispatch sources call this; tests too).
    static func dispatch(_ sig: Int32) {
        state.lock.lock()
        state.count += 1
        let first = state.count == 1
        if first { state.shuttingDown = true }
        let h = state.handler
        state.lock.unlock()
        if first, let h { h(sig) } else { exitNow(128 + sig) }
    }

    /// Replaced by tests so a shutdown path can be exercised without ending the test process.
    nonisolated(unsafe) static var exitHook: (@Sendable (Int32) -> Void)?

    /// Flush stdio and leave without running static destructors (a busy Metal
    /// queue must not race them).
    public static func exitNow(_ code: Int32) {
        fflush(stdout); fflush(stderr)
        if let hook = exitHook { hook(code); return }
        _exit(code)
    }

    /// Wait until `pending()` reaches 0 or `grace` seconds pass. True when everything drained.
    public static func waitForDrain(grace: Double, pending: @Sendable () async -> Int) async -> Bool {
        let deadline = Date().addingTimeInterval(grace)
        while await pending() > 0 {
            if Date() >= deadline { return false }
            try? await Task.sleep(nanoseconds: 25_000_000)
        }
        return true
    }
}
