import Foundation
import Logging

/// Process-wide logging setup.
///
/// Without an explicit bootstrap, swift-log installs `StreamLogHandler.standardOutput`,
/// so every `logger.info(...)` lands on **stdout** interleaved with the command's real
/// output. That produced lines like
///
///     2026-01-01T00:00:00+0000 info krill.registry: [KrillRegistry] Saved manifest for <model>
///
/// in the middle of `krill pull` and `krill list`, which is both noise for a human and
/// corruption for anything parsing stdout (`krill list | ...`).
///
/// Diagnostics belong on stderr, so stdout carries only what the user asked for.
public enum KrillLogging {
    /// Make a Krill logger, bootstrapping the logging system on first use.
    ///
    /// Every Krill logger is created through here rather than `Logger(label:)` directly.
    /// Bootstrapping at logger-creation time (instead of from the CLI entry point) keeps
    /// this independent of how the process was started: `@main` is not the only way in -
    /// the executable module is also linked into the test bundle, and taking over
    /// `main()` there made SwiftPM hand the test runner's own arguments to
    /// ArgumentParser ("Unknown option '--test-bundle-path'").
    ///
    /// Ordering is what matters: swift-log only applies a bootstrap to loggers created
    /// AFTER it, so the bootstrap must happen before the `Logger` below is constructed.
    public static func makeLogger(_ label: String) -> Logger {
        bootstrap()
        return Logger(label: label)
    }

    /// Bootstrap once. `LoggingSystem.bootstrap` traps if called twice, so the work is
    /// wrapped in a `static let` the runtime initialises exactly once.
    public static func bootstrap() { _ = once }

    private static let once: Void = {
        let level = resolvedLevel()
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardError(label: label)
            handler.logLevel = level
            return handler
        }
    }()

    /// `KRILL_LOG_LEVEL` (trace/debug/info/notice/warning/error/critical) wins; otherwise
    /// `KRILL_DEBUG` implies debug. The default is `.notice`: routine `.info` chatter such
    /// as "Saved manifest" is bookkeeping, not something a user asked to see.
    static func resolvedLevel() -> Logger.Level {
        let env = ProcessInfo.processInfo.environment
        if let raw = env["KRILL_LOG_LEVEL"]?.lowercased(),
           let level = Logger.Level(rawValue: raw) {
            return level
        }
        if env["KRILL_DEBUG"] != nil { return .debug }
        return .notice
    }
}
