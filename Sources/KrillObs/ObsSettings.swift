import Foundation

/// How much of an agent session is written to `~/.krill/sessions`.
public enum SessionLogMode: String, Sendable, CaseIterable {
    /// Text, tool arguments and tool results (each field truncated at `ObsLimits.fieldCap`).
    case full
    /// Structure only: roles, tool names, outcomes, timings, sizes. No text and no tool payloads.
    case meta
    /// Nothing is written.
    case off

    /// Lenient parse for config and environment values. nil when unrecognised.
    public static func parse(_ raw: String?) -> SessionLogMode? {
        guard let raw else { return nil }
        switch raw.trimmingCharacters(in: .whitespaces).lowercased() {
        case "full", "on", "true", "1", "yes": return .full
        case "meta", "metadata": return .meta
        case "off", "false", "0", "no", "none": return .off
        default: return nil
        }
    }
}

/// Hard limits shared by the writers.
public enum ObsLimits {
    /// Per-field cap for session transcript content (user text, tool args, tool results).
    public static let fieldCap = 16 * 1024
    /// Queued-but-unwritten lines before new ones are dropped (the writer is never allowed to
    /// grow without bound or block a request).
    public static let maxPending = 20_000
    /// One event-log file is rotated once it passes this size.
    public static let maxFileBytes = 16 * 1024 * 1024
}

/// Everything the observability layer needs to know, resolved once at process start.
public struct ObsSettings: Sendable, Equatable {
    public var eventLog: Bool
    public var sessionLog: SessionLogMode
    public var retentionDays: Int
    public var maxMB: Int
    public var root: URL

    public init(eventLog: Bool = true, sessionLog: SessionLogMode = .full,
                retentionDays: Int = 30, maxMB: Int = 1024, root: URL = ObsPaths.root()) {
        self.eventLog = eventLog
        self.sessionLog = sessionLog
        self.retentionDays = retentionDays
        self.maxMB = maxMB
        self.root = root
    }

    public var logsDir: URL { root.appendingPathComponent("logs", isDirectory: true) }
    public var sessionsDir: URL { root.appendingPathComponent("sessions", isDirectory: true) }
    public var runDir: URL { root.appendingPathComponent("run", isDirectory: true) }
}

public enum ObsPaths {
    /// `KRILL_OBS_HOME` (tests, containers) or `~/.krill`. Models, config and the rest of
    /// `~/.krill` are NOT moved by this: it only relocates logs, sessions and run markers.
    public static func root(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        if let o = environment["KRILL_OBS_HOME"], !o.isEmpty {
            return URL(fileURLWithPath: (o as NSString).expandingTildeInPath, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".krill", isDirectory: true)
    }

    /// Create `url` (and parents) readable by the owner only: logs can hold prompts' neighbours.
    @discardableResult
    public static func ensureDir(_ url: URL) -> Bool {
        do {
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            return true
        } catch { return false }
    }

    /// Local calendar date `YYYY-MM-DD`.
    public static func dayString(_ d: Date) -> String {
        var t = time_t(d.timeIntervalSince1970)
        var tmv = tm()
        localtime_r(&t, &tmv)
        return String(format: "%04d-%02d-%02d", Int(tmv.tm_year) + 1900, Int(tmv.tm_mon) + 1, Int(tmv.tm_mday))
    }

    /// Local `HHMMSS`.
    public static func timeString(_ d: Date) -> String {
        var t = time_t(d.timeIntervalSince1970)
        var tmv = tm()
        localtime_r(&t, &tmv)
        return String(format: "%02d%02d%02d", Int(tmv.tm_hour), Int(tmv.tm_min), Int(tmv.tm_sec))
    }
}
