import Foundation

/// The unclean-exit breadcrumb: a tiny JSON file written at start and removed on a clean exit.
/// A marker whose process is gone is evidence of a crash or kill.
public struct RunMarker: Sendable, Equatable {
    public var pid: Int
    public var kind: String
    public var port: Int?
    public var started: Date
    public var version: String
    public var binary: String
    public var file: URL

    public func json() -> [String: Any] {
        var d: [String: Any] = ["pid": pid, "kind": kind, "started": JSONLine.isoUTC(started),
                                "version": version, "binary": binary]
        if let port { d["port"] = port }
        return d
    }
}

public enum RunMarkers {
    @discardableResult
    public static func write(dir: URL, kind: String, port: Int?, version: String,
                             now: Date = Date(), pid: Int32 = getpid()) -> URL? {
        guard ObsPaths.ensureDir(dir) else { return nil }
        let url = dir.appendingPathComponent("\(kind)-\(pid).json")
        let m = RunMarker(pid: Int(pid), kind: kind, port: port, started: now, version: version,
                          binary: CommandLine.arguments.first.map { URL(fileURLWithPath: $0).path } ?? "",
                          file: url)
        guard let data = try? JSONSerialization.data(withJSONObject: m.json(), options: [.sortedKeys]) else { return nil }
        guard (try? data.write(to: url, options: .atomic)) != nil else { return nil }
        return url
    }

    public static func clear(_ url: URL) { try? FileManager.default.removeItem(at: url) }

    /// Markers whose process no longer exists. Each is removed as it is returned, so a crash is
    /// reported exactly once.
    public static func sweepStale(dir: URL, isAlive: (Int) -> Bool = processIsAlive) -> [RunMarker] {
        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return [] }
        var out: [RunMarker] = []
        for f in files where f.pathExtension == "json" {
            guard let m = read(f) else { try? FileManager.default.removeItem(at: f); continue }
            if m.pid == Int(getpid()) || isAlive(m.pid) { continue }
            out.append(m)
            try? FileManager.default.removeItem(at: f)
        }
        return out.sorted { $0.started < $1.started }
    }

    /// Markers currently on disk (alive or not), for doctor.
    public static func list(dir: URL) -> [RunMarker] {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return files.filter { $0.pathExtension == "json" }.compactMap(read).sorted { $0.started < $1.started }
    }

    static func read(_ f: URL) -> RunMarker? {
        guard let data = try? Data(contentsOf: f),
              let o = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let pid = o["pid"] as? Int, let kind = o["kind"] as? String,
              let started = (o["started"] as? String).flatMap(JSONLine.parseISO)
        else { return nil }
        return RunMarker(pid: pid, kind: kind, port: o["port"] as? Int, started: started,
                         version: o["version"] as? String ?? "", binary: o["binary"] as? String ?? "", file: f)
    }

    public static func processIsAlive(_ pid: Int) -> Bool {
        if kill(pid_t(pid), 0) == 0 { return true }
        return errno == EPERM
    }
}

/// What to say about a marker that outlived its process.
public enum CrashReport {
    /// Fields for the `unclean_exit` event: the marker, the dead process's last events and any
    /// macOS crash report written for a krill binary since it started.
    public static func fields(for m: RunMarker, settings: ObsSettings,
                              diagnosticsDir: URL? = defaultDiagnosticsDir()) -> [(String, LogValue)] {
        var f: [(String, LogValue)] = [
            ("dead_pid", .int(m.pid)), ("kind", .string(m.kind)),
            ("started", .string(JSONLine.isoUTC(m.started))), ("version", .string(m.version)),
            ("binary", .string(m.binary)),
        ]
        if let port = m.port { f.append(("dead_port", .int(port))) }
        let last = lastEvents(pid: m.pid, logsDir: settings.logsDir, limit: 8)
        f.append(("last_events", .string(last.joined(separator: "\n"))))
        let ips = matchingCrashReports(after: m.started, in: diagnosticsDir)
        f.append(("crash_reports", .string(ips.map(\.path).joined(separator: ","))))
        return f
    }

    /// Last `limit` lines written by `pid`, newest log file first.
    public static func lastEvents(pid: Int, logsDir: URL, limit: Int) -> [String] {
        let files = (try? FileManager.default.contentsOfDirectory(at: logsDir, includingPropertiesForKeys: nil)) ?? []
        let sorted = files.filter { $0.pathExtension == "jsonl" }.sorted { $0.lastPathComponent > $1.lastPathComponent }
        var out: [String] = []
        let needle = "\"pid\":\(pid)"
        for f in sorted.prefix(4) {
            guard let text = try? String(contentsOf: f, encoding: .utf8) else { continue }
            let mine = text.split(separator: "\n").filter { $0.contains(needle) }
            out = mine.suffix(limit).map(String.init) + out
            if out.count >= limit { break }
        }
        return Array(out.suffix(limit))
    }

    public static func defaultDiagnosticsDir() -> URL? {
        #if os(macOS)
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/DiagnosticReports", isDirectory: true)
        #else
        return nil
        #endif
    }

    /// `krill*.ips` / `krill*.crash` reports modified at or after `date` (a minute of slack).
    public static func matchingCrashReports(after date: Date, in dir: URL?) -> [URL] {
        guard let dir else { return [] }
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? []
        return files.filter {
            let n = $0.lastPathComponent.lowercased()
            guard n.hasPrefix("krill"), n.hasSuffix(".ips") || n.hasSuffix(".crash") else { return false }
            let m = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
            return (m ?? .distantPast) >= date.addingTimeInterval(-60)
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }
}
