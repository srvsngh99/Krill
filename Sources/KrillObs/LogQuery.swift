import Foundation

/// A parsed event-log line.
public struct LogRecord {
    public var ts: Date
    public var event: String
    public var raw: String
    public var fields: [String: Any]

    public var level: String { (fields["level"] as? String) ?? "" }
    public var status: Int? { fields["status"] as? Int }
    public var sessionID: String? { fields["session_id"] as? String }

    /// Errors for `krill logs --errors`: failed requests, error/critical/warning log lines, load
    /// failures, unclean exits and anything the writer flagged.
    public var isError: Bool {
        if event == "request" { return (status ?? 0) >= 400 || status == 0 || fields["error"] is String }
        if event == "log" { return ["warning", "error", "critical"].contains(level) }
        return event.hasSuffix("_failed") || event == "error" || event == "unclean_exit" || event == "log_dropped"
    }
}

public struct LogFilter: Sendable {
    public var since: Date?
    public var errorsOnly = false
    public var session: String?
    public init(since: Date? = nil, errorsOnly: Bool = false, session: String? = nil) {
        self.since = since; self.errorsOnly = errorsOnly; self.session = session
    }
    public func matches(_ r: LogRecord) -> Bool {
        if let since, r.ts < since { return false }
        if errorsOnly, !r.isError { return false }
        if let session, !(r.sessionID.map { $0 == session || $0.hasPrefix(session) } ?? false) { return false }
        return true
    }
}

public enum LogQuery {
    /// Event log files (`serve-*.jsonl`, `cli-*.jsonl`), oldest first.
    public static func eventFiles(settings: ObsSettings) -> [URL] {
        let files = (try? FileManager.default.contentsOfDirectory(at: settings.logsDir, includingPropertiesForKeys: nil)) ?? []
        return files.filter {
            let n = $0.lastPathComponent
            return n.hasSuffix(".jsonl") && (n.hasPrefix("serve-") || n.hasPrefix("cli-"))
        }.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    public static func parse(line: Substring) -> LogRecord? {
        guard let o = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
              let ev = o["event"] as? String,
              let ts = (o["ts"] as? String).flatMap(JSONLine.parseISO) else { return nil }
        return LogRecord(ts: ts, event: ev, raw: String(line), fields: o)
    }

    /// All matching records across files, time-ordered. Files older than `since` are skipped by
    /// their name (the date) before being read.
    public static func records(settings: ObsSettings, filter: LogFilter) -> [LogRecord] {
        var out: [LogRecord] = []
        for f in eventFiles(settings: settings) {
            if let since = filter.since, let d = fileDay(f), d < Calendar.current.startOfDay(for: since) { continue }
            guard let text = try? String(contentsOf: f, encoding: .utf8) else { continue }
            for line in text.split(separator: "\n") {
                if let r = parse(line: line), filter.matches(r) { out.append(r) }
            }
        }
        return out.sorted { $0.ts < $1.ts }
    }

    static func fileDay(_ f: URL) -> Date? {
        let n = f.deletingPathExtension().lastPathComponent   // serve-2026-10-22 or serve-2026-10-22.1
        let parts = n.split(separator: "-", maxSplits: 1)
        guard parts.count == 2 else { return nil }
        let day = String(parts[1].prefix(10))
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd"
        fmt.locale = Locale(identifier: "en_US_POSIX")
        return fmt.date(from: day)
    }

    /// `30m`, `2h`, `7d`, `1w`, an ISO date (`2026-10-20`) or ISO timestamp.
    public static func parseSince(_ s: String, now: Date = Date()) -> Date? {
        let t = s.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return nil }
        if let unit = t.last, "smhdw".contains(unit), let n = Double(t.dropLast()) {
            let mult: Double = ["s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800][String(unit)]!
            return now.addingTimeInterval(-n * mult)
        }
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "yyyy-MM-dd"
        if let d = day.date(from: t) { return d }
        return JSONLine.parseISO(t)
    }

    public static func startOfToday(now: Date = Date()) -> Date { Calendar.current.startOfDay(for: now) }

    /// One human line per record.
    public static func render(_ r: LogRecord) -> String {
        let f = r.fields
        let time = localTime(r.ts)
        func s(_ k: String) -> String? { f[k] as? String }
        func n(_ k: String) -> Double? { (f[k] as? Double) ?? (f[k] as? Int).map(Double.init) }
        func ms(_ k: String) -> String? { n(k).map { "\(Int($0.rounded()))ms" } }
        switch r.event {
        case "request":
            var parts = ["\(time) request", "\(s("method") ?? "") \(s("endpoint") ?? "?")"]
            if let m = s("model") { parts.append("model=\(m)") }
            parts.append(String(f["status"] as? Int ?? 0))
            if let v = ms("ttft_ms") { parts.append("ttft=\(v)") }
            if let v = ms("total_ms") { parts.append("total=\(v)") }
            if let v = ms("queue_wait_ms"), (n("queue_wait_ms") ?? 0) > 0 { parts.append("queue=\(v)") }
            if let p = f["prompt_tokens"] as? Int, let c = f["completion_tokens"] as? Int { parts.append("\(p)+\(c)tok") }
            if let t = n("tok_s") { parts.append(String(format: "%.1ftok/s", t)) }
            if let d = s("embed_device") { parts.append("device=\(d)") }
            if let e = s("error") { parts.append("error=\"\(e)\"") }
            if let id = s("request_id") { parts.append("id=\(id)") }
            if let sid = s("session_id") { parts.append("session=\(sid)") }
            return parts.joined(separator: " ")
        case "log":
            return "\(time) \(r.level) \(s("logger") ?? "") \(s("message") ?? "")"
        default:
            let skip: Set<String> = ["ts", "event", "pid"]
            let rest = f.keys.sorted().filter { !skip.contains($0) }.map { k -> String in
                let v = f[k]
                if let str = v as? String { return "\(k)=\(str.contains(" ") || str.contains("\n") ? "\"\(str.replacingOccurrences(of: "\n", with: "\\n"))\"" : str)" }
                return "\(k)=\(v.map { "\($0)" } ?? "null")"
            }
            return "\(time) \(r.event) " + rest.joined(separator: " ")
        }
    }

    public static func localTime(_ d: Date) -> String {
        var t = time_t(d.timeIntervalSince1970)
        var tmv = tm()
        localtime_r(&t, &tmv)
        return String(format: "%02d-%02d %02d:%02d:%02d", Int(tmv.tm_mon) + 1, Int(tmv.tm_mday),
                      Int(tmv.tm_hour), Int(tmv.tm_min), Int(tmv.tm_sec))
    }

    /// Sessions whose project directory is `dir` or inside it.
    public static func sessions(forProject dir: String, settings: ObsSettings) -> [SessionInfo] {
        let want = URL(fileURLWithPath: (dir as NSString).expandingTildeInPath).standardizedFileURL.path
        return SessionIndex.load(settings: settings).filter {
            guard let p = $0.project else { return false }
            let have = URL(fileURLWithPath: p).standardizedFileURL.path
            return have == want || have.hasPrefix(want + "/")
        }
    }

    public static func renderSession(_ s: SessionInfo) -> String {
        let end = s.end.map { localTime($0) } ?? "(open)"
        return "\(s.id)  \(localTime(s.start)) -> \(end)  \(s.surface)  \(s.model ?? "-")"
            + (s.provider.map { "/\($0)" } ?? "") + "  \(s.outcome ?? "no end record")  \(s.project ?? "-")"
    }
}

/// `-f`: poll the event files for appended lines. Pure and synchronous so it is testable; the CLI
/// calls `poll()` in a loop with a sleep.
public final class LogFollower {
    private let settings: ObsSettings
    private let filter: LogFilter
    private var offsets: [String: UInt64] = [:]

    public init(settings: ObsSettings, filter: LogFilter, startAtEnd: Bool = true) {
        self.settings = settings
        self.filter = filter
        if startAtEnd {
            for f in LogQuery.eventFiles(settings: settings) {
                offsets[f.path] = ((try? FileManager.default.attributesOfItem(atPath: f.path)[.size]) as? NSNumber)?.uint64Value ?? 0
            }
        }
    }

    /// New matching records since the last poll.
    public func poll() -> [LogRecord] {
        var out: [LogRecord] = []
        for f in LogQuery.eventFiles(settings: settings) {
            let size = ((try? FileManager.default.attributesOfItem(atPath: f.path)[.size]) as? NSNumber)?.uint64Value ?? 0
            var off = offsets[f.path] ?? 0
            if size < off { off = 0 }   // rotated/truncated
            guard size > off, let h = try? FileHandle(forReadingFrom: f) else { offsets[f.path] = size; continue }
            defer { try? h.close() }
            try? h.seek(toOffset: off)
            let data = h.readDataToEndOfFile()
            // Only consume complete lines.
            guard let lastNL = data.lastIndex(of: 0x0A) else { continue }
            let chunk = data.prefix(upTo: lastNL + 1)
            offsets[f.path] = off + UInt64(chunk.count)
            for line in String(decoding: chunk, as: UTF8.self).split(separator: "\n") {
                if let r = LogQuery.parse(line: line), filter.matches(r) { out.append(r) }
            }
        }
        return out.sorted { $0.ts < $1.ts }
    }
}
