import Foundation

/// Truncation helper shared by transcript fields.
public enum ObsText {
    /// `s` cut to at most `limit` UTF-8 bytes (on a scalar boundary) and how many bytes were dropped.
    public static func cap(_ s: String, limit: Int = ObsLimits.fieldCap) -> (text: String, dropped: Int) {
        let total = s.utf8.count
        guard total > limit else { return (s, 0) }
        var out = String.UnicodeScalarView()
        var used = 0
        for u in s.unicodeScalars {
            let n = String(u).utf8.count
            if used + n > limit { break }
            out.append(u); used += n
        }
        return (String(out), total - used)
    }
}

/// One agent session's transcript. Created by a surface (`run`, `code`, `ui`, `launch`), fed by
/// `AgentLoop`, closed with `end`. Every call is cheap and asynchronous (see `LineWriter`).
///
/// `full` keeps text and tool payloads (each field capped at 16 KB); `meta` keeps only structure
/// (roles, tool names, sizes, outcomes, timings). Nothing leaves the machine by itself.
public final class SessionRecorder: @unchecked Sendable {
    public let id: String
    public let surface: String
    public let mode: SessionLogMode
    public let fileURL: URL
    public let startedAt: Date

    private let transcript: LineWriter
    private let index: LineWriter
    private let settings: ObsSettings
    private let lock = NSLock()
    private var turns = 0
    private var toolCalls = 0
    private var ended = false

    /// nil when session logging is off.
    public static func start(surface: String, project: String?, model: String?, provider: String?,
                             version: String = "", extra: [(String, LogValue)] = [], id: String? = nil,
                             settings: ObsSettings = Obs.settings, now: Date = Date()) -> SessionRecorder? {
        guard settings.sessionLog != .off else { return nil }
        return SessionRecorder(surface: surface, project: project, model: model, provider: provider,
                               version: version, extra: extra, id: id, settings: settings, now: now)
    }

    public static func newID() -> String {
        String(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased().prefix(12))
    }

    init(surface: String, project: String?, model: String?, provider: String?, version: String,
         extra: [(String, LogValue)], id: String? = nil, settings: ObsSettings, now: Date) {
        self.id = id ?? Self.newID()
        self.surface = surface
        self.mode = settings.sessionLog
        self.settings = settings
        self.startedAt = now
        let parts = ObsPaths.dayString(now).split(separator: "-").map(String.init)
        let dir = settings.sessionsDir
            .appendingPathComponent(parts[0]).appendingPathComponent(parts[1]).appendingPathComponent(parts[2])
        let url = dir.appendingPathComponent("\(ObsPaths.timeString(now))-\(self.id).jsonl")
        self.fileURL = url
        self.transcript = LineWriter(label: "krill.obs.session") { _ in url }
        let idx = settings.sessionsDir.appendingPathComponent("index.jsonl")
        self.index = LineWriter(label: "krill.obs.index") { _ in idx }

        let rel = "\(parts[0])/\(parts[1])/\(parts[2])/\(url.lastPathComponent)"
        var head: [(String, LogValue)] = [
            ("id", .string(self.id)), ("surface", .string(surface)), ("project", LogValue.opt(project)),
            ("model", LogValue.opt(model)), ("provider", LogValue.opt(provider)), ("mode", .string(mode.rawValue)),
            ("version", .string(version)), ("pid", .int(Int(getpid()))),
        ]
        head.append(contentsOf: extra)
        transcript.append(LogEvent("session_start", ts: now, head))
        index.append(LogEvent("start", ts: now, [
            ("id", .string(self.id)), ("project", LogValue.opt(project)), ("surface", .string(surface)),
            ("model", LogValue.opt(model)), ("provider", LogValue.opt(provider)), ("file", .string(rel)),
        ]))
    }

    private func next(_ keyPath: ReferenceWritableKeyPath<SessionRecorder, Int>) -> Int {
        lock.lock(); defer { lock.unlock() }
        self[keyPath: keyPath] += 1
        return self[keyPath: keyPath]
    }

    private func text(_ key: String, _ s: String) -> [(String, LogValue)] {
        guard mode == .full else { return [("\(key)_chars", .int(s.count))] }
        let c = ObsText.cap(s)
        var f: [(String, LogValue)] = [(key, .string(c.text))]
        if c.dropped > 0 { f.append(("\(key)_truncated_bytes", .int(c.dropped))) }
        return f
    }

    private var turnNumber: Int { lock.lock(); defer { lock.unlock() }; return turns }

    public func user(_ s: String) {
        let t = next(\.turns)
        transcript.append(LogEvent("user", [("turn", .int(t))] + text("text", s)))
    }

    public func assistant(_ s: String, toolCalls n: Int = 0) {
        transcript.append(LogEvent("assistant", [("turn", .int(turnNumber)), ("tool_calls", .int(n))] + text("text", s)))
    }

    public func toolCall(name: String, arguments: String) {
        _ = next(\.toolCalls)
        transcript.append(LogEvent("tool_call", [("turn", .int(turnNumber)), ("tool", .string(name))]
            + (mode == .full ? text("args", arguments) : [("args_bytes", .int(arguments.utf8.count))])))
    }

    public func toolResult(name: String, content: String, isError: Bool, ms: Int) {
        transcript.append(LogEvent("tool_result", [
            ("turn", .int(turnNumber)), ("tool", .string(name)), ("is_error", .bool(isError)), ("ms", .int(ms)),
        ] + (mode == .full ? text("result", content) : [("result_bytes", .int(content.utf8.count))])))
    }

    /// decision: allow | deny | ask_approved | ask_declined | unknown_tool
    public func permission(tool: String, decision: String, reason: String? = nil) {
        transcript.append(LogEvent("permission", [
            ("turn", .int(turnNumber)), ("tool", .string(tool)), ("decision", .string(decision)),
            ("reason", .opt(reason.map { ObsText.cap($0, limit: 400).text })),
        ]))
    }

    /// kind: arg_repair | tool_name_repair | ...
    public func retry(kind: String, tool: String, detail: String? = nil) {
        transcript.append(LogEvent("retry", [
            ("turn", .int(turnNumber)), ("kind", .string(kind)), ("tool", .string(tool)),
            ("detail", .opt(detail.map { ObsText.cap($0, limit: 400).text })),
        ]))
    }

    public func error(_ message: String) {
        transcript.append(LogEvent("error", [("turn", .int(turnNumber)), ("message", .string(ObsText.cap(message, limit: 2000).text))]))
    }

    /// reason: final_answer | iteration_limit | cancelled | repeat_guard | tool_budget | error
    public func stop(reason: String) {
        transcript.append(LogEvent("stop", [("turn", .int(turnNumber)), ("reason", .string(reason))]))
    }

    /// Close the session: outcome is completed | cancelled | error | iteration_limit | ...
    public func end(outcome: String, now: Date = Date()) {
        lock.lock()
        if ended { lock.unlock(); return }
        ended = true
        let t = turns, c = toolCalls
        lock.unlock()
        let ms = Int(now.timeIntervalSince(startedAt) * 1000)
        transcript.append(LogEvent("session_end", ts: now, [
            ("outcome", .string(outcome)), ("turns", .int(t)), ("tool_calls", .int(c)), ("duration_ms", .int(ms)),
        ]))
        index.append(LogEvent("end", ts: now, [("id", .string(self.id)), ("outcome", .string(outcome)),
                                               ("turns", .int(t)), ("tool_calls", .int(c))]))
        flush()
    }

    public func flush() { transcript.flush(); index.flush() }
}

public struct SessionInfo: Sendable, Equatable {
    public var id: String
    public var start: Date
    public var end: Date?
    public var project: String?
    public var surface: String
    public var model: String?
    public var provider: String?
    public var outcome: String?
    public var file: String
}

public enum SessionIndex {
    public static func url(_ settings: ObsSettings) -> URL { settings.sessionsDir.appendingPathComponent("index.jsonl") }

    /// Merge start/end rows by id; newest first.
    public static func load(settings: ObsSettings) -> [SessionInfo] {
        guard let text = try? String(contentsOf: url(settings), encoding: .utf8) else { return [] }
        var byID: [String: SessionInfo] = [:]
        for line in text.split(separator: "\n") {
            guard let o = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
                  let id = o["id"] as? String, let ev = o["event"] as? String,
                  let ts = (o["ts"] as? String).flatMap(JSONLine.parseISO) else { continue }
            if ev == "start" {
                byID[id] = SessionInfo(id: id, start: ts, end: nil, project: o["project"] as? String,
                                       surface: o["surface"] as? String ?? "?", model: o["model"] as? String,
                                       provider: o["provider"] as? String, outcome: nil,
                                       file: o["file"] as? String ?? "")
            } else if ev == "end" {
                byID[id]?.end = ts
                byID[id]?.outcome = o["outcome"] as? String
            }
        }
        return byID.values.sorted { $0.start > $1.start }
    }

    /// Absolute path of a session's transcript.
    public static func transcriptURL(_ info: SessionInfo, settings: ObsSettings) -> URL {
        settings.sessionsDir.appendingPathComponent(info.file)
    }

    /// Find a session by full id or unique prefix.
    public static func find(id: String, settings: ObsSettings) -> SessionInfo? {
        let all = load(settings: settings)
        if let exact = all.first(where: { $0.id == id }) { return exact }
        let m = all.filter { $0.id.hasPrefix(id) }
        return m.count == 1 ? m[0] : nil
    }

    /// Drop rows whose transcript was pruned. Atomic replace; run from retention only.
    public static func compact(settings: ObsSettings) {
        let u = url(settings)
        guard let text = try? String(contentsOf: u, encoding: .utf8) else { return }
        let live = Set(load(settings: settings).filter {
            FileManager.default.fileExists(atPath: settings.sessionsDir.appendingPathComponent($0.file).path)
        }.map(\.id))
        let kept = text.split(separator: "\n").filter { line in
            guard let o = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
                  let id = o["id"] as? String else { return false }
            return live.contains(id)
        }
        let out = kept.joined(separator: "\n") + (kept.isEmpty ? "" : "\n")
        try? out.write(to: u, atomically: true, encoding: .utf8)
    }
}
