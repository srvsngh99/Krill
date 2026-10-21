import Foundation

public struct BundleEntry: Sendable, Equatable {
    public var name: String
    public var bytes: Int
    public var note: String
}

public struct BugReportResult {
    public var zipURL: URL
    public var manifest: [String: Any]
    public var entries: [BundleEntry]
    public var issueURL: URL?
    public var summary: String

    public func renderManifest() -> String {
        var s = "Bundle: \(zipURL.path)\n"
        for e in entries { s += String(format: "  %@  %8d bytes  %@\n", e.name.padding(toLength: 34, withPad: " ", startingAt: 0), e.bytes, e.note) }
        return s
    }
}

public struct BugReportOptions: Sendable {
    public var days = 3
    public var sessionID: String?
    public var includeContent = false
    public var outputDir: URL
    public init(days: Int = 3, sessionID: String? = nil, includeContent: Bool = false, outputDir: URL) {
        self.days = days; self.sessionID = sessionID; self.includeContent = includeContent; self.outputDir = outputDir
    }
}

/// Builds the share bundle. It only reads local files and writes ONE zip into `outputDir`. It
/// sends nothing anywhere; the user attaches the zip to an issue themselves.
public enum BugReport {
    public static let repo = "srvsngh99/Krill"

    public static func build(version: String, settings: ObsSettings, doctor: DoctorReport, platform: String,
                             configText: String?, environment: [String: String], options: BugReportOptions,
                             redactor: Redactor = Redactor(), now: Date = Date(),
                             diagnosticsDir: URL? = CrashReport.defaultDiagnosticsDir()) throws -> BugReportResult {
        var zip = ZipWriter()
        var entries: [BundleEntry] = []
        func add(_ name: String, _ text: String, _ note: String) {
            let d = Data(text.utf8)
            zip.add(name: name, data: d, date: now)
            entries.append(BundleEntry(name: name, bytes: d.count, note: note))
        }

        // Doctor
        add("doctor.txt", redactor.redact(doctor.render()), "krill doctor output (paths redacted)")
        // Config (secret keys blanked) and environment (KRILL_*/OLLAMA_*/HF_* only; secret-looking values hidden)
        if let configText {
            add("config.toml", redactor.redactConfig(configText), "config with keys and tokens redacted")
        }
        let env = redactor.redactedEnvironment(environment).map { "\($0.0)=\($0.1)" }.joined(separator: "\n")
        add("environment.txt", env + (env.isEmpty ? "" : "\n"), "KRILL_*, OLLAMA_*, HF_* variables; secret-looking values hidden")

        // Event logs, last N days
        let since = Calendar.current.startOfDay(for: now).addingTimeInterval(-Double(max(1, options.days) - 1) * 86_400)
        var logCount = 0
        for f in LogQuery.eventFiles(settings: settings) {
            if let d = LogQuery.fileDay(f), d < since { continue }
            guard let text = try? String(contentsOf: f, encoding: .utf8) else { continue }
            add("logs/\(f.lastPathComponent)", redactor.redact(text), "event log (no prompts or completions)")
            logCount += 1
        }

        // Crash markers still on disk and macOS crash reports
        for m in RunMarkers.list(dir: settings.runDir) {
            let text = (try? String(contentsOf: m.file, encoding: .utf8)) ?? ""
            add("crash/\(m.file.lastPathComponent)", redactor.redact(text), "run marker (a marker for a dead pid means an unclean exit)")
        }
        let ips = CrashReport.matchingCrashReports(after: since, in: diagnosticsDir)
        for u in ips.suffix(3) {
            if let text = try? String(contentsOf: u, encoding: .utf8) {
                add("crash/\(u.lastPathComponent)", redactor.redact(text), "macOS crash report (paths redacted)")
            }
        }

        // One session, only if asked for
        var sessionNote = "none (pass --session <id> to include one)"
        if let sid = options.sessionID {
            guard let info = SessionIndex.find(id: sid, settings: settings) else {
                throw BugReportError.sessionNotFound(sid)
            }
            let url = SessionIndex.transcriptURL(info, settings: settings)
            let text = (try? String(contentsOf: url, encoding: .utf8)) ?? ""
            if options.includeContent {
                add("session/\(info.id).jsonl", redactor.redact(text), "session transcript, content KEPT (--include-content); secrets and paths still redacted")
                sessionNote = "\(info.id) with content"
            } else {
                add("session/\(info.id).jsonl", stripContent(text, redactor: redactor), "session transcript, text and tool payloads removed (use --include-content to keep)")
                sessionNote = "\(info.id) structure only"
            }
        }

        let summary = Self.summary(doctor: doctor, settings: settings, logCount: logCount, now: now)
        let manifest: [String: Any] = [
            "krill_version": version, "platform": platform, "generated": JSONLine.isoUTC(now),
            "days": options.days, "session": sessionNote,
            "redaction": "api keys, bearer tokens, server_api_key and other secret-named config/env values, e-mail addresses and the home directory path are replaced",
            "sent_anywhere": false,
            "files": entries.map { ["name": $0.name, "bytes": $0.bytes, "note": $0.note] },
        ]
        let mdata = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        zip.add(name: "manifest.json", data: mdata, date: now)
        entries.insert(BundleEntry(name: "manifest.json", bytes: mdata.count, note: "what is in this bundle"), at: 0)

        ObsPaths.ensureDir(options.outputDir)
        let name = "krill-bug-report-\(ObsPaths.dayString(now))-\(ObsPaths.timeString(now)).zip"
        let zipURL = options.outputDir.appendingPathComponent(name)
        try zip.finish().write(to: zipURL, options: .atomic)
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: zipURL.path)

        return BugReportResult(zipURL: zipURL, manifest: manifest, entries: entries,
                               issueURL: issueURL(version: version, platform: platform, summary: summary),
                               summary: summary)
    }

    /// Keep keys that describe structure; drop text and payload fields.
    static func stripContent(_ jsonl: String, redactor: Redactor) -> String {
        let drop: Set<String> = ["text", "args", "result", "detail"]
        var out: [String] = []
        for line in jsonl.split(separator: "\n") {
            guard var o = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] else { continue }
            for k in drop where o[k] != nil {
                if let s = o[k] as? String { o["\(k)_chars"] = s.count }
                o.removeValue(forKey: k)
            }
            if let d = try? JSONSerialization.data(withJSONObject: o, options: [.sortedKeys]) {
                out.append(redactor.redact(String(decoding: d, as: UTF8.self)))
            }
        }
        return out.joined(separator: "\n") + (out.isEmpty ? "" : "\n")
    }

    static func summary(doctor: DoctorReport, settings: ObsSettings, logCount: Int, now: Date) -> String {
        let bad = doctor.findings.filter { $0.severity >= .warn }
        var s = "doctor: \(bad.filter { $0.severity == .fail }.count) failure(s), \(bad.filter { $0.severity == .warn }.count) warning(s)"
        let errs = LogQuery.records(settings: settings, filter: LogFilter(since: now.addingTimeInterval(-86_400), errorsOnly: true))
        s += "; \(errs.count) error event(s) in the last 24h; \(logCount) log file(s) in the bundle"
        return s
    }

    /// A prefilled "new issue" URL. Title and body only; GitHub cannot take a file upload through a URL,
    /// so the user drags the zip into the issue themselves.
    public static func issueURL(version: String, platform: String, summary: String) -> URL? {
        var c = URLComponents(string: "https://github.com/\(repo)/issues/new")
        let body = """
        **What happened**
        (describe what you were doing and what went wrong)

        **Environment**
        - Krill: \(version)
        - Platform: \(platform)
        - \(summary)

        **Bundle**
        Drag the `krill-bug-report-*.zip` created by `krill bug-report` into this issue (review it first; nothing was uploaded).
        """
        c?.queryItems = [URLQueryItem(name: "title", value: "[bug] "), URLQueryItem(name: "body", value: body)]
        return c?.url
    }
}

public enum BugReportError: Error, CustomStringConvertible {
    case sessionNotFound(String)
    public var description: String {
        switch self { case .sessionNotFound(let s): return "no session matching '\(s)' in ~/.krill/sessions/index.jsonl (try `krill logs --project .`)" }
    }
}
