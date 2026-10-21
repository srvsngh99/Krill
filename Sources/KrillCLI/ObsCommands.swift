import ArgumentParser
import Foundation
import KrillObs
import KrillRegistry

private func currentSettings() -> ObsSettings { KrillConfig.load().obsSettings }

private func runningBinaryPath() -> String {
    Bundle.main.executablePath ?? CommandLine.arguments.first ?? "krill"
}

/// `krill logs`: read the local event log and session transcripts. Read-only.
struct LogsCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "logs",
        abstract: "Show Krill's local event log and agent sessions.",
        discussion: """
            Reads ~/.krill/logs (server events) and ~/.krill/sessions (agent transcripts). \
            Nothing is sent anywhere. By default shows the last 50 events.
            """)

    @Flag(name: .long, help: "Only events from today.")
    var today = false

    @Option(name: .long, help: "Only events since a duration (30m, 2h, 7d, 1w) or a date (2026-10-20).")
    var since: String?

    @Flag(name: .long, help: "Only errors: failed requests, warnings and errors, load failures, unclean exits.")
    var errors = false

    @Option(name: .long, help: "Show one agent session: its transcript plus the server requests it made. Accepts an id prefix.")
    var session: String?

    @Option(name: .long, help: "List the agent sessions started in this project directory (or below it).")
    var project: String?

    @Flag(name: .long, help: "Print raw JSON lines.")
    var json = false

    @Flag(name: [.customShort("f"), .long], help: "Keep running and print new events as they arrive.")
    var follow = false

    @Option(name: [.customShort("n"), .long], help: "Number of events to show when no --since/--today is given.")
    var lines = 50

    func run() throws {
        let settings = currentSettings()

        if let project {
            let rows = LogQuery.sessions(forProject: project, settings: settings)
            if rows.isEmpty { print("No sessions recorded for \(project)."); return }
            for r in rows.prefix(100) {
                if json {
                    print(String(decoding: (try? JSONSerialization.data(withJSONObject: [
                        "id": r.id, "start": JSONLine.isoUTC(r.start), "end": r.end.map(JSONLine.isoUTC) as Any,
                        "project": r.project as Any, "surface": r.surface, "model": r.model as Any,
                        "provider": r.provider as Any, "outcome": r.outcome as Any,
                    ], options: [.sortedKeys])) ?? Data(), as: UTF8.self))
                } else { print(LogQuery.renderSession(r)) }
            }
            return
        }

        if let session {
            try showSession(session, settings: settings)
            return
        }

        var since = self.since.flatMap { LogQuery.parseSince($0) }
        if self.since != nil, since == nil {
            throw ValidationError("cannot read --since '\(self.since!)'. Use 30m, 2h, 7d, 1w or a date like 2026-10-20.")
        }
        if today { since = LogQuery.startOfToday() }
        let filter = LogFilter(since: since, errorsOnly: errors)
        var recs = LogQuery.records(settings: settings, filter: filter)
        if since == nil { recs = Array(recs.suffix(max(1, lines))) }
        if recs.isEmpty && !follow {
            let files = LogQuery.eventFiles(settings: settings)
            print(files.isEmpty
                ? "No event log yet in \(settings.logsDir.path). It is written by `krill serve` (event_log = on)."
                : "No matching events.")
            return
        }
        for r in recs { print(json ? r.raw : LogQuery.render(r)) }
        if follow {
            let f = LogFollower(settings: settings, filter: filter)
            fflush(stdout)
            while true {
                for r in f.poll() { print(json ? r.raw : LogQuery.render(r)) }
                fflush(stdout)
                Thread.sleep(forTimeInterval: 0.5)
            }
        }
    }

    private func showSession(_ id: String, settings: ObsSettings) throws {
        guard let info = SessionIndex.find(id: id, settings: settings) else {
            throw ValidationError("no session '\(id)'. List sessions with: krill logs --project .")
        }
        if !json { print("session \(info.id)  \(LogQuery.renderSession(info))\n") }
        var rows: [(Date, String)] = []
        if let text = try? String(contentsOf: SessionIndex.transcriptURL(info, settings: settings), encoding: .utf8) {
            for line in text.split(separator: "\n") {
                guard let r = LogQuery.parse(line: line) else { continue }
                var out = json ? r.raw : LogQuery.render(r)
                if !json, out.count > 400 { out = String(out.prefix(400)) + " ..." }
                rows.append((r.ts, out))
            }
        }
        for r in LogQuery.records(settings: settings, filter: LogFilter(session: info.id)) {
            rows.append((r.ts, json ? r.raw : LogQuery.render(r)))
        }
        for (_, l) in rows.sorted(by: { $0.0 < $1.0 }) { print(l) }
    }
}

/// `krill doctor`: one read-only screen of everything that usually explains "Krill is broken".
struct DoctorCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "doctor",
        abstract: "Check installs, launchd jobs, ports, disk and recent errors (read-only).",
        discussion: "Changes nothing: it never starts, stops, loads or unloads anything.")

    @Flag(name: .long, help: "Print JSON.")
    var json = false

    @Flag(name: .long, help: "Exit 1 when any check fails.")
    var strict = false

    func run() throws {
        let cfg = KrillConfig.load()
        let report = Doctor.run(version: KrillVersion, runningBinary: runningBinaryPath(),
                                settings: cfg.obsSettings, configPort: cfg.serverPort, probe: LiveProbe())
        if json {
            let d = try JSONSerialization.data(withJSONObject: report.jsonObject(), options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: d, as: UTF8.self))
        } else {
            print(report.render(), terminator: "")
        }
        if strict, report.worst == .fail { throw ExitCode.failure }
    }
}

/// `krill bug-report`: build a redacted zip for the user to attach to an issue themselves.
struct BugReportCommand: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "bug-report",
        abstract: "Bundle redacted logs and diagnostics into a zip you can attach to a GitHub issue.",
        discussion: """
            Writes ONE zip to the current directory and prints what is in it. Nothing is uploaded: \
            Krill sends nothing anywhere. You review the zip and attach it yourself.
            """)

    @Option(name: .long, help: "Days of server logs to include.")
    var days = 3

    @Option(name: .long, help: "Include this one agent session (id from `krill logs --project .`). Text and tool payloads are removed unless --include-content.")
    var session: String?

    @Flag(name: .long, help: "Keep the session's text and tool payloads (secrets and paths are still redacted).")
    var includeContent = false

    @Option(name: .long, help: "Directory for the zip (default: current directory).")
    var output: String?

    func run() throws {
        let cfg = KrillConfig.load()
        let settings = cfg.obsSettings
        let probe = LiveProbe()
        let doctor = Doctor.run(version: KrillVersion, runningBinary: runningBinaryPath(),
                                settings: settings, configPort: cfg.serverPort, probe: probe)
        let configText = try? String(contentsOf: KrillConfig.configPath(), encoding: .utf8)
        let outDir = URL(fileURLWithPath: output ?? FileManager.default.currentDirectoryPath, isDirectory: true)
        let result: BugReportResult
        do {
            result = try BugReport.build(
                version: KrillVersion, settings: settings, doctor: doctor, platform: probe.platform(),
                configText: configText, environment: ProcessInfo.processInfo.environment,
                options: BugReportOptions(days: days, sessionID: session, includeContent: includeContent, outputDir: outDir))
        } catch let e as BugReportError {
            throw ValidationError(e.description)
        }
        print(result.renderManifest())
        print("Redaction: API keys, bearer tokens, server_api_key and other secret-named values, e-mail addresses and your home path are replaced.")
        print("Nothing was sent anywhere. Review the zip, then attach it to the issue yourself.\n")
        guard let url = result.issueURL else { return }
        print("Prefilled issue (title and body only; no file is uploaded):\n\(url.absoluteString)\n")

        // Open the browser only if a person is here to say yes.
        guard isatty(STDIN_FILENO) != 0, isatty(STDOUT_FILENO) != 0 else { return }
        print("Open this in your browser now? [y/N] ", terminator: "")
        fflush(stdout)
        if let ans = readLine()?.trimmingCharacters(in: .whitespaces).lowercased(), ans == "y" || ans == "yes" {
            let p = Process()
            #if os(macOS)
            p.executableURL = URL(fileURLWithPath: "/usr/bin/open")
            #else
            p.executableURL = URL(fileURLWithPath: "/usr/bin/xdg-open")
            #endif
            p.arguments = [url.absoluteString]
            try? p.run()
        }
    }
}
