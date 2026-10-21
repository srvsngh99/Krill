import Foundation
#if canImport(Darwin)
import Darwin
#endif

public enum Severity: String, Sendable, Comparable {
    case ok, info, warn, fail
    public static func < (a: Severity, b: Severity) -> Bool { a.rank < b.rank }
    var rank: Int { switch self { case .ok: 0; case .info: 1; case .warn: 2; case .fail: 3 } }
    var tag: String { switch self { case .ok: "ok  "; case .info: "info"; case .warn: "WARN"; case .fail: "FAIL" } }
}

public struct Finding: Sendable, Equatable {
    public var severity: Severity
    public var section: String
    public var message: String
    public init(_ severity: Severity, _ section: String, _ message: String) {
        self.severity = severity; self.section = section; self.message = message
    }
}

public struct KrillInstall: Sendable, Equatable {
    public var path: String            // as found (e.g. /usr/local/bin/krill)
    public var resolved: String        // symlinks resolved
    public var version: String?
    public var codesign: String?
    public init(path: String, resolved: String, version: String? = nil, codesign: String? = nil) {
        self.path = path; self.resolved = resolved; self.version = version; self.codesign = codesign
    }
}

public struct LaunchdJob: Sendable, Equatable {
    public var label: String
    public var plist: String
    public var program: String
    public var arguments: [String]
    public var loaded: Bool
    public var state: String?
    public var pid: Int?
    public var runs: Int?
    public var lastExit: String?
    public var port: Int? {
        if let i = arguments.firstIndex(of: "--port"), i + 1 < arguments.count { return Int(arguments[i + 1]) }
        return arguments.compactMap { $0.hasPrefix("--port=") ? Int($0.dropFirst(7)) : nil }.first
    }
    public init(label: String, plist: String, program: String, arguments: [String] = [], loaded: Bool,
                state: String? = nil, pid: Int? = nil, runs: Int? = nil, lastExit: String? = nil) {
        self.label = label; self.plist = plist; self.program = program; self.arguments = arguments
        self.loaded = loaded; self.state = state; self.pid = pid; self.runs = runs; self.lastExit = lastExit
    }
}

public struct PortStatus: Sendable, Equatable {
    public var port: Int
    public var listening: Bool
    public var healthz: String?        // "ok model=x" / "http 503" / nil when not answering
    public var owner: String?          // process name when known
    public init(port: Int, listening: Bool, healthz: String? = nil, owner: String? = nil) {
        self.port = port; self.listening = listening; self.healthz = healthz; self.owner = owner
    }
}

/// Everything doctor reads from the machine. Read-only by construction: no method starts, stops,
/// loads, unloads or writes anything. Tests supply a fake.
public protocol SystemProbe: Sendable {
    func installs() -> [KrillInstall]
    func launchdJobs() -> [LaunchdJob]
    func portStatus(_ port: Int) -> PortStatus
    func diskFree(at path: String) -> (free: Int64, total: Int64)?
    func crashReports(limit: Int) -> [URL]
    func platform() -> String
}

public struct DoctorReport: Sendable {
    public var version: String
    public var runningBinary: String
    public var findings: [Finding]
    public var generated: Date

    public var worst: Severity { findings.map(\.severity).max() ?? .ok }

    /// One screen of text.
    public func render() -> String {
        var out = "krill doctor  v\(version)  \(runningBinary)\n"
        var section = ""
        for f in findings {
            if f.section != section { section = f.section; out += "\n\(section)\n" }
            out += "  [\(f.severity.tag)] \(f.message)\n"
        }
        let w = findings.filter { $0.severity == .warn }.count, x = findings.filter { $0.severity == .fail }.count
        out += "\n" + (x + w == 0 ? "All checks passed." : "\(x) failure(s), \(w) warning(s).")
        out += "  (read-only: doctor changes nothing)\n"
        return out
    }

    public func jsonObject() -> [String: Any] {
        ["version": version, "binary": runningBinary, "generated": JSONLine.isoUTC(generated),
         "worst": worst.rawValue,
         "findings": findings.map { ["severity": $0.severity.rawValue, "section": $0.section, "message": $0.message] }]
    }
}

public enum Doctor {
    public static let lowDiskWarnBytes: Int64 = 20 * 1_073_741_824
    public static let lowDiskFailBytes: Int64 = 5 * 1_073_741_824

    public static func run(version: String, runningBinary: String, settings: ObsSettings,
                           configPort: Int, probe: SystemProbe, now: Date = Date()) -> DoctorReport {
        var f: [Finding] = []

        // 1. Installs
        let installs = probe.installs()
        let distinct = Dictionary(grouping: installs, by: \.resolved)
        if installs.isEmpty {
            f.append(Finding(.warn, "Install", "no krill binary found on PATH or in the usual install locations"))
        }
        for i in installs {
            let link = i.path == i.resolved ? "" : " -> \(i.resolved)"
            f.append(Finding(.info, "Install", "\(i.path)\(link)  v\(i.version ?? "?")" + (i.codesign.map { "  codesign: \($0)" } ?? "")))
            if let cs = i.codesign, cs.contains("invalid") || cs.contains("not signed") || cs.contains("modified") {
                f.append(Finding(.fail, "Install", "\(i.path): code signature problem (\(cs)). macOS will SIGKILL a launchd job that runs a binary whose signature no longer matches; reinstall instead of replacing it in place."))
            }
        }
        if distinct.count > 1 {
            f.append(Finding(.warn, "Install", "\(distinct.count) different krill installs exist (\(distinct.keys.sorted().joined(separator: ", "))); `krill` on PATH and a launchd job may not be the same build"))
            let versions = Set(installs.compactMap(\.version))
            if versions.count > 1 { f.append(Finding(.warn, "Install", "installs disagree on version: \(versions.sorted().joined(separator: ", "))")) }
        }
        let runningResolved = URL(fileURLWithPath: runningBinary).resolvingSymlinksInPath().path

        // 2. launchd
        let jobs = probe.launchdJobs()
        if jobs.isEmpty {
            f.append(Finding(.info, "launchd", "no launchd job runs krill"))
        }
        for j in jobs {
            let prog = URL(fileURLWithPath: j.program).resolvingSymlinksInPath().path
            let which = installs.first(where: { $0.resolved == prog || $0.path == j.program })
            let uses = "runs \(j.program)" + (which.map { " (v\($0.version ?? "?"))" } ?? "")
                + (j.port.map { ", port \($0)" } ?? "")
            if !j.loaded {
                f.append(Finding(.fail, "launchd", "\(j.label): NOT LOADED  \(uses). It will not run or restart until it is bootstrapped; nothing records who unloaded it. (plist: \(j.plist))"))
            } else {
                var line = "\(j.label): loaded, \(j.state ?? "state unknown")" + (j.pid.map { ", pid \($0)" } ?? "")
                if let r = j.runs { line += ", runs \(r)" }
                if let e = j.lastExit { line += ", last exit: \(e)" }
                line += "; \(uses)"
                let exit = (j.lastExit ?? "").uppercased()
                let killed = exit.contains("CODESIGN") || exit.contains("SIGKILL")
                let running = j.pid != nil
                f.append(Finding(killed ? (running ? .warn : .fail) : .ok, "launchd", line))
                if killed {
                    f.append(Finding(running ? .warn : .fail, "launchd",
                        "\(j.label): its last exit was a code-signing kill (a binary replaced in place under a loaded job)"
                        + (running ? "; launchd restarted it and it is running again. Install to a new path or bootout/bootstrap around upgrades." : "; it is NOT running. Reinstall and restart the job.")))
                }
            }
            if which == nil, !j.program.isEmpty, !FileManager.default.fileExists(atPath: j.program) {
                f.append(Finding(.fail, "launchd", "\(j.label): program \(j.program) does not exist"))
            }
        }

        // 3. Ports
        var ports = Set([configPort, 57455, 57456, 57457])
        for j in jobs { if let p = j.port { ports.insert(p) } }
        for p in ports.sorted() {
            let s = probe.portStatus(p)
            let expected = jobs.contains(where: { $0.port == p })
            if s.listening {
                let ans = s.healthz.map { "answers /healthz: \($0)" } ?? "open but /healthz does not answer"
                f.append(Finding(s.healthz == nil ? .warn : .ok, "Ports", ":\(p) in use\(s.owner.map { " by \($0)" } ?? ""); \(ans)"))
            } else if expected {
                f.append(Finding(.fail, "Ports", ":\(p) nothing is listening, but a launchd job is configured for it"))
            } else if p == configPort {
                f.append(Finding(.info, "Ports", ":\(p) free (the configured default; `krill serve` is not running)"))
            }
        }

        // 4. Disk
        let dir = settings.root.path
        if let d = probe.diskFree(at: dir) {
            let pct = d.total > 0 ? Double(d.free) / Double(d.total) * 100 : 0
            let gb = Double(d.free) / 1_073_741_824
            let line = String(format: "%.1f GB free (%.1f%%) on the volume holding %@", gb, pct, dir)
            if d.free < lowDiskFailBytes { f.append(Finding(.fail, "Disk", line + "; model loads, caches and swap can fail below this")) }
            else if d.free < lowDiskWarnBytes || pct < 5 { f.append(Finding(.warn, "Disk", line + "; low")) }
            else { f.append(Finding(.ok, "Disk", line)) }
        }
        let sz = directorySize(settings.logsDir) + directorySize(settings.sessionsDir)
        f.append(Finding(.info, "Disk", String(format: "krill logs + sessions use %.1f MB (cap %d MB, kept %d days)",
                                                Double(sz) / 1_048_576, settings.maxMB, settings.retentionDays)))

        // 5. Crashes and errors
        let unclean = LogQuery.records(settings: settings, filter: LogFilter(since: now.addingTimeInterval(-14 * 86_400)))
            .filter { $0.event == "unclean_exit" }
        if let last = unclean.last {
            let ips = (last.fields["crash_reports"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            f.append(Finding(.warn, "Crashes", "unclean exit seen \(LogQuery.localTime(last.ts)) (pid \(last.fields["dead_pid"] ?? "?"), v\(last.fields["version"] ?? "?"))" + (ips.map { "; crash report: \($0)" } ?? "; no macOS crash report matched")))
        }
        let reports = probe.crashReports(limit: 3)
        if let r = reports.last {
            f.append(Finding(.warn, "Crashes", "latest macOS crash report: \(r.path) (\(reports.count) recent)"))
        }
        if unclean.isEmpty && reports.isEmpty { f.append(Finding(.ok, "Crashes", "no unclean exits or crash reports found")) }
        let errs = LogQuery.records(settings: settings, filter: LogFilter(since: now.addingTimeInterval(-86_400), errorsOnly: true))
            .filter { $0.event != "unclean_exit" }
        if errs.isEmpty {
            f.append(Finding(.ok, "Recent errors", "none in the last 24 hours" + (LogQuery.eventFiles(settings: settings).isEmpty ? " (no event log yet: run `krill serve` once with logging on)" : "")))
        } else {
            f.append(Finding(.warn, "Recent errors", "\(errs.count) in the last 24 hours; latest:"))
            for e in errs.suffix(5) { f.append(Finding(.info, "Recent errors", LogQuery.render(e))) }
        }
        _ = runningResolved
        return DoctorReport(version: version, runningBinary: runningBinary, findings: f, generated: now)
    }

    static func directorySize(_ dir: URL) -> Int {
        guard let en = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.fileSizeKey]) else { return 0 }
        var n = 0
        for case let u as URL in en { n += (try? u.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0 }
        return n
    }
}

// MARK: - Live probe

/// The real machine. Every call is a read: `which`-style path checks, `krill version`,
/// `codesign -dv`, `launchctl print` (print, never bootstrap/bootout), TCP connect + GET /healthz,
/// `statfs`, directory listings.
public struct LiveProbe: SystemProbe {
    /// Directories scanned for launchd plists as `path` (user domain unless under /Library/LaunchDaemons).
    /// `KRILL_DOCTOR_LAUNCHD_DIRS` (colon separated) replaces the defaults, for fixtures and tests.
    private let launchDirs: [String]?

    public init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        launchDirs = environment["KRILL_DOCTOR_LAUNCHD_DIRS"].map { $0.split(separator: ":").map(String.init) }
    }

    public func platform() -> String {
        let v = ProcessInfo.processInfo.operatingSystemVersion
        var u = utsname(); uname(&u)
        let machine = withUnsafePointer(to: &u.machine) { $0.withMemoryRebound(to: CChar.self, capacity: 256) { String(cString: $0) } }
        #if os(macOS)
        return "macOS \(v.majorVersion).\(v.minorVersion).\(v.patchVersion) \(machine)"
        #else
        return "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion) \(machine)"
        #endif
    }

    public func installs() -> [KrillInstall] {
        let home = NSHomeDirectory()
        var candidates = ["/usr/local/bin/krill", "/opt/homebrew/bin/krill", "\(home)/.local/bin/krill", "\(home)/.krill/bin/krill"]
        for dir in (ProcessInfo.processInfo.environment["PATH"] ?? "").split(separator: ":") {
            candidates.append("\(dir)/krill")
        }
        var seen = Set<String>()
        var out: [KrillInstall] = []
        for c in candidates where seen.insert(c).inserted {
            guard FileManager.default.isExecutableFile(atPath: c) else { continue }
            let resolved = URL(fileURLWithPath: c).resolvingSymlinksInPath().path
            let ver = Shell.run("/usr/bin/env", [resolved, "version"], timeout: 4).stdout
                .split(separator: "\n").first.map(String.init)?.trimmingCharacters(in: .whitespaces)
            let cs = Self.codesignSummary(resolved)
            out.append(KrillInstall(path: c, resolved: resolved,
                                    version: ver.map { $0.replacingOccurrences(of: "krill ", with: "").replacingOccurrences(of: "Krill ", with: "").trimmingCharacters(in: CharacterSet(charactersIn: "vV")) },
                                    codesign: cs))
        }
        return out
    }

    static func codesignSummary(_ path: String) -> String? {
        #if os(macOS)
        let verify = Shell.run("/usr/bin/codesign", ["--verify", "--strict", path], timeout: 8)
        let info = Shell.run("/usr/bin/codesign", ["-dv", path], timeout: 8)
        let flags = (info.stderr + info.stdout).split(separator: "\n").first { $0.contains("flags=") }
            .map { String($0.drop(while: { $0 != "(" }).dropFirst().prefix(while: { $0 != ")" })) } ?? "unknown"
        if verify.status == 0 { return "valid (\(flags))" }
        let why = verify.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            .replacingOccurrences(of: path, with: "binary").prefix(120)
        return "invalid: \(why)"
        #else
        return nil
        #endif
    }

    public func launchdJobs() -> [LaunchdJob] {
        #if os(macOS)
        let home = NSHomeDirectory()
        let uid = getuid()
        var jobs: [LaunchdJob] = []
        var dirs = [("\(home)/Library/LaunchAgents", "gui/\(uid)"), ("/Library/LaunchAgents", "gui/\(uid)"), ("/Library/LaunchDaemons", "system")]
        if let override = launchDirs { dirs = override.map { ($0, $0.hasPrefix("/Library/LaunchDaemons") ? "system" : "gui/\(uid)") } }
        for (dir, domain) in dirs {
            let files = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
            for f in files where f.hasSuffix(".plist") {
                let path = "\(dir)/\(f)"
                guard let d = NSDictionary(contentsOfFile: path) as? [String: Any],
                      let label = d["Label"] as? String else { continue }
                var args = d["ProgramArguments"] as? [String] ?? []
                let program = (d["Program"] as? String) ?? args.first ?? ""
                // Only jobs that run the krill binary itself (not tools that merely mention krill).
                let isKrill = URL(fileURLWithPath: program).lastPathComponent == "krill"
                    || (args.count > 1 && URL(fileURLWithPath: args[0]).lastPathComponent == "krill")
                guard isKrill else { continue }
                if args.isEmpty { args = [program] }
                let printed = Shell.run("/bin/launchctl", ["print", "\(domain)/\(label)"], timeout: 6)
                let loaded = printed.status == 0
                var job = LaunchdJob(label: label, plist: path, program: program, arguments: args, loaded: loaded)
                if loaded { Self.parseLaunchctl(printed.stdout, into: &job) }
                jobs.append(job)
            }
        }
        return jobs
        #else
        return []
        #endif
    }

    static func parseLaunchctl(_ text: String, into job: inout LaunchdJob) {
        for raw in text.split(separator: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            func val(_ key: String) -> String? {
                guard line.hasPrefix(key + " = ") else { return nil }
                return String(line.dropFirst(key.count + 3))
            }
            if let v = val("state") { job.state = v }
            if let v = val("pid") { job.pid = Int(v) }
            if let v = val("runs") { job.runs = Int(v) }
            if let v = val("last exit reason") { job.lastExit = v }
            else if let v = val("last exit code"), job.lastExit == nil { job.lastExit = "code \(v)" }
        }
    }

    public func portStatus(_ port: Int) -> PortStatus {
        let fd = socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return PortStatus(port: port, listening: false) }
        defer { close(fd) }
        var addr = sockaddr_in()
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(port).bigEndian)
        addr.sin_addr.s_addr = inet_addr("127.0.0.1")
        let flags = fcntl(fd, F_GETFL)
        _ = fcntl(fd, F_SETFL, flags | O_NONBLOCK)
        let rc = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { connect(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        var connected = rc == 0
        if rc != 0 && errno == EINPROGRESS {
            var p = pollfd(fd: fd, events: Int16(POLLOUT), revents: 0)
            if poll(&p, 1, 400) > 0 {
                var err: Int32 = 0; var len = socklen_t(MemoryLayout<Int32>.size)
                getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len)
                connected = err == 0
            }
        }
        guard connected else { return PortStatus(port: port, listening: false) }
        _ = fcntl(fd, F_SETFL, flags)
        var tv = timeval(tv_sec: 1, tv_usec: 0)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, socklen_t(MemoryLayout<timeval>.size))
        let req = "GET /healthz HTTP/1.0\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
        _ = req.withCString { send(fd, $0, strlen($0), 0) }
        var buf = [UInt8](repeating: 0, count: 4096)
        let n = recv(fd, &buf, buf.count, 0)
        var health: String?
        if n > 0 {
            let text = String(decoding: buf.prefix(n), as: UTF8.self)
            let status = text.split(separator: " ").dropFirst().first.map(String.init) ?? "?"
            if status == "200" {
                let body = text.components(separatedBy: "\r\n\r\n").last ?? ""
                if let o = (try? JSONSerialization.jsonObject(with: Data(body.utf8))) as? [String: Any] {
                    health = "ok" + ((o["model"] as? String).map { $0 == "none" ? ", no model loaded" : ", model \($0)" } ?? "")
                } else { health = "http 200" }
            } else { health = "http \(status)" }
        }
        let owner = Shell.run("/usr/sbin/lsof", ["-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-Fc"], timeout: 3).stdout
            .split(separator: "\n").first { $0.hasPrefix("c") }.map { String($0.dropFirst()) }
        return PortStatus(port: port, listening: true, healthz: health, owner: owner)
    }

    public func diskFree(at path: String) -> (free: Int64, total: Int64)? {
        var p = path
        while !FileManager.default.fileExists(atPath: p), p != "/" { p = (p as NSString).deletingLastPathComponent }
        let url = URL(fileURLWithPath: p)
        guard let v = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey, .volumeTotalCapacityKey]) else { return nil }
        let free = v.volumeAvailableCapacityForImportantUsage ?? 0
        return (free, Int64(v.volumeTotalCapacity ?? 0))
    }

    public func crashReports(limit: Int) -> [URL] {
        guard let dir = CrashReport.defaultDiagnosticsDir() else { return [] }
        return Array(CrashReport.matchingCrashReports(after: Date().addingTimeInterval(-14 * 86_400), in: dir).suffix(limit))
    }
}

/// Minimal bounded subprocess runner (read-only callers only).
enum Shell {
    struct Result { var status: Int32; var stdout: String; var stderr: String }

    static func run(_ exe: String, _ args: [String], timeout: TimeInterval) -> Result {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: exe)
        p.arguments = args
        let o = Pipe(), e = Pipe()
        p.standardOutput = o; p.standardError = e; p.standardInput = FileHandle.nullDevice
        do { try p.run() } catch { return Result(status: -1, stdout: "", stderr: "\(error)") }
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { p.waitUntilExit(); done.signal() }
        if done.wait(timeout: .now() + timeout) == .timedOut { p.terminate(); return Result(status: -2, stdout: "", stderr: "timeout") }
        let out = String(decoding: o.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        let err = String(decoding: e.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        return Result(status: p.terminationStatus, stdout: out, stderr: err)
    }
}
