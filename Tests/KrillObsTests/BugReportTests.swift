import XCTest
@testable import KrillObs

final class BugReportTests: ObsTestCase {
    private let secrets = ["sk-ant-api03-PlantedSecret0123456789abcdef", "hf_PlantedHuggingFaceToken0123456789", "ghp_PlantedGithubToken0123456789abcdef", "s3cr3t-server-key-value"]

    private func entries(_ zip: URL) -> [String: Data] {
        // Parse the central directory of a stored-only zip.
        let d = (try? Data(contentsOf: zip)) ?? Data()
        var out: [String: Data] = [:]
        func u16(_ o: Int) -> Int { Int(d[o]) | Int(d[o + 1]) << 8 }
        func u32(_ o: Int) -> Int { u16(o) | u16(o + 2) << 16 }
        var eocd = d.count - 22
        while eocd >= 0, u32(eocd) != 0x06054b50 { eocd -= 1 }
        guard eocd >= 0 else { return out }
        var p = u32(eocd + 16)
        for _ in 0..<u16(eocd + 10) {
            let size = u32(p + 24), nlen = u16(p + 28), xlen = u16(p + 30), clen = u16(p + 32), off = u32(p + 42)
            let name = String(decoding: d[(p + 46)..<(p + 46 + nlen)], as: UTF8.self)
            let lnlen = u16(off + 26), lxlen = u16(off + 28)
            let start = off + 30 + lnlen + lxlen
            out[name] = d.subdata(in: start..<(start + size))
            p += 46 + nlen + xlen + clen
        }
        return out
    }

    private func text(_ e: [String: Data], _ n: String) -> String { String(decoding: e[n] ?? Data(), as: UTF8.self) }

    private func build(session: String? = nil, include: Bool = false, home: String = "/Users/alice") throws -> (BugReportResult, [String: Data]) {
        let day = ObsPaths.dayString(Date())
        write(settings.logsDir.appendingPathComponent("serve-\(day).jsonl"), """
        {"ts":"\(JSONLine.isoUTC(Date()))","event":"request","status":500,"error":"Bearer \(secrets[0]) rejected at /Users/alice/.krill/models","prompt_tokens":12}
        {"ts":"\(JSONLine.isoUTC(Date()))","event":"log","message":"token=\(secrets[1])"}

        """)
        write(settings.logsDir.appendingPathComponent("serve-2020-01-01.jsonl"), "{\"ts\":\"2020-01-01T00:00:00.000Z\",\"event\":\"request\",\"status\":200}\n")
        let config = """
        default_model = "gemma"
        server_api_key = "\(secrets[3])"
        tavily_api_key = "\(secrets[2])"
        models_dir = "/Users/alice/models"
        """
        let probe = FakeProbe()
        let doctor = Doctor.run(version: "0.26.0", runningBinary: "/Users/alice/bin/krill", settings: settings, configPort: 57455, probe: probe)
        let out = tmp.appendingPathComponent("out")
        let r = try BugReport.build(
            version: "0.26.0", settings: settings, doctor: doctor, platform: "TestOS 1.0 arm64", configText: config,
            environment: ["KRILL_API_KEY": secrets[3], "KRILL_LOG_LEVEL": "debug", "KRILL_X": secrets[0], "HOME": "/Users/alice"],
            options: BugReportOptions(days: 3, sessionID: session, includeContent: include, outputDir: out),
            redactor: Redactor(home: home), diagnosticsDir: tmp.appendingPathComponent("no-diagnostics"))
        return (r, entries(r.zipURL))
    }

    func testBundleHasManifestDoctorConfigEnvAndRecentLogsOnly() throws {
        let (r, e) = try build()
        XCTAssertEqual(Set(e.keys), Set(["manifest.json", "doctor.txt", "config.toml", "environment.txt", "logs/serve-\(ObsPaths.dayString(Date())).jsonl"]))
        XCTAssertFalse(e.keys.contains("logs/serve-2020-01-01.jsonl"), "outside the last N days")
        let m = try XCTUnwrap(JSONSerialization.jsonObject(with: e["manifest.json"]!) as? [String: Any])
        XCTAssertEqual(m["krill_version"] as? String, "0.26.0")
        XCTAssertEqual(m["platform"] as? String, "TestOS 1.0 arm64")
        XCTAssertEqual(m["sent_anywhere"] as? Bool, false)
        XCTAssertEqual((m["files"] as? [[String: Any]])?.count, 4)
        XCTAssertEqual(m["session"] as? String, "none (pass --session <id> to include one)")
        XCTAssertTrue(r.renderManifest().contains("manifest.json"))
        XCTAssertEqual(r.zipURL.deletingLastPathComponent().lastPathComponent, "out")
    }

    func testPlantedSecretsAndHomePathAppearNowhereInTheBundle() throws {
        let (_, e) = try build()
        for (name, data) in e {
            let t = String(decoding: data, as: UTF8.self)
            for s in secrets { XCTAssertFalse(t.contains(s), "\(s) leaked in \(name)") }
            XCTAssertFalse(t.contains("/Users/alice"), "home path leaked in \(name)")
        }
        XCTAssertTrue(text(e, "config.toml").contains("server_api_key = \"[REDACTED]\""))
        XCTAssertTrue(text(e, "config.toml").contains("default_model = \"gemma\""))
        XCTAssertTrue(text(e, "environment.txt").contains("KRILL_LOG_LEVEL=debug"))
        XCTAssertTrue(text(e, "environment.txt").contains("KRILL_API_KEY=[REDACTED]"))
        XCTAssertFalse(text(e, "environment.txt").contains("HOME"))
        let log = text(e, "logs/serve-\(ObsPaths.dayString(Date())).jsonl")
        XCTAssertTrue(log.contains("\"prompt_tokens\":12"), "numbers survive redaction")
    }

    func testZipIsReadableByTheSystemUnzip() throws {
        let (r, _) = try build()
        guard FileManager.default.isExecutableFile(atPath: "/usr/bin/unzip") else { throw XCTSkip("no unzip") }
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/unzip"); p.arguments = ["-t", r.zipURL.path]
        let pipe = Pipe(); p.standardOutput = pipe; p.standardError = pipe
        try p.run(); p.waitUntilExit()
        XCTAssertEqual(p.terminationStatus, 0, String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self))
    }

    func testSessionIsExcludedByDefaultAndIncludedOnlyWhenPicked() throws {
        let s = SessionRecorder.start(surface: "code", project: "/Users/alice/p", model: "m", provider: "local", settings: settings)!
        s.user("please use key \(secrets[0]) in /Users/alice/p")
        s.toolCall(name: "bash", arguments: "{\"command\":\"echo hi\"}")
        s.toolResult(name: "bash", content: "hi", isError: false, ms: 5)
        s.end(outcome: "completed")
        let (_, none) = try build()
        XCTAssertTrue(none.keys.allSatisfy { !$0.hasPrefix("session/") })

        let (r1, stripped) = try build(session: s.id)
        let t1 = text(stripped, "session/\(s.id).jsonl")
        XCTAssertFalse(t1.contains("please use key") || t1.contains("echo hi") || t1.contains(secrets[0]))
        XCTAssertTrue(t1.contains("\"event\":\"tool_call\"") && t1.contains("\"text_chars\""))
        XCTAssertTrue(r1.entries.first { $0.name.hasPrefix("session/") }!.note.contains("removed"))

        let (_, full) = try build(session: s.id, include: true)
        let t2 = text(full, "session/\(s.id).jsonl")
        XCTAssertTrue(t2.contains("please use key"), "--include-content keeps the text")
        XCTAssertFalse(t2.contains(secrets[0]), "...but never the secret")
        XCTAssertFalse(t2.contains("/Users/alice"))
    }

    func testUnknownSessionIsAnError() {
        XCTAssertThrowsError(try build(session: "nope-nope"))
    }

    func testIssueURLIsPrefilledWithVersionPlatformAndSummaryAndNoFile() throws {
        let (r, _) = try build()
        let url = try XCTUnwrap(r.issueURL)
        XCTAssertEqual(url.host, "github.com")
        XCTAssertEqual(url.path, "/srvsngh99/Krill/issues/new")
        let q = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems!
        let body = q.first { $0.name == "body" }!.value!
        XCTAssertTrue(body.contains("Krill: 0.26.0") && body.contains("TestOS 1.0 arm64") && body.contains("doctor:"))
        XCTAssertEqual(Set(q.map(\.name)), ["title", "body"])
        XCTAssertLessThan(url.absoluteString.count, 4000)
    }

    func testCrashMarkersAreIncluded() throws {
        RunMarkers.write(dir: settings.runDir, kind: "serve", port: 57456, version: "0.26.0", pid: 99999)
        let (_, e) = try build()
        XCTAssertTrue(e.keys.contains { $0.hasPrefix("crash/serve-99999") })
    }
}

final class MetricsTests: XCTestCase {
    func testCountersHistogramAndLabels() {
        let m = ServerMetrics()
        m.begin(); m.record(endpoint: "/v1/chat/completions", status: 200, seconds: 0.2, promptTokens: 10, completionTokens: 30)
        m.begin(); m.record(endpoint: "/v1/chat/completions", status: 500, seconds: 61)
        m.begin(); m.record(endpoint: "/v1/embeddings", status: 200, seconds: 0.004)
        XCTAssertEqual(m.totalRequests, 3); XCTAssertEqual(m.totalErrors, 1)
        let t = m.prometheus()
        XCTAssertTrue(t.contains("krill_http_requests_total{endpoint=\"/v1/chat/completions\",code=\"200\"} 1"))
        XCTAssertTrue(t.contains("krill_http_errors_total{endpoint=\"/v1/chat/completions\"} 1"))
        XCTAssertTrue(t.contains("krill_http_request_duration_seconds_bucket{endpoint=\"/v1/chat/completions\",le=\"0.25\"} 1"))
        XCTAssertTrue(t.contains("krill_http_request_duration_seconds_bucket{endpoint=\"/v1/chat/completions\",le=\"60\"} 1"))
        XCTAssertTrue(t.contains("krill_http_request_duration_seconds_bucket{endpoint=\"/v1/chat/completions\",le=\"+Inf\"} 2"))
        XCTAssertTrue(t.contains("krill_http_request_duration_seconds_count{endpoint=\"/v1/embeddings\"} 1"))
        XCTAssertTrue(t.contains("krill_prompt_tokens_total 10") && t.contains("krill_completion_tokens_total 30"))
        XCTAssertTrue(t.contains("krill_http_in_flight 0"))
    }

    func testEndpointLabelsAreBounded() {
        XCTAssertEqual(ServerMetrics.endpointLabel(method: "GET", path: "/v1/models/llama-3"), "/v1/models/:id")
        XCTAssertEqual(ServerMetrics.endpointLabel(method: "GET", path: "/v1/models"), "/v1/models")
        XCTAssertEqual(ServerMetrics.endpointLabel(method: "POST", path: "/v1/agent/sessions/abc/messages"), "/v1/agent/*")
        XCTAssertEqual(ServerMetrics.endpointLabel(method: "GET", path: "/wp-login.php"), "other")
    }

    func testProcessUptimeIsTheProcessNotTheMachine() {
        let up = ProcessStats.uptimeSeconds()
        XCTAssertGreaterThanOrEqual(up, 0)
        XCTAssertLessThan(up, ProcessInfo.processInfo.systemUptime)
        XCTAssertGreaterThan(ProcessStats.physFootprintBytes(), 1_000_000)
    }

    func testSessionLogModeParsing() {
        XCTAssertEqual(SessionLogMode.parse("FULL"), .full)
        XCTAssertEqual(SessionLogMode.parse("meta"), .meta)
        XCTAssertEqual(SessionLogMode.parse("off"), .off)
        XCTAssertEqual(SessionLogMode.parse("0"), .off)
        XCTAssertNil(SessionLogMode.parse("loud"))
    }
}
