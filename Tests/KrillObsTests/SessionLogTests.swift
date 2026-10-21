import XCTest
@testable import KrillObs

final class SessionLogTests: ObsTestCase {
    private func start(_ s: ObsSettings? = nil, project: String = "/work/proj") -> SessionRecorder {
        SessionRecorder.start(surface: "code", project: project, model: "qwen", provider: "local",
                              version: "1", settings: s ?? settings)!
    }

    func testFullTranscriptHasEveryTurnToolPermissionRetryAndStop() throws {
        let r = start()
        r.user("fix the bug in main.swift")
        r.assistant("I will read it", toolCalls: 1)
        r.toolCall(name: "read_file", arguments: "{\"path\":\"main.swift\"}")
        r.permission(tool: "read_file", decision: "allow")
        r.retry(kind: "arg_repair", tool: "read_file", detail: "accepted")
        r.toolResult(name: "read_file", content: "print(1)", isError: false, ms: 12)
        r.error("boom")
        r.stop(reason: "final_answer")
        r.end(outcome: "completed")
        let evs = lines(r.fileURL).map { json($0) }
        XCTAssertEqual(evs.map { $0["event"] as? String },
                       ["session_start", "user", "assistant", "tool_call", "permission", "retry", "tool_result", "error", "stop", "session_end"])
        XCTAssertEqual(evs[1]["text"] as? String, "fix the bug in main.swift")
        XCTAssertEqual(evs[3]["args"] as? String, "{\"path\":\"main.swift\"}")
        XCTAssertEqual(evs[6]["result"] as? String, "print(1)")
        XCTAssertEqual(evs[6]["ms"] as? Int, 12)
        XCTAssertEqual(evs[9]["outcome"] as? String, "completed")
        XCTAssertEqual(evs[9]["tool_calls"] as? Int, 1)
    }

    func testFileLivesUnderYearMonthDayWithTimeAndID() {
        let r = start()
        r.end(outcome: "completed")
        let parts = ObsPaths.dayString(Date()).split(separator: "-").map(String.init)
        XCTAssertTrue(r.fileURL.path.contains("/sessions/\(parts[0])/\(parts[1])/\(parts[2])/"))
        XCTAssertTrue(r.fileURL.lastPathComponent.hasSuffix("-\(r.id).jsonl"))
        XCTAssertEqual(r.fileURL.lastPathComponent.prefix(6).allSatisfy(\.isNumber), true)
    }

    func testMetaModeDropsTextAndToolPayloads() {
        var m = settings!; m.sessionLog = .meta
        let r = start(m)
        r.user("my secret plan")
        r.assistant("here is the answer", toolCalls: 1)
        r.toolCall(name: "bash", arguments: "{\"command\":\"cat ~/.ssh/id_rsa\"}")
        r.toolResult(name: "bash", content: "-----BEGIN KEY-----", isError: true, ms: 3)
        r.end(outcome: "completed")
        let raw = (try? String(contentsOf: r.fileURL, encoding: .utf8)) ?? ""
        XCTAssertFalse(raw.contains("secret plan"))
        XCTAssertFalse(raw.contains("id_rsa"))
        XCTAssertFalse(raw.contains("BEGIN KEY"))
        XCTAssertFalse(raw.contains("here is the answer"))
        let evs = lines(r.fileURL).map { json($0) }
        XCTAssertEqual(evs[1]["text_chars"] as? Int, 14)
        XCTAssertEqual(evs[3]["tool"] as? String, "bash")
        XCTAssertEqual(evs[4]["is_error"] as? Bool, true)
        XCTAssertNotNil(evs[4]["result_bytes"])
    }

    func testOffWritesNothingAndReturnsNil() {
        var o = settings!; o.sessionLog = .off
        XCTAssertNil(SessionRecorder.start(surface: "run", project: nil, model: nil, provider: nil, settings: o))
        XCTAssertFalse(FileManager.default.fileExists(atPath: o.sessionsDir.path))
    }

    func testFieldsAreTruncatedAtTheCapOnAScalarBoundary() {
        let r = start()
        r.toolResult(name: "bash", content: String(repeating: "é", count: 20_000), isError: false, ms: 1)
        r.end(outcome: "completed")
        let ev = json(lines(r.fileURL)[1])
        let kept = (ev["result"] as? String) ?? ""
        XCTAssertLessThanOrEqual(kept.utf8.count, ObsLimits.fieldCap)
        XCTAssertGreaterThan(kept.utf8.count, ObsLimits.fieldCap - 4)
        XCTAssertEqual(ev["result_truncated_bytes"] as? Int, 40_000 - kept.utf8.count)
        XCTAssertFalse(kept.contains("\u{FFFD}"))
    }

    func testIndexRowCarriesIdStartEndProjectSurfaceModelProviderOutcome() throws {
        let a = start(project: "/work/a"); a.user("x"); a.end(outcome: "completed")
        let b = start(project: "/work/b/sub")   // never ended: crashed or still open
        b.flush()
        let all = SessionIndex.load(settings: settings)
        XCTAssertEqual(all.count, 2)
        let ra = try XCTUnwrap(all.first { $0.id == a.id })
        XCTAssertEqual(ra.project, "/work/a"); XCTAssertEqual(ra.surface, "code")
        XCTAssertEqual(ra.model, "qwen"); XCTAssertEqual(ra.provider, "local")
        XCTAssertEqual(ra.outcome, "completed"); XCTAssertNotNil(ra.end)
        let rb = try XCTUnwrap(all.first { $0.id == b.id })
        XCTAssertNil(rb.end); XCTAssertNil(rb.outcome)
        XCTAssertEqual(SessionIndex.find(id: String(a.id.prefix(5)), settings: settings)?.id, a.id)
    }

    func testProjectFilterMatchesTheDirectoryAndBelow() {
        let a = start(project: "/work/a"); a.end(outcome: "completed")
        let b = start(project: "/work/b/sub"); b.end(outcome: "completed")
        XCTAssertEqual(LogQuery.sessions(forProject: "/work/a", settings: settings).map(\.id), [a.id])
        XCTAssertEqual(LogQuery.sessions(forProject: "/work/b", settings: settings).map(\.id), [b.id])
        XCTAssertEqual(LogQuery.sessions(forProject: "/work", settings: settings).count, 2)
        XCTAssertEqual(LogQuery.sessions(forProject: "/work/ab", settings: settings).count, 0)
    }

    func testRecordingNeverBlocks() {
        let r = start()
        let t0 = DispatchTime.now().uptimeNanoseconds
        for i in 0..<2000 { r.toolResult(name: "t", content: "result \(i)", isError: false, ms: 1) }
        let perCallUs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000 / 2000
        r.end(outcome: "completed")
        XCTAssertLessThan(perCallUs, 100)
    }
}

final class RetentionTests: ObsTestCase {
    func testOldLogsAndSessionsAreDeletedAndIndexIsCompacted() {
        let oldLog = settings.logsDir.appendingPathComponent("serve-2026-08-01.jsonl")
        let newLog = settings.logsDir.appendingPathComponent("serve-2026-10-21.jsonl")
        write(oldLog, "{}\n"); write(newLog, "{}\n")
        touch(oldLog, daysAgo: 45); touch(newLog, daysAgo: 1)
        let old = SessionRecorder.start(surface: "run", project: "/p", model: "m", provider: "local", settings: settings)!
        old.end(outcome: "completed")
        let fresh = SessionRecorder.start(surface: "run", project: "/p", model: "m", provider: "local", settings: settings)!
        fresh.end(outcome: "completed")
        touch(old.fileURL, daysAgo: 40)

        let r = Retention.prune(settings: settings)
        XCTAssertEqual(r.deletedFiles, 2)
        XCTAssertFalse(FileManager.default.fileExists(atPath: oldLog.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: newLog.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: old.fileURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: fresh.fileURL.path))
        XCTAssertEqual(SessionIndex.load(settings: settings).map(\.id), [fresh.id])
    }

    func testRetentionDaysIsConfigurable() {
        let f = settings.logsDir.appendingPathComponent("serve-2026-10-15.jsonl")
        write(f, "{}\n"); touch(f, daysAgo: 7)
        var s = settings!; s.retentionDays = 30
        XCTAssertEqual(Retention.prune(settings: s).deletedFiles, 0)
        s.retentionDays = 3
        XCTAssertEqual(Retention.prune(settings: s).deletedFiles, 1)
    }

    func testTotalSizeCapDeletesOldestFirstAndSparesActiveFiles() {
        var s = settings!; s.maxMB = 1
        let mb = String(repeating: "x", count: 600_000)
        let names = ["serve-2026-10-01.jsonl", "serve-2026-10-10.jsonl", "serve-2026-10-20.jsonl", "serve-2026-10-22.jsonl"]
        for (i, n) in names.enumerated() {
            let f = s.logsDir.appendingPathComponent(n)
            write(f, mb)
            touch(f, daysAgo: Double(20 - i * 6) + 0.5)   // 20.5, 14.5, 8.5, 2.5 days
        }
        let active = s.logsDir.appendingPathComponent("serve-today.jsonl")
        write(active, mb)   // just written: must survive
        let r = Retention.prune(settings: s)
        XCTAssertGreaterThanOrEqual(r.deletedFiles, 3)
        XCTAssertTrue(FileManager.default.fileExists(atPath: active.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: s.logsDir.appendingPathComponent(names[0]).path))
        XCTAssertLessThanOrEqual(r.remainingBytes, 1_048_576 + 600_000)
    }

    func testPruneIfDueSkipsWhenItRanRecently() {
        let f = settings.logsDir.appendingPathComponent("serve-2026-08-01.jsonl")
        write(f, "{}\n"); touch(f, daysAgo: 60)
        Retention.pruneIfDue(settings: settings)           // runs, deletes, stamps
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.path))
        write(f, "{}\n"); touch(f, daysAgo: 60)
        Retention.pruneIfDue(settings: settings)           // stamped within the hour: skipped
        XCTAssertTrue(FileManager.default.fileExists(atPath: f.path))
        Retention.pruneIfDue(settings: settings, now: Date().addingTimeInterval(7200))
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.path))
    }

    func testPruneIsCheapOnAThousandFiles() {
        for i in 0..<1000 { write(settings.logsDir.appendingPathComponent(String(format: "serve-2026-10-%04d.jsonl", i)), "{}\n") }
        let t0 = Date()
        _ = Retention.prune(settings: settings)
        XCTAssertLessThan(Date().timeIntervalSince(t0), 2)
    }
}
