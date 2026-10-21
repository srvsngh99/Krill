import XCTest
@testable import KrillObs

final class EventLogTests: ObsTestCase {
    func testWritesOneJSONObjectPerLineInADailyFile() throws {
        Obs.configure(kind: "serve", settings: settings, port: 57499, version: "9.9.9", prune: false)
        Obs.emit("server_start", [("version", "9.9.9"), ("n", 3), ("ok", true), ("x", 1.5), ("none", .null)])
        Obs.emit("request", [("endpoint", "/v1/chat/completions"), ("status", 200)])
        Obs.flush()
        let day = ObsPaths.dayString(Date())
        let file = settings.logsDir.appendingPathComponent("serve-\(day).jsonl")
        let ls = lines(file)
        XCTAssertEqual(ls.count, 2)
        let first = json(ls[0])
        XCTAssertEqual(first["event"] as? String, "server_start")
        XCTAssertEqual(first["version"] as? String, "9.9.9")
        XCTAssertEqual(first["port"] as? Int, 57499)
        XCTAssertEqual(first["pid"] as? Int, Int(getpid()))
        XCTAssertEqual(first["ok"] as? Bool, true)
        XCTAssertTrue(first["none"] is NSNull)
        XCTAssertNotNil(JSONLine.parseISO(first["ts"] as! String))
    }

    func testNothingIsWrittenWhenTheEventLogIsOff() {
        var off = settings!; off.eventLog = false
        Obs.configure(kind: "serve", settings: off, version: "x", prune: false)
        Obs.emit("request", [("status", 200)])
        Obs.flush()
        XCTAssertFalse(FileManager.default.fileExists(atPath: off.logsDir.path))
        XCTAssertFalse(Obs.eventsEnabled)
    }

    func testStringsAreEscapedAndControlCharactersSurvive() {
        let line = JSONLine.encode(LogEvent("e", [("m", .string("a\"b\\c\nd\u{01}é"))]))
        let o = json(line)
        XCTAssertEqual(o["m"] as? String, "a\"b\\c\nd\u{01}é")
    }

    func testRotationStartsANewFileAtTheSizeCap() throws {
        let dir = tmp.appendingPathComponent("rot")
        let w = LineWriter(maxFileBytes: 400) { _ in dir.appendingPathComponent("serve-2026-10-22.jsonl") }
        for i in 0..<40 { w.append(LogEvent("request", [("i", .int(i)), ("pad", .string(String(repeating: "x", count: 30)))])) }
        w.flush()
        let names = files(dir).map(\.lastPathComponent)
        XCTAssertTrue(names.contains("serve-2026-10-22.jsonl"))
        XCTAssertTrue(names.contains("serve-2026-10-22.1.jsonl"), "\(names)")
        let total = names.reduce(0) { $0 + lines(dir.appendingPathComponent($1)).count }
        XCTAssertEqual(total, 40, "rotation must not lose lines")
        for n in names {
            let size = (try FileManager.default.attributesOfItem(atPath: dir.appendingPathComponent(n).path)[.size] as! NSNumber).intValue
            XCTAssertLessThanOrEqual(size, 400 + 150)
        }
    }

    func testEventsAreRoutedToTheFileOfTheirOwnDay() {
        let dir = tmp.appendingPathComponent("days")
        let w = LineWriter { d in dir.appendingPathComponent("serve-\(ObsPaths.dayString(d)).jsonl") }
        let yesterday = Date().addingTimeInterval(-86_400)
        w.append(LogEvent("a", ts: yesterday)); w.append(LogEvent("b")); w.flush()
        XCTAssertEqual(files(dir).count, 2)
    }

    func testAppendIsFastAndNeverBlocksOnIO() {
        let dir = tmp.appendingPathComponent("perf")
        let w = LineWriter { _ in dir.appendingPathComponent("p.jsonl") }
        let n = 5000
        let t0 = DispatchTime.now().uptimeNanoseconds
        for i in 0..<n { w.append(LogEvent("request", [("i", .int(i)), ("endpoint", "/v1/chat/completions"), ("status", 200)])) }
        let perCallUs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000 / Double(n)
        w.flush()
        XCTAssertLessThan(perCallUs, 100, "append must stay far below 1 ms (was \(perCallUs) us)")
        XCTAssertEqual(lines(dir.appendingPathComponent("p.jsonl")).count, n)
    }

    func testShutdownFlushesAndClearsTheMarker() {
        Obs.configure(kind: "serve", settings: settings, port: 57499, version: "1", prune: false)
        XCTAssertEqual(RunMarkers.list(dir: settings.runDir).count, 1)
        Obs.emit("server_stop")
        Obs.shutdown()
        XCTAssertEqual(RunMarkers.list(dir: settings.runDir).count, 0)
        XCTAssertEqual(lines(settings.logsDir.appendingPathComponent("serve-\(ObsPaths.dayString(Date())).jsonl")).count, 1)
        Obs.shutdown()   // twice is harmless
    }

    func testStaleMarkerOfADeadProcessBecomesAnUncleanExitEventWithLastEvents() throws {
        // A process that really existed and is gone.
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/true"); try p.run(); p.waitUntilExit()
        let deadPid = p.processIdentifier
        let started = Date().addingTimeInterval(-120)
        RunMarkers.write(dir: settings.runDir, kind: "serve", port: 57456, version: "0.26.0", now: started, pid: deadPid)
        // Its last words, in the log, plus noise from another pid that must not be attributed to it.
        let day = ObsPaths.dayString(Date())
        write(settings.logsDir.appendingPathComponent("serve-\(day).jsonl"), """
        {"ts":"2026-10-22T01:00:00.000Z","event":"model_load","pid":\(deadPid),"model":"m"}
        {"ts":"2026-10-22T01:00:01.000Z","event":"request","pid":1,"status":200}
        {"ts":"2026-10-22T01:00:02.000Z","event":"request","pid":\(deadPid),"status":200}

        """)
        let diag = tmp.appendingPathComponent("DiagnosticReports")
        write(diag.appendingPathComponent("krill-2026-10-22-010000.ips"), "{}")
        write(diag.appendingPathComponent("other-app.ips"), "{}")

        let stale = RunMarkers.sweepStale(dir: settings.runDir)
        XCTAssertEqual(stale.count, 1)
        let f = CrashReport.fields(for: stale[0], settings: settings, diagnosticsDir: diag)
        let d = Dictionary(uniqueKeysWithValues: f.map { ($0.0, $0.1) })
        XCTAssertEqual(d["dead_pid"], .int(Int(deadPid)))
        guard case .string(let last)? = d["last_events"] else { return XCTFail("no last_events") }
        XCTAssertTrue(last.contains("model_load") && last.contains("\"status\":200"))
        XCTAssertFalse(last.contains("\"pid\":1,"))
        guard case .string(let ips)? = d["crash_reports"] else { return XCTFail("no ips") }
        XCTAssertTrue(ips.contains("krill-2026-10-22-010000.ips"))
        XCTAssertFalse(ips.contains("other-app"))
        XCTAssertEqual(RunMarkers.sweepStale(dir: settings.runDir).count, 0, "reported once, then gone")
    }

    func testConfigureReportsAnUncleanExitFromAPreviousRun() throws {
        let p = Process(); p.executableURL = URL(fileURLWithPath: "/usr/bin/true"); try p.run(); p.waitUntilExit()
        RunMarkers.write(dir: settings.runDir, kind: "serve", port: 57456, version: "0.26.0", pid: p.processIdentifier)
        Obs.configure(kind: "serve", settings: settings, port: 57456, version: "0.26.1", prune: false)
        Obs.flush()
        let recs = LogQuery.records(settings: settings, filter: LogFilter())
        XCTAssertTrue(recs.contains { $0.event == "unclean_exit" && ($0.fields["dead_pid"] as? Int) == Int(p.processIdentifier) })
        XCTAssertTrue(recs.first { $0.event == "unclean_exit" }!.isError)
    }

    func testALiveProcessMarkerIsNotACrash() {
        RunMarkers.write(dir: settings.runDir, kind: "serve", port: nil, version: "1", pid: getppid())
        XCTAssertEqual(RunMarkers.sweepStale(dir: settings.runDir).count, 0)
    }
}
