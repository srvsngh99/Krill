import XCTest
@testable import KrillObs

final class LogQueryTests: ObsTestCase {
    private func seed() {
        let now = Date()
        func ts(_ ago: TimeInterval) -> String { JSONLine.isoUTC(now.addingTimeInterval(-ago)) }
        let day = ObsPaths.dayString(now)
        write(settings.logsDir.appendingPathComponent("serve-\(day).jsonl"), """
        {"ts":"\(ts(7200))","event":"server_start","pid":1,"version":"0.26.0"}
        {"ts":"\(ts(3600))","event":"request","pid":1,"method":"POST","endpoint":"/v1/chat/completions","model":"m","status":200,"ttft_ms":120,"total_ms":900,"prompt_tokens":10,"completion_tokens":40,"tok_s":46.7,"request_id":"aaa","session_id":"sess-a"}
        {"ts":"\(ts(1800))","event":"request","pid":1,"method":"POST","endpoint":"/v1/embeddings","model":"e","status":500,"error":"embedding failed: boom","total_ms":50,"embed_device":"gpu","request_id":"bbb"}
        {"ts":"\(ts(900))","event":"log","pid":1,"level":"warning","logger":"krill.server","message":"ANE package failed"}
        {"ts":"\(ts(600))","event":"model_load_failed","pid":1,"model":"x","error":"oom"}
        {"ts":"\(ts(300))","event":"request","pid":1,"method":"POST","endpoint":"/api/chat","model":"m","status":200,"request_id":"ccc","session_id":"sess-b"}
        not json at all

        """)
    }

    func testDefaultReturnsEverythingAndSkipsGarbageLines() {
        seed()
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter()).count, 6)
    }

    func testErrorsFilterKeepsFailedRequestsWarningsAndFailures() {
        seed()
        let recs = LogQuery.records(settings: settings, filter: LogFilter(errorsOnly: true))
        XCTAssertEqual(recs.map(\.event), ["request", "log", "model_load_failed"])
        XCTAssertEqual(recs[0].status, 500)
    }

    func testSinceFiltersByTime() {
        seed()
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter(since: Date().addingTimeInterval(-2000))).count, 4)
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter(since: Date().addingTimeInterval(-400))).count, 1)
    }

    func testSessionFilterMatchesPrefixAndOnlyThatSessionsRequests() {
        seed()
        let r = LogQuery.records(settings: settings, filter: LogFilter(session: "sess-a"))
        XCTAssertEqual(r.map { $0.fields["request_id"] as? String }, ["aaa"])
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter(session: "sess")).count, 2)
    }

    func testFilesOlderThanSinceAreNotEvenRead() {
        write(settings.logsDir.appendingPathComponent("serve-2026-01-01.jsonl"),
              "{\"ts\":\"2026-01-01T00:00:00.000Z\",\"event\":\"request\",\"status\":500}\n")
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter(since: Date().addingTimeInterval(-86_400))).count, 0)
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter(errorsOnly: true)).count, 1)
    }

    func testParseSinceAcceptsDurationsAndDates() throws {
        let now = Date()
        XCTAssertEqual(try XCTUnwrap(LogQuery.parseSince("30m", now: now)).timeIntervalSince(now), -1800, accuracy: 1)
        XCTAssertEqual(try XCTUnwrap(LogQuery.parseSince("2h", now: now)).timeIntervalSince(now), -7200, accuracy: 1)
        XCTAssertEqual(try XCTUnwrap(LogQuery.parseSince("7d", now: now)).timeIntervalSince(now), -604_800, accuracy: 1)
        XCTAssertEqual(try XCTUnwrap(LogQuery.parseSince("1w", now: now)).timeIntervalSince(now), -604_800, accuracy: 1)
        let d = try XCTUnwrap(LogQuery.parseSince("2026-10-20"))
        XCTAssertEqual(ObsPaths.dayString(d), "2026-10-20")
        XCTAssertNotNil(LogQuery.parseSince("2026-10-20T10:00:00Z"))
        XCTAssertNil(LogQuery.parseSince("yesterdayish"))
        XCTAssertNil(LogQuery.parseSince(""))
    }

    func testRenderedLinesAreHumanReadable() {
        seed()
        let recs = LogQuery.records(settings: settings, filter: LogFilter())
        let line = LogQuery.render(recs[1])
        XCTAssertTrue(line.contains("POST /v1/chat/completions"), line)
        XCTAssertTrue(line.contains("model=m") && line.contains("200") && line.contains("ttft=120ms") && line.contains("10+40tok"), line)
        XCTAssertTrue(line.contains("46.7tok/s") && line.contains("session=sess-a"), line)
        XCTAssertTrue(LogQuery.render(recs[2]).contains("error=\"embedding failed: boom\""))
    }

    func testFollowerReturnsOnlyNewCompleteLines() throws {
        seed()
        let f = LogFollower(settings: settings, filter: LogFilter(errorsOnly: true))
        XCTAssertTrue(f.poll().isEmpty)
        let file = settings.logsDir.appendingPathComponent("serve-\(ObsPaths.dayString(Date())).jsonl")
        let h = try FileHandle(forWritingTo: file); h.seekToEndOfFile()
        h.write(Data("{\"ts\":\"\(JSONLine.isoUTC(Date()))\",\"event\":\"request\",\"status\":503,\"endpoint\":\"/x\"}\n".utf8))
        h.write(Data("{\"ts\":\"\(JSONLine.isoUTC(Date()))\",\"event\":\"req".utf8))   // half-written line
        try h.close()
        XCTAssertEqual(f.poll().count, 1)
        XCTAssertTrue(f.poll().isEmpty)
    }
}
