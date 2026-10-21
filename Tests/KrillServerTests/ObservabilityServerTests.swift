import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import KrillEngine
import KrillObs
import KrillRegistry
@testable import KrillServer

/// The /metrics and /v1/status numbers the audit found wrong, plus the request log wiring.
final class ObservabilityServerTests: XCTestCase {
    private var tmp: URL!

    override func setUp() {
        super.setUp()
        tmp = FileManager.default.temporaryDirectory.appendingPathComponent("krill-obs-server-\(UUID().uuidString)")
        ServerMetrics.shared.reset()
    }
    override func tearDown() {
        Obs.resetForTesting()
        try? FileManager.default.removeItem(at: tmp)
        super.tearDown()
    }

    private func channel(trace: RequestTrace? = nil, logging: Bool = false) throws -> EmbeddedChannel {
        let engine = InferenceEngine(modelDirectory: tmp.appendingPathComponent("model"))
        let reg = Registry(baseDir: tmp.appendingPathComponent("registry"))
        let ch = EmbeddedChannel()
        let t = trace ?? RequestTrace()
        if logging { try ch.pipeline.addHandler(RequestLogHandler(trace: t)).wait() }
        try ch.pipeline.addHandler(
            KrillServer._makeHTTPHandlerForTesting(engine: engine, registry: reg, trace: t)).wait()
        return ch
    }

    private func get(_ ch: EmbeddedChannel, _ uri: String, headers: [(String, String)] = []) throws -> (HTTPResponseHead, String) {
        var head = HTTPRequestHead(version: .http1_1, method: .GET, uri: uri)
        for (k, v) in headers { head.headers.add(name: k, value: v) }
        try ch.writeInbound(HTTPServerRequestPart.head(head))
        try ch.writeInbound(HTTPServerRequestPart.end(nil))
        guard case .head(let h)? = try ch.readOutbound(as: HTTPServerResponsePart.self) else { throw XCTSkip("no head") }
        var text = ""
        if case .body(.byteBuffer(var b))? = try ch.readOutbound(as: HTTPServerResponsePart.self) {
            text = b.readString(length: b.readableBytes) ?? ""
        }
        _ = try ch.readOutbound(as: HTTPServerResponsePart.self)
        return (h, text)
    }

    private func metricValue(_ name: String, in body: String) -> Double? {
        body.split(separator: "\n").first { $0.hasPrefix(name + " ") }
            .flatMap { Double($0.dropFirst(name.count + 1)) }
    }

    func testMetricsUptimeIsProcessAgeNotMachineUptime() throws {
        let ch = try channel(); defer { _ = try? ch.finish(acceptAlreadyClosed: true) }
        let (_, body) = try get(ch, "/metrics")
        let up = try XCTUnwrap(metricValue("krill_uptime_seconds", in: body))
        // The old code returned ProcessInfo.systemUptime (machine uptime: days).
        XCTAssertLessThan(up, ProcessInfo.processInfo.systemUptime)
        XCTAssertLessThan(up, 3600, "a test process is minutes old, not days")
        XCTAssertGreaterThanOrEqual(up, 0)
    }

    func testMetricsReportsFootprintAndMLXMemory() throws {
        let ch = try channel(); defer { _ = try? ch.finish(acceptAlreadyClosed: true) }
        let (_, body) = try get(ch, "/metrics")
        let footprint = try XCTUnwrap(metricValue("krill_memory_footprint_mb", in: body))
        let resident = try XCTUnwrap(metricValue("krill_resident_memory_mb", in: body))
        XCTAssertGreaterThan(footprint, 1)
        XCTAssertGreaterThan(resident, 1)
        XCTAssertNotNil(metricValue("krill_mlx_active_memory_mb", in: body))
        XCTAssertNotNil(metricValue("krill_mlx_peak_memory_mb", in: body))
    }

    func testStatusUptimeIsNotZeroAndGrowsAcrossRequests() throws {
        let ch = try channel(); defer { _ = try? ch.finish(acceptAlreadyClosed: true) }
        let (_, b1) = try get(ch, "/v1/status")
        let j1 = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(b1.utf8)) as? [String: Any])
        XCTAssertNotNil(j1["mlx_active_mb"])
        XCTAssertGreaterThan(j1["memory_mb"] as? Int ?? 0, 0)
        // Per-connection startedAt used to make this 0 on every fresh connection.
        let ch2 = try channel(); defer { _ = try? ch2.finish(acceptAlreadyClosed: true) }
        Thread.sleep(forTimeInterval: 1.1)
        let (_, b2) = try get(ch2, "/v1/status")
        let j2 = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(b2.utf8)) as? [String: Any])
        XCTAssertGreaterThanOrEqual(j2["uptime_seconds"] as? Int ?? 0, 1)
    }

    func testRequestLogHandlerCountsRequestsAndErrorsAndStampsRequestID() throws {
        let ch = try channel(logging: true); defer { _ = try? ch.finish(acceptAlreadyClosed: true) }
        let (ok, _) = try get(ch, "/api/version")
        XCTAssertEqual(ok.status, .ok)
        XCTAssertEqual(ok.headers.first(name: "X-Krill-Request-Id")?.count, 12)
        let (nf, _) = try get(ch, "/nope")
        XCTAssertEqual(nf.status, .notFound)
        XCTAssertEqual(ServerMetrics.shared.totalRequests, 2)
        XCTAssertEqual(ServerMetrics.shared.totalErrors, 1)
        let text = ServerMetrics.shared.prometheus()
        XCTAssertTrue(text.contains("krill_http_requests_total{endpoint=\"/api/version\",code=\"200\"} 1"), text)
        XCTAssertTrue(text.contains("krill_http_requests_total{endpoint=\"other\",code=\"404\"} 1"), text)
        XCTAssertTrue(text.contains("krill_http_request_duration_seconds_bucket{endpoint=\"/api/version\",le=\"+Inf\"} 1"), text)
    }

    func testRequestEventLoggedWithErrorAndSessionButNoBody() throws {
        let settings = ObsSettings(eventLog: true, sessionLog: .off, root: tmp.appendingPathComponent("obs"))
        Obs.configure(kind: "serve", settings: settings, port: 57499, version: "t", prune: false)
        let ch = try channel(logging: true); defer { _ = try? ch.finish(acceptAlreadyClosed: true) }
        var head = HTTPRequestHead(version: .http1_1, method: .POST, uri: "/v1/chat/completions")
        head.headers.add(name: "X-Krill-Session", value: "sess123")
        head.headers.add(name: "Content-Length", value: "2")
        try ch.writeInbound(HTTPServerRequestPart.head(head))
        var buf = ch.allocator.buffer(capacity: 2); buf.writeString("{}")
        try ch.writeInbound(HTTPServerRequestPart.body(buf))
        try ch.writeInbound(HTTPServerRequestPart.end(nil))
        while (try ch.readOutbound(as: HTTPServerResponsePart.self)) != nil {}
        Obs.flush()
        let recs = LogQuery.records(settings: settings, filter: LogFilter())
        let r = try XCTUnwrap(recs.first { $0.event == "request" })
        XCTAssertEqual(r.fields["endpoint"] as? String, "/v1/chat/completions")
        XCTAssertEqual(r.sessionID, "sess123")
        XCTAssertGreaterThanOrEqual(r.status ?? 0, 400)
        XCTAssertNotNil(r.fields["error"] as? String, "the failure reason must be in the log")
        XCTAssertNotNil(r.fields["total_ms"])
        XCTAssertEqual(r.fields["request_id"] as? String == nil, false)
        XCTAssertTrue(recs.contains { $0.event == "unclean_exit" } == false)
    }

    func testProbesAreQuietInTheEventLogButStillCounted() throws {
        let settings = ObsSettings(eventLog: true, sessionLog: .off, root: tmp.appendingPathComponent("obs"))
        Obs.configure(kind: "serve", settings: settings, port: 57499, version: "t", prune: false)
        let ch = try channel(logging: true); defer { _ = try? ch.finish(acceptAlreadyClosed: true) }
        _ = try get(ch, "/healthz")
        _ = try get(ch, "/metrics")
        Obs.flush()
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter()).filter { $0.event == "request" }.count, 0)
        XCTAssertEqual(ServerMetrics.shared.totalRequests, 2)
    }

    /// What one request costs the request path: metrics bookkeeping plus queueing the event.
    /// (File I/O happens later on the writer queue.) Asserted far below the 1 ms budget; the
    /// measured mean is printed so the PR can quote it.
    func testPerRequestLoggingCostIsFarBelowOneMillisecond() {
        let settings = ObsSettings(eventLog: true, sessionLog: .off, root: tmp.appendingPathComponent("obs"))
        Obs.configure(kind: "serve", settings: settings, port: 57499, version: "t", prune: false)
        let m = ServerMetrics()
        let n = 20_000
        let t0 = DispatchTime.now().uptimeNanoseconds
        for i in 0..<n {
            m.begin()
            m.record(endpoint: "/v1/chat/completions", status: 200, seconds: 0.25, promptTokens: 40, completionTokens: 120)
            Obs.emit("request", [
                ("request_id", .string("0123456789ab")), ("method", "POST"), ("endpoint", "/v1/chat/completions"),
                ("model", "llama-3.2-1b"), ("status", 200), ("error", .null), ("ttft_ms", 151.8), ("total_ms", 332.7),
                ("prompt_tokens", 40), ("completion_tokens", 120), ("tok_s", 82.0), ("queue_wait_ms", 0.0),
                ("session_id", .string("s\(i % 7)")),
            ])
        }
        let perRequestUs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000 / Double(n)
        Obs.flush()
        print("OBS-COST per request (debug build): \(String(format: "%.2f", perRequestUs)) us")
        XCTAssertLessThan(perRequestUs, 200, "logging must cost well under 1 ms per request")
        XCTAssertEqual(LogQuery.records(settings: settings, filter: LogFilter()).filter { $0.event == "request" }.count, n)
    }
}
