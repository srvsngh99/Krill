@preconcurrency import NIOCore
@preconcurrency import NIOHTTP1
import Foundation
import KrillObs

/// Per-request facts, written by the HTTP handler as it learns them and read by
/// ``RequestLogHandler`` when the response ends. One instance per connection (HTTP/1.1 is one
/// request at a time per connection); the lock only guards the async generation tasks that
/// set fields from another thread.
final class RequestTrace: @unchecked Sendable {
    struct Snapshot {
        var requestID = ""
        var method = ""
        var endpoint = ""
        var startNs: UInt64 = 0
        var model: String?
        var promptTokens: Int?
        var completionTokens: Int?
        var prefillSeconds: Double?
        var decodeSeconds: Double?
        var queueWaitMs: Double?
        var embedDevice: String?
        var error: String?
        var sessionID: String?
        var inputs: Int?
    }

    private let lock = NSLock()
    private var s = Snapshot()
    private var statsProvider: (@Sendable () -> (Int, Int, Double, Double)?)?

    func setStatsProvider(_ p: @escaping @Sendable () -> (Int, Int, Double, Double)?) {
        lock.lock(); statsProvider = p; lock.unlock()
    }

    /// Final snapshot: if no handler recorded token stats, read them from the engine now.
    func finalSnapshot() -> Snapshot {
        lock.lock()
        let missing = s.completionTokens == nil && s.embedDevice == nil
        let provider = statsProvider
        lock.unlock()
        if missing, let v = provider?() { setStats(prompt: v.0, completion: v.1, prefillSeconds: v.2, decodeSeconds: v.3) }
        return snapshot()
    }

    func begin(requestID: String, method: String, endpoint: String, sessionID: String?) {
        lock.lock()
        s = Snapshot(requestID: requestID, method: method, endpoint: endpoint,
                     startNs: DispatchTime.now().uptimeNanoseconds, sessionID: sessionID)
        statsProvider = nil
        lock.unlock()
    }
    func snapshot() -> Snapshot { lock.lock(); defer { lock.unlock() }; return s }

    func setModel(_ m: String?) { guard let m, !m.isEmpty else { return }; lock.lock(); s.model = String(m.prefix(200)); lock.unlock() }
    func setStats(prompt: Int, completion: Int, prefillSeconds: Double, decodeSeconds: Double) {
        lock.lock()
        s.promptTokens = prompt; s.completionTokens = completion
        s.prefillSeconds = prefillSeconds; s.decodeSeconds = decodeSeconds
        lock.unlock()
    }
    func setEmbed(device: String, promptTokens: Int, inputs: Int) {
        lock.lock(); s.embedDevice = device; s.promptTokens = promptTokens; s.inputs = inputs; lock.unlock()
    }
    func addQueueWait(ms: Double) { lock.lock(); s.queueWaitMs = (s.queueWaitMs ?? 0) + ms; lock.unlock() }
    func setError(_ e: String) {
        lock.lock(); if s.error == nil { s.error = String(e.prefix(500)) }; lock.unlock()
    }
}

/// Sits in front of ``HTTPHandler`` in the pipeline and sees every request head inbound and every
/// response part outbound. It stamps `X-Krill-Request-Id` on the response, feeds the Prometheus
/// counters and queues one `request` event per request. It never blocks and never fails a
/// request: all file work happens on the log writer's queue.
final class RequestLogHandler: ChannelDuplexHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias InboundOut = HTTPServerRequestPart
    typealias OutboundIn = HTTPServerResponsePart
    typealias OutboundOut = HTTPServerResponsePart

    /// Probes and static assets that would drown the log; still counted in metrics, and logged
    /// when they fail.
    static let quietEndpoints: Set<String> = ["/healthz", "/health", "/metrics", "/ui", "/ui/*", "/"]

    let trace: RequestTrace
    private var active = false
    private var status = 0
    private var streaming = false
    private var firstBodyNs: UInt64?
    private var endpoint = ""

    init(trace: RequestTrace) { self.trace = trace }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        if case .head(let head) = unwrapInboundIn(data) {
            if active { finish(closedEarly: false) }
            let path = head.uri.split(separator: "?").first.map(String.init) ?? head.uri
            endpoint = ServerMetrics.endpointLabel(method: head.method.rawValue, path: path)
            let session = head.headers.first(name: "X-Krill-Session").map { String($0.prefix(64)) }
            trace.begin(requestID: Self.newRequestID(), method: head.method.rawValue,
                        endpoint: endpoint, sessionID: session)
            ServerMetrics.shared.begin()
            active = true; status = 0; streaming = false; firstBodyNs = nil
        }
        context.fireChannelRead(data)
    }

    func write(context: ChannelHandlerContext, data: NIOAny, promise: EventLoopPromise<Void>?) {
        switch unwrapOutboundIn(data) {
        case .head(var head) where active:
            status = Int(head.status.code)
            let ct = head.headers.first(name: "Content-Type") ?? ""
            streaming = ct.contains("event-stream") || ct.contains("ndjson")
            head.headers.replaceOrAdd(name: "X-Krill-Request-Id", value: trace.snapshot().requestID)
            context.write(wrapOutboundOut(.head(head)), promise: promise)
            return
        case .body where active:
            if firstBodyNs == nil { firstBodyNs = DispatchTime.now().uptimeNanoseconds }
        case .end where active:
            context.write(data, promise: promise)
            finish(closedEarly: false)
            return
        default: break
        }
        context.write(data, promise: promise)
    }

    func channelInactive(context: ChannelHandlerContext) {
        if active { finish(closedEarly: true) }
        context.fireChannelInactive()
    }

    private func finish(closedEarly: Bool) {
        active = false
        let now = DispatchTime.now().uptimeNanoseconds
        var snap = trace.finalSnapshot()
        let totalMs = Double(now &- snap.startNs) / 1e6
        let code = closedEarly && status == 0 ? 0 : status
        if closedEarly && snap.error == nil { snap.error = "client closed the connection before the response finished" }
        ServerMetrics.shared.record(endpoint: endpoint, status: code, seconds: totalMs / 1000,
                                    promptTokens: snap.promptTokens ?? 0, completionTokens: snap.completionTokens ?? 0)
        guard Obs.eventsEnabled else { return }
        if Self.quietEndpoints.contains(endpoint), code > 0, code < 400 { return }

        var ttftMs: Double?
        if streaming, let fb = firstBodyNs { ttftMs = Double(fb &- snap.startNs) / 1e6 }
        else if let pre = snap.prefillSeconds { ttftMs = (snap.queueWaitMs ?? 0) + pre * 1000 }
        var tokS: Double?
        if let c = snap.completionTokens, c > 0 {
            if let d = snap.decodeSeconds, d > 0 { tokS = Double(c) / d }
            else if totalMs > 0 { tokS = Double(c) / (totalMs / 1000) }
        }
        var f: [(String, LogValue)] = [
            ("request_id", .string(snap.requestID)), ("method", .string(snap.method)),
            ("endpoint", .string(endpoint)), ("model", .opt(snap.model)), ("status", .int(code)),
            ("error", .opt(snap.error)), ("ttft_ms", .opt(ttftMs)), ("total_ms", .double(totalMs)),
            ("prompt_tokens", .opt(snap.promptTokens)), ("completion_tokens", .opt(snap.completionTokens)),
            ("tok_s", .opt(tokS)), ("queue_wait_ms", .opt(snap.queueWaitMs)),
        ]
        if let d = snap.embedDevice { f.append(("embed_device", .string(d))) }
        if let n = snap.inputs { f.append(("inputs", .int(n))) }
        if streaming { f.append(("stream", .bool(true))) }
        if let sid = snap.sessionID { f.append(("session_id", .string(sid))) }
        Obs.emit("request", f)
    }

    static func newRequestID() -> String {
        var g = SystemRandomNumberGenerator()
        return String(format: "%012llx", g.next() & 0xFFFF_FFFF_FFFF)
    }
}
