import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Process facts that the old `/metrics` got wrong.
public enum ProcessStats {
    /// When this process started (kernel start time), not machine boot.
    public static let startDate: Date = {
        #if canImport(Darwin)
        var mib: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        if sysctl(&mib, 3, &info, &size, nil, 0) == 0 {
            let tv = info.kp_proc.p_un.__p_starttime
            if tv.tv_sec > 0 { return Date(timeIntervalSince1970: Double(tv.tv_sec) + Double(tv.tv_usec) / 1e6) }
        }
        #endif
        return Date()
    }()

    public static func uptimeSeconds(now: Date = Date()) -> Double { max(0, now.timeIntervalSince(startDate)) }

    /// Physical footprint in bytes: what Activity Monitor calls "Memory". Unlike `resident_size`
    /// it includes Metal/MLX buffers and compressed pages. Falls back to RSS where unavailable.
    public static func physFootprintBytes() -> Int {
        #if canImport(Darwin)
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS { return Int(info.phys_footprint) }
        #endif
        return residentBytes()
    }

    public static func residentBytes() -> Int {
        #if canImport(Darwin)
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS { return Int(info.resident_size) }
        #endif
        return 0
    }
}

/// Request counters and a latency histogram in a few locked dictionaries. Prometheus text out,
/// loopback/bearer-auth only (it is served by `/metrics`). Cardinality is bounded: `endpoint`
/// is a normalised route, never a raw path.
public final class ServerMetrics: @unchecked Sendable {
    public static let shared = ServerMetrics()
    public static let buckets: [Double] = [0.005, 0.025, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60]

    private let lock = NSLock()
    private var requests: [String: Int] = [:]          // "endpoint|code"
    private var errors: [String: Int] = [:]            // endpoint
    private var hist: [String: [Int]] = [:]            // endpoint -> cumulative-free bucket counts (+Inf last)
    private var sum: [String: Double] = [:]
    private var promptTokens = 0
    private var completionTokens = 0
    private var inFlight = 0

    public init() {}

    public func begin() { lock.lock(); inFlight += 1; lock.unlock() }

    public func record(endpoint: String, status: Int, seconds: Double, promptTokens p: Int = 0, completionTokens c: Int = 0) {
        lock.lock(); defer { lock.unlock() }
        inFlight = max(0, inFlight - 1)
        requests["\(endpoint)|\(status)", default: 0] += 1
        if status >= 400 || status == 0 { errors[endpoint, default: 0] += 1 }
        var h = hist[endpoint] ?? Array(repeating: 0, count: Self.buckets.count + 1)
        var placed = false
        for (i, b) in Self.buckets.enumerated() where seconds <= b { h[i] += 1; placed = true; break }
        if !placed { h[Self.buckets.count] += 1 }
        hist[endpoint] = h
        sum[endpoint, default: 0] += seconds
        promptTokens += p
        completionTokens += c
    }

    public func reset() {
        lock.lock(); requests = [:]; errors = [:]; hist = [:]; sum = [:]; promptTokens = 0; completionTokens = 0; inFlight = 0; lock.unlock()
    }

    public var totalRequests: Int { lock.lock(); defer { lock.unlock() }; return requests.values.reduce(0, +) }
    public var totalErrors: Int { lock.lock(); defer { lock.unlock() }; return errors.values.reduce(0, +) }

    /// Prometheus text exposition for the counters and histogram.
    public func prometheus() -> String {
        lock.lock()
        let req = requests, err = errors, h = hist, s = sum
        let pt = promptTokens, ct = completionTokens, fl = inFlight
        lock.unlock()
        var o = ""
        o += "# HELP krill_http_requests_total HTTP requests handled, by endpoint and status code\n"
        o += "# TYPE krill_http_requests_total counter\n"
        for k in req.keys.sorted() {
            let p = k.split(separator: "|", omittingEmptySubsequences: false)
            o += "krill_http_requests_total{endpoint=\"\(esc(String(p[0])))\",code=\"\(p[1])\"} \(req[k]!)\n"
        }
        o += "# HELP krill_http_errors_total HTTP requests that ended with status >= 400 (or a dropped connection)\n"
        o += "# TYPE krill_http_errors_total counter\n"
        for k in err.keys.sorted() { o += "krill_http_errors_total{endpoint=\"\(esc(k))\"} \(err[k]!)\n" }
        o += "# HELP krill_http_request_duration_seconds Request latency from first byte of the request to the last byte of the response\n"
        o += "# TYPE krill_http_request_duration_seconds histogram\n"
        for k in h.keys.sorted() {
            var cum = 0
            for (i, b) in Self.buckets.enumerated() {
                cum += h[k]![i]
                o += "krill_http_request_duration_seconds_bucket{endpoint=\"\(esc(k))\",le=\"\(fmt(b))\"} \(cum)\n"
            }
            cum += h[k]![Self.buckets.count]
            o += "krill_http_request_duration_seconds_bucket{endpoint=\"\(esc(k))\",le=\"+Inf\"} \(cum)\n"
            o += "krill_http_request_duration_seconds_sum{endpoint=\"\(esc(k))\"} \(String(format: "%.6f", s[k] ?? 0))\n"
            o += "krill_http_request_duration_seconds_count{endpoint=\"\(esc(k))\"} \(cum)\n"
        }
        o += "# HELP krill_prompt_tokens_total Prompt tokens processed\n# TYPE krill_prompt_tokens_total counter\nkrill_prompt_tokens_total \(pt)\n"
        o += "# HELP krill_completion_tokens_total Completion tokens generated\n# TYPE krill_completion_tokens_total counter\nkrill_completion_tokens_total \(ct)\n"
        o += "# HELP krill_http_in_flight Requests currently being handled\n# TYPE krill_http_in_flight gauge\nkrill_http_in_flight \(fl)\n"
        return o
    }

    private func fmt(_ d: Double) -> String { d == d.rounded() ? String(Int(d)) : String(d) }
    private func esc(_ s: String) -> String {
        s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"")
    }

    /// Normalise a raw request path to a bounded route label.
    public static func endpointLabel(method: String, path: String) -> String {
        let known: Set<String> = [
            "/v1/chat/completions", "/v1/completions", "/v1/models", "/v1/rerank", "/v1/embeddings",
            "/v1/messages", "/v1/responses", "/v1/models/load", "/v1/models/unload", "/research",
            "/v1/status", "/v1/catalog", "/api/chat", "/api/generate", "/api/tags", "/api/version",
            "/api/ps", "/api/show", "/api/embed", "/api/embeddings", "/api/pull", "/api/create",
            "/api/delete", "/api/copy", "/healthz", "/health", "/metrics", "/ui", "/",
        ]
        if known.contains(path) { return path }
        if path.hasPrefix("/v1/agent/") { return "/v1/agent/*" }
        if path.hasPrefix("/v1/models/") { return "/v1/models/:id" }
        if path.hasPrefix("/api/blobs/") { return "/api/blobs/:digest" }
        if path.hasPrefix("/ui/") { return "/ui/*" }
        return "other"
    }
}
