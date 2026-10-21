import Foundation

/// A JSON scalar for a log field. Kept tiny and `Sendable` so an event can be built on a
/// request thread and encoded later on the writer queue.
public enum LogValue: Sendable, Equatable, ExpressibleByStringLiteral, ExpressibleByIntegerLiteral,
                      ExpressibleByFloatLiteral, ExpressibleByBooleanLiteral {
    case string(String)
    case int(Int)
    case double(Double)
    case bool(Bool)
    case null

    public init(stringLiteral value: String) { self = .string(value) }
    public init(integerLiteral value: Int) { self = .int(value) }
    public init(floatLiteral value: Double) { self = .double(value) }
    public init(booleanLiteral value: Bool) { self = .bool(value) }

    public static func opt(_ s: String?) -> LogValue { s.map { .string($0) } ?? .null }
    public static func opt(_ i: Int?) -> LogValue { i.map { .int($0) } ?? .null }
    public static func opt(_ d: Double?) -> LogValue { d.map { .double($0) } ?? .null }
}

/// One log record: a timestamp, an event name and ordered fields.
public struct LogEvent: Sendable {
    public var ts: Date
    public var event: String
    public var fields: [(String, LogValue)]

    public init(_ event: String, ts: Date = Date(), _ fields: [(String, LogValue)] = []) {
        self.ts = ts
        self.event = event
        self.fields = fields
    }
}

/// Hand-rolled JSON line encoding: deterministic key order (ts, event, then fields as given),
/// no Foundation encoder allocations on the hot path, and it runs on the writer queue.
public enum JSONLine {
    public static func encode(_ e: LogEvent) -> String {
        var s = "{\"ts\":\""
        s += isoUTC(e.ts)
        s += "\",\"event\":"
        appendString(&s, e.event)
        for (k, v) in e.fields {
            s += ","
            appendString(&s, k)
            s += ":"
            append(&s, v)
        }
        s += "}"
        return s
    }

    public static func append(_ s: inout String, _ v: LogValue) {
        switch v {
        case .string(let x): appendString(&s, x)
        case .int(let x): s += String(x)
        case .double(let x):
            if x.isFinite { s += trimmed(x) } else { s += "null" }
        case .bool(let x): s += x ? "true" : "false"
        case .null: s += "null"
        }
    }

    static func trimmed(_ x: Double) -> String {
        let r = (x * 1000).rounded() / 1000
        if r == r.rounded() && abs(r) < 1e15 { return String(Int64(r)) }
        return String(r)
    }

    public static func appendString(_ s: inout String, _ x: String) {
        s += "\""
        for u in x.unicodeScalars {
            switch u {
            case "\"": s += "\\\""
            case "\\": s += "\\\\"
            case "\n": s += "\\n"
            case "\r": s += "\\r"
            case "\t": s += "\\t"
            default:
                if u.value < 0x20 { s += String(format: "\\u%04x", u.value) } else { s.unicodeScalars.append(u) }
            }
        }
        s += "\""
    }

    /// `2026-10-22T10:11:12.123Z` without a DateFormatter.
    public static func isoUTC(_ d: Date) -> String {
        let secs = d.timeIntervalSince1970
        var t = time_t(secs.rounded(.down))
        var tmv = tm()
        gmtime_r(&t, &tmv)
        let ms = Int(((secs - secs.rounded(.down)) * 1000).rounded(.down))
        return String(format: "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                      Int(tmv.tm_year) + 1900, Int(tmv.tm_mon) + 1, Int(tmv.tm_mday),
                      Int(tmv.tm_hour), Int(tmv.tm_min), Int(tmv.tm_sec), ms)
    }

    public static func parseISO(_ s: String) -> Date? {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let d = f.date(from: s) { return d }
        f.formatOptions = [.withInternetDateTime]
        return f.date(from: s)
    }
}
