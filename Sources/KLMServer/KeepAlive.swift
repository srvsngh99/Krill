import Foundation

/// Parse an Ollama `keep_alive` value: a number of seconds, a duration
/// string ("5m", "30s", "1h", "1h30m"), `0` (unload right after), or a
/// negative value (keep loaded indefinitely).
///
/// Returns seconds; `nil` means "not specified, use the server default".
/// A returned `< 0` means "never evict"; `0` means "evict immediately".
public enum KeepAliveParse {
    public static func seconds(from any: Any?) -> Int? {
        guard let any else { return nil }
        if let i = any as? Int { return i }
        if let d = any as? Double { return Int(d) }
        if let s = any as? String { return duration(s) }
        return nil
    }

    public static func duration(_ s: String) -> Int? {
        let t = s.trimmingCharacters(in: .whitespaces).lowercased()
        if let i = Int(t) { return i }
        if t == "0" { return 0 }
        var total = 0
        var num = ""
        var matched = false
        for ch in t {
            if ch.isNumber || ch == "-" {
                num.append(ch)
            } else {
                guard let n = Int(num) else { return matched ? total : nil }
                switch ch {
                case "s": total += n
                case "m": total += n * 60
                case "h": total += n * 3600
                case "d": total += n * 86400
                default: return matched ? total : nil
                }
                matched = true
                num = ""
            }
        }
        if let trailing = Int(num), !matched { return trailing }
        return matched ? total : nil
    }
}

/// Owns the model-eviction deadline (WS-E / T1-4). Thread-safe via actor.
///
/// `touch` is called on every generation; a background loop in
/// ``KLMServer/start()`` unloads the engine once the deadline passes.
/// Negative keep-alive ⇒ never evict (deadline = nil). Zero ⇒ evict on the
/// next tick after the request.
public actor KeepAliveController {
    private let defaultSeconds: Int
    private var deadline: Date?
    private var evictImmediately = false

    public init(defaultSeconds: Int) {
        self.defaultSeconds = defaultSeconds
        self.deadline = Date().addingTimeInterval(TimeInterval(defaultSeconds))
    }

    /// Record activity. `override` (seconds) is the request's `keep_alive`:
    /// nil ⇒ default, <0 ⇒ pin loaded, 0 ⇒ evict after this request.
    public func touch(override: Int?) {
        let secs = override ?? defaultSeconds
        if secs < 0 {
            deadline = nil
            evictImmediately = false
        } else if secs == 0 {
            evictImmediately = true
            deadline = Date()
        } else {
            evictImmediately = false
            deadline = Date().addingTimeInterval(TimeInterval(secs))
        }
    }

    /// Whether the model should be evicted now.
    public func shouldEvict(now: Date = Date()) -> Bool {
        if evictImmediately { return true }
        guard let deadline else { return false }
        return now >= deadline
    }

    public func expiresAt() -> Date? { deadline }

    public func markEvicted() {
        deadline = nil
        evictImmediately = false
    }
}
