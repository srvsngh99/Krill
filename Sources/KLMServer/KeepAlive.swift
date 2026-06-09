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
/// Negative keep-alive ⇒ never evict (deadline = nil). Zero ⇒ evict once the
/// in-flight request that carried it has drained — never before or during it.
///
/// In-flight tracking (`beginRequest`/`endRequest`) makes eviction
/// request-safe: the model is never unloaded while a request still holds it,
/// even if a deadline elapses mid-generation. `keep_alive:0` matches Ollama's
/// "unload right after this request" rather than racing the active call.
public actor KeepAliveController {
    private let defaultSeconds: Int
    private var deadline: Date?
    private var evictImmediately = false
    /// `keep_alive:0` seen; evict as soon as the last request drains.
    private var evictAfterDrain = false
    /// Requests currently holding the model (queue slot held).
    private var inFlight = 0

    public init(defaultSeconds: Int) {
        self.defaultSeconds = defaultSeconds
        // A negative default means "never evict" (e.g. `krillm launch` pins the
        // model for an agent session). Mirror `touch()`'s semantics at init so a
        // freshly-loaded model with a negative default isn't handed a deadline
        // in the past (now + negative) and evicted before its first request.
        self.deadline = defaultSeconds < 0
            ? nil
            : Date().addingTimeInterval(TimeInterval(defaultSeconds))
    }

    /// Record activity. `override` (seconds) is the request's `keep_alive`:
    /// nil ⇒ default, <0 ⇒ pin loaded, 0 ⇒ evict after this request drains.
    public func touch(override: Int?) {
        let secs = override ?? defaultSeconds
        if secs < 0 {
            deadline = nil
            evictImmediately = false
            evictAfterDrain = false
        } else if secs == 0 {
            // Defer: don't evict until the request that set this drains.
            // Resolved in `endRequest()` (or `shouldEvict` if nothing is
            // in flight yet — e.g. a bare keep_alive:0 unload ping).
            evictAfterDrain = true
            evictImmediately = false
            deadline = Date()
        } else {
            evictImmediately = false
            evictAfterDrain = false
            deadline = Date().addingTimeInterval(TimeInterval(secs))
        }
    }

    /// Mark a request as holding the model (call after acquiring a gen slot).
    public func beginRequest() { inFlight += 1 }

    /// Release a request's hold. When the last in-flight request drains and
    /// `keep_alive:0` was requested, eviction is armed for the next tick.
    public func endRequest() {
        inFlight = max(0, inFlight - 1)
        if inFlight == 0 && evictAfterDrain { evictImmediately = true }
    }

    /// Whether the model should be evicted now. Never evicts while a request
    /// is in flight — the active request must complete first.
    public func shouldEvict(now: Date = Date()) -> Bool {
        if inFlight > 0 { return false }
        if evictImmediately { return true }
        // `keep_alive:0` with nothing in flight (bare unload ping): evict.
        if evictAfterDrain { return true }
        guard let deadline else { return false }
        return now >= deadline
    }

    public func expiresAt() -> Date? { deadline }

    public func markEvicted() {
        deadline = nil
        evictImmediately = false
        evictAfterDrain = false
    }
}
