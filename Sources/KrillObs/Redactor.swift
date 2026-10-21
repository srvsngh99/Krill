import Foundation

/// Scrubs secrets and personal paths from text before it leaves the machine (bug reports).
///
/// Order matters: structured forms (`Bearer x`, `key = "x"`) go first, then free-standing token
/// shapes, then home paths. Everything replaced becomes `[REDACTED]` (or `~` for a home path).
public struct Redactor: Sendable {
    public static let mask = "[REDACTED]"

    public let homePaths: [String]

    public init(home: String = NSHomeDirectory()) {
        var h: [String] = []
        if !home.isEmpty, home != "/" { h.append(home) }
        self.homePaths = h
    }

    private static let rules: [(NSRegularExpression, String)] = {
        func re(_ p: String, _ o: NSRegularExpression.Options = []) -> NSRegularExpression {
            // Patterns are literals in this file; a bad one is a programmer error caught by tests.
            try! NSRegularExpression(pattern: p, options: o)
        }
        return [
            // Authorization headers and bearer tokens.
            (re("(?i)(bearer|basic)\\s+[A-Za-z0-9._~+/=-]{6,}"), "$1 \(mask)"),
            // JSON/TOML/env style assignments to secret-named keys:  "api_key": "x"  /  token = x  /  FOO_SECRET=x
            (re("(?i)([\"']?[A-Za-z0-9_.-]*(?:api[_-]?key|apikey|secret|token(?!s)|password|passwd|credential|authorization|private[_-]?key|access[_-]?key)[A-Za-z0-9_.-]*[\"']?\\s*[:=]\\s*)(\"[^\"]+\"|'[^']+'|[^\\s,;}\\]\"']+)"),
             "$1\"\(mask)\""),
            // Well-known token shapes.
            (re("\\bsk-[A-Za-z0-9_-]{12,}"), mask),
            (re("\\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"), mask),
            (re("\\bgithub_pat_[A-Za-z0-9_]{20,}"), mask),
            (re("\\bhf_[A-Za-z0-9]{20,}"), mask),
            (re("\\bAKIA[0-9A-Z]{16}\\b"), mask),
            (re("\\bAIza[0-9A-Za-z_-]{30,}"), mask),
            (re("\\bxox[abposr]-[A-Za-z0-9-]{10,}"), mask),
            (re("\\bey[A-Za-z0-9_-]{8,}\\.[A-Za-z0-9_-]{8,}\\.[A-Za-z0-9_-]{8,}"), mask),
            (re("-----BEGIN [A-Z ]*PRIVATE KEY-----[\\s\\S]*?-----END [A-Z ]*PRIVATE KEY-----"), mask),
            // Email addresses.
            (re("[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}"), "[EMAIL]"),
        ]
    }()

    private static let genericHome: [NSRegularExpression] = [
        try! NSRegularExpression(pattern: "/Users/[^/\\s\"'\\\\:]+"),
        try! NSRegularExpression(pattern: "/home/[^/\\s\"'\\\\:]+"),
    ]

    public func redact(_ text: String) -> String {
        var s = text
        for (re, template) in Self.rules {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: template)
        }
        for h in homePaths { s = s.replacingOccurrences(of: h, with: "~") }
        for re in Self.genericHome {
            s = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "~")
        }
        return s
    }

    /// An environment value that does not carry a secret-ish NAME but looks like one: long,
    /// no whitespace, mixed letters and digits.
    public static func looksSecret(_ value: String) -> Bool {
        guard value.count >= 24, !value.contains(" "), !value.contains("/") else { return false }
        let hasDigit = value.contains { $0.isNumber }
        let hasAlpha = value.contains { $0.isLetter }
        return hasDigit && hasAlpha
    }

    public static func secretName(_ name: String) -> Bool {
        let n = name.lowercased()
        return ["key", "token", "secret", "password", "passwd", "credential", "auth"].contains { n.contains($0) }
    }

    /// `KRILL_*` / `OLLAMA_*` environment, values hidden when the name or the value looks secret.
    public func redactedEnvironment(_ env: [String: String]) -> [(String, String)] {
        env.keys.sorted().filter { $0.hasPrefix("KRILL_") || $0.hasPrefix("OLLAMA_") || $0.hasPrefix("HF_") }
            .map { k in
                let v = env[k] ?? ""
                let hidden = Self.secretName(k) || Self.looksSecret(v)
                return (k, hidden ? Self.mask : redact(v))
            }
    }

    /// config.toml text: every line whose key is secret-ish loses its value.
    public func redactConfig(_ toml: String) -> String {
        toml.components(separatedBy: "\n").map { line -> String in
            let t = line.trimmingCharacters(in: .whitespaces)
            guard !t.hasPrefix("#"), let eq = t.firstIndex(of: "=") else { return redact(line) }
            let key = String(t[..<eq]).trimmingCharacters(in: .whitespaces)
            let value = String(t[t.index(after: eq)...]).trimmingCharacters(in: .whitespaces)
            if Self.secretName(key) && !value.isEmpty && value != "\"\"" { return "\(key) = \"\(Self.mask)\"" }
            return redact(line)
        }.joined(separator: "\n")
    }
}
