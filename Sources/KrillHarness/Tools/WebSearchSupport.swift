import Foundation

/// Small, dependency-free text helpers shared by the HTML/JSON search backends
/// (`DuckDuckGoBackend` scrapes HTML; Brave/Tavily snippets embed `<strong>`
/// highlight tags). Kept internal so they are unit-testable without a network.
enum WebSearchText {
    /// Strip HTML tags and collapse whitespace into a single-line plain string.
    static func stripHTML(_ s: String) -> String {
        var out = ""
        var inTag = false
        for ch in s {
            if ch == "<" { inTag = true; continue }
            if ch == ">" { inTag = false; out.append(" "); continue }
            if !inTag { out.append(ch) }
        }
        return decodeEntities(out)
            .replacingOccurrences(of: "\u{00A0}", with: " ")
            .split(whereSeparator: { $0 == " " || $0 == "\n" || $0 == "\t" || $0 == "\r" })
            .joined(separator: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Decode the handful of HTML entities that show up in search titles/snippets.
    static func decodeEntities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var r = s
        let named: [(String, String)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""),
            ("&#x27;", "'"), ("&#39;", "'"), ("&apos;", "'"), ("&nbsp;", " "),
            ("&#x2F;", "/"), ("&#47;", "/"),
        ]
        for (k, v) in named { r = r.replacingOccurrences(of: k, with: v) }
        return r
    }

    /// All capture-group-1 substrings matching `pattern` (case-insensitive,
    /// dot-matches-newline), in order. Best-effort: returns `[]` on a bad pattern.
    static func captures(_ pattern: String, in text: String) -> [[String]] {
        guard let re = try? NSRegularExpression(
            pattern: pattern, options: [.caseInsensitive, .dotMatchesLineSeparators]) else {
            return []
        }
        let ns = text as NSString
        let matches = re.matches(in: text, range: NSRange(location: 0, length: ns.length))
        return matches.map { m in
            (0..<m.numberOfRanges).map { i in
                let r = m.range(at: i)
                return r.location == NSNotFound ? "" : ns.substring(with: r)
            }
        }
    }
}
