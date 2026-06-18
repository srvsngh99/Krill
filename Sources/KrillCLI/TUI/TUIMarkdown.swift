import Foundation
import KrillTUI

/// Render markdown-lite text into styled, width-wrapped display lines for the
/// TUI conversation pane: fenced code blocks, ATX headings, inline `code` and
/// `**bold**`. Stateless per call but tracks code-fence state across the lines
/// of one message.
enum TUIMarkdown {
    static func render(_ text: String, width: Int) -> [String] {
        var out: [String] = []
        var inFence = false
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(raw)
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") {
                inFence.toggle()
                out.append(Ansi.dim(line))
                continue
            }
            if inFence {
                for w in Layout.wrap(line, width: width) { out.append(Ansi.green(w)) }
                continue
            }
            if let r = trimmed.range(of: #"^#{1,6}\s+"#, options: .regularExpression) {
                let heading = String(trimmed[r.upperBound...])
                for w in Layout.wrap(heading, width: width) { out.append(Ansi.bold(w)) }
                continue
            }
            for w in Layout.wrap(line, width: width) {
                out.append(styleInline(w))
            }
        }
        return out
    }

    /// Style inline `code` (cyan) then `**bold**`. Code first so `**` inside a
    /// code span is left alone.
    static func styleInline(_ s: String) -> String {
        guard Ansi.enabled else { return s }
        var out = applyRegex(s, pattern: "`([^`]+)`") { Ansi.cyan($0) }
        out = applyRegex(out, pattern: #"\*\*(.+?)\*\*"#) { Ansi.bold($0) }
        return out
    }

    private static func applyRegex(_ s: String, pattern: String, style: (String) -> String) -> String {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return s }
        let ns = s as NSString
        let matches = re.matches(in: s, range: NSRange(location: 0, length: ns.length))
        guard !matches.isEmpty else { return s }
        var result = ""
        var last = 0
        for m in matches {
            result += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            result += style(ns.substring(with: m.range(at: 1)))
            last = m.range.location + m.range.length
        }
        result += ns.substring(from: last)
        return result
    }
}
