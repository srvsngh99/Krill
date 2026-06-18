import Foundation

/// Pure text-layout helpers for the TUI conversation pane.
public enum Layout {
    /// Word-wrap `text` to `width` columns, preserving explicit newlines and
    /// hard-breaking words longer than the width. Returns the display lines.
    public static func wrap(_ text: String, width: Int) -> [String] {
        guard width > 0 else { return [text] }
        var out: [String] = []
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            if line.isEmpty { out.append(""); continue }
            var current = ""
            for word in line.split(separator: " ", omittingEmptySubsequences: true).map(String.init) {
                if current.isEmpty {
                    current = word.count <= width ? word : hardBreak(word, width: width, into: &out)
                } else if current.count + 1 + word.count <= width {
                    current += " " + word
                } else {
                    out.append(current)
                    current = word.count <= width ? word : hardBreak(word, width: width, into: &out)
                }
            }
            out.append(current)
        }
        return out
    }

    /// Emit full-width chunks of an over-long word and return the remainder.
    private static func hardBreak(_ word: String, width: Int, into out: inout [String]) -> String {
        var rem = Substring(word)
        while rem.count > width {
            out.append(String(rem.prefix(width)))
            rem = rem.dropFirst(width)
        }
        return String(rem)
    }
}
