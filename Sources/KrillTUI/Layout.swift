import Foundation

/// Pure, ANSI-aware text-layout helpers shared by terminal surfaces.
public enum Layout {
    /// Word-wrap `text` to `width` columns, preserving explicit newlines and
    /// hard-breaking words longer than the width. ANSI sequences occupy no cells.
    public static func wrap(_ text: String, width: Int) -> [String] {
        guard width > 0 else { return [text] }
        var out: [String] = []
        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            if line.isEmpty { out.append(""); continue }
            var current = ""
            for word in line.split(separator: " ", omittingEmptySubsequences: true).map(String.init) {
                let wordWidth = visibleWidth(word)
                if current.isEmpty {
                    current = wordWidth <= width ? word : hardBreak(word, width: width, into: &out)
                } else if visibleWidth(current) + 1 + wordWidth <= width {
                    current += " " + word
                } else {
                    out.append(current)
                    current = wordWidth <= width ? word : hardBreak(word, width: width, into: &out)
                }
            }
            out.append(current)
        }
        return out
    }

    /// Number of terminal cells after ignoring CSI/OSC control sequences.
    /// Krill's UI glyphs are single-cell; this intentionally avoids locale- or
    /// terminal-specific guesses for emoji width.
    public static func visibleWidth(_ text: String) -> Int {
        tokens(text).reduce(0) { $0 + ($1.isEscape ? 0 : 1) }
    }

    /// Clip to visible terminal columns while preserving complete ANSI sequences.
    /// An ellipsis is optional because row composition generally wants a hard edge.
    public static func clip(_ text: String, width: Int, ellipsis: Bool = false) -> String {
        guard width > 0 else { return "" }
        guard visibleWidth(text) > width else { return text }
        let target = max(0, width - (ellipsis && width > 1 ? 1 : 0))
        var result = ""
        var cells = 0
        var sawEscape = false
        for token in tokens(text) {
            if token.isEscape {
                result += token.text
                sawEscape = true
            } else if cells < target {
                result += token.text
                cells += 1
            } else { break }
        }
        if ellipsis && width > 1 { result += "…" }
        // A clipped styled span must not bleed into the adjacent pane.
        if sawEscape { result += "\u{1B}[0m" }
        return result
    }

    /// Compose a row whose right value is flush-right within `width`; left is
    /// clipped first so the two panes can never overlap.
    public static func join(left: String, right: String, width: Int) -> String {
        guard width > 0 else { return "" }
        let fittedRight = clip(right, width: width)
        let rightWidth = visibleWidth(fittedRight)
        let leftBudget = max(0, width - rightWidth)
        let fittedLeft = clip(left, width: leftBudget)
        let padding = max(0, leftBudget - visibleWidth(fittedLeft))
        return fittedLeft + String(repeating: " ", count: padding) + fittedRight
    }

    /// Position content without clearing the row. Row and column are 1-based.
    public static func positioned(_ content: String, row: Int, col: Int = 1) -> String {
        "\u{1B}[\(max(1, row));\(max(1, col))H" + content
    }

    private struct Token { let text: String; let isEscape: Bool }

    private static func tokens(_ text: String) -> [Token] {
        let chars = Array(text)
        var result: [Token] = []
        var i = 0
        while i < chars.count {
            guard chars[i] == "\u{1B}" else {
                result.append(Token(text: String(chars[i]), isEscape: false)); i += 1; continue
            }
            let start = i
            i += 1
            if i < chars.count, chars[i] == "[" { // CSI: final byte @ through ~
                i += 1
                while i < chars.count {
                    let scalar = chars[i].unicodeScalars.first?.value ?? 0
                    i += 1
                    if scalar >= 0x40 && scalar <= 0x7e { break }
                }
            } else if i < chars.count, chars[i] == "]" { // OSC: BEL or ST
                i += 1
                while i < chars.count {
                    if chars[i] == "\u{07}" { i += 1; break }
                    if chars[i] == "\u{1B}", i + 1 < chars.count, chars[i + 1] == "\\" { i += 2; break }
                    i += 1
                }
            } else if i < chars.count { i += 1 }
            result.append(Token(text: String(chars[start..<i]), isEscape: true))
        }
        return result
    }

    private static func hardBreak(_ word: String, width: Int, into out: inout [String]) -> String {
        var remainder = word
        while visibleWidth(remainder) > width {
            let chunk = clip(remainder, width: width)
            out.append(chunk)
            // hardBreak is used for ordinary prose; remove the consumed Characters.
            remainder = String(remainder.dropFirst(max(1, chunk.count)))
        }
        return remainder
    }
}
