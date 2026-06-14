import Foundation

/// Sourav AI Labs / KrillLM brand surface for the TUI, rendered in the brand's
/// monochrome (ink-on-paper) identity: bold for the wordmark, dim for secondary
/// text, an inverse-video bar for the masthead. The `>_` device that wraps the
/// wordmark is the SAI mark (see assets/krillm-lockup-ink.png).
enum Brand {
    static let product = "KrillLM"
    static let lab = "Sourav AI Labs"
    static let tagline = "A fast, lean LLM runtime, built for Mac."
    static let site = "souravailabs.ai"
    static let chips = ["text \u{00B7} vision \u{00B7} audio", "agentic", "macOS-native"]

    /// The `>_ KrillLM` wordmark (plain text; caller styles it).
    static var wordmark: String { ">_ \(product)" }

    // MARK: - Masthead (persistent top bar)

    /// A full-width inverse-video header bar: wordmark on the left, the loaded
    /// model on the right. Mirrors the ink/paper inversion of the brand lockup.
    static func header(width: Int, model: String) -> String {
        let left = " \(wordmark)  \(lab) "
        let right = "\(model) "
        let pad = max(1, width - visibleCount(left) - visibleCount(right))
        let bar = left + String(repeating: " ", count: pad) + right
        return Ansi.inverse(Ansi.bold(clip(bar, width: width)))
    }

    /// A dim footer line: status on the left, the lab site on the right.
    static func footer(width: Int, status: String) -> String {
        let left = " \(status)"
        let right = "\(lab) \u{00B7} \(site) "
        let pad = max(1, width - visibleCount(left) - visibleCount(right))
        let line = left + String(repeating: " ", count: pad) + right
        return Ansi.dim(clip(line, width: width))
    }

    // MARK: - Launch splash

    /// Centered launch splash: wordmark, tagline, capability chips, lab line.
    /// Returns styled lines sized to `width`.
    static func splash(width: Int) -> [String] {
        func center(_ s: String, _ vis: Int) -> String {
            let pad = max(0, (width - vis) / 2)
            return String(repeating: " ", count: pad) + s
        }
        let chipRow = chips.map { "[ \($0) ]" }.joined(separator: "   ")
        let labLine = "a \(lab) project \u{00B7} \(site)"
        return [
            "",
            center(Ansi.bold(wordmark), visibleCount(wordmark)),
            "",
            center(Ansi.dim(tagline), visibleCount(tagline)),
            "",
            center(Ansi.inverse(chipRow), visibleCount(chipRow)),
            "",
            center(Ansi.dim(labLine), visibleCount(labLine)),
            "",
        ]
    }

    // MARK: - Helpers

    /// Visible character count (ANSI styling here never reaches this since the
    /// inputs are plain text, but guard anyway by stripping CSI sequences).
    static func visibleCount(_ s: String) -> Int {
        stripAnsi(s).count
    }

    static func stripAnsi(_ s: String) -> String {
        guard s.contains("\u{1B}") else { return s }
        var out = ""
        var inEsc = false
        for ch in s {
            if inEsc {
                if ch.isLetter { inEsc = false }
            } else if ch == "\u{1B}" {
                inEsc = true
            } else {
                out.append(ch)
            }
        }
        return out
    }

    private static func clip(_ s: String, width: Int) -> String {
        s.count <= width ? s : String(s.prefix(width))
    }
}
