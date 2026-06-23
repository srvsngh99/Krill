import Foundation
import KrillTUI

/// Sourav AI Labs / Krill brand surface for the TUI, rendered in the brand's
/// monochrome (ink-on-paper) identity: bold for the wordmark, dim for secondary
/// text, an inverse-video bar for the masthead. The `>_` device that wraps the
/// wordmark is the SAI mark (see assets/krill-lockup-ink.png).
enum Brand {
    static let product = "Krill"
    static let lab = "Sourav AI Labs"
    static let labMark = "> SAI_"            // the SAI lockup device (matches the brand lockup)
    static let labTagline = "INDEPENDENT AI LAB"
    static let tagline = "A fast, lean LLM runtime, built for Mac."
    static let site = "souravailabs.ai"
    static let chips = ["text \u{00B7} vision \u{00B7} audio", "agentic", "macOS-native"]

    /// The `>_ Krill` wordmark (plain text; caller styles it).
    static var wordmark: String { ">_ \(product)" }

    // MARK: - Masthead (persistent top bar)

    /// A light two-part masthead line (NOT a solid inverse bar): the bold
    /// wordmark + lab on the left, the loaded model dim on the right. Pair with
    /// `headerRule` on the row beneath for the underline. Degrades on narrow
    /// terminals so the line never overflows onto the rule row: drop the model
    /// (still shown in the footer) when there is no room, then clip the wordmark.
    static func header(width: Int, model: String) -> String {
        // Product-only masthead: the `>_ Krill` wordmark on the left, the loaded
        // model dim on the right. The full Sourav AI Labs lockup lives on the
        // launch splash, not the persistent bar.
        let leftPlain = "  \(wordmark)"
        let rightPlain = "\(model)  "
        let styledLeft = "  " + Ansi.bold(Ansi.ember(wordmark))
        if width >= leftPlain.count + rightPlain.count + 1 {
            let pad = width - leftPlain.count - rightPlain.count
            return styledLeft + String(repeating: " ", count: pad) + Ansi.chrome(model) + "  "
        }
        if width >= leftPlain.count { return styledLeft }
        return Ansi.bold(String(leftPlain.prefix(max(0, width))))
    }

    /// A dim full-width rule drawn under the masthead.
    static func headerRule(width: Int) -> String {
        Ansi.chrome(String(repeating: "\u{2500}", count: max(0, width)))
    }

    /// A dim footer line: `left` status on the left, `right` session info on the
    /// right.
    static func footer(width: Int, left: String, right: String) -> String {
        let l = "  \(left)"
        let r = "\(right)  "
        let pad = max(1, width - l.count - r.count)
        let line = l + String(repeating: " ", count: pad) + r
        return Ansi.chrome(clip(line, width: width))
    }

    // MARK: - Launch splash

    /// The Krill wordmark as a scaled-up line-drawing ASCII banner (figlet
    /// "big", pure ASCII so it stays inside the house ASCII rule). The hero of
    /// the splash, echoing the wordmark on the social-preview brand asset.
    static let banner: [String] = Banner.krill

    /// Centered launch splash in the brand identity: the block wordmark (or a
    /// plain `>_ Krill` fallback on terminals too narrow for the banner), a
    /// terminal-style tagline (the `>_` device typing the brand line, with "Mac"
    /// reverse-highlighted exactly as the social preview highlights it),
    /// capability chips, and the lab/site line.
    static func splash(width: Int) -> [String] {
        func center(_ s: String, _ vis: Int) -> String {
            String(repeating: " ", count: Chrome.centerPad(visibleWidth: vis, totalWidth: width)) + s
        }
        // Hero = the solid-block "KRILL" wordmark, centered on its own. Falls
        // back to the plain `>_ Krill` line on terminals too narrow to fit it.
        let bannerWidth = Banner.width(banner)
        let heroRows: [String]
        if width >= bannerWidth {
            heroRows = Ansi.emberGradient(banner).map { center($0, bannerWidth) }
        } else {
            heroRows = [center(Ansi.bold(Ansi.ember(wordmark)), visibleCount(wordmark))]
        }

        // Tagline split on "Mac" so it can be reverse-highlighted like the brand
        // asset, with the `>_` device prefixed (the line reads as a terminal
        // prompt typing the brand line).
        let parts = tagline.components(separatedBy: "Mac")
        let head = parts.first ?? tagline
        let tail = parts.count > 1 ? parts[1] : ""
        let taglineVis = 3 + head.count + (parts.count > 1 ? 3 : 0) + tail.count
        var styledTagline = Ansi.ember(">_ ") + Ansi.chrome(head)
        if parts.count > 1 { styledTagline += Ansi.inverse(Ansi.bold("Mac")) + Ansi.chrome(tail) }
        let taglineLine = center(styledTagline, taglineVis)

        let chipRow = chips.map { " \($0) " }.joined(separator: "  ")
        // The Sourav AI Labs lockup, rendered like the brand mark: the `> SAI_`
        // device bold beside the lab name, with the "INDEPENDENT AI LAB" line
        // and site beneath.
        let lockupPlain = "\(labMark)  \(lab)"
        let styledLockup = Ansi.ember(Ansi.bold(labMark)) + "  " + Ansi.chrome(lab)
        let tagPlain = "\(labTagline)  \u{00B7}  \(site)"

        var out: [String] = [""]
        out.append(contentsOf: heroRows)
        out.append("")
        out.append(taglineLine)
        out.append("")
        out.append(center(Ansi.inverse(chipRow), visibleCount(chipRow)))
        out.append("")
        out.append(center(styledLockup, lockupPlain.count))
        out.append(center(Ansi.chrome(tagPlain), tagPlain.count))
        out.append("")
        return out
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
