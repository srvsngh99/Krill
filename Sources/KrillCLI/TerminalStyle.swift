import Foundation
import KrillTUI

// MARK: - ANSI styling

/// Minimal ANSI styling, disabled automatically when stdout is not a TTY or
/// `NO_COLOR` is set (https://no-color.org). All helpers are no-ops then, so
/// piped / redirected output stays clean plain text.
enum Ansi {
    static let enabled: Bool = {
        isatty(fileno(stdout)) != 0
            && ProcessInfo.processInfo.environment["NO_COLOR"] == nil
            && ProcessInfo.processInfo.environment["TERM"] != "dumb"
    }()

    private static func wrap(_ s: String, _ code: String) -> String {
        enabled ? "\u{1B}[\(code)m\(s)\u{1B}[0m" : s
    }

    static func bold(_ s: String) -> String { wrap(s, "1") }
    static func dim(_ s: String) -> String { wrap(s, "2") }
    /// Faded + italic - for unobtrusive contextual hints.
    static func hint(_ s: String) -> String { wrap(s, "2;3") }
    static func inverse(_ s: String) -> String { wrap(s, "7") }
    /// Slow-blink attribute (SGR 5) - the terminal animates it, so a pulsing
    /// glyph needs no render timer. Used for the live reply bullet while the
    /// model is still producing the turn.
    static func blink(_ s: String) -> String { wrap(s, "5") }
    static func underline(_ s: String) -> String { wrap(s, "4") }
    static func white(_ s: String) -> String { wrap(s, "97") }
    static func cyan(_ s: String) -> String { wrap(s, "36") }
    static func green(_ s: String) -> String { wrap(s, "32") }
    static func red(_ s: String) -> String { wrap(s, "31") }
    static func yellow(_ s: String) -> String { wrap(s, "33") }
    static func magenta(_ s: String) -> String { wrap(s, "35") }
    static func gray(_ s: String) -> String { wrap(s, "90") }

    // MARK: - Ember accent (Krill's owned palette — the one place colour lives)

    /// Single-line Ember accent (coral mid-stop #FF7D5C) for marks, prompts, spinner.
    static func ember(_ s: String) -> String { wrap(s, "38;2;255;125;92") }
    /// Colour a multi-row banner top→bottom amber #FFC04D → coral #FF7D5C → magenta #E0457D.
    static func emberGradient(_ rows: [String]) -> [String] {
        guard rows.count > 1 else { return rows.map { ember($0) } }
        let stops = ["38;2;255;192;77", "38;2;255;160;84", "38;2;255;125;92", "38;2;240;94;104", "38;2;224;69;125"]
        return rows.enumerated().map { i, r in
            let t = Double(i) / Double(rows.count - 1)
            return wrap(r, stops[min(stops.count - 1, Int((t * Double(stops.count - 1)).rounded()))])
        }
    }

    /// Gray-tint an already-styled string: set gray as the base color and
    /// re-enter gray after every inner reset, so embedded spans (code, bold)
    /// keep their own styling but plain text reads dim. Used to calm the model's
    /// reply against the user's bright-white turn.
    static func dimStyled(_ s: String) -> String {
        guard enabled else { return s }
        let reEnter = "\u{1B}[0m\u{1B}[90m"
        return "\u{1B}[90m" + s.replacingOccurrences(of: "\u{1B}[0m", with: reEnter) + "\u{1B}[0m"
    }

    // MARK: - Background-adaptive shade roles

    /// The resolved palette for the three speaker/chrome shade roles. Defaults to
    /// the always-readable "unknown" palette and is set once at TUI startup after
    /// the terminal background is detected (see `Theme`). Single-threaded: written
    /// before the first render, read only from the render task.
    nonisolated(unsafe) static var theme: Palette = Theme.palette(for: .unknown)

    private static func roleWrap(_ s: String, _ code: String?) -> String {
        guard enabled, let code else { return s }
        return "\u{1B}[\(code)m\(s)\u{1B}[0m"
    }

    /// User-turn shade (bright-white on dark, bold default on light/unknown).
    static func user(_ s: String) -> String { roleWrap(s, theme.userSGR) }

    /// Secondary chrome shade (masthead, footer, rules, borders, notes).
    static func chrome(_ s: String) -> String { roleWrap(s, theme.chromeSGR) }

    /// Like `chrome`, but re-enters the chrome shade after every inner reset, so
    /// embedded colour spans (ember signal accents) keep their colour while the
    /// rest of the line stays dim. Used by the footer, which carries ember spans.
    static func chromeStyled(_ s: String) -> String {
        guard enabled, let code = theme.chromeSGR else { return s }
        let reEnter = "\u{1B}[0m\u{1B}[\(code)m"
        return "\u{1B}[\(code)m" + s.replacingOccurrences(of: "\u{1B}[0m", with: reEnter) + "\u{1B}[0m"
    }

    /// Model-turn shade: tint plain text with the model color while preserving
    /// embedded spans (re-enter the color after every inner reset), like
    /// `dimStyled` but driven by the palette. When the palette has no model color
    /// (unknown background), the terminal's own foreground is used unchanged.
    static func model(_ s: String) -> String {
        guard enabled, let code = theme.modelSGR else { return s }
        let reEnter = "\u{1B}[0m\u{1B}[\(code)m"
        return "\u{1B}[\(code)m" + s.replacingOccurrences(of: "\u{1B}[0m", with: reEnter) + "\u{1B}[0m"
    }

    /// Render a user-turn line as a subtle full-width bar: the plain text padded
    /// to `width` and wrapped in the palette's bar SGR (fg+bg). When the palette
    /// has no bar (unknown background), fall back to the bright user shade so the
    /// turn still stands apart. `text` must be plain (no embedded resets).
    static func userBar(_ text: String, width: Int) -> String {
        let pad = max(0, width - text.count)
        let full = text + String(repeating: " ", count: pad)
        guard enabled, let code = theme.userBarSGR else { return user(text) }
        return "\u{1B}[\(code)m" + full + "\u{1B}[0m"
    }

    /// Clear the current line and return the cursor to column 0.
    static var clearLine: String { enabled ? "\r\u{1B}[2K" : "\r" }

    /// Color a libedit/readline prompt. The escape sequences are wrapped in the
    /// readline "ignore" markers (\001 .. \002) so the line editor counts the
    /// prompt as zero-width and keeps cursor math correct while editing.
    static func prompt(_ s: String, _ code: String) -> String {
        guard enabled else { return s }
        return "\u{01}\u{1B}[\(code)m\u{02}\(s)\u{01}\u{1B}[0m\u{02}"
    }
}

// MARK: - Spinner

/// A pre-first-token "thinking" spinner. ASCII frames only (no non-ASCII glyphs)
/// so it renders everywhere. Animates on a background task while the caller
/// awaits the model's first token, then `stop()` clears the line.
final class Spinner: @unchecked Sendable {
    private let label: String
    private var task: Task<Void, Never>?

    init(_ label: String) { self.label = label }

    func start() {
        guard Ansi.enabled else { return }
        let label = self.label
        task = Task {
            let frames = ["|", "/", "-", "\\"]
            var i = 0
            while !Task.isCancelled {
                print("\r\(Ansi.ember(frames[i % frames.count])) \(Ansi.dim(label))", terminator: "")
                fflush(stdout)
                i += 1
                try? await Task.sleep(nanoseconds: 90_000_000)
            }
        }
    }

    /// Stop animating and clear the line. Awaits the animation task's exit
    /// first, so no stray frame can print after the line is cleared.
    func stop() async {
        task?.cancel()
        await task?.value
        task = nil
        if Ansi.enabled {
            print(Ansi.clearLine, terminator: "")
            fflush(stdout)
        }
    }
}

// MARK: - Markdown-lite streaming renderer

/// Line-buffered markdown-lite styler for streamed model output. Completed lines
/// are styled and emitted as they arrive; the trailing partial line is held
/// until the next newline or `finish()`. Handles fenced code blocks, ATX
/// headings, inline `code`, and `**bold**` - enough to read cleanly without a
/// full markdown engine.
final class MarkdownStream {
    private var buffer = ""
    private var inFence = false

    /// Feed streamed text; returns styled text ready to print (only fully
    /// received lines are styled; a partial last line is buffered).
    func consume(_ text: String) -> String {
        buffer += text
        var out = ""
        while let nl = buffer.firstIndex(of: "\n") {
            let line = String(buffer[buffer.startIndex..<nl])
            buffer.removeSubrange(buffer.startIndex...nl)
            out += renderLine(line) + "\n"
        }
        return out
    }

    /// Flush and style any remaining partial line.
    func finish() -> String {
        guard !buffer.isEmpty else { return "" }
        let line = buffer
        buffer = ""
        return renderLine(line)
    }

    private func renderLine(_ line: String) -> String {
        guard Ansi.enabled else { return line }
        let trimmed = line.trimmingCharacters(in: .whitespaces)

        // Fenced code block: toggle on ``` and color the block body.
        if trimmed.hasPrefix("```") {
            inFence.toggle()
            return Ansi.gray(line)
        }
        if inFence { return Ansi.green(line) }

        // ATX heading.
        if let hashRange = trimmed.range(of: #"^#{1,6}\s+"#, options: .regularExpression) {
            return Ansi.bold(Ansi.magenta(String(trimmed[hashRange.upperBound...])))
        }

        return styleInline(line)
    }

    /// Style inline `code` (colored) then `**bold**`. Code first so `**` inside
    /// a code span is not mistaken for bold.
    private func styleInline(_ s: String) -> String {
        var out = applyRegex(s, pattern: "`([^`]+)`") { Ansi.cyan($0) }
        out = applyRegex(out, pattern: #"\*\*(.+?)\*\*"#) { Ansi.bold($0) }
        return out
    }

    private func applyRegex(_ s: String, pattern: String, style: (String) -> String) -> String {
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
