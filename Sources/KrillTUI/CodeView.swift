import Foundation

/// Style role for a rendered transcript line in the `code` TUI. Kept as a tag
/// (not ANSI) so this geometry stays pure and unit-testable; the executable
/// layer maps each role to a color.
public enum CodeStyle: Sendable, Equatable {
    case user        // the task the user gave
    case assistant   // the model's prose
    case toolName    // a tool-call chip
    case toolOk      // a successful tool observation
    case toolError   // a failed / denied tool observation
    case diffAdd     // an added line in a diff
    case diffDel     // a removed line in a diff
    case note        // a status note ([cancelled], [iteration limit], ...)
    case dim         // de-emphasized chrome (truncation hints)
}

/// One styled display line: width-fitted text plus its role.
public struct CodeLine: Sendable, Equatable {
    public let text: String
    public let style: CodeStyle
    public init(_ text: String, _ style: CodeStyle) {
        self.text = text
        self.style = style
    }
}

/// Pure formatters that turn agentic-transcript pieces into width-fitted,
/// role-tagged lines. No ANSI, no terminal - deterministic and testable. The
/// runtime re-runs these every frame at the current width, so a resize rewraps
/// correctly (it never caches lines at a stale width).
public enum CodeView {
    /// The user's task, hanging-indented under a `>` marker.
    public static func userTask(_ text: String, width: Int) -> [CodeLine] {
        hanging(prefix: "> ", text: text, width: width).map { CodeLine($0, .user) }
    }

    /// The model's prose for a turn, word-wrapped.
    public static func assistantText(_ text: String, width: Int) -> [CodeLine] {
        Layout.wrap(text, width: max(1, width)).map { CodeLine($0, .assistant) }
    }

    /// A status note (cancelled, iteration limit), word-wrapped.
    public static func note(_ text: String, width: Int) -> [CodeLine] {
        Layout.wrap(text, width: max(1, width)).map { CodeLine($0, .note) }
    }

    /// A tool-call chip: `* name(args)` on one line, ellipsized to the width
    /// (tool-arg JSON wraps poorly, so a single clipped line reads cleanly).
    public static func toolCall(name: String, argumentsJSON: String, width: Int) -> [CodeLine] {
        let chip = "* \(name)(\(argumentsJSON))"
        return [CodeLine(clip(chip, width: max(1, width)), .toolName)]
    }

    /// A tool observation, indented under its chip. Diff lines (`+ `/`- `) are
    /// tagged for color; everything else takes ok/error per `isError`. At most
    /// `maxLines` are shown, with a dim "N more lines" hint when truncated.
    public static func toolResult(
        content: String, isError: Bool, width: Int, maxLines: Int
    ) -> [CodeLine] {
        let indent = "    "
        let body = max(1, width - indent.count)
        let raw = content.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        let shown = maxLines > 0 ? Array(raw.prefix(maxLines)) : raw

        var out: [CodeLine] = []
        for line in shown {
            let style: CodeStyle
            if line.hasPrefix("+ ") || line == "+" { style = .diffAdd }
            else if line.hasPrefix("- ") || line == "-" { style = .diffDel }
            else { style = isError ? .toolError : .toolOk }
            // Wrap long observation lines so nothing runs off-screen.
            for w in Layout.wrap(line, width: body) {
                out.append(CodeLine(indent + w, style))
            }
        }
        let hidden = raw.count - shown.count
        if hidden > 0 {
            out.append(CodeLine(indent + "... (\(hidden) more line\(hidden == 1 ? "" : "s"))", .dim))
        }
        return out
    }

    // MARK: - helpers

    /// Wrap `text` to `width`, prefixing the first line with `prefix` and
    /// continuation lines with an equal-width blank hang.
    private static func hanging(prefix: String, text: String, width: Int) -> [String] {
        let body = max(1, width - prefix.count)
        let wrapped = Layout.wrap(text, width: body)
        guard !wrapped.isEmpty else { return [prefix] }
        let hang = String(repeating: " ", count: prefix.count)
        return wrapped.enumerated().map { i, line in (i == 0 ? prefix : hang) + line }
    }

    /// Clip `s` to `width` columns, marking the cut with a trailing ellipsis.
    private static func clip(_ s: String, width: Int) -> String {
        guard s.count > width else { return s }
        guard width > 1 else { return String(s.prefix(width)) }
        return String(s.prefix(width - 1)) + "\u{2026}"
    }
}
