import Foundation

/// A shell escape typed straight at the chat prompt.
///
/// `!<command>` runs `command` in a shell and shows its output in the
/// transcript; whether the model also sees that output is the
/// `shell_output_to_model` setting. `!!<command>` runs it the same way but
/// always keeps the output local — the model never sees it, whatever the
/// setting says. So the double bang is the private form, and the setting only
/// governs the single one.
///
/// Parsing is pure so the TUI's dispatch is unit-testable without a terminal.
public struct BangCommand: Equatable, Sendable {
    /// The shell command to run. Empty when the user typed a bare `!` / `!!`,
    /// which the caller answers with usage text rather than a subprocess.
    public let command: String

    /// True for `!!` — this run's output stays out of the model's context
    /// regardless of `shell_output_to_model`.
    public let isPrivate: Bool

    public init(command: String, isPrivate: Bool) {
        self.command = command
        self.isPrivate = isPrivate
    }

    /// Parse a submitted line as a shell escape.
    ///
    /// Returns nil when the line is not one, so the caller can fall through to
    /// its normal slash-command / prompt handling. That covers two cases: a
    /// line with no leading `!` at all, and a line escaped with `\!`, which is
    /// how you send a message that genuinely starts with an exclamation mark
    /// (the caller strips the backslash with `unescape`).
    public static func parse(_ line: String) -> BangCommand? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("!") else { return nil }
        let isPrivate = trimmed.hasPrefix("!!")
        let body = trimmed.dropFirst(isPrivate ? 2 : 1)
        return BangCommand(
            command: body.trimmingCharacters(in: .whitespaces),
            isPrivate: isPrivate)
    }

    /// Strip the `\!` escape so an ordinary message can start with a literal
    /// exclamation mark. Callers apply this to any line `parse` declined, so
    /// `\!important` is sent to the model as `!important`. Only the leading
    /// marker is touched; a backslash anywhere else is the user's own text.
    public static func unescape(_ line: String) -> String {
        let leading = line.prefix { $0 == " " || $0 == "\t" }
        let rest = line.dropFirst(leading.count)
        guard rest.hasPrefix("\\!") else { return line }
        return String(leading) + String(rest.dropFirst())
    }

    /// One-line reminder shown for a bare `!` / `!!`, and in `/help`.
    public static let usage =
        "!<command> runs a shell command and shows its output; "
        + "!!<command> keeps that output out of the model's context "
        + "(\\! sends a literal leading exclamation mark to the model)."
}
