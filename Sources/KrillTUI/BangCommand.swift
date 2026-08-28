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
    /// Returns nil when the line is not one (no leading `!`), so the caller can
    /// fall through to its normal slash-command / prompt handling. A leading
    /// `!` always means "run this", so a message that genuinely starts with an
    /// exclamation mark needs a leading space or a rephrase.
    public static func parse(_ line: String) -> BangCommand? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("!") else { return nil }
        let isPrivate = trimmed.hasPrefix("!!")
        let body = trimmed.dropFirst(isPrivate ? 2 : 1)
        return BangCommand(
            command: body.trimmingCharacters(in: .whitespaces),
            isPrivate: isPrivate)
    }

    /// One-line reminder shown for a bare `!` / `!!`, and in `/help`.
    public static let usage =
        "!<command> runs a shell command and shows its output; "
        + "!!<command> keeps that output out of the model's context."
}
