import CEditLine
import Foundation

/// Thin wrapper over macOS libedit's readline-compatible API for the
/// interactive REPL: line editing (Ctrl-A/E/K/U, left/right), history (up/down),
/// and tab completion - none of which plain `Swift.readLine()` provides.
///
/// `readLine(prompt:)` returns the entered line (without trailing newline), or
/// `nil` on EOF (Ctrl-D on an empty line). The caller owns history: call
/// `addHistory` for lines worth recalling.
enum LineEditor {
    /// Read one edited line. Returns nil on EOF (Ctrl-D).
    static func readLine(prompt: String) -> String? {
        guard let raw = CEditLine.readline(prompt) else { return nil }
        defer { free(raw) }
        return String(cString: raw)
    }

    /// Append a line to the in-memory history (skips blank lines).
    static func addHistory(_ line: String) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard !trimmed.isEmpty else { return }
        add_history(line)
    }
}
