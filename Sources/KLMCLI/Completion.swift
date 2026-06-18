import CEditLine
import Foundation

/// Installs Tab completion for the REPL via libedit's readline-compatible
/// completion hooks: slash-command names at the start of a line, and filesystem
/// paths everywhere else (so `/image <Tab>`, `@<Tab>`, and bare dragged paths
/// all complete files).
enum ReplCompletion {
    static func install() {
        krill_set_attempted_completion(krillAttemptedCompletion)
    }
}

/// Commands offered for completion. Kept in sync with InteractiveSession's
/// command switch; a stale entry only affects Tab hints, never behavior.
let krillReplCommands: [String] = [
    "/help", "/quit", "/exit", "/clear", "/reset", "/history", "/system",
    "/model", "/save", "/attach", "/remove", "/image", "/img", "/audio", "/mic",
]

// Generator state for the command completer. Single-threaded (readline runs on
// the REPL thread), so a plain global index is safe.
private nonisolated(unsafe) var commandGenIndex = 0

private func krillCommandGenerator(_ text: UnsafePointer<CChar>?, _ state: Int32) -> UnsafeMutablePointer<CChar>? {
    let prefix = text.map { String(cString: $0) } ?? ""
    if state == 0 { commandGenIndex = 0 }
    while commandGenIndex < krillReplCommands.count {
        let cmd = krillReplCommands[commandGenIndex]
        commandGenIndex += 1
        if cmd.hasPrefix(prefix) { return strdup(cmd) }
    }
    return nil
}

private func krillAttemptedCompletion(
    _ text: UnsafePointer<CChar>?,
    _ start: Int32,
    _ end: Int32
) -> UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>? {
    let word = text.map { String(cString: $0) } ?? ""
    // A "/command" at the very start of the line -> complete command names and
    // suppress the default filename fallback.
    if start == 0, word.hasPrefix("/") {
        krill_set_completion_over(1)
        return rl_completion_matches(text, krillCommandGenerator)
    }
    // Anywhere else, let libedit run its built-in filename completion.
    krill_set_completion_over(0)
    return nil
}
