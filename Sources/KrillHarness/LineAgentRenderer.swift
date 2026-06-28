import Foundation

/// Renders an agent run to a line-based sink (stdout by default) by folding each
/// `AgentEvent` through the shared `foldAgentEvent` mapping — the SAME seam the
/// full-screen `ChatTUI` (`applyAgentEvent`) and background `AgentSession` use —
/// then printing the `AgentEntry` transcript it produces. Used by
/// `krill code --classic` / non-interactive runs.
///
/// Why this exists: the classic `code` path used to hand-roll its own `switch`
/// over `AgentEvent`, so a new event case had to be handled in two places (there
/// and `foldAgentEvent`) or the classic output would silently drift from the TUI.
/// Routing through the fold keeps the event→transcript semantics single-sourced;
/// only the per-frontend *rendering* of an `AgentEntry` to a text line lives here.
///
/// `AgentLoop` invokes `onEvent` serially from its single run task, so the mutable
/// rendering state below is never touched concurrently — the same assumption the
/// `EventQueue` hand-off relies on (hence `@unchecked Sendable`).
public final class LineAgentRenderer: @unchecked Sendable {
    private let maxResultLines: Int
    private let emit: (String) -> Void
    private var chipShown = false
    /// A `.toolCall` is buffered until its `.toolResult` arrives so the chip can be
    /// printed on one line with the success/error marker (`[*]`/`[x]`), matching the
    /// long-standing classic format. `foldAgentEvent` always emits the result
    /// immediately after the call, so the buffer holds at most one entry.
    private var pendingCall: (name: String, args: String)?

    public init(maxResultLines: Int = 20, emit: @escaping (String) -> Void = { print($0) }) {
        self.maxResultLines = maxResultLines
        self.emit = emit
    }

    /// Fold one event and render whatever transcript entries it produces.
    public func handle(_ event: AgentEvent) {
        var produced: [AgentEntry] = []
        foldAgentEvent(event, into: &produced, chipShown: &chipShown)
        for entry in produced { render(entry) }
    }

    private func render(_ entry: AgentEntry) {
        switch entry {
        case .user(let text):
            emit("> \(text)")
        case .assistant(let text):
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { emit(text) }
        case .toolCall(let name, let args):
            pendingCall = (name, args)
        case .toolResult(let content, let isError):
            let marker = isError ? "x" : "*"
            if let call = pendingCall {
                emit("  [\(marker)] \(call.name)(\(call.args))")
                pendingCall = nil
            }
            let lines = content
                .split(separator: "\n", omittingEmptySubsequences: false)
                .prefix(maxResultLines)
            for line in lines { emit("      \(line)") }
        case .note(let text):
            emit(text)
        }
    }
}
