import Foundation

/// One entry in an agent transcript - the rendering shape of a run. `foldAgentEvent`
/// is the single mapping from `AgentEvent` to these, shared by the foreground turn
/// and background sessions so the two never drift.
public enum AgentEntry: Equatable, Sendable {
    case user(String)
    case assistant(String)
    case toolCall(name: String, args: String)
    case toolResult(content: String, isError: Bool)
    case note(String)
}

/// Fold one `AgentEvent` into a transcript, appending the entries it produces.
/// `chipShown` tracks whether the in-flight tool call already emitted a chip, so
/// a denied/unknown tool (which skips `toolStarted`) still gets one before its
/// observation. The single source of truth for event -> transcript mapping.
public func foldAgentEvent(
    _ event: AgentEvent, into entries: inout [AgentEntry], chipShown: inout Bool
) {
    switch event {
    case .assistantTurn(let text):
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty { entries.append(.assistant(t)) }
    case .toolStarted(let name, let args):
        entries.append(.toolCall(name: name, args: args))
        chipShown = true
    case .toolFinished(let inv):
        if !chipShown {
            entries.append(.toolCall(name: inv.name, args: inv.argumentsJSON))
        }
        entries.append(.toolResult(content: inv.result.content, isError: inv.result.isError))
        chipShown = false
    case .finalAnswer(let text):
        entries.append(.assistant(text))
    case .iterationLimitReached:
        entries.append(.note("[stopped at the iteration limit without a final answer]"))
    case .cancelled:
        entries.append(.note("(cancelled)"))
    }
}
