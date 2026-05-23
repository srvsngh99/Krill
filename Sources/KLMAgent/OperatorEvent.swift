import Foundation

/// One observable step in an `OperatorLoop` run.
///
/// The loop emits events on an `AsyncStream` so the CLI can render them
/// incrementally (text + tool-call notices + confirmation prompts) and so
/// the `--json` mode can dump one JSON object per line.
///
/// Cases are designed to round-trip through JSON for the `--json`
/// scripting surface (see `OperatorEvent.encodeJSON`).
public enum OperatorEvent: Equatable, Sendable {
    /// The loop accepted the goal and is about to start step 1.
    case goalStarted(goal: String)

    /// The model finished a turn with assistant-visible text. Empty
    /// strings are dropped before this is emitted.
    case assistantMessage(String)

    /// The model emitted a tool call. Arguments are the raw JSON string
    /// as the model wrote it (so a JSON consumer can re-parse it
    /// instead of going through a typed bridge).
    case toolCallStarted(name: String, argumentsJSON: String)

    /// The tool call finished. `content` is what's fed back to the
    /// model; `isError` is set when the tool threw.
    case toolCallResult(name: String, content: String, isError: Bool)

    /// A hardware-fit / scope warning. `severity` is one of
    /// `info` / `warn` / `risky` / `wontFit`.
    case warning(severity: WarningSeverity, message: String)

    /// The loop is asking the user to confirm an action before
    /// proceeding. The CLI renders this as `[y/n]`; `--yes` short-
    /// circuits to "yes" without emitting the event.
    case confirmationNeeded(action: String, prompt: String)

    /// The loop finished cleanly with a summary line.
    case goalCompleted(summary: String)

    /// The loop terminated because the same tool call with the same
    /// arguments fired three times in a row (probable model loop).
    case stuck(reason: String)

    /// The loop terminated because the budget was exhausted (max
    /// steps, max tool calls, or max output tokens).
    case budgetExhausted(detail: String)

    /// The loop was cancelled (Ctrl-C or external cancel).
    case cancelled
}

/// Severity tier for `OperatorEvent.warning`. Maps onto the
/// hardware-fit classification in `HardwareInfo.classifyFit`.
public enum WarningSeverity: String, Equatable, Sendable, Codable {
    case info
    case warn
    case risky
    case wontFit = "wont_fit"
}

public extension OperatorEvent {
    /// Stable string tag used by the JSON event surface. Lets `jq`
    /// pipelines filter by `.type` without parsing the case payload.
    var typeTag: String {
        switch self {
        case .goalStarted: return "goal_started"
        case .assistantMessage: return "assistant_message"
        case .toolCallStarted: return "tool_call_started"
        case .toolCallResult: return "tool_call_result"
        case .warning: return "warning"
        case .confirmationNeeded: return "confirmation_needed"
        case .goalCompleted: return "goal_completed"
        case .stuck: return "stuck"
        case .budgetExhausted: return "budget_exhausted"
        case .cancelled: return "cancelled"
        }
    }

    /// Encode the event as a single-line JSON object for the `--json`
    /// scripting mode. Field shape is stable per `typeTag`.
    func encodeJSON() -> String {
        var obj: [String: Any] = ["type": typeTag]
        switch self {
        case .goalStarted(let goal):
            obj["goal"] = goal
        case .assistantMessage(let text):
            obj["text"] = text
        case .toolCallStarted(let name, let argsJSON):
            obj["name"] = name
            if let data = argsJSON.data(using: .utf8),
               let parsed = try? JSONSerialization.jsonObject(with: data) {
                obj["arguments"] = parsed
            } else {
                obj["arguments_raw"] = argsJSON
            }
        case .toolCallResult(let name, let content, let isError):
            obj["name"] = name
            obj["content"] = content
            obj["is_error"] = isError
        case .warning(let severity, let message):
            obj["severity"] = severity.rawValue
            obj["message"] = message
        case .confirmationNeeded(let action, let prompt):
            obj["action"] = action
            obj["prompt"] = prompt
        case .goalCompleted(let summary):
            obj["summary"] = summary
        case .stuck(let reason):
            obj["reason"] = reason
        case .budgetExhausted(let detail):
            obj["detail"] = detail
        case .cancelled:
            break
        }
        guard let data = try? JSONSerialization.data(
            withJSONObject: obj, options: [.sortedKeys])
        else {
            return "{\"type\":\"\(typeTag)\"}"
        }
        return String(data: data, encoding: .utf8) ?? "{\"type\":\"\(typeTag)\"}"
    }
}

/// Hard upper bounds for a single `OperatorLoop.run` call.
///
/// The defaults match §2.6 of the strategic plan: operator-agent goals
/// are short ("pull this model and load it", "tell me what fits"), so
/// the loop bails after a small number of steps. Long autonomous runs
/// are an explicit non-goal — they belong in a coding agent.
public struct OperatorBudget: Equatable, Sendable {
    /// Maximum number of model turns (one prompt + one decode = one step).
    public let maxSteps: Int
    /// Maximum number of tool calls across the run.
    public let maxToolCalls: Int
    /// Maximum tokens the loop will accept from the model across the run.
    public let maxOutputTokens: Int

    public init(maxSteps: Int = 15, maxToolCalls: Int = 30, maxOutputTokens: Int = 4096) {
        self.maxSteps = maxSteps
        self.maxToolCalls = maxToolCalls
        self.maxOutputTokens = maxOutputTokens
    }
}
