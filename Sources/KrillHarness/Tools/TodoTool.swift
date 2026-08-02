import Foundation
import KrillTooling

/// A session-scoped checklist the agent maintains for multi-step work — the
/// local-model equivalent of Claude Code's todo list. Passing `items` replaces
/// the whole list (the model re-states it with finished steps marked done);
/// calling with no items just shows it. State lives in the tool instance, so
/// each surface (code TUI, voice panel, a background agent) keeps its own list.
///
/// Read-only by the permission layer's definition: it never touches files or
/// runs commands, so planning stays available even in plan posture.
public final class TodoTool: Tool, @unchecked Sendable {
    public let name = "todo"
    public let isReadOnly = true
    public let description =
        "Track your task checklist for this session. Pass the FULL list in `items` to "
        + "replace it (mark finished steps with done:true); pass no items to view it. "
        + "Use it to plan multi-step work and tick steps off as you complete them."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "items":{"type":"array","description":"The full checklist, in order.","items":{\
    "type":"object","properties":{\
    "text":{"type":"string","description":"The step."},\
    "done":{"type":"boolean","description":"true once the step is complete."}},\
    "required":["text"]}}},\
    "required":[]}
    """

    private let lock = NSLock()
    private var items: [(text: String, done: Bool)] = []

    public init() {}

    public func run(argumentsJSON: String) async -> ToolResult {
        let obj = jsonObject(argumentsJSON) ?? [:]
        return ToolResult(
            content: renderApplying(obj["items"] as? [[String: Any]]), isError: false)
    }

    /// Synchronous seam: `NSLock` may not be held across (or taken inside) an
    /// async body, and this tool's critical section is pure in-memory state.
    private func renderApplying(_ raw: [[String: Any]]?) -> String {
        lock.lock()
        defer { lock.unlock() }
        if let raw {
            items = raw.compactMap { entry in
                guard let text = entry["text"] as? String,
                      !text.trimmingCharacters(in: .whitespaces).isEmpty else { return nil }
                return (text: text, done: entry["done"] as? Bool ?? false)
            }
        }
        guard !items.isEmpty else { return "(todo list is empty)" }
        let done = items.filter(\.done).count
        let rows = items.map { "[\($0.done ? "x" : " ")] \($0.text)" }
        return "Todo (\(done)/\(items.count) done):\n" + rows.joined(separator: "\n")
    }
}
