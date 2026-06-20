import Foundation

/// A request to spawn a background agent, produced by `DispatchTool` and drained
/// by the TUI, which creates and starts an independent `AgentSession`.
public struct SpawnRequest: Sendable, Equatable {
    public let title: String
    public let task: String
    public init(title: String, task: String) {
        self.title = title
        self.task = task
    }
}

/// Thread-safe hand-off of spawn requests from a tool's `run` (which executes on
/// the agent loop's task) to the main TUI task that owns session creation.
public final class SpawnQueue: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [SpawnRequest] = []
    public init() {}
    public func push(_ r: SpawnRequest) { lock.lock(); pending.append(r); lock.unlock() }
    public func drain() -> [SpawnRequest] {
        lock.lock(); defer { lock.unlock() }
        let out = pending; pending.removeAll(); return out
    }
}

/// `dispatch_agent` - let the model spawn a focused background agent for a
/// sub-task. The new agent runs independently (it does NOT report back into this
/// conversation - it is a background worker, not a return-into-parent subagent);
/// the user can attach to it from the agent switcher to watch and steer it.
///
/// The tool itself only enqueues the request (hence `isReadOnly` at the parent's
/// permission layer, so spawning never prompts); the child inherits the parent's
/// permission posture, so the child's own edits/commands are gated in its session.
public struct DispatchTool: Tool {
    public let name = "dispatch_agent"
    public let isReadOnly = true
    public let description =
        "Spawn a background agent to work on a focused sub-task independently. It runs on its own "
        + "and does not report back here; the user can attach to it to watch its progress. Use for "
        + "parallel or self-contained work (e.g. 'explore the auth module', 'add tests for X')."
    public let parametersJSON = """
    {"type":"object","properties":{\
    "task":{"type":"string","description":"The full task for the background agent."},\
    "title":{"type":"string","description":"A short label for the agent (a few words)."}},\
    "required":["task"]}
    """

    private let queue: SpawnQueue
    public init(queue: SpawnQueue) { self.queue = queue }

    public func run(argumentsJSON: String) async -> ToolResult {
        guard let obj = jsonObject(argumentsJSON),
              let task = obj["task"] as? String,
              !task.trimmingCharacters(in: .whitespaces).isEmpty
        else {
            return ToolResult(content: "Error: dispatch_agent requires a 'task'.", isError: true)
        }
        let rawTitle = (obj["title"] as? String)?.trimmingCharacters(in: .whitespaces) ?? ""
        let title = rawTitle.isEmpty ? Self.deriveTitle(task) : rawTitle
        queue.push(SpawnRequest(title: title, task: task))
        return ToolResult(
            content: "Spawned background agent '\(title)'. It runs independently; "
                + "the user can attach to it to watch its progress.",
            isError: false)
    }

    /// A short label from the task's first few words when none was supplied.
    public static func deriveTitle(_ task: String) -> String {
        let words = task.split(whereSeparator: { $0 == " " || $0 == "\n" }).prefix(5)
        let s = words.joined(separator: " ")
        return s.count > 40 ? String(s.prefix(40)) + "\u{2026}" : s
    }
}
