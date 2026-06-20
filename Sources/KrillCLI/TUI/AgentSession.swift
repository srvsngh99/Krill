import Foundation
import KrillEngine
import KrillHarness

/// A background agent: its own transcript, message history, permission posture,
/// approver, and run Task. Created by `/bg` or the `dispatch_agent` tool, pumped
/// by the main loop each tick, and attachable from the agent switcher so the
/// user can watch it, answer its approval prompts, and continue it.
final class AgentSession {
    enum Status: Equatable { case running, waiting, idle, cancelled }

    let id: Int
    var title: String
    private(set) var status: Status = .idle
    private(set) var entries: [AgentEntry] = []
    let posture: PermissionMode
    let approver = TUIApprover()

    private let engine: InferenceEngine
    private let maxTokens: Int
    private let tools: ToolRegistry
    private var messages: [[String: String]] = []   // priorMessages seam, carried across turns

    private let queue = EventQueue()
    private var runTask: Task<AgentTranscript, Never>?
    private var chipShown = false
    private(set) var startedAt = CFAbsoluteTimeGetCurrent()

    init(id: Int, title: String, engine: InferenceEngine, maxTokens: Int,
         posture: PermissionMode, tools: ToolRegistry) {
        self.id = id
        self.title = title
        self.engine = engine
        self.maxTokens = maxTokens
        self.posture = posture
        self.tools = tools
    }

    var isRunning: Bool { status == .running || status == .waiting }
    var elapsed: Double { CFAbsoluteTimeGetCurrent() - startedAt }

    /// Start (or continue) the session on `task`. Continuation reuses the prior
    /// transcript via the loop's `priorMessages` seam.
    func start(task: String) {
        entries.append(.user(task))
        status = .running
        chipShown = false
        startedAt = CFAbsoluteTimeGetCurrent()

        let steered = posture == .plan
            ? "(Plan mode: read-only. Investigate with the read-only tools and propose a clear, "
              + "step-by-step plan. Do not edit files or run commands.)\n\n\(task)"
            : task

        let loop = AgentLoop(
            generator: EngineGenerator(engine: engine, maxTokens: maxTokens),
            tools: tools,
            permission: PermissionPolicy(mode: posture),
            gate: approver)
        // Capture only locals (loop/queue/strings) so the run closure stays
        // Sendable - no `self` capture. The transcript is stashed in the queue.
        let q = queue, prior = messages
        runTask = Task {
            let t = await loop.run(user: steered, priorMessages: prior, onEvent: { q.push($0) })
            q.finish(t)
            return t
        }
    }

    /// Drain pending events into the transcript and update status. Returns true
    /// if anything changed (so the caller re-renders). Synchronous - safe to call
    /// every tick from the main loop.
    @discardableResult
    func pump() -> Bool {
        var changed = false
        for ev in queue.drain() {
            foldAgentEvent(ev, into: &entries, chipShown: &chipShown)
            changed = true
        }
        if isRunning {
            let next: Status = approver.pending() != nil ? .waiting : .running
            if next != status { status = next; changed = true }
        }
        if queue.isFinished, runTask != nil, let t = queue.finishedResult {
            messages = t.messages
            status = t.wasCancelled ? .cancelled : .idle
            runTask = nil
            changed = true
        }
        return changed
    }

    /// Cancel the run (Ctrl-C / quit). Also resolves any pending approval so the
    /// suspended loop never hangs.
    func cancel() {
        approver.resolve(false)
        runTask?.cancel()
    }

    /// One-line status label for the switcher (e.g. "running 8s", "waiting").
    func statusLabel() -> String {
        switch status {
        case .running: return "running \(ChatTUI.formatElapsed(elapsed))"
        case .waiting: return "needs approval"
        case .idle: return "done"
        case .cancelled: return "cancelled"
        }
    }
}
