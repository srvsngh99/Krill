import Foundation
import KrillEngine
import KrillHarness

/// A background agent: its own transcript, message history, permissions,
/// approver, and run Task. Created by `/bg` or the `dispatch_agent` tool, pumped
/// by the main loop each tick, and attachable from the agent switcher so the
/// user can watch it, answer its approval prompts, and continue it.
final class AgentSession {
    enum Status: Equatable { case running, waiting, idle, cancelled }

    let id: Int
    var title: String
    private(set) var status: Status = .idle
    private(set) var entries: [AgentEntry] = []
    let permissionBox: PermissionBox
    private(set) var permissions: PermissionMode
    let approver = TUIApprover()
    let asker = TUIQuestionAsker()
    let todoTool = TodoTool()

    private let engine: InferenceEngine
    private let maxTokens: Int
    private var tools: ToolRegistry?
    private var messages: [[String: String]] = []   // priorMessages seam, carried across turns

    private let queue = EventQueue()
    private var runTask: Task<AgentTranscript, Never>?
    private var chipShown = false
    private(set) var startedAt = CFAbsoluteTimeGetCurrent()
    private(set) var lastStats: GenerationStats?

    private final class StatsBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value: GenerationStats?
        func put(_ stats: GenerationStats) { lock.lock(); value = stats; lock.unlock() }
        func take() -> GenerationStats? {
            lock.lock(); defer { lock.unlock() }
            let result = value; value = nil; return result
        }
    }
    private let statsBox = StatsBox()

    init(id: Int, title: String, engine: InferenceEngine, maxTokens: Int,
         permissions: PermissionMode, effectivePolicy: PermissionPolicy? = nil) {
        self.id = id
        self.title = title
        self.engine = engine
        self.maxTokens = maxTokens
        self.permissions = permissions
        self.permissionBox = PermissionBox(
            origin: permissions,
            policy: effectivePolicy ?? PermissionPolicy(mode: permissions.initialEffective))
    }

    /// Session-owned tools must be assembled after the session so its asker,
    /// permission box, and todo instance cannot leak into another agent.
    func configureTools(_ tools: ToolRegistry) { self.tools = tools }

    func cyclePermissions() {
        permissions = permissions.next
        permissionBox.setPolicy(mode: permissions)
    }

    var isRunning: Bool { status == .running || status == .waiting }
    var elapsed: Double { CFAbsoluteTimeGetCurrent() - startedAt }

    /// Whether `start` can actually run a turn. The caller checks this before
    /// spending anything it would have to put back (banked `!` output).
    var canStart: Bool { tools != nil }

    /// Start (or continue) the session on `task`. Continuation reuses the prior
    /// transcript via the loop's `priorMessages` seam.
    ///
    /// `displayAs` is the on-screen bubble; the model always receives `task`.
    /// They differ when banked `!` shell output is riding along with the turn.
    func start(task: String, displayAs: String? = nil) {
        guard let tools else {
            entries.append(.note("Agent tools were not configured."))
            status = .cancelled
            return
        }
        entries.append(.user(displayAs ?? task))
        status = .running
        chipShown = false
        startedAt = CFAbsoluteTimeGetCurrent()

        var turnDirectives: [String] = messages.isEmpty
            ? [AgentEnvironment.askUserDirective] : []
        if permissionBox.isPlanning {
            turnDirectives.append(AgentEnvironment.planTurnPrefix)
            if permissions == .adaptive {
                turnDirectives.append(AgentEnvironment.adaptivePlanTail)
            }
        }
        let steered = (turnDirectives + [task]).joined(separator: "\n\n")

        var generator = EngineGenerator(engine: engine, maxTokens: maxTokens)
        generator.onStats = { [statsBox] in statsBox.put($0) }
        let loop = AgentLoop(
            generator: generator,
            tools: tools,
            permission: permissionBox.policy,
            permissionBox: permissionBox,
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
        if let stats = statsBox.take() { lastStats = stats; changed = true }
        for ev in queue.drain() {
            foldAgentEvent(ev, into: &entries, chipShown: &chipShown)
            changed = true
        }
        if isRunning {
            let next: Status = (asker.pending() != nil || approver.pending() != nil) ? .waiting : .running
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
        asker.cancelPending()
        approver.resolve(false)
        runTask?.cancel()
    }

    /// One-line status label for the switcher (e.g. "running 8s", "waiting").
    func statusLabel() -> String {
        switch status {
        case .running: return "running \(ChatTUI.formatElapsed(elapsed))"
        case .waiting: return asker.pending() != nil ? "needs answer" : "needs approval"
        case .idle: return "done"
        case .cancelled: return "cancelled"
        }
    }
}
