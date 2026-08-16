import Foundation
import KrillEngine
import KrillHarness
import KrillTooling

// MARK: - Server-side harness generator

/// `HarnessGenerator` for agent sessions hosted by `krill serve`: the same
/// binding as the CLI's `EngineGenerator`, but serialized through the server's
/// `GenerationQueue` so agent turns and API chat requests never decode
/// concurrently on the single GPU.
struct ServerHarnessGenerator: HarnessGenerator {
    let engine: InferenceEngine
    let genQueue: GenerationQueue
    let maxTokens: Int

    var toolFormat: ToolCalling.ToolFormat {
        ToolCalling.ToolFormat.forFamily(engine.family)
    }

    func complete(messages: [[String: String]]) async -> String {
        await generate(messages: messages, format: nil, maxTokens: maxTokens)
    }

    func complete(
        messages: [[String: String]], constrainingToolNames toolNames: [String]
    ) async -> String {
        let sentinels = ToolCallSentinels.sentinels(for: toolFormat)
        guard !sentinels.isEmpty, !toolNames.isEmpty else {
            return await complete(messages: messages)
        }
        return await generate(
            messages: messages,
            format: .toolNames(
                sentinels: sentinels,
                nameKey: ToolCallSentinels.nameKey(for: toolFormat),
                names: toolNames),
            maxTokens: maxTokens)
    }

    func completeConstrained(messages: [[String: String]], jsonSchema: String) async -> String {
        await generate(messages: messages, format: .jsonSchemaCompact(jsonSchema), maxTokens: 256)
    }

    private func generate(
        messages: [[String: String]], format: OutputFormat?, maxTokens: Int
    ) async -> String {
        // One queue slot per completion (not per run), so chat requests can
        // interleave between agent turns instead of stalling behind a long run.
        do {
            try await genQueue.enter()
        } catch {
            return "Error: server busy (generation queue full); try again shortly."
        }
        let (stream, _) = engine.generate(
            messages: messages, params: .greedy, maxTokens: maxTokens, format: format)
        var out = ""
        for await event in stream {
            if Task.isCancelled { break }
            if event.isEnd { break }
            out += event.text
        }
        await genQueue.leave()
        return out
    }
}

// MARK: - Remote approval gate

/// `PermissionGate` for remote clients: an `.ask` decision parks the loop on a
/// continuation (exactly the TUI approver's shape) and surfaces the pending
/// request through `onRequest`, so a client can render an approve/deny sheet
/// and answer over HTTP.
final class RemoteApprover: PermissionGate, @unchecked Sendable {
    struct Request: Equatable, Sendable {
        let id: String
        let toolName: String
        let argumentsJSON: String
    }

    private let lock = NSLock()
    private var request: Request?
    private var continuation: CheckedContinuation<Bool, Never>?
    /// Tools the client chose to "always allow" for the rest of the session.
    private var sticky: Set<String> = []
    /// Called (outside the lock) when a request parks / resolves, so the
    /// session can emit events to its subscribers.
    var onRequest: (@Sendable (Request) -> Void)?
    var onResolve: (@Sendable (Request, Bool) -> Void)?

    func approve(toolName: String, argumentsJSON: String) async -> Bool {
        await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            register(toolName: toolName, argumentsJSON: argumentsJSON, cont: cont)
        }
    }

    private func register(
        toolName: String, argumentsJSON: String, cont: CheckedContinuation<Bool, Never>
    ) {
        lock.lock()
        if sticky.contains(toolName) {
            lock.unlock()
            cont.resume(returning: true)
            return
        }
        let req = Request(id: UUID().uuidString, toolName: toolName, argumentsJSON: argumentsJSON)
        request = req
        continuation = cont
        let notify = onRequest
        lock.unlock()
        notify?(req)
    }

    /// The request awaiting a decision, if any.
    func pending() -> Request? {
        lock.lock(); defer { lock.unlock() }
        return request
    }

    /// Answer the pending request. When `id` is non-nil it must match the
    /// pending request (a stale answer from a reconnecting client is ignored).
    /// Returns false when nothing (or a different request) was pending.
    @discardableResult
    func resolve(id: String?, allow: Bool, always: Bool = false) -> Bool {
        lock.lock()
        guard let cont = continuation, let req = request else { lock.unlock(); return false }
        if let id, id != req.id { lock.unlock(); return false }
        if allow, always { sticky.insert(req.toolName) }
        continuation = nil
        request = nil
        let notify = onResolve
        lock.unlock()
        cont.resume(returning: allow)
        notify?(req, allow)
        return true
    }
}

// MARK: - Remote agent session

/// One hosted `krill code` conversation: its transcript (as a seq-numbered
/// event log SSE clients can replay and tail), the `priorMessages` seam that
/// carries context across turns, the approval gate, and the run task.
///
/// Lock-guarded like the TUI's session/event-queue pair: the loop's `onEvent`
/// pushes from the run task, HTTP handlers read and subscribe from event-loop
/// threads. Subscriber callbacks are invoked outside the lock and must be
/// thread-safe (the SSE sink writes via `writeOnLoop`, which is).
final class RemoteAgentSession: @unchecked Sendable {
    enum Status: String, Sendable { case idle, running, waiting, cancelled }

    let id: String
    let createdAt = Date()
    let workspace: URL
    /// Registry model name; nil pins the run to the server's active engine.
    let modelName: String?
    let mode: PermissionMode
    let maxTokens: Int
    let maxIterations: Int
    let approver = RemoteApprover()
    /// One todo list per session (the tool instance carries the state).
    let todoTool = TodoTool()

    private let lock = NSLock()
    private var _title: String
    private var _status: Status = .idle
    /// The loop's `priorMessages` seam, updated when a run finishes.
    private var _messages: [[String: String]] = []
    /// Seq-numbered, pre-serialized JSON event frames (no SSE framing).
    private var _events: [(seq: Int, frame: String)] = []
    private var nextSeq = 1
    private var subscribers: [UUID: @Sendable (Int, String) -> Void] = [:]
    private var runTask: Task<Void, Never>?
    /// Mirrors `foldAgentEvent`'s chip tracking: whether the in-flight tool
    /// call already emitted a `tool_call` event (a denied/unknown tool skips
    /// `toolStarted`, so its result must synthesize one).
    private var chipShown = false

    init(id: String, title: String, workspace: URL, modelName: String?,
         mode: PermissionMode, maxTokens: Int, maxIterations: Int) {
        self.id = id
        self._title = title
        self.workspace = workspace
        self.modelName = modelName
        self.mode = mode
        self.maxTokens = maxTokens
        self.maxIterations = maxIterations
        approver.onRequest = { [weak self] req in
            guard let self else { return }
            self.setStatus(.waiting)
            self.emit([
                "type": "approval_request", "id": req.id,
                "tool": req.toolName, "args": req.argumentsJSON,
            ])
        }
        approver.onResolve = { [weak self] req, allow in
            guard let self else { return }
            if self.status == .waiting { self.setStatus(.running) }
            self.emit(["type": "approval_resolved", "id": req.id, "allow": allow])
        }
    }

    var title: String { lock.lock(); defer { lock.unlock() }; return _title }
    var status: Status { lock.lock(); defer { lock.unlock() }; return _status }
    var isRunning: Bool { let s = status; return s == .running || s == .waiting }
    var lastSeq: Int { lock.lock(); defer { lock.unlock() }; return nextSeq - 1 }

    /// Session metadata for list/detail endpoints.
    func summary() -> [String: Any] {
        lock.lock(); defer { lock.unlock() }
        return [
            "id": id,
            "title": _title,
            "status": _status.rawValue,
            "model": modelName ?? "",
            "workspace": workspace.path,
            "permission_mode": mode.rawValue,
            "created_at": ISO8601DateFormatter().string(from: createdAt),
            "last_seq": nextSeq - 1,
            "pending_approval": approver.pending().map {
                ["id": $0.id, "tool": $0.toolName, "args": $0.argumentsJSON]
            } ?? NSNull(),
        ]
    }

    /// All event frames (parsed back to objects) for the detail endpoint, so a
    /// client can render the transcript in one GET before tailing the SSE.
    func eventObjects() -> [[String: Any]] {
        lock.lock(); defer { lock.unlock() }
        return _events.compactMap { seq, frame in
            guard let data = frame.data(using: .utf8),
                  var obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            else { return nil }
            obj["seq"] = seq
            return obj
        }
    }

    // MARK: Run lifecycle (called from the HTTP handler / run task)

    /// Begin a turn: append the user event, flip to running, and return the
    /// prior message history for the loop. Nil when a run is already active.
    func beginRun(userText: String) -> [[String: String]]? {
        lock.lock()
        guard _status != .running, _status != .waiting else { lock.unlock(); return nil }
        _status = .running
        chipShown = false
        if _title.isEmpty {
            _title = String(userText.prefix(64))
        }
        let prior = _messages
        lock.unlock()
        emit(["type": "user", "text": userText])
        emit(["type": "status", "status": Status.running.rawValue])
        return prior
    }

    func setRunTask(_ task: Task<Void, Never>) {
        lock.lock(); runTask = task; lock.unlock()
    }

    /// Fold a loop event into the session's event log (the wire mirror of
    /// `foldAgentEvent`).
    func push(_ event: AgentEvent) {
        switch event {
        case .assistantTurn(let text):
            let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { emit(["type": "assistant", "text": t]) }
        case .toolStarted(let name, let args):
            lock.lock(); chipShown = true; lock.unlock()
            emit(["type": "tool_call", "name": name, "args": args])
        case .toolFinished(let inv):
            lock.lock()
            let hadChip = chipShown
            chipShown = false
            lock.unlock()
            if !hadChip {
                emit(["type": "tool_call", "name": inv.name, "args": inv.argumentsJSON])
            }
            emit([
                "type": "tool_result", "name": inv.name,
                "content": inv.result.content, "is_error": inv.result.isError,
            ])
        case .finalAnswer(let text):
            emit(["type": "assistant_final", "text": text])
        case .iterationLimitReached:
            emit(["type": "note", "text": "Stopped at the iteration limit without a final answer."])
        case .cancelled:
            emit(["type": "note", "text": "Run cancelled."])
        }
    }

    /// A run finished: persist the message seam and emit the terminal status.
    func finishRun(_ transcript: AgentTranscript) {
        lock.lock()
        _messages = transcript.messages
        _status = transcript.wasCancelled ? .cancelled : .idle
        runTask = nil
        lock.unlock()
        emit(["type": "status", "status": status.rawValue])
    }

    /// A run failed before the loop could start (no engine).
    func failRun(_ message: String) {
        emit(["type": "note", "text": message])
        lock.lock(); _status = .idle; runTask = nil; lock.unlock()
        emit(["type": "status", "status": Status.idle.rawValue])
    }

    /// Cancel the active run; resolves any parked approval first so the
    /// suspended loop never hangs.
    func cancel() {
        approver.resolve(id: nil, allow: false)
        lock.lock(); let task = runTask; lock.unlock()
        task?.cancel()
    }

    // MARK: Event log + subscribers

    private func setStatus(_ s: Status) {
        lock.lock()
        guard _status != s else { lock.unlock(); return }
        _status = s
        lock.unlock()
        emit(["type": "status", "status": s.rawValue])
    }

    /// Append an event frame and fan it out to live subscribers.
    func emit(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj),
              let frame = String(data: data, encoding: .utf8) else { return }
        lock.lock()
        let seq = nextSeq
        nextSeq += 1
        _events.append((seq, frame))
        // Cap the log so a marathon session cannot grow without bound; SSE
        // replay from before the cap starts at the oldest retained event.
        if _events.count > 5000 { _events.removeFirst(_events.count - 5000) }
        let sinks = Array(subscribers.values)
        lock.unlock()
        for sink in sinks { sink(seq, frame) }
    }

    /// Register a live sink, first replaying every retained event with
    /// `seq > since`. Replay happens under the lock so no event can slip
    /// between replay and registration.
    func subscribe(
        from since: Int, _ sink: @escaping @Sendable (Int, String) -> Void
    ) -> UUID {
        let token = UUID()
        lock.lock()
        for (seq, frame) in _events where seq > since { sink(seq, frame) }
        subscribers[token] = sink
        lock.unlock()
        return token
    }

    func unsubscribe(_ token: UUID) {
        lock.lock(); subscribers.removeValue(forKey: token); lock.unlock()
    }
}

// MARK: - Store

/// The server's shared session table (one per `KrillServer`, injected into
/// every connection handler).
public final class AgentSessionStore: @unchecked Sendable {
    private let lock = NSLock()
    private var sessions: [String: RemoteAgentSession] = [:]
    private var order: [String] = []

    public init() {}

    func create(title: String, workspace: URL, modelName: String?,
                mode: PermissionMode, maxTokens: Int, maxIterations: Int) -> RemoteAgentSession {
        let id = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12)).lowercased()
        let session = RemoteAgentSession(
            id: id, title: title, workspace: workspace, modelName: modelName,
            mode: mode, maxTokens: maxTokens, maxIterations: maxIterations)
        lock.lock()
        sessions[id] = session
        order.append(id)
        lock.unlock()
        return session
    }

    func get(_ id: String) -> RemoteAgentSession? {
        lock.lock(); defer { lock.unlock() }
        return sessions[id]
    }

    /// Sessions in creation order, newest first.
    func list() -> [RemoteAgentSession] {
        lock.lock(); defer { lock.unlock() }
        return order.reversed().compactMap { sessions[$0] }
    }

    @discardableResult
    func remove(_ id: String) -> RemoteAgentSession? {
        lock.lock()
        let session = sessions.removeValue(forKey: id)
        order.removeAll { $0 == id }
        lock.unlock()
        session?.cancel()
        return session
    }
}
