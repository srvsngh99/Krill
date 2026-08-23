import XCTest
import KrillHarness
import KrillTooling
@testable import KrillServer

/// Unit coverage for the agent-session plumbing that backs `/v1/agent/*`:
/// `RemoteApprover` (the HTTP-answerable `PermissionGate`), `RemoteAgentSession`
/// (event folding + subscriber fan-out), and `AgentSessionStore`. No HTTP layer
/// here — see `AgentAPITests` for the EmbeddedChannel-level coverage.
final class AgentSessionTests: XCTestCase {

    // MARK: - RemoteApprover

    /// Poll until a request parks (the `Task` wrapping `approve()` needs a
    /// scheduling turn before `register` runs), bounded so a real regression
    /// fails fast instead of hanging.
    private func waitForPending(
        _ approver: RemoteApprover, timeout: TimeInterval = 5
    ) async -> RemoteApprover.Request? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let req = approver.pending() { return req }
            await Task.yield()
        }
        return approver.pending()
    }

    func testApproveParksAndPendingShowsTheRequest() async {
        let approver = RemoteApprover()
        let task = Task { await approver.approve(toolName: "bash", argumentsJSON: #"{"command":"ls"}"#) }

        guard let req = await waitForPending(approver) else {
            return XCTFail("approve() never parked a request")
        }
        XCTAssertEqual(req.toolName, "bash")
        XCTAssertEqual(req.argumentsJSON, #"{"command":"ls"}"#)

        XCTAssertTrue(approver.resolve(id: nil, allow: true))
        let allowed = await task.value
        XCTAssertTrue(allowed)
        XCTAssertNil(approver.pending())
    }

    func testResolveWithWrongIdDoesNotResolve() async {
        let approver = RemoteApprover()
        let task = Task { await approver.approve(toolName: "bash", argumentsJSON: "{}") }
        _ = await waitForPending(approver)

        XCTAssertFalse(approver.resolve(id: "not-the-real-id", allow: true))
        XCTAssertNotNil(approver.pending(), "a mismatched id must leave the request pending")

        // Clean up so the parked task doesn't leak past the test.
        XCTAssertTrue(approver.resolve(id: nil, allow: false))
        _ = await task.value
    }

    func testStickyAlwaysAllowAutoApprovesNextSameNameCall() async {
        let approver = RemoteApprover()
        let task1 = Task { await approver.approve(toolName: "bash", argumentsJSON: "{}") }
        _ = await waitForPending(approver)
        XCTAssertTrue(approver.resolve(id: nil, allow: true, always: true))
        let firstAllowed = await task1.value
        XCTAssertTrue(firstAllowed)

        // Same tool name again: sticky always-allow must resolve immediately,
        // with no request ever parking.
        let allowed = await approver.approve(toolName: "bash", argumentsJSON: #"{"command":"ls -la"}"#)
        XCTAssertTrue(allowed)
        XCTAssertNil(approver.pending(), "sticky auto-allow must not park a request")
    }

    func testResolveWithNothingPendingReturnsFalse() {
        let approver = RemoteApprover()
        XCTAssertFalse(approver.resolve(id: nil, allow: true))
    }

    func testRemoteApproverTimesOutToDeny() async {
        let approver = RemoteApprover(timeout: 0.02)
        let allowed = await approver.approve(toolName: "bash", argumentsJSON: "{}")
        XCTAssertFalse(allowed)
        XCTAssertNil(approver.pending())
    }

    // MARK: - RemoteQuestionAsker

    private func waitForPending(
        _ asker: RemoteQuestionAsker, timeout: TimeInterval = 5
    ) async -> RemoteQuestionAsker.Request? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let req = asker.pending() { return req }
            await Task.yield()
        }
        return asker.pending()
    }

    func testQuestionParksAndMatchingAnswerResolves() async {
        let asker = RemoteQuestionAsker()
        let question = UserQuestion(
            header: "Database", question: "Which store?", body: "Choose one.",
            options: ["SQLite", "Postgres"])
        let task = Task { await asker.ask(question) }

        guard let req = await waitForPending(asker) else {
            return XCTFail("ask() never parked a question")
        }
        XCTAssertEqual(req.question, question)
        let answer = UserAnswer(text: "Postgres", optionIndex: 1)
        XCTAssertTrue(asker.resolve(id: req.id, answer: answer))
        let resolved = await task.value
        XCTAssertEqual(resolved, answer)
        XCTAssertNil(asker.pending())
    }

    func testQuestionStaleIdLeavesPending() async {
        let asker = RemoteQuestionAsker()
        let task = Task { await asker.ask(UserQuestion(header: "H", question: "Q")) }
        guard let req = await waitForPending(asker) else {
            return XCTFail("ask() never parked a question")
        }

        XCTAssertFalse(asker.resolve(id: "stale-id", answer: UserAnswer(text: "wrong")))
        XCTAssertEqual(asker.pending()?.id, req.id)
        asker.cancelPending()
        let declined = await task.value
        XCTAssertTrue(declined.declined)
    }

    func testQuestionCancelPendingResumesDeclined() async {
        let asker = RemoteQuestionAsker()
        let task = Task { await asker.ask(UserQuestion(header: "H", question: "Q")) }
        _ = await waitForPending(asker)
        asker.cancelPending()

        let answer = await task.value
        XCTAssertTrue(answer.declined)
        XCTAssertNil(asker.pending())
    }

    // MARK: - RemoteAgentSession: event folding

    private func makeSession(mode: PermissionMode = .acceptAll) -> RemoteAgentSession {
        RemoteAgentSession(
            id: "test-session", title: "", workspace: FileManager.default.temporaryDirectory,
            modelName: nil, mode: mode, maxTokens: 512, maxIterations: 8)
    }

    func testSummaryReportsOriginEffectivePosturePhaseAndPendingQuestion() async {
        let session = makeSession(mode: .adaptive)
        var summary = session.summary()
        XCTAssertEqual(summary["permission_mode"] as? String, "adaptive")
        XCTAssertEqual(summary["effective_permission_mode"] as? String, "plan")
        XCTAssertEqual(summary["permission_phase"] as? String, "planning")

        XCTAssertTrue(session.permissionBox.promote(to: .acceptEdits))
        summary = session.summary()
        XCTAssertEqual(summary["permission_mode"] as? String, "adaptive")
        XCTAssertEqual(summary["effective_permission_mode"] as? String, "accept-edits")
        XCTAssertEqual(summary["permission_phase"] as? String, "executing")

        let task = Task {
            await session.asker.ask(UserQuestion(
                header: "Choice", question: "Pick one", body: "Context", options: ["A", "B"]))
        }
        guard let req = await waitForPending(session.asker) else {
            return XCTFail("question never became pending")
        }
        summary = session.summary()
        let pending = summary["pending_question"] as? [String: Any]
        XCTAssertEqual(pending?["id"] as? String, req.id)
        XCTAssertEqual(pending?["options"] as? [String], ["A", "B"])
        session.asker.cancelPending()
        _ = await task.value
        let questionEvents = session.eventObjects().filter {
            ($0["type"] as? String)?.hasPrefix("question_") == true
        }
        XCTAssertEqual(questionEvents.count, 2)
        XCTAssertEqual(questionEvents[0]["type"] as? String, "question_request")
        XCTAssertEqual(questionEvents[1]["type"] as? String, "question_answered")
        XCTAssertEqual(questionEvents[1]["declined"] as? Bool, true)
    }

    func testHostedRegistryIncludesInteractiveTools() {
        let registry = HTTPHandler.agentToolRegistry(for: makeSession(mode: .plan))
        XCTAssertTrue(registry.names.contains("ask_user"))
        XCTAssertTrue(registry.names.contains("request_execute"))
    }

    func testRemoteSystemPromptIncludesQuestionAndAdaptivePlanningDirectives() {
        let askPrompt = HTTPHandler.agentSystemPrompt(mode: .ask, modelName: nil)
        XCTAssertTrue(askPrompt.contains(AgentEnvironment.askUserDirective))
        XCTAssertFalse(askPrompt.contains(AgentEnvironment.planSystemSteer))

        let adaptivePrompt = HTTPHandler.agentSystemPrompt(mode: .adaptive, modelName: nil)
        XCTAssertTrue(adaptivePrompt.contains(AgentEnvironment.askUserDirective))
        XCTAssertTrue(adaptivePrompt.contains(AgentEnvironment.planSystemSteer))
        XCTAssertTrue(adaptivePrompt.contains(AgentEnvironment.adaptivePlanTail))
    }

    func testBeginRunEmitsUserThenStatusEvents() {
        let session = makeSession()
        let prior = session.beginRun(userText: "hello")
        XCTAssertEqual(prior, [])

        let events = session.eventObjects()
        XCTAssertEqual(events.count, 2)
        XCTAssertEqual(events[0]["type"] as? String, "user")
        XCTAssertEqual(events[0]["text"] as? String, "hello")
        XCTAssertEqual(events[0]["seq"] as? Int, 1)
        XCTAssertEqual(events[1]["type"] as? String, "status")
        XCTAssertEqual(events[1]["status"] as? String, "running")
        XCTAssertEqual(events[1]["seq"] as? Int, 2)
        XCTAssertEqual(session.status, .running)
    }

    func testPushFoldsAgentEventsIntoDocumentedWireTypes() {
        let session = makeSession()
        _ = session.beginRun(userText: "go")  // seq 1, 2

        session.push(.assistantTurn(text: "Let me check."))  // seq 3
        session.push(.toolStarted(name: "bash", argumentsJSON: #"{"command":"ls"}"#))  // seq 4
        let inv = ToolInvocation(
            name: "bash", argumentsJSON: #"{"command":"ls"}"#,
            result: ToolResult(content: "a.txt", isError: false))
        session.push(.toolFinished(inv))  // seq 5
        session.push(.finalAnswer("Done."))  // seq 6

        let events = session.eventObjects()
        XCTAssertEqual(events.count, 6)

        XCTAssertEqual(events[2]["type"] as? String, "assistant")
        XCTAssertEqual(events[2]["text"] as? String, "Let me check.")
        XCTAssertEqual(events[2]["seq"] as? Int, 3)

        XCTAssertEqual(events[3]["type"] as? String, "tool_call")
        XCTAssertEqual(events[3]["name"] as? String, "bash")
        XCTAssertEqual(events[3]["args"] as? String, #"{"command":"ls"}"#)
        XCTAssertEqual(events[3]["seq"] as? Int, 4)

        XCTAssertEqual(events[4]["type"] as? String, "tool_result")
        XCTAssertEqual(events[4]["name"] as? String, "bash")
        XCTAssertEqual(events[4]["content"] as? String, "a.txt")
        XCTAssertEqual(events[4]["is_error"] as? Bool, false)
        XCTAssertEqual(events[4]["seq"] as? Int, 5)

        XCTAssertEqual(events[5]["type"] as? String, "assistant_final")
        XCTAssertEqual(events[5]["text"] as? String, "Done.")
        XCTAssertEqual(events[5]["seq"] as? Int, 6)
    }

    func testPermissionChangedEmitsDedicatedSSEFrame() {
        let session = makeSession(mode: .adaptive)
        XCTAssertTrue(session.permissionBox.promote(to: .acceptEdits))
        session.push(.permissionChanged(from: .plan, to: .acceptEdits))

        let event = session.eventObjects().last
        XCTAssertEqual(event?["type"] as? String, "permission_changed")
        XCTAssertEqual(event?["from"] as? String, "plan")
        XCTAssertEqual(event?["to"] as? String, "accept-edits")
        XCTAssertEqual(event?["origin"] as? String, "adaptive")
        XCTAssertEqual(event?["phase"] as? String, "executing")
    }

    func testToolFinishedWithoutPriorToolStartedSynthesizesToolCall() {
        // A denied/unknown tool never emits `toolStarted`; the fold must still
        // produce a `tool_call` chip before the `tool_result` so the client
        // renders a consistent chip+result pair.
        let session = makeSession()
        _ = session.beginRun(userText: "go")  // seq 1, 2
        let inv = ToolInvocation(
            name: "bash", argumentsJSON: "{}",
            result: ToolResult(content: "Permission denied: the user declined.", isError: true))
        session.push(.toolFinished(inv))  // synthesized tool_call (seq 3) + tool_result (seq 4)

        let events = session.eventObjects()
        XCTAssertEqual(events.count, 4)
        XCTAssertEqual(events[2]["type"] as? String, "tool_call")
        XCTAssertEqual(events[2]["name"] as? String, "bash")
        XCTAssertEqual(events[2]["seq"] as? Int, 3)
        XCTAssertEqual(events[3]["type"] as? String, "tool_result")
        XCTAssertEqual(events[3]["is_error"] as? Bool, true)
        XCTAssertEqual(events[3]["seq"] as? Int, 4)
    }

    // MARK: - Subscribe / replay / live tail

    /// Lock-guarded seq collector: `subscribe`'s sink type is `@Sendable`, so a
    /// plain captured `var` cannot be mutated from it under strict concurrency.
    private final class SeqCollector: @unchecked Sendable {
        private let lock = NSLock()
        private var seqs: [Int] = []
        func append(_ seq: Int) { lock.lock(); seqs.append(seq); lock.unlock() }
        var all: [Int] { lock.lock(); defer { lock.unlock() }; return seqs }
    }

    func testSubscribeReplaysOnlyEventsAfterSinceThenTailsLiveEvents() {
        let session = makeSession()
        _ = session.beginRun(userText: "go")  // seq 1, 2
        session.push(.assistantTurn(text: "hi"))  // seq 3

        let received = SeqCollector()
        let token = session.subscribe(from: 2) { seq, _ in received.append(seq) }
        XCTAssertEqual(received.all, [3], "replay must skip everything at or before `since`")

        session.push(.finalAnswer("done"))  // seq 4, delivered live
        XCTAssertEqual(received.all, [3, 4])

        session.unsubscribe(token)
        session.push(.iterationLimitReached)  // seq 5, must NOT reach the unsubscribed sink
        XCTAssertEqual(received.all, [3, 4], "unsubscribe must stop further delivery")
    }

    // MARK: - Run lifecycle

    func testBeginRunWhileRunningReturnsNil() {
        let session = makeSession()
        XCTAssertNotNil(session.beginRun(userText: "first"))
        XCTAssertNil(session.beginRun(userText: "second"), "a run already in flight must refuse a second beginRun")
    }

    func testFinishRunStoresMessagesSoNextBeginRunReturnsThemAsPrior() {
        let session = makeSession()
        _ = session.beginRun(userText: "first")
        let transcript = AgentTranscript(
            steps: [], finalText: "answer", hitIterationLimit: false, wasCancelled: false,
            messages: [
                ["role": "user", "content": "first"],
                ["role": "assistant", "content": "answer"],
            ])
        session.finishRun(transcript)
        XCTAssertEqual(session.status, .idle)

        let prior = session.beginRun(userText: "second")
        XCTAssertEqual(prior, transcript.messages)
    }

    // MARK: - AgentSessionStore

    func testStoreCreateListGetRemove() {
        let store = AgentSessionStore()
        let a = store.create(
            title: "A", workspace: FileManager.default.temporaryDirectory,
            modelName: nil, mode: .acceptAll, maxTokens: 512, maxIterations: 8)
        let b = store.create(
            title: "B", workspace: FileManager.default.temporaryDirectory,
            modelName: nil, mode: .acceptAll, maxTokens: 512, maxIterations: 8)

        XCTAssertEqual(store.get(a.id)?.id, a.id)
        XCTAssertEqual(store.list().map(\.id), [b.id, a.id], "list must be newest-first")

        let removed = store.remove(a.id)
        XCTAssertEqual(removed?.id, a.id)
        XCTAssertNil(store.get(a.id))
        XCTAssertEqual(store.list().map(\.id), [b.id])
    }

    // MARK: - End-to-end: AgentLoop -> RemoteAgentSession

    private actor SessionMockGenerator: HarnessGenerator {
        nonisolated let toolFormat: ToolCalling.ToolFormat = .hermes
        private let responses: [String]
        private var idx = 0
        init(responses: [String]) { self.responses = responses }
        func complete(messages: [[String: String]]) async -> String {
            defer { idx += 1 }
            return idx < responses.count ? responses[idx] : "done"
        }
    }

    private struct EchoBashTool: Tool {
        let name = "bash"
        let description = "test stub"
        let parametersJSON = #"{"type":"object","properties":{"command":{"type":"string"}}}"#
        func run(argumentsJSON: String) async -> ToolResult { ToolResult(content: "ok") }
    }

    func testEndToEndLoopDrivenBySessionPushProducesExpectedTranscriptShape() async {
        let gen = SessionMockGenerator(responses: [
            "Checking.\n<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\":\"ls\"}}</tool_call>",
            "All done.",
        ])
        let loop = AgentLoop(generator: gen, tools: ToolRegistry([EchoBashTool()]))
        let session = makeSession()
        _ = session.beginRun(userText: "go")  // seq 1, 2

        let transcript = await loop.run(user: "go", onEvent: { session.push($0) })
        session.finishRun(transcript)

        let types = session.eventObjects().compactMap { $0["type"] as? String }
        XCTAssertEqual(
            types,
            ["user", "status", "assistant", "tool_call", "tool_result", "assistant_final", "status"])
        XCTAssertEqual(session.eventObjects().last?["status"] as? String, "idle")
        XCTAssertEqual(session.status, .idle)
    }
}
