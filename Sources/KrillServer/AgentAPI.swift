import Foundation
import KrillEngine
import KrillHarness
import KrillTooling
import NIOCore
import NIOHTTP1

/// The agent-session HTTP surface: hosted `krill code` conversations a remote
/// client (the `/ui` phone app, or any HTTP client) can create, drive, tail
/// over SSE, and answer approval prompts for.
///
///   POST   /v1/agent/sessions                 create {workspace?, model?, permission_mode?, ...}
///   GET    /v1/agent/sessions                 list
///   GET    /v1/agent/sessions/{id}            meta + full event replay
///   DELETE /v1/agent/sessions/{id}            cancel + remove
///   POST   /v1/agent/sessions/{id}/messages   {text} -> starts a turn (202)
///   GET    /v1/agent/sessions/{id}/events     SSE tail (replay via ?since= / Last-Event-ID)
///   POST   /v1/agent/sessions/{id}/approvals  {id?, allow, always?}
///   POST   /v1/agent/sessions/{id}/questions  {id?, text?, option_index?, was_free_text?, declined?}
///   POST   /v1/agent/sessions/{id}/cancel     stop the active turn
///   GET    /v1/agent/workspaces               shallow directory browser for the workspace picker
extension HTTPHandler {

    func handleAgentRoute(
        context: ChannelHandlerContext, head: HTTPRequestHead, path: String, body: ByteBuffer
    ) {
        let rest = path.dropFirst("/v1/agent/".count)
        let parts = rest.split(separator: "/").map(String.init)

        switch (head.method, parts.count) {
        case (.GET, 1) where parts[0] == "sessions":
            sendJSON(context: context, status: .ok, body: [
                "sessions": agentSessions.list().map { $0.summary() }
            ])
        case (.POST, 1) where parts[0] == "sessions":
            handleAgentCreate(context: context, body: body)
        case (.GET, 1) where parts[0] == "workspaces":
            handleAgentWorkspaces(context: context, uri: head.uri)
        case (.GET, 2) where parts[0] == "sessions":
            guard let session = agentSessions.get(parts[1]) else {
                return sendAgentNotFound(context, parts[1])
            }
            var detail = session.summary()
            detail["events"] = session.eventObjects()
            sendJSON(context: context, status: .ok, body: detail)
        case (.DELETE, 2) where parts[0] == "sessions":
            guard agentSessions.remove(parts[1]) != nil else {
                return sendAgentNotFound(context, parts[1])
            }
            sendJSON(context: context, status: .ok, body: ["ok": true])
        case (.POST, 3) where parts[0] == "sessions" && parts[2] == "messages":
            guard let session = agentSessions.get(parts[1]) else {
                return sendAgentNotFound(context, parts[1])
            }
            handleAgentMessage(context: context, session: session, body: body)
        case (.GET, 3) where parts[0] == "sessions" && parts[2] == "events":
            guard let session = agentSessions.get(parts[1]) else {
                return sendAgentNotFound(context, parts[1])
            }
            handleAgentEvents(context: context, head: head, session: session)
        case (.POST, 3) where parts[0] == "sessions" && parts[2] == "approvals":
            guard let session = agentSessions.get(parts[1]) else {
                return sendAgentNotFound(context, parts[1])
            }
            let json = parseJSON(body) ?? [:]
            let allow = (json["allow"] as? Bool) ?? false
            let always = (json["always"] as? Bool) ?? false
            let resolved = session.approver.resolve(
                id: json["id"] as? String, allow: allow, always: always)
            if resolved {
                sendJSON(context: context, status: .ok, body: ["ok": true])
            } else {
                sendJSON(context: context, status: .conflict,
                         body: ["error": "no matching pending approval"])
            }
        case (.POST, 3) where parts[0] == "sessions" && parts[2] == "questions":
            guard let session = agentSessions.get(parts[1]) else {
                return sendAgentNotFound(context, parts[1])
            }
            let json = parseJSON(body) ?? [:]
            let optionIndex = json["option_index"] as? Int
            let pending = session.asker.pending()
            let suppliedText = (json["text"] as? String) ?? ""
            let text: String
            if suppliedText.isEmpty, let optionIndex,
               let options = pending?.question.options,
               options.indices.contains(optionIndex) {
                text = options[optionIndex]
            } else {
                text = suppliedText
            }
            let answer = UserAnswer(
                text: text,
                optionIndex: optionIndex,
                wasFreeText: (json["was_free_text"] as? Bool) ?? (optionIndex == nil && !text.isEmpty),
                declined: (json["declined"] as? Bool) ?? false)
            let resolved = session.asker.resolve(id: json["id"] as? String, answer: answer)
            if resolved {
                sendJSON(context: context, status: .ok, body: ["ok": true])
            } else {
                sendJSON(context: context, status: .conflict,
                         body: ["error": "no matching pending question"])
            }
        case (.POST, 3) where parts[0] == "sessions" && parts[2] == "cancel":
            guard let session = agentSessions.get(parts[1]) else {
                return sendAgentNotFound(context, parts[1])
            }
            session.cancel()
            sendJSON(context: context, status: .ok, body: ["ok": true])
        default:
            sendJSON(context: context, status: .notFound,
                     body: ["error": "Not found: \(head.method) \(path)"])
        }
    }

    private func sendAgentNotFound(_ context: ChannelHandlerContext, _ id: String) {
        sendJSON(context: context, status: .notFound,
                 body: ["error": "no agent session '\(id)'"])
    }

    // MARK: - Create

    private func handleAgentCreate(context: ChannelHandlerContext, body: ByteBuffer) {
        let json = parseJSON(body) ?? [:]

        let rawWorkspace = (json["workspace"] as? String)
            .flatMap { $0.isEmpty ? nil : $0 } ?? FileManager.default.currentDirectoryPath
        let workspace = URL(fileURLWithPath: (rawWorkspace as NSString).expandingTildeInPath)
            .standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: workspace.path, isDirectory: &isDir),
              isDir.boolValue else {
            return sendJSON(context: context, status: .badRequest,
                            body: ["error": "workspace is not a directory: \(workspace.path)"])
        }

        var modelName: String? = (json["model"] as? String).flatMap { $0.isEmpty ? nil : $0 }
        if let name = modelName {
            // Resolve against the registry; the active engine's name is also
            // accepted (it may have been loaded from a path, not the registry).
            if !registry.hasModel(name) {
                if activeRef.current?.modelName == name {
                    modelName = nil  // pin to the active engine
                } else {
                    return sendJSON(context: context, status: .badRequest, body: [
                        "error": "model '\(name)' is not installed; pick one from GET /v1/models"
                    ])
                }
            }
        }

        // Default to ask: the safe interactive posture for a remote client
        // (mutating tools park on an approval the phone answers).
        let mode = (json["permission_mode"] as? String).flatMap(PermissionMode.parse) ?? .ask
        let maxTokens = max(128, min((json["max_tokens"] as? Int) ?? 1024, 8192))
        let maxIterations = max(1, min((json["max_iterations"] as? Int) ?? 24, 64))
        let title = (json["title"] as? String) ?? ""

        let session = agentSessions.create(
            title: title, workspace: workspace, modelName: modelName,
            mode: mode, maxTokens: maxTokens, maxIterations: maxIterations)
        sendJSON(context: context, status: .ok, body: session.summary())
    }

    // MARK: - Send a message (start a turn)

    private func handleAgentMessage(
        context: ChannelHandlerContext, session: RemoteAgentSession, body: ByteBuffer
    ) {
        guard let json = parseJSON(body),
              let raw = json["text"] as? String,
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return sendJSON(context: context, status: .badRequest,
                            body: ["error": "messages requires a non-empty 'text'"])
        }
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)

        guard let prior = session.beginRun(userText: text) else {
            return sendJSON(context: context, status: .conflict,
                            body: ["error": "session is already running; cancel it first"])
        }

        let engines = self.engines
        let genQueue = self.genQueue
        let registry = self.registry
        let activeRef = self.activeRef
        let task = Task {
            // Resolve the session's engine: registry model (loaded on demand,
            // like every generate route) or the active resident engine.
            var resolved: InferenceEngine?
            var loadError: String?
            if let name = session.modelName, registry.hasModel(name) {
                do { resolved = try await engines.activate(name: name) }
                catch { loadError = String(describing: error).prefix(200).description }
            } else {
                resolved = activeRef.current
            }
            guard let eng = resolved, eng.isLoaded else {
                session.failRun(
                    "No model available for this session"
                    + (loadError.map { ": \($0)" }
                        ?? "; load one (krill serve <model>) or pick an installed model."))
                return
            }
            await engines.retain(eng)

            let generator = ServerHarnessGenerator(
                engine: eng, genQueue: genQueue, maxTokens: session.maxTokens)
            let loop = AgentLoop(
                generator: generator,
                tools: Self.agentToolRegistry(for: session),
                maxIterations: session.maxIterations,
                permission: session.permissionBox.policy,
                permissionBox: session.permissionBox,
                gate: session.approver)

            // First turn: synthesize the same system prompt `krill code` uses,
            // with the ambient cwd + Krill.md resolved inside the session's
            // workspace via the task-local.
            let transcript = await AgentWorkspace.$root.withValue(session.workspace) {
                let system: String? = prior.isEmpty
                    ? Self.agentSystemPrompt(mode: session.mode, modelName: eng.modelName)
                    : nil
                return await loop.run(
                    user: text, system: system, priorMessages: prior,
                    onEvent: { session.push($0) })
            }
            await engines.release(eng)
            session.finishRun(transcript)
        }
        session.setRunTask(task)
        sendJSON(context: context, status: .accepted,
                 body: ["ok": true, "status": session.status.rawValue, "last_seq": session.lastSeq])
    }

    /// The hosted toolset: the TUI's agent toolset minus `dispatch_agent`
    /// (background fan-out stays a TUI feature for now). The permission
    /// posture — not this list — governs what actually runs.
    static func agentToolRegistry(for session: RemoteAgentSession) -> ToolRegistry {
        ToolRegistry([
            ReadTool(), ListTool(), GlobTool(), GrepTool(), WebFetchTool(), WebSearchTool(),
            EditTool(), MultiEditTool(), WriteTool(), BashTool(),
            NowTool(), session.todoTool, RepoMapTool(),
            AskUserTool(gate: session.asker),
            RequestExecuteTool(permissionBox: session.permissionBox, gate: session.asker),
        ])
    }

    /// Mirror of `CodeCommand`'s system synthesis: ambient facts, project
    /// brief, plan steering, and the anti-over-calling directive. Must be
    /// called with `AgentWorkspace.root` bound to the session workspace.
    static func agentSystemPrompt(mode: PermissionMode, modelName: String?) -> String {
        var parts = [AgentEnvironment.contextLine(modelName: modelName)]
        if let brief = AgentEnvironment.projectBrief() { parts.append(brief) }
        parts.append(contentsOf: AgentEnvironment.permissionDirectives(for: mode))
        parts.append(AgentEnvironment.toolDirective)
        return parts.joined(separator: "\n\n")
    }

    // MARK: - SSE event tail

    private func handleAgentEvents(
        context: ChannelHandlerContext, head: HTTPRequestHead, session: RemoteAgentSession
    ) {
        // Replay position: ?since=N, or the standard Last-Event-ID header a
        // reconnecting SSE client sends. 0 replays everything retained.
        let since = Self.queryParams(head.uri)["since"].flatMap(Int.init)
            ?? head.headers.first(name: "Last-Event-ID").flatMap(Int.init)
            ?? 0

        // SSE head synchronously within the channelRead chain (NIO requires
        // the response begin here; same discipline as streaming completions).
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/event-stream")
        headers.add(name: "Cache-Control", value: "no-cache")
        headers.add(name: "Connection", value: "keep-alive")
        for (k, v) in corsHeaders() { headers.add(name: k, value: v) }
        let responseHead = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        context.write(wrapOutboundOut(.head(responseHead)), promise: nil)
        context.flush()

        // One live tail per connection: a repeat request on the same channel
        // replaces (and tears down) the previous subscription.
        if let old = agentEventsSub {
            old.session.unsubscribe(old.token)
            old.ping.cancel()
            agentEventsSub = nil
        }

        nonisolated(unsafe) let ctx = context
        let token = session.subscribe(from: since) { seq, frame in
            self.writeRaw(ctx, "id: \(seq)\ndata: \(frame)\n\n")
        }
        // Comment-frame heartbeat so NATs/proxies (and the client's stall
        // detector) see a live stream even while the agent is quiet. Keep it
        // on the channel's event loop: ChannelHandlerContext is intentionally
        // not Sendable and must never cross a Swift-concurrency Task boundary.
        let ping = context.eventLoop.scheduleRepeatedTask(
            initialDelay: .seconds(25), delay: .seconds(25)
        ) { [weak self] _ in
            self?.writeRaw(ctx, ": ping\n\n")
        }
        agentEventsSub = (session, token, ping)
    }

    // MARK: - Workspace browser

    /// Shallow directory listing for the phone's workspace picker:
    /// `GET /v1/agent/workspaces?path=~/code`. Marks git repos so the UI can
    /// surface them first. Authenticated like every data route.
    private func handleAgentWorkspaces(context: ChannelHandlerContext, uri: String) {
        let raw = Self.queryParams(uri)["path"] ?? "~"
        let base = URL(fileURLWithPath: (raw as NSString).expandingTildeInPath).standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: base.path, isDirectory: &isDir),
              isDir.boolValue else {
            return sendJSON(context: context, status: .badRequest,
                            body: ["error": "not a directory: \(base.path)"])
        }
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(
            at: base, includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])) ?? []
        let dirs = entries
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .sorted { $0.lastPathComponent.localizedCaseInsensitiveCompare($1.lastPathComponent) == .orderedAscending }
            .prefix(200)
            .map { url -> [String: Any] in
                [
                    "name": url.lastPathComponent,
                    "path": url.path,
                    "is_repo": fm.fileExists(atPath: url.appendingPathComponent(".git").path),
                ]
            }
        sendJSON(context: context, status: .ok, body: [
            "path": base.path,
            "parent": base.path == "/" ? NSNull() : base.deletingLastPathComponent().path,
            "dirs": Array(dirs),
        ])
    }

    // MARK: - Small helpers

    /// Minimal query-string parser (percent-decoded values).
    static func queryParams(_ uri: String) -> [String: String] {
        guard let q = uri.split(separator: "?", maxSplits: 1).dropFirst().first else { return [:] }
        var out: [String: String] = [:]
        for pair in q.split(separator: "&") {
            let kv = pair.split(separator: "=", maxSplits: 1).map(String.init)
            guard let key = kv.first, !key.isEmpty else { continue }
            let value = kv.count > 1 ? kv[1] : ""
            out[key.removingPercentEncoding ?? key] =
                value.replacingOccurrences(of: "+", with: " ").removingPercentEncoding ?? value
        }
        return out
    }
}
