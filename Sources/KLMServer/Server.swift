import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1
import Logging
import KLMEngine
import KLMCore
import KLMRegistry
import KLMSampler

/// KrillLM HTTP server providing OpenAI-compatible and Ollama-compatible endpoints.
///
/// Endpoints:
///   POST /v1/chat/completions   - OpenAI chat API (streaming SSE)
///   GET  /v1/models             - OpenAI models list
///   POST /api/chat              - Ollama compatibility
///   POST /api/generate          - Ollama compatibility
///   GET  /api/tags              - Ollama model list
///   GET  /healthz               - Health check
public final class KLMServer: Sendable {
    private let host: String
    private let port: Int
    private let engine: InferenceEngine
    private let registry: Registry
    private let compat: CompatMode
    private let embedEngine: EmbeddingEngine
    private let logger = Logger(label: "krillm.server")

    private let corsOrigins: [String]
    private let keepAlive: KeepAliveController
    private let defaultContextLimit: Int?
    private let genQueue: GenerationQueue

    public init(host: String = "127.0.0.1", port: Int = 11435,
                compat: CompatMode = .both,
                engine: InferenceEngine, registry: Registry,
                embedEngine: EmbeddingEngine = EmbeddingEngine(),
                corsOrigins: [String] = ["http://localhost", "http://127.0.0.1", "https://localhost"],
                keepAliveDefaultSeconds: Int = 300,
                defaultContextLimit: Int? = nil,
                numParallel: Int = 1, maxQueue: Int = 512) {
        self.host = host
        self.port = port
        self.compat = compat
        self.corsOrigins = corsOrigins
        self.engine = engine
        self.registry = registry
        self.embedEngine = embedEngine
        self.keepAlive = KeepAliveController(defaultSeconds: keepAliveDefaultSeconds)
        self.defaultContextLimit = defaultContextLimit
        self.genQueue = GenerationQueue(numParallel: numParallel, maxQueue: maxQueue)
    }

    internal static func _makeHTTPHandlerForTesting(
        engine: InferenceEngine,
        registry: Registry,
        compat: CompatMode = .both,
        embedEngine: EmbeddingEngine = EmbeddingEngine(),
        maxBodySizeOverride: Int? = nil
    ) -> any ChannelHandler & Sendable {
        HTTPHandler(engine: engine, registry: registry, compat: compat,
                    embedEngine: embedEngine,
                    keepAlive: KeepAliveController(defaultSeconds: 300),
                    genQueue: GenerationQueue(numParallel: 1, maxQueue: 512),
                    maxBodySizeOverride: maxBodySizeOverride)
    }

    /// Start the HTTP server (blocks until shutdown).
    public func start() async throws {
        let group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)

        let bootstrap = ServerBootstrap(group: group)
            .serverChannelOption(.backlog, value: 256)
            .serverChannelOption(.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline().flatMap {
                    channel.pipeline.addHandler(
                        HTTPHandler(engine: self.engine, registry: self.registry,
                                    compat: self.compat, embedEngine: self.embedEngine,
                                    keepAlive: self.keepAlive,
                                    corsOrigins: self.corsOrigins,
                                    defaultContextLimit: self.defaultContextLimit,
                                    genQueue: self.genQueue)
                    )
                }
            }
            .childChannelOption(.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(.maxMessagesPerRead, value: 1)

        let channel = try await bootstrap.bind(host: host, port: port).get()
        logger.info("KrillLM server listening on http://\(host):\(port)")
        print("KrillLM server listening on http://\(host):\(port)  (compat: \(compat.rawValue))")
        if compat.openAIEnabled {
            print("OpenAI API: http://\(host):\(port)/v1/chat/completions")
        }
        if compat.ollamaEnabled {
            print("Ollama API: http://\(host):\(port)/api/chat")
        }
        if port != 11434 {
            print("Note: default port is \(port). For Ollama drop-in, run with --port 11434 (default flip to 11434 deferred until full parity - see docs/OLLAMA_MAC_PARITY_PLAN.md).")
        }
        print("Press Ctrl+C to stop.")

        // WS-E: background auto-unload when the keep-alive deadline passes.
        let eng = engine
        let ka = keepAlive
        let evictor = Task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 10 * 1_000_000_000)
                if eng.isLoaded, await ka.shouldEvict() {
                    eng.unload()
                    await ka.markEvicted()
                    self.logger.info("[KLMServer] auto-unloaded model (keep_alive expired)")
                }
            }
        }
        defer { evictor.cancel() }

        try await channel.closeFuture.get()
        try await group.shutdownGracefully()
    }
}

// MARK: - HTTP Handler

@preconcurrency
private final class HTTPHandler: ChannelInboundHandler, @unchecked Sendable {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart

    private let engine: InferenceEngine
    private let registry: Registry
    private let compat: CompatMode
    private let embedEngine: EmbeddingEngine
    private let logger = Logger(label: "krillm.http")
    private let startedAt = Date()

    private let maxBodySize: Int

    private var requestHead: HTTPRequestHead?
    private var body: ByteBuffer = ByteBuffer()

    private let corsOrigins: [String]
    private let keepAlive: KeepAliveController
    private let defaultContextLimit: Int?
    private let genQueue: GenerationQueue
    private var currentOrigin: String?

    init(engine: InferenceEngine, registry: Registry,
         compat: CompatMode = .both, embedEngine: EmbeddingEngine = EmbeddingEngine(),
         keepAlive: KeepAliveController = KeepAliveController(defaultSeconds: 300),
         corsOrigins: [String] = ["http://localhost", "http://127.0.0.1", "https://localhost"],
         defaultContextLimit: Int? = nil,
         genQueue: GenerationQueue = GenerationQueue(numParallel: 1, maxQueue: 512),
         maxBodySizeOverride: Int? = nil) {
        self.engine = engine
        self.registry = registry
        self.compat = compat
        self.embedEngine = embedEngine
        self.keepAlive = keepAlive
        self.corsOrigins = corsOrigins
        self.defaultContextLimit = defaultContextLimit
        self.genQueue = genQueue
        self.maxBodySize = maxBodySizeOverride ?? ServerLimits.maxBodySize
    }

    /// Record activity for WS-E auto-unload; honors a per-request
    /// `keep_alive` override (seconds; nil=default, <0=pin, 0=evict-after).
    private func touchKeepAlive(_ override: Int?) {
        let ka = keepAlive
        Task { await ka.touch(override: override) }
    }

    /// Resolve the `Access-Control-Allow-Origin` value for a request Origin.
    /// `*` in the allowlist (or no Origin header) yields `*`; an allowed
    /// Origin is echoed back; anything else gets no CORS grant.
    private func allowedOrigin(for origin: String?) -> String? {
        if corsOrigins.contains("*") { return "*" }
        guard let origin else { return nil }
        return corsOrigins.contains(origin) ? origin : nil
    }

    private func corsHeaders() -> [(String, String)] {
        guard let ao = allowedOrigin(for: currentOrigin) else { return [] }
        return [
            ("Access-Control-Allow-Origin", ao),
            ("Access-Control-Allow-Methods", "GET, POST, DELETE, HEAD, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
            ("Vary", "Origin"),
        ]
    }

    /// Best-effort load of file bytes for a path. Returns nil if the path is
    /// nil or the read fails.
    static func loadDataIfPath(_ path: String?) -> Data? {
        guard let path else { return nil }
        return try? Data(contentsOf: URL(fileURLWithPath: path))
    }

    /// True iff the file looks like a RIFF/WAVE container. The native audio
    /// frontend is WAV-PCM-only; mp3/flac/ogg/m4a payloads (which the
    /// server still accepts) must stay on the `mlx-vlm` bridge that can
    /// decode them, instead of being silently dropped (PR #21 review P1).
    static func audioIsNativeWAV(_ path: String?) -> Bool {
        guard let path,
              let fh = FileHandle(forReadingAtPath: path) else { return false }
        defer { try? fh.close() }
        guard let head = try? fh.read(upToCount: 12) else { return false }
        return AudioPreprocessor.isWAV(head)
    }

    /// Audio routes natively only when the model+flag select it AND the
    /// payload is WAV the native frontend can actually decode; otherwise
    /// the bridge handles it (or, if the bridge is unavailable, the native
    /// path surfaces an explicit decode error rather than text-only).
    func audioRoutesNative(_ media: DecodedMedia?) -> Bool {
        guard let media, media.audioPath != nil else { return false }
        return engine.canUseNativeAudio
            && Self.audioIsNativeWAV(media.audioPath)
    }

    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let part = unwrapInboundIn(data)

        switch part {
        case .head(let head):
            requestHead = head
            body.clear()
        case .body(var buf):
            body.writeBuffer(&buf)
            if body.readableBytes > maxBodySize {
                let head = HTTPResponseHead(version: .http1_1, status: .payloadTooLarge)
                context.write(wrapOutboundOut(.head(head)), promise: nil)
                context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
                context.close(promise: nil)
                requestHead = nil
                return
            }
        case .end:
            guard let head = requestHead else { return }
            handleRequest(context: context, head: head, body: body)
            requestHead = nil
        }
    }

    private func handleRequest(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer) {
        let path = head.uri.split(separator: "?").first.map(String.init) ?? head.uri
        currentOrigin = head.headers.first(name: "Origin")

        // CORS preflight (WS-G / T3-1): answer any OPTIONS with the grant.
        if head.method == .OPTIONS {
            var headers = HTTPHeaders()
            for (k, v) in corsHeaders() { headers.add(name: k, value: v) }
            headers.add(name: "Content-Length", value: "0")
            let h = HTTPResponseHead(version: .http1_1, status: .noContent, headers: headers)
            context.write(wrapOutboundOut(.head(h)), promise: nil)
            context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
            return
        }

        // Path-parameter routes (matched before the exact-match switch).
        if path.hasPrefix("/api/blobs/") {
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleBlob(context: context, method: head.method,
                       digest: String(path.dropFirst("/api/blobs/".count)), body: body)
            return
        }
        if head.method == .GET, path.hasPrefix("/v1/models/"),
           path != "/v1/models/load", path != "/v1/models/unload" {
            guard compat.openAIEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleModelDetail(context: context, id: String(path.dropFirst("/v1/models/".count)))
            return
        }

        switch (head.method, path) {
        // OpenAI endpoints
        case (.POST, "/v1/chat/completions"):
            guard compat.openAIEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleChatCompletions(context: context, body: body)
        case (.POST, "/v1/completions"):
            guard compat.openAIEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleCompletions(context: context, body: body)
        case (.GET, "/v1/models"):
            guard compat.openAIEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleModels(context: context)
        case (.POST, "/v1/embeddings"):
            guard compat.openAIEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleEmbeddings(context: context, body: body, style: .openAI)
        case (.POST, "/v1/messages"):
            guard compat.openAIEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleAnthropicMessages(context: context, body: body)

        // Model management (KrillLM-native, available in any compat mode)
        case (.POST, "/v1/models/load"):
            handleLoadModel(context: context, body: body)
        case (.POST, "/v1/models/unload"):
            handleUnloadModel(context: context)

        // Status
        case (.GET, "/v1/status"):
            handleStatus(context: context)

        // Ollama endpoints
        case (.POST, "/api/chat"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaChat(context: context, body: body)
        case (.POST, "/api/generate"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaGenerate(context: context, body: body)
        case (.GET, "/api/tags"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaTags(context: context)

        // Ollama discovery (WS-A2)
        case (.GET, "/api/version"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            sendJSON(context: context, status: .ok, body: OllamaCompat.versionPayload())
        case (.GET, "/api/ps"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaPS(context: context)
        case (.POST, "/api/show"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaShow(context: context, body: body)
        case (.POST, "/api/embed"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleEmbeddings(context: context, body: body, style: .ollamaBatch)
        case (.POST, "/api/embeddings"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleEmbeddings(context: context, body: body, style: .ollamaLegacy)

        // Ollama lifecycle (WS-A3)
        case (.POST, "/api/pull"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaPull(context: context, body: body)
        case (.POST, "/api/create"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaCreate(context: context, body: body)
        case (.DELETE, "/api/delete"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaDelete(context: context, body: body)
        case (.POST, "/api/copy"):
            guard compat.ollamaEnabled else { return sendCompatDisabled(context: context, path: path) }
            handleOllamaCopy(context: context, body: body)

        // Health and metrics (always available)
        case (.GET, "/healthz"), (.GET, "/health"):
            sendJSON(context: context, status: .ok, body: [
                "status": "ok",
                "model_loaded": engine.isLoaded,
                "model": engine.modelName ?? "none",
                "family": engine.family ?? "none"
            ])
        case (.GET, "/metrics"):
            handleMetrics(context: context)

        default:
            sendJSON(context: context, status: .notFound,
                     body: ["error": "Not found: \(head.method) \(path)"])
        }
    }

    /// WS-E queue: acquire a generation slot, or return false having sent a
    /// 503. Callers must `leaveQueue()` (deferred) once done. Concurrent
    /// requests are queued up to `maxQueue`; beyond that they get 503 so the
    /// single-flight engine + prefix/KV caches are never entered in
    /// parallel.
    private func enterQueueOr503(_ ctx: ChannelHandlerContext,
                                 _ eventLoop: EventLoop) async -> Bool {
        // Register the model hold BEFORE queueing (PR #20 review P1): a
        // request accepted while the model is loaded must keep it loaded
        // even while it waits behind another request. If beginRequest() ran
        // only after genQueue.enter() returned, a waiter would not count as
        // a hold; when the active request drained, inFlight could hit 0 with
        // keep_alive:0 / an elapsed deadline armed and the background evictor
        // could unload the model in the slot-handoff window — the resumed
        // waiter would then generate against an unloaded engine. Holding
        // from acceptance keeps inFlight >= 1 for the whole accepted set.
        await keepAlive.beginRequest()
        do {
            try await genQueue.enter()
            return true
        } catch {
            await keepAlive.endRequest()   // rejected: balance the hold
            sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                           status: .serviceUnavailable,
                           body: ["error": "server busy: max queue exceeded (KRILL_MAX_QUEUE)"])
            return false
        }
    }

    private func leaveQueue() {
        let q = genQueue
        let ka = keepAlive
        // Release the queue slot first so the next waiter resumes, then drop
        // this request's hold. Any queued waiter already holds its own
        // beginRequest()-registered hold (taken before it queued), so
        // inFlight never transiently hits 0 during the handoff and the
        // evictor can't unload between requests (PR #20 review P1).
        Task { await q.leave(); await ka.endRequest() }
    }

    /// Slot acquire that just returns success/failure (for streaming paths
    /// whose 200 head is already on the wire, so a JSON 503 is impossible -
    /// the caller closes the stream with a terminal error instead).
    private func tryEnterQueue() async -> Bool {
        await keepAlive.beginRequest()     // hold from acceptance (P1)
        do {
            try await genQueue.enter()
            return true
        } catch {
            await keepAlive.endRequest()   // rejected: balance the hold
            return false
        }
    }

    /// Apply a created model's Modelfile `PARAMETER` overrides (WS-C) as
    /// *defaults*: a value is taken from the Modelfile only when the request
    /// left that knob at its parse default, so an explicit client value
    /// still wins (matches Ollama's Modelfile-as-defaults semantics).
    private func applyModelParams(
        sampling: SamplingParams, maxTokens: Int, contextLimit: Int?
    ) -> (SamplingParams, Int, Int?) {
        guard let name = engine.modelName,
              let p = registry.getModel(name)?.overrides?.parameters,
              !p.isEmpty
        else { return (sampling, maxTokens, contextLimit) }

        var s = sampling
        var mt = maxTokens
        var ctx = contextLimit
        func f(_ k: String) -> Float? { p[k].flatMap { Float($0) } }
        func i(_ k: String) -> Int? { p[k].flatMap { Int($0) } }

        if s.temperature == 0.0, let v = f("temperature") { s.temperature = v }
        if s.topP == 1.0, let v = f("top_p") { s.topP = v }
        if s.topK == 0, let v = i("top_k") { s.topK = v }
        if s.repetitionPenalty == 1.0, let v = f("repeat_penalty") { s.repetitionPenalty = v }
        if s.minP == 0.0, let v = f("min_p") { s.minP = v }
        if s.presencePenalty == 0.0, let v = f("presence_penalty") { s.presencePenalty = v }
        if s.frequencyPenalty == 0.0, let v = f("frequency_penalty") { s.frequencyPenalty = v }
        if s.mirostat == 0, let v = i("mirostat") { s.mirostat = v }
        if let v = i("repeat_last_n") { s.repeatLastN = v }
        if ctx == nil, let v = i("num_ctx") { ctx = v }
        if (mt == 512 || mt == 2048), let v = i("num_predict"), v > 0 { mt = v }
        if s.seed == nil, let seed = i("seed") { s.seed = UInt64(max(0, seed)) }
        return (s, mt, ctx)
    }

    /// Apply a created model's Modelfile `SYSTEM` override (WS-C): if the
    /// loaded model has a system override and the request carries no system
    /// message, prepend it. `TEMPLATE` round-trips via `show`/`/api/show`;
    /// runtime template application is a tracked follow-up.
    private func applyModelSystemOverride(
        _ messages: [[String: String]]
    ) -> [[String: String]] {
        guard let name = engine.modelName,
              let sys = registry.getModel(name)?.overrides?.system,
              !sys.isEmpty,
              !messages.contains(where: { $0["role"] == "system" })
        else { return messages }
        return [["role": "system", "content": sys]] + messages
    }

    /// 404 with a clear reason when an endpoint exists but its compat family
    /// is disabled by `--compat`. 404 (not 403) keeps client probing logic
    /// behaving as if the server simply doesn't speak that protocol.
    private func sendCompatDisabled(context: ChannelHandlerContext, path: String) {
        sendJSON(context: context, status: .notFound, body: [
            "error": "Endpoint \(path) is disabled in compat mode '\(compat.rawValue)'. Restart krillm serve with --compat both."
        ])
    }

    /// Guard: returns false and sends 503 if no model is loaded or a swap is in progress.
    private func requireModel(context: ChannelHandlerContext) -> Bool {
        if engine.isSwapping {
            sendJSON(context: context, status: .serviceUnavailable, body: [
                "error": "Model swap in progress. Please retry shortly."
            ])
            return false
        }
        if engine.isLoaded { return true }
        sendJSON(context: context, status: .serviceUnavailable, body: [
            "error": "No model loaded. POST /v1/models/load to load one, or restart with --model."
        ])
        return false
    }

    // MARK: - OpenAI: POST /v1/chat/completions

    private func handleChatCompletions(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }

        let request: ServerChatRequest
        do {
            request = try ServerParsing.openAIChatRequest(from: json)
        } catch let error as ServerRequestError {
            sendJSON(context: context, status: .badRequest, body: ["error": error.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid request"])
            return
        }

        // Fix 7: Model validation — reject if a specific model is requested but doesn't match loaded model.
        if let requestedModel = request.requestedModel,
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        // Multimodal shape validation runs before model gate so it is
        // observable even when no model is loaded.
        do { try validateMediaShape(request.media) } catch let e as MediaDecodeError {
            sendJSON(context: context,
                     status: HTTPResponseStatus(statusCode: e.httpStatus),
                     body: ["error": e.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid media payload"])
            return
        }

        guard requireModel(context: context) else { return }
        touchKeepAlive(request.keepAlive)

        guard !request.messages.isEmpty else {
            sendJSON(context: context, status: .badRequest, body: ["error": "No valid messages"])
            return
        }

        // Multimodal: validate against loaded model and decode payloads.
        let decodedMedia: DecodedMedia?
        if request.media.isEmpty {
            decodedMedia = nil
        } else {
            do {
                decodedMedia = try decodeMediaForRequest(request.media)
            } catch let error as MediaDecodeError {
                sendJSON(
                    context: context,
                    status: HTTPResponseStatus(statusCode: error.httpStatus),
                    body: ["error": error.message]
                )
                return
            } catch {
                sendJSON(context: context, status: .internalServerError,
                         body: ["error": "Failed to decode media payload"])
                return
            }
        }

        if let media = decodedMedia, media.audioPath != nil,
           !audioRoutesNative(media) {
            // Audio -> Python bridge fallback. With KRILL_NATIVE_AUDIO=1 the
            // native Swift+MLX path handles it inline (see eng.generate).
            handleBridgeChat(
                context: context,
                messages: request.messages,
                media: media,
                maxTokens: request.maxTokens,
                bridgeStyle: .openAI,
                requestStream: request.stream
            )
            return
        }

        if !request.tools.isEmpty {
            handleToolChat(
                context: context, request: request,
                media: decodedMedia, style: .openAI)
            return
        }

        let genMessages = StructuredOutput.injectFormatSystem(
            into: applyModelSystemOverride(request.messages),
            format: request.responseFormat)

        let (effParams, effMax, effCtx) = applyModelParams(
            sampling: request.sampling.samplingParams,
            maxTokens: request.maxTokens,
            contextLimit: request.contextLimit ?? defaultContextLimit)
        if request.stream {
            handleStreamingCompletion(
                context: context, messages: genMessages,
                params: effParams, maxTokens: effMax,
                media: decodedMedia, contextLimit: effCtx)
        } else {
            handleNonStreamingCompletion(
                context: context, messages: genMessages,
                params: effParams, maxTokens: effMax,
                media: decodedMedia, responseFormat: request.responseFormat,
                contextLimit: effCtx)
        }
    }

    // MARK: - Anthropic: POST /v1/messages (WS-F)

    private func handleAnthropicMessages(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }
        let p = AnthropicCompat.parse(json)
        guard requireModel(context: context) else { return }
        touchKeepAlive(nil)

        var msgs = ToolCalling.injectToolSystem(into: p.messages, tools: p.tools)
        if p.thinking {
            msgs = StructuredOutput.injectFormatSystem(into: msgs, format: nil)
            msgs.insert(["role": "system",
                         "content": "Think step by step inside <thinking>...</thinking> before your final answer."],
                        at: 0)
        }
        let modelName = engine.modelName ?? p.model ?? "unknown"
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let wantStream = p.stream
        let (params, maxTokens, ctxLimit) = applyModelParams(
            sampling: p.sampling.samplingParams, maxTokens: p.maxTokens,
            contextLimit: defaultContextLimit)

        Task {
            guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            defer { self.leaveQueue() }
            let (tokenStream, getStats) = eng.generate(
                messages: msgs, params: params, maxTokens: maxTokens,
                contextLimit: ctxLimit)
            var full = ""
            for await ev in tokenStream { if ev.isEnd { break }; full += ev.text }

            // Split out <thinking> if present.
            var thinking: String? = nil
            var visible = full
            if let s = full.range(of: "<thinking>"),
               let e = full.range(of: "</thinking>") {
                thinking = String(full[s.upperBound ..< e.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                visible.removeSubrange(s.lowerBound ..< e.upperBound)
            }
            let (calls, cleaned) = ToolCalling.extractToolCalls(from: visible)
            let stats = getStats()
            let inTok = stats?.promptTokens ?? 0
            let outTok = stats?.generatedTokens ?? 0

            if !wantStream {
                let resp = AnthropicCompat.response(
                    model: modelName, text: cleaned, toolCalls: calls,
                    thinking: thinking, inputTokens: inTok, outputTokens: outTok)
                self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                    status: .ok, body: resp)
                return
            }

            // Minimal but valid Anthropic SSE event sequence.
            self.writeOnLoop(ctx, .head(ServerResponseHeads.openAIStreaming(cors: self.corsHeaders())))
            func sse(_ event: String, _ obj: [String: Any]) {
                if let d = try? JSONSerialization.data(withJSONObject: obj),
                   let s = String(data: d, encoding: .utf8) {
                    self.writeRaw(ctx, "event: \(event)\ndata: \(s)\n\n")
                }
            }
            let msgId = "msg_\(UUID().uuidString.prefix(12))"
            sse("message_start", [
                "type": "message_start",
                "message": ["id": msgId, "type": "message", "role": "assistant",
                            "model": modelName, "content": [[String: Any]](),
                            "stop_reason": NSNull(),
                            "usage": ["input_tokens": inTok, "output_tokens": 0]],
            ])
            if calls.isEmpty {
                sse("content_block_start", ["type": "content_block_start", "index": 0,
                    "content_block": ["type": "text", "text": ""]])
                sse("content_block_delta", ["type": "content_block_delta", "index": 0,
                    "delta": ["type": "text_delta", "text": cleaned]])
                sse("content_block_stop", ["type": "content_block_stop", "index": 0])
            } else {
                for (i, c) in calls.enumerated() {
                    let input = (try? JSONSerialization.jsonObject(
                        with: Data(c.argumentsJSON.utf8))) ?? [String: Any]()
                    sse("content_block_start", ["type": "content_block_start", "index": i,
                        "content_block": ["type": "tool_use",
                                          "id": "toolu_\(UUID().uuidString.prefix(8))",
                                          "name": c.name, "input": input]])
                    sse("content_block_stop", ["type": "content_block_stop", "index": i])
                }
            }
            sse("message_delta", ["type": "message_delta",
                "delta": ["stop_reason": calls.isEmpty ? "end_turn" : "tool_use"],
                "usage": ["output_tokens": outTok]])
            sse("message_stop", ["type": "message_stop"])
            self.writeOnLoop(ctx, .end(nil), flush: true)
        }
    }

    // MARK: - Tool/function calling (WS-D D1)

    private enum ToolChatStyle { case openAI, ollama }

    /// Buffered (non-token-streaming) generation for tool-enabled chat.
    /// Tool definitions are injected as a system turn; the completed text is
    /// scanned for tool-call sentinels and shaped into the OpenAI/Ollama
    /// `tool_calls` response. Honors `stream` by emitting the assembled
    /// result as a single well-formed SSE / NDJSON sequence (token-level
    /// tool-call deltas are Phase 4 per the parity plan).
    private func handleToolChat(
        context: ChannelHandlerContext,
        request: ServerChatRequest,
        media: DecodedMedia?,
        style: ToolChatStyle
    ) {
        let messages = ToolCalling.injectToolSystem(
            into: request.messages, tools: request.tools)
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let imageData = Self.loadDataIfPath(media?.imagePath)
        let audioData = Self.loadDataIfPath(media?.audioPath)
        let mediaCopy = media
        let modelName = engine.modelName ?? request.requestedModel ?? "unknown"
        let (params, maxTokens, toolCtx) = applyModelParams(
            sampling: request.sampling.samplingParams,
            maxTokens: request.maxTokens,
            contextLimit: request.contextLimit ?? defaultContextLimit)
        let wantStream = request.stream
        let started = CFAbsoluteTimeGetCurrent()

        Task {
            defer { mediaCopy?.cleanup() }
            guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            defer { self.leaveQueue() }
            let (tokenStream, getStats) = eng.generate(
                messages: messages, params: params,
                maxTokens: maxTokens, imageData: imageData,
                audioData: audioData,
                contextLimit: toolCtx)

            var full = ""
            for await event in tokenStream {
                if event.isEnd { break }
                full += event.text
            }
            let (calls, cleaned) = ToolCalling.extractToolCalls(from: full)
            let stats = getStats()
            let totalNs = Int64((CFAbsoluteTimeGetCurrent() - started) * 1_000_000_000)

            let response: [String: Any]
            switch style {
            case .openAI:
                var message: [String: Any] = ["role": "assistant"]
                if calls.isEmpty {
                    message["content"] = cleaned
                } else {
                    message["content"] = NSNull()
                    message["tool_calls"] = ToolCalling.openAIToolCalls(calls)
                }
                response = [
                    "id": "chatcmpl-\(UUID().uuidString.prefix(8))",
                    "object": "chat.completion",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": modelName,
                    "choices": [[
                        "index": 0,
                        "message": message,
                        "finish_reason": calls.isEmpty ? "stop" : "tool_calls",
                    ]],
                    "usage": [
                        "prompt_tokens": stats?.promptTokens ?? 0,
                        "completion_tokens": stats?.generatedTokens ?? 0,
                        "total_tokens": (stats?.promptTokens ?? 0) + (stats?.generatedTokens ?? 0),
                    ],
                ]
            case .ollama:
                var message: [String: Any] = ["role": "assistant",
                                              "content": calls.isEmpty ? cleaned : ""]
                if !calls.isEmpty {
                    message["tool_calls"] = ToolCalling.ollamaToolCalls(calls)
                }
                response = [
                    "model": modelName,
                    "message": message,
                    "done": true,
                    "done_reason": calls.isEmpty ? "stop" : "tool_calls",
                    "total_duration": totalNs,
                    "prompt_eval_count": stats?.promptTokens ?? 0,
                    "eval_count": stats?.generatedTokens ?? 0,
                ]
            }

            if !wantStream {
                self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                    status: .ok, body: response)
                return
            }

            // Streaming clients: emit the assembled result as one chunk.
            switch style {
            case .openAI:
                let choice = (response["choices"] as? [[String: Any]])?.first ?? [:]
                let msg = choice["message"] as? [String: Any] ?? [:]
                var delta: [String: Any] = ["role": "assistant"]
                if let tc = msg["tool_calls"] { delta["tool_calls"] = tc }
                if let c = msg["content"] as? String { delta["content"] = c }
                let chunk: [String: Any] = [
                    "id": response["id"] ?? "chatcmpl",
                    "object": "chat.completion.chunk",
                    "created": Int(Date().timeIntervalSince1970),
                    "model": modelName,
                    "choices": [[
                        "index": 0, "delta": delta,
                        "finish_reason": choice["finish_reason"] ?? "stop",
                    ]],
                ]
                self.writeOnLoop(ctx, .head(ServerResponseHeads.openAIStreaming(cors: self.corsHeaders())))
                self.writeSSEJSON(ctx, chunk)
                self.writeRaw(ctx, "data: [DONE]\n\n")
                self.writeOnLoop(ctx, .end(nil), flush: true)
            case .ollama:
                self.writeOnLoop(ctx, .head(ServerResponseHeads.ollamaStreaming(cors: self.corsHeaders())))
                self.writeNDJSON(ctx, response)
                self.writeOnLoop(ctx, .end(nil), flush: true)
            }
        }
    }

    private func writeSSEJSON(_ ctx: ChannelHandlerContext, _ obj: [String: Any]) {
        guard let d = try? JSONSerialization.data(withJSONObject: obj),
              let s = String(data: d, encoding: .utf8) else { return }
        writeRaw(ctx, "data: \(s)\n\n")
    }

    private func writeNDJSON(_ ctx: ChannelHandlerContext, _ obj: [String: Any]) {
        guard let d = try? JSONSerialization.data(withJSONObject: obj) else { return }
        var b = ByteBufferAllocator().buffer(capacity: d.count + 1)
        b.writeBytes(d); b.writeString("\n")
        writeOnLoop(ctx, .body(.byteBuffer(b)), flush: true)
    }

    private func writeRaw(_ ctx: ChannelHandlerContext, _ s: String) {
        var b = ByteBufferAllocator().buffer(capacity: s.utf8.count)
        b.writeString(s)
        writeOnLoop(ctx, .body(.byteBuffer(b)), flush: true)
    }

    // MARK: - Media decoding shared helper

    /// Pre-flight checks that don't require a loaded model.
    private func validateMediaShape(_ payload: ServerMediaPayload) throws {
        if payload.images.count > 1 {
            throw MediaDecodeError.tooManyImages
        }
        try ServerMultimodal.validatePayloadSizes(payload)
    }

    /// Validate a parsed media payload against the loaded model and decode it
    /// to temp files. Caller is responsible for ``DecodedMedia.cleanup()``.
    private func decodeMediaForRequest(_ payload: ServerMediaPayload) throws -> DecodedMedia {
        try validateMediaShape(payload)
        // Capability check.
        if !payload.images.isEmpty && !engine.supportsNativeImage {
            throw MediaDecodeError.mediaNotSupported(
                reason: "Loaded model does not support image input. Image input requires a Gemma 4 model."
            )
        }
        if payload.audio != nil && !engine.supportsAudio {
            throw MediaDecodeError.mediaNotSupported(
                reason: "Loaded model does not support audio input. Audio input requires a Gemma 4 model."
            )
        }

        var imagePath: String? = nil
        var audioPath: String? = nil

        if let img = payload.images.first {
            imagePath = try ServerMultimodal.decodeAndWrite(
                base64: img, field: "images", sniffImage: true
            )
        }
        if let aud = payload.audio {
            let ext = (payload.audioFormat?.lowercased()).flatMap {
                ["wav", "mp3", "flac", "ogg", "m4a"].contains($0) ? $0 : nil
            } ?? "wav"
            audioPath = try ServerMultimodal.decodeAndWrite(
                base64: aud, field: "audio", preferredExtension: ext
            )
        }

        return DecodedMedia(imagePath: imagePath, audioPath: audioPath)
    }

    /// Distill a chat history into a single prompt + system pair for the
    /// Python bridge, which doesn't take chat structure today.
    private func bridgePrompt(from messages: [[String: String]]) -> (prompt: String, system: String?) {
        var system: String? = nil
        var userTurns: [String] = []
        for m in messages {
            let role = m["role"] ?? "user"
            let content = m["content"] ?? ""
            if role == "system" {
                system = (system.map { $0 + "\n" } ?? "") + content
            } else if role == "user" {
                userTurns.append(content)
            }
        }
        return (userTurns.joined(separator: "\n"), system)
    }

    /// Reply style for the bridge (audio) path.
    private enum BridgeStyle {
        case openAI
        case ollamaChat
        case ollamaGenerate
    }

    /// Run a multimodal request through the Python bridge (audio path).
    private func handleBridgeChat(
        context: ChannelHandlerContext,
        messages: [[String: String]],
        media: DecodedMedia,
        maxTokens: Int,
        bridgeStyle: BridgeStyle,
        requestStream: Bool,
        promptOverride: String? = nil,
        systemOverride: String? = nil
    ) {
        guard let modelDir = engine.modelDirectoryPath else {
            sendJSON(context: context, status: .serviceUnavailable,
                     body: ["error": "No model loaded."])
            media.cleanup()
            return
        }
        let availability = PythonFallback.checkAvailability()
        guard availability.isAvailable else {
            sendJSON(context: context, status: .serviceUnavailable, body: [
                "error": "Audio input requires the mlx-vlm Python bridge: \(availability.detail). Install via `make setup-mlx-vlm`."
            ])
            media.cleanup()
            return
        }

        let (composedPrompt, composedSystem) = bridgePrompt(from: messages)
        let prompt = promptOverride ?? composedPrompt
        let system = systemOverride ?? composedSystem
        let modelName = engine.modelName ?? "unknown"

        nonisolated(unsafe) let ctx = context
        let eventLoop = context.eventLoop
        let imagePath = media.imagePath
        let audioPath = media.audioPath
        let style = bridgeStyle

        // Audio bridge does not stream; if the client asked for streaming we
        // emit a single payload chunk + final done chunk for compatibility.
        if requestStream {
            switch style {
            case .ollamaChat, .ollamaGenerate:
                context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming(cors: self.corsHeaders()))), promise: nil)
            case .openAI:
                context.write(wrapOutboundOut(.head(ServerResponseHeads.openAIStreaming(cors: self.corsHeaders()))), promise: nil)
            }
        }

        let requestStart = CFAbsoluteTimeGetCurrent()

        Task {
            defer { media.cleanup() }
            do {
                let fallback = PythonFallback(modelPath: modelDir)
                let composed: String
                if let sys = system, !sys.isEmpty {
                    composed = sys + "\n\n" + prompt
                } else {
                    composed = prompt
                }
                let output = try await fallback.generate(
                    prompt: composed,
                    maxTokens: maxTokens,
                    imagePath: imagePath,
                    audioPath: audioPath
                )
                let totalNs = Int64((CFAbsoluteTimeGetCurrent() - requestStart) * 1_000_000_000)
                switch style {
                case .openAI:
                    let chatId = "chatcmpl-\(UUID().uuidString.prefix(8))"
                    if requestStream {
                        let contentChunk = sseChunk(id: chatId, content: output, finishReason: nil)
                        var buf1 = ByteBufferAllocator().buffer(capacity: contentChunk.utf8.count)
                        buf1.writeString(contentChunk)
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf1)), flush: true)
                        let stopChunk = sseChunk(id: chatId, content: nil, finishReason: "stop")
                        var buf2 = ByteBufferAllocator().buffer(capacity: stopChunk.utf8.count)
                        buf2.writeString(stopChunk)
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf2)), flush: true)
                        let done = "data: [DONE]\n\n"
                        var buf3 = ByteBufferAllocator().buffer(capacity: done.utf8.count)
                        buf3.writeString(done)
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf3)), flush: true)
                        self.writeOnLoop(ctx, .end(nil), flush: true)
                    } else {
                        let response: [String: Any] = [
                            "id": chatId,
                            "object": "chat.completion",
                            "created": Int(Date().timeIntervalSince1970),
                            "choices": [[
                                "index": 0,
                                "message": ["role": "assistant", "content": output],
                                "finish_reason": "stop"
                            ]],
                            "usage": [
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            ],
                            "krillm_path": "mlx-vlm-bridge"
                        ]
                        self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
                    }
                case .ollamaChat:
                    if requestStream {
                        // Single content chunk
                        let body: [String: Any] = [
                            "model": modelName,
                            "message": ["role": "assistant", "content": output],
                            "done": false
                        ]
                        let data1 = try? JSONSerialization.data(withJSONObject: body)
                        if let data1 {
                            var buf = ByteBufferAllocator().buffer(capacity: data1.count + 1)
                            buf.writeBytes(data1)
                            buf.writeString("\n")
                            self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        }
                        let final: [String: Any] = [
                            "model": modelName,
                            "message": ["role": "assistant", "content": ""],
                            "done": true,
                            "total_duration": totalNs,
                            "prompt_eval_count": 0,
                            "prompt_eval_duration": 0,
                            "eval_count": 0,
                            "eval_duration": 0,
                            "krillm_path": "mlx-vlm-bridge"
                        ]
                        let data2 = try? JSONSerialization.data(withJSONObject: final)
                        if let data2 {
                            var buf = ByteBufferAllocator().buffer(capacity: data2.count + 1)
                            buf.writeBytes(data2)
                            buf.writeString("\n")
                            self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        }
                        self.writeOnLoop(ctx, .end(nil), flush: true)
                    } else {
                        let response: [String: Any] = [
                            "model": modelName,
                            "message": ["role": "assistant", "content": output],
                            "done": true,
                            "total_duration": totalNs,
                            "prompt_eval_count": 0,
                            "prompt_eval_duration": 0,
                            "eval_count": 0,
                            "eval_duration": 0,
                            "krillm_path": "mlx-vlm-bridge"
                        ]
                        self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
                    }
                case .ollamaGenerate:
                    if requestStream {
                        let body: [String: Any] = [
                            "model": modelName,
                            "response": output,
                            "done": false
                        ]
                        let data1 = try? JSONSerialization.data(withJSONObject: body)
                        if let data1 {
                            var buf = ByteBufferAllocator().buffer(capacity: data1.count + 1)
                            buf.writeBytes(data1)
                            buf.writeString("\n")
                            self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        }
                        let final: [String: Any] = [
                            "model": modelName,
                            "response": "",
                            "done": true,
                            "total_duration": totalNs,
                            "prompt_eval_count": 0,
                            "prompt_eval_duration": 0,
                            "eval_count": 0,
                            "eval_duration": 0,
                            "krillm_path": "mlx-vlm-bridge"
                        ]
                        let data2 = try? JSONSerialization.data(withJSONObject: final)
                        if let data2 {
                            var buf = ByteBufferAllocator().buffer(capacity: data2.count + 1)
                            buf.writeBytes(data2)
                            buf.writeString("\n")
                            self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        }
                        self.writeOnLoop(ctx, .end(nil), flush: true)
                    } else {
                        let response: [String: Any] = [
                            "model": modelName,
                            "response": output,
                            "done": true,
                            "total_duration": totalNs,
                            "prompt_eval_count": 0,
                            "prompt_eval_duration": 0,
                            "eval_count": 0,
                            "eval_duration": 0,
                            "krillm_path": "mlx-vlm-bridge"
                        ]
                        self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
                    }
                }
            } catch {
                let msg = String(describing: error).prefix(500)
                eventLoop.execute {
                    self.sendJSON(context: ctx, status: .internalServerError,
                                  body: ["error": "mlx-vlm bridge failed: \(msg)"])
                }
            }
        }
    }

    private func handleStreamingCompletion(
        context: ChannelHandlerContext,
        messages: [[String: String]],
        params: SamplingParams, maxTokens: Int,
        media: DecodedMedia? = nil,
        contextLimit: Int? = nil
    ) {
        // Write the SSE head synchronously within the channelRead call
        // chain (NIO requires the response begin here, not from a detached
        // Task - doing the latter trips a ChannelPipeline precondition).
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/event-stream")
        headers.add(name: "Cache-Control", value: "no-cache")
        headers.add(name: "Connection", value: "keep-alive")
        for (k, v) in corsHeaders() { headers.add(name: k, value: v) }
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)

        nonisolated(unsafe) let ctx = context
        let eng = engine
        let imageData = Self.loadDataIfPath(media?.imagePath)
        let mediaCopy = media

        Task {
            defer { mediaCopy?.cleanup() }
            // Hold a generation slot for the whole token loop (WS-E). The
            // 200 SSE head is already sent, so on overflow we close the
            // stream with a valid terminal error chunk + [DONE].
            guard await self.tryEnterQueue() else {
                let errChunk = "data: {\"error\":\"server busy: max queue exceeded\"}\n\n"
                var b = ByteBufferAllocator().buffer(capacity: errChunk.utf8.count)
                b.writeString(errChunk + "data: [DONE]\n\n")
                self.writeOnLoop(ctx, .body(.byteBuffer(b)), flush: true)
                self.writeOnLoop(ctx, .end(nil), flush: true)
                return
            }
            defer { self.leaveQueue() }

            let (tokenStream, _) = eng.generate(
                messages: messages,
                params: params, maxTokens: maxTokens,
                imageData: imageData, contextLimit: contextLimit)

            let id = "chatcmpl-\(UUID().uuidString.prefix(8))"

            for await event in tokenStream {
                if event.isEnd {
                    let chunk = sseChunk(id: id, content: nil, finishReason: "stop")
                    var buf = ByteBufferAllocator().buffer(capacity: chunk.utf8.count)
                    buf.writeString(chunk)
                    self.writeOnLoop(ctx, .body(.byteBuffer(buf)))

                    // Send [DONE]
                    let done = "data: [DONE]\n\n"
                    var doneBuf = ByteBufferAllocator().buffer(capacity: done.utf8.count)
                    doneBuf.writeString(done)
                    self.writeOnLoop(ctx, .body(.byteBuffer(doneBuf)))
                    break
                }

                let chunk = sseChunk(id: id, content: event.text, finishReason: nil)
                var buf = ByteBufferAllocator().buffer(capacity: chunk.utf8.count)
                buf.writeString(chunk)
                self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
            }

            self.writeOnLoop(ctx, .end(nil), flush: true)
        }
    }

    private func handleNonStreamingCompletion(
        context: ChannelHandlerContext,
        messages: [[String: String]],
        params: SamplingParams, maxTokens: Int,
        media: DecodedMedia? = nil,
        responseFormat: ResponseFormat? = nil,
        contextLimit: Int? = nil
    ) {
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let imageData = Self.loadDataIfPath(media?.imagePath)
        let mediaCopy = media

        Task {
            defer { mediaCopy?.cleanup() }
            guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            defer { self.leaveQueue() }
            let (tokenStream, getStats) = eng.generate(
                messages: messages,
                params: params, maxTokens: maxTokens,
                imageData: imageData, contextLimit: contextLimit)

            var fullContent = ""
            for await event in tokenStream {
                if event.isEnd { break }
                fullContent += event.text
            }
            fullContent = StructuredOutput.coerce(fullContent, format: responseFormat)

            let stats = getStats()
            let response: [String: Any] = [
                "id": "chatcmpl-\(UUID().uuidString.prefix(8))",
                "object": "chat.completion",
                "created": Int(Date().timeIntervalSince1970),
                "choices": [[
                    "index": 0,
                    "message": ["role": "assistant", "content": fullContent],
                    "finish_reason": "stop"
                ]],
                "usage": [
                    "prompt_tokens": stats?.promptTokens ?? 0,
                    "completion_tokens": stats?.generatedTokens ?? 0,
                    "total_tokens": (stats?.promptTokens ?? 0) + (stats?.generatedTokens ?? 0)
                ]
            ]
            self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
        }
    }

    // MARK: - OpenAI: POST /v1/completions

    private func handleCompletions(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }

        let request: ServerCompletionRequest
        do {
            request = try ServerParsing.openAICompletionRequest(from: json)
        } catch let error as ServerRequestError {
            sendJSON(context: context, status: .badRequest, body: ["error": error.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid request"])
            return
        }

        if let requestedModel = request.requestedModel,
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        guard requireModel(context: context) else { return }
        touchKeepAlive(nil)

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
            // WS-E: /v1/completions is engine-touching too - serialize it.
            guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            defer { self.leaveQueue() }
            let (tokenStream, getStats) = eng.generate(
                prompt: request.prompt,
                params: request.sampling.samplingParams,
                maxTokens: request.maxTokens)

            var fullText = ""
            for await event in tokenStream {
                if event.isEnd { break }
                fullText += event.text
            }

            let stats = getStats()
            let response: [String: Any] = [
                "id": "cmpl-\(UUID().uuidString.prefix(8))",
                "object": "text_completion",
                "created": Int(Date().timeIntervalSince1970),
                "choices": [[
                    "text": fullText,
                    "index": 0,
                    "finish_reason": "stop"
                ]],
                "usage": [
                    "prompt_tokens": stats?.promptTokens ?? 0,
                    "completion_tokens": stats?.generatedTokens ?? 0,
                    "total_tokens": (stats?.promptTokens ?? 0) + (stats?.generatedTokens ?? 0)
                ]
            ]
            self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
        }
    }

    // MARK: - Metrics: GET /metrics

    private func handleMetrics(context: ChannelHandlerContext) {
        // Prometheus text format
        let uptime = ProcessInfo.processInfo.systemUptime
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let residentMB = Double(info.resident_size) / 1_048_576

        let body = """
        # HELP krillm_up Whether the server is up
        # TYPE krillm_up gauge
        krillm_up 1
        # HELP krillm_model_loaded Whether a model is loaded
        # TYPE krillm_model_loaded gauge
        krillm_model_loaded \(engine.isLoaded ? 1 : 0)
        # HELP krillm_resident_memory_mb Process resident memory in MB
        # TYPE krillm_resident_memory_mb gauge
        krillm_resident_memory_mb \(String(format: "%.1f", residentMB))
        # HELP krillm_uptime_seconds Process uptime in seconds
        # TYPE krillm_uptime_seconds counter
        krillm_uptime_seconds \(String(format: "%.0f", uptime))
        """

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/plain; version=0.0.4")
        let data = body.data(using: .utf8) ?? Data()
        headers.add(name: "Content-Length", value: "\(data.count)")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)

        context.write(wrapOutboundOut(.head(head)), promise: nil)
        var buf = ByteBufferAllocator().buffer(capacity: data.count)
        buf.writeBytes(data)
        context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }

    // MARK: - OpenAI: GET /v1/models

    private func handleModels(context: ChannelHandlerContext) {
        let models = registry.listModels().map { manifest -> [String: Any] in
            [
                "id": manifest.name,
                "object": "model",
                "owned_by": "local",
                "created": Int(manifest.pulledAt.timeIntervalSince1970)
            ]
        }
        sendJSON(context: context, status: .ok, body: ["data": models, "object": "list"])
    }

    // MARK: - Ollama: POST /api/chat

    private func handleOllamaChat(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }

        let request: ServerChatRequest
        do {
            request = try ServerParsing.ollamaChatRequest(from: json)
        } catch let error as ServerRequestError {
            sendJSON(context: context, status: .badRequest, body: ["error": error.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid request"])
            return
        }

        // Fix 7: Model validation — reject if a specific model is requested but doesn't match loaded model.
        if let requestedModel = request.requestedModel,
           requestedModel != "unknown",
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        do { try validateMediaShape(request.media) } catch let e as MediaDecodeError {
            sendJSON(context: context,
                     status: HTTPResponseStatus(statusCode: e.httpStatus),
                     body: ["error": e.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid media payload"])
            return
        }

        guard requireModel(context: context) else { return }
        touchKeepAlive(request.keepAlive)

        guard !request.messages.isEmpty else {
            sendJSON(context: context, status: .badRequest, body: ["error": "No valid messages"])
            return
        }

        // Multimodal: validate against loaded model and decode payloads.
        let decodedMedia: DecodedMedia?
        if request.media.isEmpty {
            decodedMedia = nil
        } else {
            do {
                decodedMedia = try decodeMediaForRequest(request.media)
            } catch let error as MediaDecodeError {
                sendJSON(
                    context: context,
                    status: HTTPResponseStatus(statusCode: error.httpStatus),
                    body: ["error": error.message]
                )
                return
            } catch {
                sendJSON(context: context, status: .internalServerError,
                         body: ["error": "Failed to decode media payload"])
                return
            }
        }

        if let media = decodedMedia, media.audioPath != nil,
           !audioRoutesNative(media) {
            handleBridgeChat(
                context: context,
                messages: request.messages,
                media: media,
                maxTokens: request.maxTokens,
                bridgeStyle: .ollamaChat,
                requestStream: request.stream
            )
            return
        }

        if !request.tools.isEmpty {
            handleToolChat(context: context, request: request,
                           media: decodedMedia, style: .ollama)
            return
        }

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let modelName = engine.modelName ?? request.requestedModel ?? "unknown"
        let imageData = Self.loadDataIfPath(decodedMedia?.imagePath)
        let audioData = Self.loadDataIfPath(decodedMedia?.audioPath)
        let mediaCopy = decodedMedia

        let requestStart = CFAbsoluteTimeGetCurrent()
        let wantStream = request.stream

        let genMessages = StructuredOutput.injectFormatSystem(
            into: applyModelSystemOverride(request.messages),
            format: request.responseFormat)
        let respFormat = request.responseFormat
        let (ocParams, ocMax, ocCtx) = applyModelParams(
            sampling: request.sampling.samplingParams,
            maxTokens: request.maxTokens,
            contextLimit: request.contextLimit ?? defaultContextLimit)
        if wantStream {
            // NDJSON head must begin in the channelRead call chain (NIO).
            context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming(cors: self.corsHeaders()))), promise: nil)
        }
        Task {
            defer { mediaCopy?.cleanup() }
            // Hold a generation slot for the whole token loop (WS-E).
            if wantStream {
                guard await self.tryEnterQueue() else {
                    self.writeNDJSON(ctx, ["model": modelName,
                        "error": "server busy: max queue exceeded", "done": true])
                    self.writeOnLoop(ctx, .end(nil), flush: true)
                    return
                }
            } else {
                guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            }
            defer { self.leaveQueue() }
            let (tokenStream, getStats) = eng.generate(
                messages: genMessages,
                params: ocParams,
                maxTokens: ocMax,
                imageData: imageData,
                audioData: audioData,
                contextLimit: ocCtx)

            if request.stream {
                var firstTokenTime: Double?
                var generatedCount = 0
                for await event in tokenStream {
                    if !event.isEnd && firstTokenTime == nil {
                        firstTokenTime = CFAbsoluteTimeGetCurrent()
                    }
                    if !event.isEnd { generatedCount += 1 }

                    if event.isEnd {
                        let totalNs = Int64((CFAbsoluteTimeGetCurrent() - requestStart) * 1_000_000_000)
                        let stats = getStats()
                        let prefillNs = Int64((stats?.prefillTime ?? 0) * 1_000_000_000)
                        let decodeNs = Int64((stats?.decodeTime ?? 0) * 1_000_000_000)
                        let promptTokens = stats?.promptTokens ?? 0
                        var finalChunk: [String: Any] = [
                            "model": modelName,
                            "message": ["role": "assistant", "content": ""],
                            "done": true,
                            "total_duration": totalNs,
                            "prompt_eval_count": promptTokens,
                            "prompt_eval_duration": prefillNs,
                            "eval_count": generatedCount,
                            "eval_duration": decodeNs,
                        ]
                        if let ftt = firstTokenTime {
                            finalChunk["ttft_ns"] = Int64((ftt - requestStart) * 1_000_000_000)
                        }
                        let data = try! JSONSerialization.data(withJSONObject: finalChunk)
                        var buf = ByteBufferAllocator().buffer(capacity: data.count + 1)
                        buf.writeBytes(data)
                        buf.writeString("\n")
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        break
                    }

                    // Fast path: build JSON string directly instead of JSONSerialization
                    let escaped = escapeJSON(event.text)
                    let line = "{\"model\":\"\(modelName)\",\"message\":{\"role\":\"assistant\",\"content\":\"\(escaped)\"},\"done\":false}\n"
                    var buf = ByteBufferAllocator().buffer(capacity: line.utf8.count)
                    buf.writeString(line)
                    // Fix 4: dispatch writes onto the event loop.
                    self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                }
                self.writeOnLoop(ctx, .end(nil), flush: true)
            } else {
                var fullContent = ""
                for await event in tokenStream {
                    if event.isEnd { break }
                    fullContent += event.text
                }
                fullContent = StructuredOutput.coerce(fullContent, format: respFormat)

                let totalNs = Int64((CFAbsoluteTimeGetCurrent() - requestStart) * 1_000_000_000)
                let stats = getStats()
                let prefillNs = Int64((stats?.prefillTime ?? 0) * 1_000_000_000)
                let decodeNs = Int64((stats?.decodeTime ?? 0) * 1_000_000_000)
                let response: [String: Any] = [
                    "model": modelName,
                    "message": ["role": "assistant", "content": fullContent],
                    "done": true,
                    "total_duration": totalNs,
                    "prompt_eval_count": stats?.promptTokens ?? 0,
                    "prompt_eval_duration": prefillNs,
                    "eval_count": stats?.generatedTokens ?? 0,
                    "eval_duration": decodeNs,
                ]
                self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
            }
        }
    }

    // MARK: - Ollama: POST /api/generate

    private func handleOllamaGenerate(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }

        let request: ServerGenerateRequest
        do {
            request = try ServerParsing.ollamaGenerateRequest(from: json)
        } catch let error as ServerRequestError {
            sendJSON(context: context, status: .badRequest, body: ["error": error.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid request"])
            return
        }

        // Fix 7: Model validation.
        if let requestedModel = request.requestedModel,
           requestedModel != "unknown",
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        do { try validateMediaShape(request.media) } catch let e as MediaDecodeError {
            sendJSON(context: context,
                     status: HTTPResponseStatus(statusCode: e.httpStatus),
                     body: ["error": e.message])
            return
        } catch {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid media payload"])
            return
        }

        guard requireModel(context: context) else { return }
        touchKeepAlive(request.keepAlive)

        let modelName = engine.modelName ?? request.requestedModel ?? "unknown"

        // Multimodal: validate and decode media.
        let decodedMedia: DecodedMedia?
        if request.media.isEmpty {
            decodedMedia = nil
        } else {
            do {
                decodedMedia = try decodeMediaForRequest(request.media)
            } catch let error as MediaDecodeError {
                sendJSON(
                    context: context,
                    status: HTTPResponseStatus(statusCode: error.httpStatus),
                    body: ["error": error.message]
                )
                return
            } catch {
                sendJSON(context: context, status: .internalServerError,
                         body: ["error": "Failed to decode media payload"])
                return
            }
        }

        if let media = decodedMedia, media.audioPath != nil,
           !audioRoutesNative(media) {
            // Audio path: bridge through Python (entire request, even with
            // image). Native Swift+MLX audio (KRILL_NATIVE_AUDIO=1) instead
            // falls through to the native generate path below.
            let messages: [[String: String]] = [
                ["role": "user", "content": request.prompt]
            ]
            handleBridgeChat(
                context: context,
                messages: messages,
                media: media,
                maxTokens: request.maxTokens,
                bridgeStyle: .ollamaGenerate,
                requestStream: request.stream,
                promptOverride: request.prompt,
                systemOverride: request.system
            )
            return
        }

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let imageData = Self.loadDataIfPath(decodedMedia?.imagePath)
        let audioData = Self.loadDataIfPath(decodedMedia?.audioPath)
        let mediaCopy = decodedMedia

        let requestStart = CFAbsoluteTimeGetCurrent()
        let wantStream = request.stream

        let respFormat = request.responseFormat
        let genSystem: String? = respFormat.map { fmt in
            (request.system.map { $0 + "\n\n" } ?? "")
                + StructuredOutput.systemPrompt(for: fmt)
        } ?? request.system
        let (ogParams, ogMax, ogCtx) = applyModelParams(
            sampling: request.sampling.samplingParams,
            maxTokens: request.maxTokens,
            contextLimit: request.contextLimit ?? defaultContextLimit)
        if wantStream {
            context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming(cors: self.corsHeaders()))), promise: nil)
        }
        Task {
            defer { mediaCopy?.cleanup() }
            if wantStream {
                guard await self.tryEnterQueue() else {
                    self.writeNDJSON(ctx, ["error": "server busy: max queue exceeded",
                                           "done": true])
                    self.writeOnLoop(ctx, .end(nil), flush: true)
                    return
                }
            } else {
                guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            }
            defer { self.leaveQueue() }
            let (tokenStream, getStats) = eng.generate(
                prompt: request.prompt,
                systemPrompt: genSystem,
                params: ogParams,
                maxTokens: ogMax,
                imageData: imageData,
                audioData: audioData,
                contextLimit: ogCtx)

            if request.stream {
                var firstTokenTime: Double?
                var generatedCount = 0
                for await event in tokenStream {
                    if !event.isEnd && firstTokenTime == nil {
                        firstTokenTime = CFAbsoluteTimeGetCurrent()
                    }
                    if !event.isEnd { generatedCount += 1 }

                    if event.isEnd {
                        // Final chunk with Ollama-compatible timing fields
                        let totalNs = Int64((CFAbsoluteTimeGetCurrent() - requestStart) * 1_000_000_000)
                        let stats = getStats()
                        let prefillNs = Int64((stats?.prefillTime ?? 0) * 1_000_000_000)
                        let decodeNs = Int64((stats?.decodeTime ?? 0) * 1_000_000_000)
                        let promptTokens = stats?.promptTokens ?? 0
                        var finalChunk: [String: Any] = [
                            "model": modelName,
                            "response": "",
                            "done": true,
                            "total_duration": totalNs,
                            "prompt_eval_count": promptTokens,
                            "prompt_eval_duration": prefillNs,
                            "eval_count": generatedCount,
                            "eval_duration": decodeNs,
                        ]
                        if let ftt = firstTokenTime {
                            finalChunk["ttft_ns"] = Int64((ftt - requestStart) * 1_000_000_000)
                        }
                        let data = try! JSONSerialization.data(withJSONObject: finalChunk)
                        var buf = ByteBufferAllocator().buffer(capacity: data.count + 1)
                        buf.writeBytes(data)
                        buf.writeString("\n")
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        break
                    }

                    // Fast path: build JSON string directly instead of JSONSerialization
                    let escaped = escapeJSON(event.text)
                    let line = "{\"model\":\"\(modelName)\",\"response\":\"\(escaped)\",\"done\":false}\n"
                    var buf = ByteBufferAllocator().buffer(capacity: line.utf8.count)
                    buf.writeString(line)
                    self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                }
                self.writeOnLoop(ctx, .end(nil), flush: true)
            } else {
                var fullResponse = ""
                for await event in tokenStream {
                    if event.isEnd { break }
                    fullResponse += event.text
                }
                fullResponse = StructuredOutput.coerce(fullResponse, format: respFormat)

                let totalNs = Int64((CFAbsoluteTimeGetCurrent() - requestStart) * 1_000_000_000)
                let stats = getStats()
                let prefillNs = Int64((stats?.prefillTime ?? 0) * 1_000_000_000)
                let decodeNs = Int64((stats?.decodeTime ?? 0) * 1_000_000_000)
                let response: [String: Any] = [
                    "model": modelName,
                    "response": fullResponse,
                    "done": true,
                    "total_duration": totalNs,
                    "prompt_eval_count": stats?.promptTokens ?? 0,
                    "prompt_eval_duration": prefillNs,
                    "eval_count": stats?.generatedTokens ?? 0,
                    "eval_duration": decodeNs,
                ]
                self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop, status: .ok, body: response)
            }
        }
    }

    // MARK: - Ollama: GET /api/tags

    private func handleOllamaTags(context: ChannelHandlerContext) {
        let models = registry.listModels().map { manifest -> [String: Any] in
            [
                "name": manifest.name,
                "size": manifest.sizeBytes,
                "details": [
                    "family": manifest.family.rawValue,
                    "parameter_size": manifest.params,
                    "quantization_level": manifest.quant
                ]
            ]
        }
        sendJSON(context: context, status: .ok, body: ["models": models])
    }

    // MARK: - Embeddings: /api/embed, /api/embeddings, /v1/embeddings

    private enum EmbedStyle {
        case openAI       // POST /v1/embeddings
        case ollamaBatch  // POST /api/embed
        case ollamaLegacy // POST /api/embeddings (single "prompt")
    }

    /// Extract the input texts for an embeddings request across the three
    /// shapes (`input` string|[string] for OpenAI/Ollama-batch; `prompt`
    /// string for the Ollama legacy endpoint).
    private func embedInputs(_ json: [String: Any], style: EmbedStyle) -> [String]? {
        if style == .ollamaLegacy {
            if let p = json["prompt"] as? String { return [p] }
            return nil
        }
        if let s = json["input"] as? String { return [s] }
        if let arr = json["input"] as? [Any] {
            let strs = arr.compactMap { $0 as? String }
            return strs.count == arr.count ? strs : nil
        }
        return nil
    }

    private func handleEmbeddings(context: ChannelHandlerContext,
                                  body: ByteBuffer, style: EmbedStyle) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }
        guard let name = ollamaModelName(json) else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'model' field"])
            return
        }
        guard let inputs = embedInputs(json, style: style), !inputs.isEmpty else {
            sendJSON(context: context, status: .badRequest, body: [
                "error": style == .ollamaLegacy
                    ? "Missing 'prompt' (string)"
                    : "Missing 'input' (string or array of strings)"
            ])
            return
        }
        guard let manifest = registry.getModel(name) else {
            sendJSON(context: context, status: .notFound, body: [
                "error": "embedding model '\(name)' not found. Install with: krillm pull \(name)"
            ])
            return
        }
        guard manifest.family == .bert else {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "'\(name)' (family \(manifest.family.rawValue)) is not a sentence-embedding model. Use a dedicated embedding model, e.g. krillm pull bge-small-en"
            ])
            return
        }

        let dir = registry.modelPath(name)
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let embed = embedEngine
        let modelName = manifest.name

        let started = CFAbsoluteTimeGetCurrent()
        Task {
            guard await self.enterQueueOr503(ctx, eventLoop) else { return }
            defer { self.leaveQueue() }
            do {
                try await embed.load(directory: dir)
                let result = try embed.embed(inputs)
                let totalNs = Int64((CFAbsoluteTimeGetCurrent() - started) * 1_000_000_000)
                let response: [String: Any]
                switch style {
                case .openAI:
                    response = [
                        "object": "list",
                        "model": modelName,
                        "data": result.vectors.enumerated().map { i, v in
                            ["object": "embedding", "index": i, "embedding": v]
                        },
                        "usage": [
                            "prompt_tokens": result.promptTokens,
                            "total_tokens": result.promptTokens,
                        ],
                    ]
                case .ollamaBatch:
                    response = [
                        "model": modelName,
                        "embeddings": result.vectors,
                        "total_duration": totalNs,
                        "prompt_eval_count": result.promptTokens,
                    ]
                case .ollamaLegacy:
                    response = ["embedding": result.vectors.first ?? []]
                }
                self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                    status: .ok, body: response)
            } catch {
                let msg = String(describing: error).prefix(300)
                self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                    status: .internalServerError,
                                    body: ["error": "embedding failed: \(msg)"])
            }
        }
    }

    // MARK: - Ollama discovery: GET /api/ps

    /// Extract the model name from an Ollama request body (`model` or the
    /// legacy `name` field).
    private func ollamaModelName(_ json: [String: Any]?) -> String? {
        guard let json else { return nil }
        if let m = json["model"] as? String, !m.isEmpty { return m }
        if let n = json["name"] as? String, !n.isEmpty { return n }
        return nil
    }

    private func handleOllamaPS(context: ChannelHandlerContext) {
        guard engine.isLoaded, let modelName = engine.modelName else {
            sendJSON(context: context, status: .ok, body: ["models": [[String: Any]]()])
            return
        }
        let manifest = registry.getModel(modelName)
        let size = manifest?.sizeBytes ?? 0
        let fallback = (engine.loadedAt ?? Date())
            .addingTimeInterval(TimeInterval(KrillConfig.load().idleTimeout))
        let ka = keepAlive
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        Task {
            // Prefer the live keep-alive deadline; fall back to idle timeout.
            let expiresAt = (await ka.expiresAt()) ?? fallback
            let entry = OllamaCompat.psEntry(
                manifest: manifest, modelName: modelName,
                sizeBytes: size, expiresAt: expiresAt)
            self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                status: .ok, body: ["models": [entry]])
        }
    }

    // MARK: - Ollama discovery: POST /api/show

    private func handleOllamaShow(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let name = ollamaModelName(parseJSON(body)) else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'model' field"])
            return
        }
        let verbose = (parseJSON(body)?["verbose"] as? Bool) ?? false
        guard let manifest = registry.getModel(name) else {
            sendJSON(context: context, status: .notFound,
                     body: ["error": "model '\(name)' not found"])
            return
        }
        sendJSON(context: context, status: .ok,
                 body: OllamaCompat.showPayload(for: manifest, verbose: verbose))
    }

    // MARK: - OpenAI: GET /v1/models/{id}

    private func handleModelDetail(context: ChannelHandlerContext, id: String) {
        guard let manifest = registry.getModel(id) else {
            sendJSON(context: context, status: .notFound,
                     body: ["error": "model '\(id)' not found"])
            return
        }
        sendJSON(context: context, status: .ok, body: [
            "id": manifest.name,
            "object": "model",
            "owned_by": "local",
            "created": Int(manifest.pulledAt.timeIntervalSince1970),
        ])
    }

    // MARK: - Ollama lifecycle: DELETE /api/delete

    private func handleOllamaDelete(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let name = ollamaModelName(parseJSON(body)) else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'model' field"])
            return
        }
        guard registry.hasModel(name) else {
            sendJSON(context: context, status: .notFound,
                     body: ["error": "model '\(name)' not found"])
            return
        }
        do {
            try registry.removeModel(name)
            sendJSON(context: context, status: .ok, body: ["status": "success"])
        } catch {
            sendJSON(context: context, status: .internalServerError,
                     body: ["error": "failed to delete '\(name)': \(error)"])
        }
    }

    // MARK: - Ollama lifecycle: POST /api/copy

    private func handleOllamaCopy(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body),
              let source = (json["source"] as? String) ?? (json["src"] as? String),
              let destination = (json["destination"] as? String) ?? (json["dst"] as? String),
              !source.isEmpty, !destination.isEmpty else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'source' or 'destination'"])
            return
        }
        guard Registry.isValidModelName(source),
              Registry.isValidModelName(destination) else {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Invalid model name: must not contain path separators, '..', or a leading '.'"
            ])
            return
        }
        guard let srcManifest = registry.getModel(source) else {
            sendJSON(context: context, status: .notFound,
                     body: ["error": "model '\(source)' not found"])
            return
        }
        let fm = FileManager.default
        let srcDir = registry.modelPath(source)
        let dstDir = registry.modelPath(destination)
        do {
            if fm.fileExists(atPath: dstDir.path) {
                try fm.removeItem(at: dstDir)
            }
            if fm.fileExists(atPath: srcDir.path) {
                try fm.copyItem(at: srcDir, to: dstDir)
            }
            let copied = ModelManifest(
                name: destination, family: srcManifest.family,
                params: srcManifest.params, quant: srcManifest.quant,
                source: srcManifest.source, context: srcManifest.context,
                files: srcManifest.files, draftPair: srcManifest.draftPair,
                chatTemplate: srcManifest.chatTemplate,
                sizeBytes: srcManifest.sizeBytes, pulledAt: Date(),
                // Preserve the source's Modelfile PARAMETER/TEMPLATE/SYSTEM
                // overrides — `/api/copy` must match `krillm cp` (PR #18
                // rereview); dropping them silently de-customized the copy.
                overrides: srcManifest.overrides)
            try registry.saveManifest(copied)
            sendJSON(context: context, status: .ok, body: ["status": "success"])
        } catch {
            sendJSON(context: context, status: .internalServerError,
                     body: ["error": "copy failed: \(error)"])
        }
    }

    // MARK: - Ollama lifecycle: POST /api/blobs/:digest

    /// Digest blob store backing `ollama create` uploads. `/api/create`
    /// itself is Phase 2 (Modelfile); this provides shape-correct
    /// HEAD/POST so blob-precheck clients don't hard-fail.
    private func blobStorePath(_ digest: String) -> URL {
        let safe = digest.replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "..", with: "_")
        return registry.modelsDir.appendingPathComponent("blobs-sha")
            .appendingPathComponent(safe)
    }

    private func handleBlob(context: ChannelHandlerContext, method: HTTPMethod,
                            digest: String, body: ByteBuffer) {
        let path = blobStorePath(digest)
        switch method {
        case .HEAD:
            let exists = FileManager.default.fileExists(atPath: path.path)
            sendJSON(context: context, status: exists ? .ok : .notFound, body: [:])
        case .POST:
            do {
                try FileManager.default.createDirectory(
                    at: path.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                var buf = body
                let bytes = buf.readBytes(length: buf.readableBytes) ?? []
                try Data(bytes).write(to: path)
                sendJSON(context: context, status: .created, body: [:])
            } catch {
                sendJSON(context: context, status: .internalServerError,
                         body: ["error": "blob store failed: \(error)"])
            }
        default:
            sendJSON(context: context, status: .methodNotAllowed,
                     body: ["error": "blobs supports HEAD and POST"])
        }
    }

    // MARK: - Ollama lifecycle: POST /api/pull

    private func handleOllamaPull(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body), let name = ollamaModelName(json) else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'model' field"])
            return
        }
        let stream = (json["stream"] as? Bool) ?? true
        guard let resolved = AliasMap.resolve(name) else {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Cannot resolve '\(name)'. Use a known alias or an org/repo HuggingFace path."
            ])
            return
        }
        // Defense in depth: a crafted ref like "x/.." resolves to a name
        // that would escape the registry root. Reject before any pull.
        guard Registry.isValidModelName(resolved.name) else {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Invalid model name: must not contain path separators, '..', or a leading '.'"
            ])
            return
        }

        let reg = registry
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context

        let ndjson: @Sendable ([String: Any]) -> Void = { obj in
            guard let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
            var b = ByteBufferAllocator().buffer(capacity: data.count + 1)
            b.writeBytes(data)
            b.writeString("\n")
            self.writeOnLoop(ctx, .body(.byteBuffer(b)), flush: true)
        }

        if stream {
            context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming(cors: self.corsHeaders()))), promise: nil)
        }

        Task {
            do {
                let puller = Puller(registry: reg)
                let progress: Puller.ProgressHandler = { done, total, file in
                    if stream {
                        if file == "done" {
                            ndjson(["status": "verifying sha256 digest"])
                        } else {
                            ndjson([
                                "status": "downloading \(file)",
                                "digest": file,
                                "total": total,
                                "completed": done,
                            ])
                        }
                    }
                }
                if stream { ndjson(["status": "pulling manifest"]) }
                _ = try await puller.pull(resolved, force: false, progress: progress)
                if stream {
                    ndjson(["status": "success"])
                    self.writeOnLoop(ctx, .end(nil), flush: true)
                } else {
                    self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                        status: .ok, body: ["status": "success"])
                }
            } catch {
                let msg = String(describing: error).prefix(300)
                if stream {
                    ndjson(["error": "pull failed: \(msg)"])
                    self.writeOnLoop(ctx, .end(nil), flush: true)
                } else {
                    self.sendJSONOnLoop(context: ctx, eventLoop: eventLoop,
                                        status: .internalServerError,
                                        body: ["error": "pull failed: \(msg)"])
                }
            }
        }
    }

    // MARK: - Ollama lifecycle: POST /api/create (WS-C)

    private func handleOllamaCreate(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body), let name = ollamaModelName(json) else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'model' field"])
            return
        }
        // Accept the `modelfile` string form (most common). The structured
        // {from, system, parameters,...} form maps onto the same parser by
        // synthesizing a Modelfile.
        var mfText = json["modelfile"] as? String
        if mfText == nil, let from = json["from"] as? String {
            var lines = ["FROM \(from)"]
            if let s = json["system"] as? String { lines.append("SYSTEM \(s)") }
            if let p = json["parameters"] as? [String: Any] {
                for (k, v) in p { lines.append("PARAMETER \(k) \(v)") }
            }
            if let t = json["template"] as? String { lines.append("TEMPLATE \(t)") }
            mfText = lines.joined(separator: "\n")
        }
        guard let mfText else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'modelfile' or 'from'"])
            return
        }
        let stream = (json["stream"] as? Bool) ?? true
        let reg = registry
        nonisolated(unsafe) let ctx = context

        let ndjson: @Sendable ([String: Any]) -> Void = { obj in
            guard let d = try? JSONSerialization.data(withJSONObject: obj) else { return }
            var b = ByteBufferAllocator().buffer(capacity: d.count + 1)
            b.writeBytes(d); b.writeString("\n")
            self.writeOnLoop(ctx, .body(.byteBuffer(b)), flush: true)
        }

        do {
            let mf = try ModelfileParser.parse(mfText)
            if stream {
                context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming(cors: self.corsHeaders()))), promise: nil)
                ndjson(["status": "reading modelfile"])
                ndjson(["status": "creating model layer"])
            }
            _ = try reg.createModel(name: name, from: mf)
            if stream {
                if let w = mf.adapterWarning { ndjson(["status": "warning: \(w)"]) }
                ndjson(["status": "success"])
                writeOnLoop(ctx, .end(nil), flush: true)
            } else {
                sendJSON(context: context, status: .ok, body: ["status": "success"])
            }
        } catch {
            let msg = String(describing: error).prefix(300)
            if stream {
                ndjson(["error": "\(msg)"])
                writeOnLoop(ctx, .end(nil), flush: true)
            } else {
                sendJSON(context: context, status: .badRequest,
                         body: ["error": "\(msg)"])
            }
        }
    }

    // MARK: - Model Management: POST /v1/models/load

    private func handleLoadModel(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body),
              let modelName = json["model"] as? String else {
            sendJSON(context: context, status: .badRequest,
                     body: ["error": "Missing 'model' field"])
            return
        }

        // Only allow loading models from the registry — no arbitrary filesystem paths.
        let reg = registry
        guard reg.hasModel(modelName) else {
            sendJSON(context: context, status: .notFound,
                     body: ["error": "Model '\(modelName)' not found. Install with: krillm pull \(modelName)"])
            return
        }
        let modelDir = reg.modelPath(modelName)

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
            do {
                try await eng.swap(modelDirectory: modelDir)
                let model = eng.modelName ?? modelName
                let family = eng.family ?? "unknown"
                eventLoop.execute {
                    self.sendJSON(context: ctx, status: .ok, body: [
                        "status": "loaded",
                        "model": model,
                        "family": family
                    ])
                }
            } catch {
                let errMsg = String(describing: error).prefix(200)
                eventLoop.execute {
                    self.sendJSON(context: ctx, status: .internalServerError,
                                  body: ["error": "Failed to load model: \(errMsg)"])
                }
            }
        }
    }

    // MARK: - Model Management: POST /v1/models/unload

    private func handleUnloadModel(context: ChannelHandlerContext) {
        engine.unload()
        sendJSON(context: context, status: .ok, body: [
            "status": "unloaded"
        ])
    }

    // MARK: - Status: GET /v1/status

    private func handleStatus(context: ChannelHandlerContext) {
        let serverUptime = Date().timeIntervalSince(startedAt)

        // Process memory
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let _ = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        let residentMB = Double(info.resident_size) / 1_048_576

        // Installed models
        let installed = registry.listModels().map { $0.name }

        var response: [String: Any] = [
            "status": engine.isLoaded ? "ready" : "idle",
            "model_loaded": engine.isLoaded,
            "uptime_seconds": Int(serverUptime),
            "memory_mb": Int(residentMB),
            "installed_models": installed,
            "version": "0.2.0"
        ]

        if engine.isLoaded {
            response["model"] = engine.modelName ?? "unknown"
            response["family"] = engine.family ?? "unknown"
            if let loadedAt = engine.loadedAt {
                response["model_loaded_at"] = ISO8601DateFormatter().string(from: loadedAt)
                response["model_uptime_seconds"] = Int(Date().timeIntervalSince(loadedAt))
            }
        }

        sendJSON(context: context, status: .ok, body: response)
    }

    // MARK: - Helpers

    /// Write a response part on the channel's event loop.
    /// If already on the event loop, writes immediately; otherwise dispatches via execute.
    private func writeOnLoop(_ context: ChannelHandlerContext, _ part: HTTPServerResponsePart, flush: Bool = false) {
        if context.eventLoop.inEventLoop {
            if flush {
                context.writeAndFlush(wrapOutboundOut(part), promise: nil)
            } else {
                context.write(wrapOutboundOut(part), promise: nil)
            }
        } else {
            nonisolated(unsafe) let ctx = context
            context.eventLoop.execute {
                if flush {
                    ctx.writeAndFlush(self.wrapOutboundOut(part), promise: nil)
                } else {
                    ctx.write(self.wrapOutboundOut(part), promise: nil)
                }
            }
        }
    }

    private func parseJSON(_ buffer: ByteBuffer) -> [String: Any]? {
        ServerParsing.jsonObject(from: buffer)
    }

    private func sendJSON(context: ChannelHandlerContext, status: HTTPResponseStatus, body: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: body) else {
            context.close(promise: nil)
            return
        }
        sendJSONData(context: context, status: status, data: data)
    }

    private func sendJSONOnLoop(
        context: ChannelHandlerContext,
        eventLoop: EventLoop,
        status: HTTPResponseStatus,
        body: [String: Any]
    ) {
        guard let data = try? JSONSerialization.data(withJSONObject: body) else {
            writeOnLoop(context, .end(nil), flush: true)
            return
        }
        nonisolated(unsafe) let ctx = context
        eventLoop.execute {
            self.sendJSONData(context: ctx, status: status, data: data)
        }
    }

    private func sendJSONData(context: ChannelHandlerContext, status: HTTPResponseStatus, data: Data) {
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(data.count)")
        for (k, v) in corsHeaders() { headers.add(name: k, value: v) }
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)

        context.write(wrapOutboundOut(.head(head)), promise: nil)
        var buf = ByteBufferAllocator().buffer(capacity: data.count)
        buf.writeBytes(data)
        context.write(wrapOutboundOut(.body(.byteBuffer(buf))), promise: nil)
        context.writeAndFlush(wrapOutboundOut(.end(nil)), promise: nil)
    }
}

// MARK: - SSE Formatting

private func sseChunk(id: String, content: String?, finishReason: String?) -> String {
    var delta: [String: Any] = [:]
    if let content { delta["content"] = content }
    delta["role"] = "assistant"

    var choice: [String: Any] = ["index": 0, "delta": delta]
    if let reason = finishReason {
        choice["finish_reason"] = reason
        choice["delta"] = [String: Any]()
    }

    let payload: [String: Any] = [
        "id": id,
        "object": "chat.completion.chunk",
        "created": Int(Date().timeIntervalSince1970),
        "choices": [choice]
    ]

    guard let data = try? JSONSerialization.data(withJSONObject: payload),
          let json = String(data: data, encoding: .utf8) else {
        return ""
    }
    return "data: \(json)\n\n"
}

/// Escape a string for safe embedding inside a JSON string value.
/// Handles backslash, double-quote, and control characters.
private func escapeJSON(_ s: String) -> String {
    var result = ""
    result.reserveCapacity(s.utf8.count)
    for c in s {
        switch c {
        case "\"": result += "\\\""
        case "\\": result += "\\\\"
        case "\n": result += "\\n"
        case "\r": result += "\\r"
        case "\t": result += "\\t"
        default:
            if c.asciiValue != nil && c.asciiValue! < 0x20 {
                result += String(format: "\\u%04x", c.asciiValue!)
            } else {
                result.append(c)
            }
        }
    }
    return result
}
