import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1
import Logging
import KLMEngine
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
    private let logger = Logger(label: "krillm.server")

    public init(host: String = "127.0.0.1", port: Int = 11435,
                engine: InferenceEngine, registry: Registry) {
        self.host = host
        self.port = port
        self.engine = engine
        self.registry = registry
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
                        HTTPHandler(engine: self.engine, registry: self.registry)
                    )
                }
            }
            .childChannelOption(.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(.maxMessagesPerRead, value: 1)

        let channel = try await bootstrap.bind(host: host, port: port).get()
        logger.info("KrillLM server listening on http://\(host):\(port)")
        print("KrillLM server listening on http://\(host):\(port)")
        print("OpenAI API: http://\(host):\(port)/v1/chat/completions")
        print("Ollama API: http://\(host):\(port)/api/chat")
        print("Press Ctrl+C to stop.")

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
    private let logger = Logger(label: "krillm.http")
    private let startedAt = Date()

    private let maxBodySize = 10 * 1024 * 1024  // 10 MB

    private var requestHead: HTTPRequestHead?
    private var body: ByteBuffer = ByteBuffer()

    init(engine: InferenceEngine, registry: Registry) {
        self.engine = engine
        self.registry = registry
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

        switch (head.method, path) {
        // OpenAI endpoints
        case (.POST, "/v1/chat/completions"):
            guard requireModel(context: context) else { return }
            handleChatCompletions(context: context, body: body)
        case (.POST, "/v1/completions"):
            guard requireModel(context: context) else { return }
            handleCompletions(context: context, body: body)
        case (.GET, "/v1/models"):
            handleModels(context: context)

        // Model management
        case (.POST, "/v1/models/load"):
            handleLoadModel(context: context, body: body)
        case (.POST, "/v1/models/unload"):
            handleUnloadModel(context: context)

        // Status
        case (.GET, "/v1/status"):
            handleStatus(context: context)

        // Ollama endpoints
        case (.POST, "/api/chat"):
            guard requireModel(context: context) else { return }
            handleOllamaChat(context: context, body: body)
        case (.POST, "/api/generate"):
            guard requireModel(context: context) else { return }
            handleOllamaGenerate(context: context, body: body)
        case (.GET, "/api/tags"):
            handleOllamaTags(context: context)

        // Health and metrics
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

        let messages = json["messages"] as? [[String: Any]] ?? []
        let stream = json["stream"] as? Bool ?? false
        let maxTokens = json["max_tokens"] as? Int ?? 512

        // Fix 7: Model validation — reject if a specific model is requested but doesn't match loaded model.
        if let requestedModel = json["model"] as? String,
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        // Fix 7: Parse additional sampling parameters.
        let temperature = json["temperature"] as? Double ?? 0.0
        let topP = json["top_p"] as? Double ?? 1.0
        let topK = json["top_k"] as? Int ?? 0
        let params = SamplingParams(
            temperature: Float(temperature),
            topP: Float(topP),
            topK: topK
        )

        // Fix 7: Build full conversation prompt from all messages.
        let systemPrompt = messages.first(where: { ($0["role"] as? String) == "system" })?["content"] as? String
        let conversationParts: [String] = messages.compactMap { msg in
            guard let role = msg["role"] as? String,
                  let content = msg["content"] as? String,
                  role != "system" else { return nil }
            return "\(role): \(content)"
        }
        let prompt = conversationParts.joined(separator: "\n")

        guard !prompt.isEmpty else {
            sendJSON(context: context, status: .badRequest, body: ["error": "No user message"])
            return
        }

        if stream {
            handleStreamingCompletion(
                context: context, prompt: prompt, systemPrompt: systemPrompt,
                params: params, maxTokens: maxTokens)
        } else {
            handleNonStreamingCompletion(
                context: context, prompt: prompt, systemPrompt: systemPrompt,
                params: params, maxTokens: maxTokens)
        }
    }

    private func handleStreamingCompletion(
        context: ChannelHandlerContext,
        prompt: String, systemPrompt: String?,
        params: SamplingParams, maxTokens: Int
    ) {
        // Send SSE headers
        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "text/event-stream")
        headers.add(name: "Cache-Control", value: "no-cache")
        headers.add(name: "Connection", value: "keep-alive")
        let head = HTTPResponseHead(version: .http1_1, status: .ok, headers: headers)
        context.write(wrapOutboundOut(.head(head)), promise: nil)

        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
            let (tokenStream, _) = eng.generate(
                prompt: prompt, systemPrompt: systemPrompt,
                params: params, maxTokens: maxTokens)

            let id = "chatcmpl-\(UUID().uuidString.prefix(8))"

            for await event in tokenStream {
                if event.isEnd {
                    let chunk = sseChunk(id: id, content: nil, finishReason: "stop")
                    var buf = ctx.channel.allocator.buffer(capacity: chunk.utf8.count)
                    buf.writeString(chunk)
                    self.writeOnLoop(ctx, .body(.byteBuffer(buf)))

                    // Send [DONE]
                    let done = "data: [DONE]\n\n"
                    var doneBuf = ctx.channel.allocator.buffer(capacity: done.utf8.count)
                    doneBuf.writeString(done)
                    self.writeOnLoop(ctx, .body(.byteBuffer(doneBuf)))
                    break
                }

                let chunk = sseChunk(id: id, content: event.text, finishReason: nil)
                var buf = ctx.channel.allocator.buffer(capacity: chunk.utf8.count)
                buf.writeString(chunk)
                self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
            }

            self.writeOnLoop(ctx, .end(nil), flush: true)
        }
    }

    private func handleNonStreamingCompletion(
        context: ChannelHandlerContext,
        prompt: String, systemPrompt: String?,
        params: SamplingParams, maxTokens: Int
    ) {
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
            let (tokenStream, getStats) = eng.generate(
                prompt: prompt, systemPrompt: systemPrompt,
                params: params, maxTokens: maxTokens)

            var fullContent = ""
            for await event in tokenStream {
                if event.isEnd { break }
                fullContent += event.text
            }

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
            eventLoop.execute {
                self.sendJSON(context: ctx, status: .ok, body: response)
            }
        }
    }

    // MARK: - OpenAI: POST /v1/completions

    private func handleCompletions(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }

        let prompt = json["prompt"] as? String ?? ""
        let maxTokens = json["max_tokens"] as? Int ?? 256
        let temperature = json["temperature"] as? Double ?? 0.0
        let params = SamplingParams(temperature: Float(temperature))

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
            let (tokenStream, getStats) = eng.generate(
                prompt: prompt, params: params, maxTokens: maxTokens)

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
            eventLoop.execute {
                self.sendJSON(context: ctx, status: .ok, body: response)
            }
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
        var buf = context.channel.allocator.buffer(capacity: data.count)
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

        let messages = json["messages"] as? [[String: Any]] ?? []
        let stream = json["stream"] as? Bool ?? true

        // Fix 7: Model validation — reject if a specific model is requested but doesn't match loaded model.
        if let requestedModel = json["model"] as? String,
           requestedModel != "unknown",
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        // Fix 7: Build full conversation prompt from all messages.
        let systemPrompt = messages.first(where: { ($0["role"] as? String) == "system" })?["content"] as? String
        let conversationParts: [String] = messages.compactMap { msg in
            guard let role = msg["role"] as? String,
                  let content = msg["content"] as? String,
                  role != "system" else { return nil }
            return "\(role): \(content)"
        }
        let prompt = conversationParts.joined(separator: "\n")

        guard !prompt.isEmpty else {
            sendJSON(context: context, status: .badRequest, body: ["error": "No user message"])
            return
        }

        // Fix 7: Parse additional sampling parameters.
        let options = json["options"] as? [String: Any] ?? [:]
        let temperature = options["temperature"] as? Double ?? 0.0
        let topP = options["top_p"] as? Double ?? 1.0
        let topK = options["top_k"] as? Int ?? 0
        let params = SamplingParams(
            temperature: Float(temperature),
            topP: Float(topP),
            topK: topK
        )

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let modelName = engine.modelName ?? json["model"] as? String ?? "unknown"

        if stream {
            // Fix 5: Send HTTP response head before streaming body chunks.
            var streamHeaders = HTTPHeaders()
            streamHeaders.add(name: "Content-Type", value: "application/x-ndjson")
            streamHeaders.add(name: "Transfer-Encoding", value: "chunked")
            let streamHead = HTTPResponseHead(version: .http1_1, status: .ok, headers: streamHeaders)
            context.write(wrapOutboundOut(.head(streamHead)), promise: nil)
        }

        Task {
            let (tokenStream, _) = eng.generate(
                prompt: prompt, systemPrompt: systemPrompt, params: params, maxTokens: 2048)

            if stream {
                for await event in tokenStream {
                    let chunk: [String: Any] = [
                        "model": modelName,
                        "message": ["role": "assistant", "content": event.text],
                        "done": event.isEnd
                    ]
                    let data = try! JSONSerialization.data(withJSONObject: chunk)
                    var buf = ctx.channel.allocator.buffer(capacity: data.count + 1)
                    buf.writeBytes(data)
                    buf.writeString("\n")
                    // Fix 4: dispatch writes onto the event loop.
                    self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                    if event.isEnd { break }
                }
                self.writeOnLoop(ctx, .end(nil), flush: true)
            } else {
                var fullContent = ""
                for await event in tokenStream {
                    if event.isEnd { break }
                    fullContent += event.text
                }
                let response: [String: Any] = [
                    "model": modelName,
                    "message": ["role": "assistant", "content": fullContent],
                    "done": true
                ]
                eventLoop.execute {
                    self.sendJSON(context: ctx, status: .ok, body: response)
                }
            }
        }
    }

    // MARK: - Ollama: POST /api/generate

    private func handleOllamaGenerate(context: ChannelHandlerContext, body: ByteBuffer) {
        guard let json = parseJSON(body) else {
            sendJSON(context: context, status: .badRequest, body: ["error": "Invalid JSON"])
            return
        }

        let prompt = json["prompt"] as? String ?? ""
        let system = json["system"] as? String

        // Fix 7: Model validation.
        if let requestedModel = json["model"] as? String,
           requestedModel != "unknown",
           let loadedModel = engine.modelName,
           requestedModel != loadedModel {
            sendJSON(context: context, status: .badRequest, body: [
                "error": "Requested model '\(requestedModel)' is not loaded. Currently loaded: '\(loadedModel)'."
            ])
            return
        }

        let modelName = engine.modelName ?? json["model"] as? String ?? "unknown"

        // Fix 7: Parse additional sampling parameters.
        let options = json["options"] as? [String: Any] ?? [:]
        let temperature = options["temperature"] as? Double ?? 0.0
        let topP = options["top_p"] as? Double ?? 1.0
        let topK = options["top_k"] as? Int ?? 0
        let params = SamplingParams(
            temperature: Float(temperature),
            topP: Float(topP),
            topK: topK
        )

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
            let (tokenStream, _) = eng.generate(
                prompt: prompt, systemPrompt: system, params: params, maxTokens: 2048)

            var fullResponse = ""
            for await event in tokenStream {
                if event.isEnd { break }
                fullResponse += event.text
            }

            let response: [String: Any] = [
                "model": modelName,
                "response": fullResponse,
                "done": true
            ]
            // Fix 4: dispatch write back onto the event loop.
            eventLoop.execute {
                self.sendJSON(context: ctx, status: .ok, body: response)
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
            context.eventLoop.execute {
                if flush {
                    context.writeAndFlush(self.wrapOutboundOut(part), promise: nil)
                } else {
                    context.write(self.wrapOutboundOut(part), promise: nil)
                }
            }
        }
    }

    private func parseJSON(_ buffer: ByteBuffer) -> [String: Any]? {
        var buf = buffer
        guard let bytes = buf.readBytes(length: buf.readableBytes) else {
            return nil
        }
        let data = Data(bytes)
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    private func sendJSON(context: ChannelHandlerContext, status: HTTPResponseStatus, body: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: body) else {
            context.close(promise: nil)
            return
        }

        var headers = HTTPHeaders()
        headers.add(name: "Content-Type", value: "application/json")
        headers.add(name: "Content-Length", value: "\(data.count)")
        let head = HTTPResponseHead(version: .http1_1, status: status, headers: headers)

        context.write(wrapOutboundOut(.head(head)), promise: nil)
        var buf = context.channel.allocator.buffer(capacity: data.count)
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
