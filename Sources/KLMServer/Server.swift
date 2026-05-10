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

    internal static func _makeHTTPHandlerForTesting(
        engine: InferenceEngine,
        registry: Registry
    ) -> any ChannelHandler & Sendable {
        HTTPHandler(engine: engine, registry: registry)
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

    private let maxBodySize = ServerLimits.maxBodySize

    private var requestHead: HTTPRequestHead?
    private var body: ByteBuffer = ByteBuffer()

    init(engine: InferenceEngine, registry: Registry) {
        self.engine = engine
        self.registry = registry
    }

    /// Best-effort load of file bytes for a path. Returns nil if the path is
    /// nil or the read fails.
    static func loadDataIfPath(_ path: String?) -> Data? {
        guard let path else { return nil }
        return try? Data(contentsOf: URL(fileURLWithPath: path))
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
            handleChatCompletions(context: context, body: body)
        case (.POST, "/v1/completions"):
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
            handleOllamaChat(context: context, body: body)
        case (.POST, "/api/generate"):
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

        if let media = decodedMedia, media.audioPath != nil {
            // Audio (with or without image) -> Python bridge, non-streaming bridge response.
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

        if request.stream {
            handleStreamingCompletion(
                context: context, messages: request.messages,
                params: request.sampling.samplingParams, maxTokens: request.maxTokens,
                media: decodedMedia)
        } else {
            handleNonStreamingCompletion(
                context: context, messages: request.messages,
                params: request.sampling.samplingParams, maxTokens: request.maxTokens,
                media: decodedMedia)
        }
    }

    // MARK: - Media decoding shared helper

    /// Pre-flight checks that don't require a loaded model.
    private func validateMediaShape(_ payload: ServerMediaPayload) throws {
        if payload.images.count > 1 {
            throw MediaDecodeError.tooManyImages
        }
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
        if requestStream && (style == .ollamaChat || style == .ollamaGenerate) {
            context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming())), promise: nil)
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
                    let response: [String: Any] = [
                        "id": "chatcmpl-\(UUID().uuidString.prefix(8))",
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
                            var buf = ctx.channel.allocator.buffer(capacity: data1.count + 1)
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
                            var buf = ctx.channel.allocator.buffer(capacity: data2.count + 1)
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
                            var buf = ctx.channel.allocator.buffer(capacity: data1.count + 1)
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
                            var buf = ctx.channel.allocator.buffer(capacity: data2.count + 1)
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
        media: DecodedMedia? = nil
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
        let imageData = Self.loadDataIfPath(media?.imagePath)
        let mediaCopy = media

        Task {
            defer { mediaCopy?.cleanup() }
            let (tokenStream, _) = eng.generate(
                messages: messages,
                params: params, maxTokens: maxTokens,
                imageData: imageData)

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
        messages: [[String: String]],
        params: SamplingParams, maxTokens: Int,
        media: DecodedMedia? = nil
    ) {
        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let imageData = Self.loadDataIfPath(media?.imagePath)
        let mediaCopy = media

        Task {
            defer { mediaCopy?.cleanup() }
            let (tokenStream, getStats) = eng.generate(
                messages: messages,
                params: params, maxTokens: maxTokens,
                imageData: imageData)

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

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine

        Task {
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

        if let media = decodedMedia, media.audioPath != nil {
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

        let eventLoop = context.eventLoop
        nonisolated(unsafe) let ctx = context
        let eng = engine
        let modelName = engine.modelName ?? request.requestedModel ?? "unknown"
        let imageData = Self.loadDataIfPath(decodedMedia?.imagePath)
        let mediaCopy = decodedMedia

        let requestStart = CFAbsoluteTimeGetCurrent()

        if request.stream {
            context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming())), promise: nil)
        }

        Task {
            defer { mediaCopy?.cleanup() }
            let (tokenStream, getStats) = eng.generate(
                messages: request.messages,
                params: request.sampling.samplingParams,
                maxTokens: request.maxTokens,
                imageData: imageData)

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
                        var buf = ctx.channel.allocator.buffer(capacity: data.count + 1)
                        buf.writeBytes(data)
                        buf.writeString("\n")
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        break
                    }

                    // Fast path: build JSON string directly instead of JSONSerialization
                    let escaped = escapeJSON(event.text)
                    let line = "{\"model\":\"\(modelName)\",\"message\":{\"role\":\"assistant\",\"content\":\"\(escaped)\"},\"done\":false}\n"
                    var buf = ctx.channel.allocator.buffer(capacity: line.utf8.count)
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

        if let media = decodedMedia, media.audioPath != nil {
            // Audio path: bridge through Python (entire request, even with image).
            // Synthesize a single-message conversation for the bridge.
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
        let mediaCopy = decodedMedia

        if request.stream {
            context.write(wrapOutboundOut(.head(ServerResponseHeads.ollamaStreaming())), promise: nil)
        }

        let requestStart = CFAbsoluteTimeGetCurrent()

        Task {
            defer { mediaCopy?.cleanup() }
            let (tokenStream, getStats) = eng.generate(
                prompt: request.prompt,
                systemPrompt: request.system,
                params: request.sampling.samplingParams,
                maxTokens: request.maxTokens,
                imageData: imageData)

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
                        var buf = ctx.channel.allocator.buffer(capacity: data.count + 1)
                        buf.writeBytes(data)
                        buf.writeString("\n")
                        self.writeOnLoop(ctx, .body(.byteBuffer(buf)), flush: true)
                        break
                    }

                    // Fast path: build JSON string directly instead of JSONSerialization
                    let escaped = escapeJSON(event.text)
                    let line = "{\"model\":\"\(modelName)\",\"response\":\"\(escaped)\",\"done\":false}\n"
                    var buf = ctx.channel.allocator.buffer(capacity: line.utf8.count)
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
