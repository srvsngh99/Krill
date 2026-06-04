import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
@testable import KLMEngine
import KLMRegistry
@testable import KLMServer

/// HTTP-level tests for the multimodal endpoints. These tests exercise both
/// the parsing layer (request shape validation) and the handler layer (model
/// capability gating, payload size limits).
///
/// Tests that depend on a real Gemma 4 model or mlx-vlm install use the same
/// env-var skip pattern as ``Gemma4SmokeTests`` in
/// ``Tests/KLMEngineTests/Gemma4SmokeTests.swift``: set
/// ``KLM_GEMMA4_MODEL_PATH`` to enable; absent or invalid path leads to
/// ``XCTSkip``.
final class MultimodalEndpointsTests: XCTestCase {

    // MARK: - Parsing-only tests (no model required)

    func testOpenAIChatRequestAcceptsImageURLDataURL() throws {
        let request = try ServerParsing.openAIChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "text", "text": "describe"],
                    [
                        "type": "image_url",
                        "image_url": ["url": "data:image/png;base64,iVBORw0KGgo="]
                    ]
                ]
            ]]
        ])
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages.first?["content"], "describe")
        XCTAssertEqual(request.media.images.count, 1)
        XCTAssertTrue(request.media.images.first?.hasPrefix("data:") ?? false)
    }

    func testOpenAIChatRequestAcceptsInputAudio() throws {
        let request = try ServerParsing.openAIChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "text", "text": "transcribe"],
                    [
                        "type": "input_audio",
                        "input_audio": ["data": "RIFF=", "format": "wav"]
                    ]
                ]
            ]]
        ])
        XCTAssertEqual(request.media.audio, "RIFF=")
        XCTAssertEqual(request.media.audioFormat, "wav")
    }

    func testOpenAIChatRequestRejectsNonDataImageURL() {
        XCTAssertThrowsError(try ServerParsing.openAIChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "image_url", "image_url": ["url": "https://example.com/x.png"]]
                ]
            ]]
        ])) { error in
            // We require data: URLs for images.
            guard let req = error as? ServerRequestError else {
                XCTFail("Expected ServerRequestError, got \(error)")
                return
            }
            if case .invalidValue = req {} else {
                XCTFail("Expected invalidValue, got \(req)")
            }
        }
    }

    func testOllamaChatRequestAcceptsImagesPerMessage() throws {
        let request = try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": "what's this?",
                "images": ["dGVzdA=="]
            ]]
        ])
        XCTAssertEqual(request.media.images, ["dGVzdA=="])
    }

    func testOllamaGenerateAcceptsAudioField() throws {
        let request = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "gemma-4-e2b",
            "prompt": "describe",
            "audio": "UklGRg==",
            "audio_format": "wav"
        ])
        XCTAssertEqual(request.media.audio, "UklGRg==")
        XCTAssertEqual(request.media.audioFormat, "wav")
    }

    /// Regression for BENCHMARK_ISSUES #1: the message-level `audio` field on
    /// `/api/chat` must flow into the request-level media payload so the
    /// handler can decode it and thread `audioData` to the engine. (The
    /// benchmark's HTTP probe wrongly reported this path as "accepted but not
    /// ingested"; reproduction on current `main` shows the audio frames reach
    /// prefill. This locks the parse → media wiring so it can't silently
    /// regress.)
    func testOllamaChatRequestAcceptsAudioPerMessage() throws {
        let request = try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": "transcribe this",
                "audio": "UklGRg=="
            ]]
        ])
        XCTAssertEqual(request.media.audio, "UklGRg==",
                       "message-level audio must populate request.media.audio")
        XCTAssertEqual(request.messages.count, 1)
        XCTAssertEqual(request.messages.first?["content"], "transcribe this")
    }

    /// A single request must not carry two audio clips: a second `audio` field
    /// across messages is rejected rather than silently overwriting the first.
    func testOllamaChatRequestRejectsTwoAudioClips() {
        XCTAssertThrowsError(try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [
                ["role": "user", "content": "a", "audio": "UklGRg=="],
                ["role": "user", "content": "b", "audio": "UklGRh=="],
            ]
        ]))
    }

    // MARK: - Ollama /api/chat content-block array form (BENCHMARK_ISSUES #5)

    /// Regression for BENCHMARK_ISSUES #5: the Ollama-compat `/api/chat` endpoint
    /// rejected the OpenAI content-block array form ("content must be a string"),
    /// which also blocked the `input_audio` path. It must now accept the same
    /// blocks the OpenAI endpoint does, routing media into the request payload.
    func testOllamaChatRequestAcceptsContentBlockInputAudio() throws {
        let request = try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "text", "text": "transcribe"],
                    ["type": "input_audio", "input_audio": ["data": "UklGRg==", "format": "wav"]],
                ]
            ]]
        ])
        XCTAssertEqual(request.media.audio, "UklGRg==")
        XCTAssertEqual(request.media.audioFormat, "wav")
        XCTAssertEqual(request.messages.first?["content"], "transcribe",
                       "text blocks become the message content")
    }

    func testOllamaChatRequestAcceptsContentBlockImageURL() throws {
        let request = try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "text", "text": "what is this?"],
                    ["type": "image_url", "image_url": ["url": "data:image/png;base64,iVBORw0KGgo="]],
                ]
            ]]
        ])
        XCTAssertEqual(request.media.images.count, 1)
        XCTAssertTrue(request.media.images.first?.hasPrefix("data:") ?? false)
        XCTAssertEqual(request.messages.first?["content"], "what is this?")
    }

    /// A non-`data:` image URL must be rejected on the Ollama path too (parity
    /// with the OpenAI parser; we only accept base64 data URLs).
    func testOllamaChatRequestRejectsNonDataImageURLInContentBlock() {
        XCTAssertThrowsError(try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "image_url", "image_url": ["url": "https://example.com/x.png"]],
                ]
            ]]
        ]))
    }

    /// A content-block audio clip plus a message-level `audio` field is still a
    /// two-clip request and must be rejected, not silently overwrite one.
    func testOllamaChatRequestRejectsContentBlockPlusMessageLevelAudio() {
        XCTAssertThrowsError(try ServerParsing.ollamaChatRequest(from: [
            "model": "gemma-4-e2b",
            "messages": [[
                "role": "user",
                "content": [
                    ["type": "input_audio", "input_audio": ["data": "UklGRg==", "format": "wav"]],
                ],
                "audio": "UklGRh==",
            ]]
        ]))
    }

    // MARK: - Handler tests (gated by capability checks)

    func testOllamaGenerateRejectsMoreThanOneImage() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/api/generate",
            body: [
                "model": "local-model",
                "prompt": "describe",
                "images": [
                    base64Bytes(repeating: 0x41, count: 32),
                    base64Bytes(repeating: 0x42, count: 32),
                ]
            ]
        )

        let head = try readResponseHead(from: channel)
        XCTAssertEqual(head.status, .badRequest)
        let body = try readJSONResponseBody(from: channel)
        XCTAssertEqual(body["error"] as? String, "Only one image per request is supported")
    }

    func testOllamaGenerateRejectsImageWhenModelNotLoaded() throws {
        // No model loaded -> request is rejected with 503 by requireModel.
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/api/generate",
            body: [
                "model": "local-model",
                "prompt": "describe",
                "images": [base64Bytes(repeating: 0x41, count: 32)]
            ]
        )

        let head = try readResponseHead(from: channel)
        XCTAssertEqual(head.status, .serviceUnavailable)
    }

    func testOpenAICompletionsRemainsTextOnly() throws {
        // /v1/completions has no `media` field and no support for content
        // blocks; only the bare `prompt` form is valid. Confirm parsing
        // succeeds and the resulting struct has no multimodal hooks.
        let request = try ServerParsing.openAICompletionRequest(from: [
            "model": "local-model",
            "prompt": "hello"
        ])
        XCTAssertEqual(request.prompt, "hello")
        // ServerCompletionRequest has no `media` property — verified by the
        // type definition; this assertion documents the contract.
        let mirror = Mirror(reflecting: request)
        XCTAssertFalse(mirror.children.contains(where: { $0.label == "media" }))
    }

    /// Regression for PR #9 medium #4: OpenAI streaming bridge requests must
    /// receive an SSE response head, not application/x-ndjson or JSON.
    func testOpenAIStreamingHeadIsSSE() {
        let head = ServerResponseHeads.openAIStreaming()
        XCTAssertEqual(head.status, .ok)
        XCTAssertEqual(head.headers.first(name: "Content-Type"), "text/event-stream")
        XCTAssertEqual(head.headers.first(name: "Cache-Control"), "no-cache")
    }

    func testOllamaStreamingHeadIsNDJSON() {
        let head = ServerResponseHeads.ollamaStreaming()
        XCTAssertEqual(head.status, .ok)
        XCTAssertEqual(head.headers.first(name: "Content-Type"), "application/x-ndjson")
        XCTAssertEqual(head.headers.first(name: "Transfer-Encoding"), "chunked")
    }

    /// Regression for PR #9 medium #5: per-item size validation must fire BEFORE
    /// the model-loaded check so oversized payloads are rejected with 413
    /// regardless of server state. Pre-fix this test got 503 (no model loaded).
    func testOversizedImageReturns413BeforeModelCheck() throws {
        let raw = Data(repeating: 0x42, count: ServerMultimodal.maxPayloadBytes + 1024)
        let b64 = raw.base64EncodedString()

        let channel = try makeChannel(maxBodySizeOverride: 64 * 1024 * 1024)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/api/generate",
            body: [
                "model": "local-model",
                "prompt": "describe",
                "images": [b64]
            ]
        )

        let head = try readResponseHead(from: channel)
        XCTAssertEqual(head.status, .payloadTooLarge,
                       "oversized image must return 413 even with no model loaded")
    }

    func testOversizedImageReturns413() throws {
        // Create a >25MB base64 string.
        let raw = Data(repeating: 0x42, count: 26 * 1024 * 1024)
        let b64 = raw.base64EncodedString()

        let channel = try makeChannel(maxBodySizeOverride: 64 * 1024 * 1024)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/api/generate",
            body: [
                "model": "local-model",
                "prompt": "describe",
                "images": [b64]
            ]
        )

        let head = try readResponseHead(from: channel)
        // The HTTP body limit may trip first when the request body itself
        // exceeds ServerLimits.maxBodySize, returning 413. If the body fits
        // but the decoded payload exceeds the per-item limit, we also return
        // 413 from the handler. Either path satisfies the contract that
        // oversized media is refused with 413.
        XCTAssertEqual(head.status, .payloadTooLarge)
    }

    // MARK: - Chat path placeholder injection (PR #9 blocker 1)

    /// Regression for PR #9: the chat path (used by /v1/chat/completions and
    /// /api/chat) silently dropped image placeholders, so multimodal forward
    /// had no positions to inject vision embeddings into. Verify that
    /// injectMediaPlaceholders prepends N copies of `<|image|>` to the first
    /// user message's content.
    func testChatPathInjectsImagePlaceholdersIntoFirstUserMessage() {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-mm-placeholder-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))

        let pngData = makeTinyPNG(width: 4, height: 4, r: 200, g: 50, b: 50)
        XCTAssertFalse(pngData.isEmpty, "Test PNG generation should succeed")

        let expectedCount = engine.computeImageTokenCount(imageData: pngData)
        XCTAssertGreaterThan(expectedCount, 0)

        let messages: [[String: String]] = [
            ["role": "system", "content": "you are helpful"],
            ["role": "user", "content": "describe this"],
        ]
        let prepared = engine.injectMediaPlaceholders(
            into: messages, imageData: pngData, audioData: nil)

        XCTAssertEqual(prepared.count, messages.count)
        XCTAssertEqual(prepared[0]["content"], "you are helpful", "system msg untouched")

        let userContent = prepared[1]["content"] ?? ""
        let placeholder = "<|image|>"
        let occurrences = userContent.components(separatedBy: placeholder).count - 1
        XCTAssertEqual(occurrences, expectedCount,
                       "Expected \(expectedCount) image placeholders, got \(occurrences)")
        XCTAssertTrue(userContent.hasSuffix("describe this"),
                      "Original prompt content must be preserved at end")
    }

    func testChatPathInjectsAudioPlaceholderIntoFirstUserMessage() {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-mm-audio-placeholder-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))

        let messages: [[String: String]] = [["role": "user", "content": "transcribe"]]
        let prepared = engine.injectMediaPlaceholders(
            into: messages, imageData: nil, audioData: Data([0x52, 0x49, 0x46, 0x46]))

        let userContent = prepared[0]["content"] ?? ""
        XCTAssertTrue(userContent.hasPrefix("<|audio|>"),
                      "Audio placeholder must precede prompt text")
        XCTAssertTrue(userContent.hasSuffix("transcribe"))
    }

    func testChatPathLeavesMessagesUnchangedWithoutMedia() {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-mm-nomedia-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))
        let messages: [[String: String]] = [["role": "user", "content": "hello"]]
        let prepared = engine.injectMediaPlaceholders(
            into: messages, imageData: nil, audioData: nil)
        XCTAssertEqual(prepared, messages)
    }

    // MARK: - Capability gating (PR #9 blocker 2)

    /// supportsNativeImage must be false when no model is loaded. This guards
    /// the regression where a text-only checkpoint claimed image capability.
    func testSupportsNativeImageFalseWhenNoModelLoaded() {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-mm-cap-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))
        XCTAssertFalse(engine.supportsNativeImage,
                       "No model loaded -> no image capability")
        XCTAssertFalse(engine.supportsAudio,
                       "No model loaded -> no audio capability")
    }

    /// /api/generate with image payload must be rejected (503 because no model
    /// is loaded). With the tighter capability check, even a text-only
    /// gemma4 checkpoint would yield the same rejection at decodeMediaForRequest.
    /// This test documents the capability gating contract from the server side.
    func testImageRequestRejectedWithoutMultimodalModel() throws {
        // Without any model loaded, requireModel returns 503 first; this is
        // covered by testOllamaGenerateRejectsImageWhenModelNotLoaded above.
        // Here we additionally verify the engine-level capability is false.
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-mm-cap2-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))
        XCTAssertFalse(engine.supportsNativeImage)
    }

    // MARK: - Live: two distinct images must change output

    /// Live regression for PR #9 blocker 1: two visually different images with
    /// the same prompt must produce different outputs. If placeholder injection
    /// is missing, the model never sees the image and both outputs match.
    func testTwoDifferentImagesProduceDifferentOutputs() async throws {
        try requireGemma4Env()
        let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"]!
        let dir = URL(fileURLWithPath: path, isDirectory: true)
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsNativeImage else {
            throw XCTSkip("Loaded checkpoint is not multimodal Gemma 4")
        }

        let redPNG = makeTinyPNG(width: 64, height: 64, r: 230, g: 20, b: 20)
        let bluePNG = makeTinyPNG(width: 64, height: 64, r: 20, g: 20, b: 230)
        XCTAssertFalse(redPNG.isEmpty)
        XCTAssertFalse(bluePNG.isEmpty)

        func runOnce(image: Data) async -> String {
            let messages: [[String: String]] = [
                ["role": "user", "content": "What single color do you see?"]
            ]
            let (stream, _) = engine.generate(
                messages: messages,
                params: .greedy,
                maxTokens: 24,
                useSpeculative: false,
                usePrefixCache: false,
                imageData: image,
                audioData: nil)
            var out = ""
            for await event in stream {
                if event.isEnd { break }
                out += event.text
            }
            return out
        }

        let redOut = await runOnce(image: redPNG)
        let blueOut = await runOnce(image: bluePNG)
        XCTAssertFalse(redOut.isEmpty)
        XCTAssertFalse(blueOut.isEmpty)
        XCTAssertNotEqual(
            redOut, blueOut,
            "Identical outputs across visually-different images suggest the image is not conditioning the model")
    }

    // MARK: - Live Gemma 4 path (skipped without env var)

    func testOllamaGenerateImageHappyPathRequiresGemma4() throws {
        try requireGemma4Env()
        throw XCTSkip("Live Gemma 4 server multimodal path is exercised by the multimodal benchmark; this test only verifies prerequisites.")
    }

    func testOllamaGenerateAudioHappyPathRequiresGemma4() throws {
        try requireGemma4Env()
        throw XCTSkip("Live Gemma 4 native audio path is exercised by the multimodal benchmark and Gemma4SmokeTests; this test only verifies prerequisites.")
    }

    // MARK: - Helpers

    private func requireGemma4Env() throws {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set; skipping live multimodal endpoint test")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH does not point to a directory: \(path)")
        }
    }

    /// Build a base64 string from `count` repeated `byte` values.
    private func base64Bytes(repeating byte: UInt8, count: Int) -> String {
        Data(repeating: byte, count: count).base64EncodedString()
    }

    private func makeChannel(maxBodySizeOverride: Int? = nil) throws -> EmbeddedChannel {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-server-mm-tests-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        let channel = EmbeddedChannel()
        try channel.pipeline.addHandler(
            KLMServer._makeHTTPHandlerForTesting(
                engine: engine,
                registry: registry,
                maxBodySizeOverride: maxBodySizeOverride)
        ).wait()
        return channel
    }

    private func writeJSONRequest(
        to channel: EmbeddedChannel,
        method: HTTPMethod,
        uri: String,
        body: [String: Any]
    ) throws {
        let data = try JSONSerialization.data(withJSONObject: body)
        var head = HTTPRequestHead(version: .http1_1, method: method, uri: uri)
        head.headers.add(name: "Content-Type", value: "application/json")
        head.headers.add(name: "Content-Length", value: "\(data.count)")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        var buffer = channel.allocator.buffer(capacity: data.count)
        buffer.writeBytes(data)
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.body(buffer)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
    }

    private func readResponseHead(from channel: EmbeddedChannel) throws -> HTTPResponseHead {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .head(let head) = part else {
            XCTFail("Expected response head, got \(part)")
            throw TestError.unexpectedResponsePart
        }
        return head
    }

    private func readResponseBody(from channel: EmbeddedChannel) throws -> String {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .body(.byteBuffer(var buffer)) = part,
              let body = buffer.readString(length: buffer.readableBytes) else {
            XCTFail("Expected response body, got \(part)")
            throw TestError.unexpectedResponsePart
        }
        return body
    }

    private func readJSONResponseBody(from channel: EmbeddedChannel) throws -> [String: Any] {
        let body = try readResponseBody(from: channel)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any])
    }

    private enum TestError: Error {
        case unexpectedResponsePart
    }

    /// Guard for the multi-image plumbing: every generate handler loads its
    /// images through `DecodedMedia.loadImages()`, which must return BOTH the
    /// first image (single-image runtimes) and the full ordered list (mllama). A
    /// regression here would silently drop all but the first image of a
    /// multi-image Llama-3.2-Vision request while the prompt still emits N
    /// `<|image|>` tokens.
    func testLoadImagesReturnsFirstAndAll() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-loadimages-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        var paths: [String] = []
        for i in 0 ..< 3 {
            let p = dir.appendingPathComponent("img\(i).bin").path
            try Data([UInt8(i), UInt8(i), UInt8(i)]).write(to: URL(fileURLWithPath: p))
            paths.append(p)
        }

        let (first, all) = DecodedMedia(imagePaths: paths).loadImages()
        XCTAssertEqual(all.count, 3, "all three images must load")
        XCTAssertEqual(first, all.first, "first must equal the head of the list")
        XCTAssertEqual(Array(all[0]), [0, 0, 0])
        XCTAssertEqual(Array(all[2]), [2, 2, 2], "order preserved")

        // Empty media yields no images.
        XCTAssertNil(DecodedMedia().loadImages().first)
        XCTAssertTrue(DecodedMedia().loadImages().all.isEmpty)
    }

    /// Build a minimal solid-color PNG via CoreGraphics + ImageIO.
    fileprivate func makeTinyPNG(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) -> Data {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return Data() }
        ctx.setFillColor(CGColor(
            red: CGFloat(r) / 255.0,
            green: CGFloat(g) / 255.0,
            blue: CGFloat(b) / 255.0,
            alpha: 1.0))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        guard let image = ctx.makeImage() else { return Data() }
        let mutableData = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            mutableData, "public.png" as CFString, 1, nil) else { return Data() }
        CGImageDestinationAddImage(dest, image, nil)
        CGImageDestinationFinalize(dest)
        return mutableData as Data
    }
}
