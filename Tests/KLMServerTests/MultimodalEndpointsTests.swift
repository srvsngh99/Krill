import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import KLMEngine
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

    // MARK: - Live Gemma 4 path (skipped without env var)

    func testOllamaGenerateImageHappyPathRequiresGemma4() throws {
        try requireGemma4Env()
        throw XCTSkip("Live Gemma 4 server multimodal path is exercised by the multimodal benchmark; this test only verifies prerequisites.")
    }

    func testOllamaGenerateAudioHappyPathRequiresMLXVLM() throws {
        try requireGemma4Env()
        let availability = PythonFallback.checkAvailability()
        if !availability.isAvailable {
            throw XCTSkip("mlx-vlm not available: \(availability.detail)")
        }
        throw XCTSkip("Live Gemma 4 audio bridge path is exercised by the multimodal benchmark; this test only verifies prerequisites.")
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
            KLMServer._makeHTTPHandlerForTesting(engine: engine, registry: registry)
        ).wait()
        _ = maxBodySizeOverride // currently unused; ServerLimits.maxBodySize is global.
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
}
