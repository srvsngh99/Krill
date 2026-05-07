import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import KLMEngine
import KLMRegistry
@testable import KLMServer

final class ServerTests: XCTestCase {
    func testMessageConversionPreservesStructuredConversation() {
        let rawMessages: [[String: Any]] = [
            ["role": "system", "content": "You answer tersely."],
            ["role": "user", "content": "What is 2+2?"],
            ["role": "assistant", "content": "4"],
            ["role": "user", "content": "What is 3+3?"],
        ]

        let messages = ServerParsing.structuredMessages(from: rawMessages)

        XCTAssertEqual(messages, [
            ["role": "system", "content": "You answer tersely."],
            ["role": "user", "content": "What is 2+2?"],
            ["role": "assistant", "content": "4"],
            ["role": "user", "content": "What is 3+3?"],
        ])
    }

    func testOpenAIChatRequestParsesSamplingOptions() throws {
        let request = try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "stream": true,
            "max_tokens": 77,
            "temperature": 0.7,
            "top_p": 0.82,
            "top_k": 41,
        ])

        XCTAssertEqual(request.requestedModel, "local-model")
        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.maxTokens, 77)
        XCTAssertEqual(request.sampling.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topP, 0.82, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topK, 41)
    }

    func testOpenAIChatRequestParsesJSONSerializationNumbers() throws {
        let request = try ServerParsing.openAIChatRequest(from: jsonObject("""
        {
          "model": "local-model",
          "messages": [{"role": "user", "content": "hello"}],
          "stream": true,
          "max_tokens": 77,
          "temperature": 0.7,
          "top_p": 0.82,
          "top_k": 41
        }
        """))

        XCTAssertTrue(request.stream)
        XCTAssertEqual(request.maxTokens, 77)
        XCTAssertEqual(request.sampling.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topP, 0.82, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topK, 41)
    }

    func testOpenAIChatRequestParsesMaxCompletionTokensAlias() throws {
        let request = try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "max_completion_tokens": 33,
        ])

        XCTAssertEqual(request.maxTokens, 33)
    }

    func testOpenAIChatRequestRejectsUnsupportedTools() {
        XCTAssertThrowsError(try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "tools": [],
        ])) { error in
            XCTAssertEqual(error as? ServerRequestError, .unsupportedField("tools"))
        }
    }

    func testOpenAIChatRequestRejectsInvalidTokenLimit() {
        XCTAssertThrowsError(try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "max_tokens": 0,
        ])) { error in
            XCTAssertEqual(
                error as? ServerRequestError,
                .invalidValue(field: "max_tokens", reason: "must be greater than 0")
            )
        }
    }

    func testOpenAIChatRequestRejectsInvalidMessageContent() {
        XCTAssertThrowsError(try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": ["type": "text"]]],
        ])) { error in
            XCTAssertEqual(
                error as? ServerRequestError,
                .invalidType(field: "messages[0].content", expected: "a string")
            )
        }
    }

    func testOllamaChatRequestParsesSamplingOptions() throws {
        let options: [String: Any] = [
            "temperature": 0.35,
            "top_p": 0.91,
            "top_k": 12,
            "num_predict": 64,
            "repeat_penalty": 1.15,
            "seed": 123,
        ]
        let request = try ServerParsing.ollamaChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "stream": false,
            "options": options,
        ])

        XCTAssertEqual(request.requestedModel, "local-model")
        XCTAssertFalse(request.stream)
        XCTAssertEqual(request.maxTokens, 64)
        XCTAssertEqual(request.sampling.temperature, 0.35, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topP, 0.91, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topK, 12)
        XCTAssertEqual(request.sampling.repetitionPenalty, 1.15, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.seed, 123)
    }

    func testOllamaGenerateRequestParsesNumPredictAndMaxTokens() throws {
        let numPredict = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "local-model",
            "prompt": "hello",
            "stream": false,
            "options": ["num_predict": 27],
        ])

        let maxTokens = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "local-model",
            "prompt": "hello",
            "stream": false,
            "options": ["max_tokens": 31],
        ])

        XCTAssertEqual(numPredict.maxTokens, 27)
        XCTAssertEqual(maxTokens.maxTokens, 31)
    }

    func testOllamaGenerateRequestParsesJSONSerializationNumbers() throws {
        let request = try ServerParsing.ollamaGenerateRequest(from: jsonObject("""
        {
          "model": "local-model",
          "prompt": "hello",
          "stream": false,
          "options": {
            "num_predict": 8,
            "temperature": 0.3,
            "top_p": 0.9,
            "top_k": 12
          }
        }
        """))

        XCTAssertFalse(request.stream)
        XCTAssertEqual(request.maxTokens, 8)
        XCTAssertEqual(request.sampling.temperature, 0.3, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topP, 0.9, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topK, 12)
    }

    func testOllamaGenerateRequestRejectsConflictingTokenLimits() {
        XCTAssertThrowsError(try ServerParsing.ollamaGenerateRequest(from: [
            "model": "local-model",
            "prompt": "hello",
            "num_predict": 12,
            "options": ["max_tokens": 24],
        ])) { error in
            XCTAssertEqual(
                error as? ServerRequestError,
                .invalidValue(field: "max_tokens", reason: "conflicting top-level and options token limits")
            )
        }
    }

    func testOllamaGenerateRequestRejectsUnsupportedImages() {
        XCTAssertThrowsError(try ServerParsing.ollamaGenerateRequest(from: [
            "model": "local-model",
            "prompt": "describe this",
            "images": ["base64"],
        ])) { error in
            XCTAssertEqual(error as? ServerRequestError, .unsupportedField("images"))
        }
    }

    func testOversizedBodyReturnsPayloadTooLarge() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        var head = HTTPRequestHead(version: .http1_1, method: .POST, uri: "/v1/chat/completions")
        head.headers.add(name: "Content-Length", value: "\(ServerLimits.maxBodySize + 1)")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))

        var body = channel.allocator.buffer(capacity: ServerLimits.maxBodySize + 1)
        body.writeBytes(Array(repeating: UInt8(120), count: ServerLimits.maxBodySize + 1))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.body(body)))

        let responseHead = try readResponseHead(from: channel)
        XCTAssertEqual(responseHead.status, .payloadTooLarge)
        try readResponseEnd(from: channel)
    }

    func testChatCompletionsUnsupportedToolsReturnsBadRequestBeforeModelCheck() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/v1/chat/completions",
            body: [
                "model": "local-model",
                "messages": [["role": "user", "content": "hello"]],
                "tools": [],
            ]
        )

        let responseHead = try readResponseHead(from: channel)
        XCTAssertEqual(responseHead.status, .badRequest)

        let body = try readJSONResponseBody(from: channel)
        XCTAssertEqual(body["error"] as? String, "Field 'tools' is not supported by this endpoint")

        try readResponseEnd(from: channel)
    }

    func testOllamaGenerateInvalidNumPredictReturnsBadRequestBeforeModelCheck() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/api/generate",
            body: [
                "model": "local-model",
                "prompt": "hello",
                "options": ["num_predict": -1],
            ]
        )

        let responseHead = try readResponseHead(from: channel)
        XCTAssertEqual(responseHead.status, .badRequest)

        let body = try readJSONResponseBody(from: channel)
        XCTAssertEqual(body["error"] as? String, "Field 'num_predict' is invalid: must be greater than 0")

        try readResponseEnd(from: channel)
    }

    func testHealthEndpointWithoutModelReturnsJSONModelLoadedFalse() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        let head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/healthz")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))

        let responseHead = try readResponseHead(from: channel)
        XCTAssertEqual(responseHead.status, .ok)
        XCTAssertEqual(responseHead.headers.first(name: "Content-Type"), "application/json")

        let body = try readResponseBody(from: channel)
        let json = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any]
        )
        XCTAssertEqual(json["status"] as? String, "ok")
        XCTAssertEqual(json["model_loaded"] as? Bool, false)
        XCTAssertEqual(json["model"] as? String, "none")

        try readResponseEnd(from: channel)
    }

    func testOllamaStreamingResponseHead() {
        let head = ServerResponseHeads.ollamaStreaming()

        XCTAssertEqual(head.status, .ok)
        XCTAssertEqual(head.headers.first(name: "Content-Type"), "application/x-ndjson")
        XCTAssertEqual(head.headers.first(name: "Transfer-Encoding"), "chunked")
    }

    private func makeChannel() throws -> EmbeddedChannel {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-server-tests-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: baseDir.appendingPathComponent("model"))
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        let channel = EmbeddedChannel()
        try channel.pipeline.addHandler(
            KLMServer._makeHTTPHandlerForTesting(engine: engine, registry: registry)
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

    private func readResponseEnd(from channel: EmbeddedChannel) throws {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .end = part else {
            XCTFail("Expected response end, got \(part)")
            throw TestError.unexpectedResponsePart
        }
    }

    private func jsonObject(_ text: String) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    private enum TestError: Error {
        case unexpectedResponsePart
    }
}
