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

    func testOpenAIChatRequestParsesSamplingOptions() {
        let request = ServerParsing.openAIChatRequest(from: [
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

    func testOllamaChatRequestParsesSamplingOptions() {
        let options: [String: Any] = [
            "temperature": 0.35,
            "top_p": 0.91,
            "top_k": 12,
        ]
        let request = ServerParsing.ollamaChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "stream": false,
            "options": options,
        ])

        XCTAssertEqual(request.requestedModel, "local-model")
        XCTAssertFalse(request.stream)
        XCTAssertEqual(request.maxTokens, 2048)
        XCTAssertEqual(request.sampling.temperature, 0.35, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topP, 0.91, accuracy: 0.0001)
        XCTAssertEqual(request.sampling.topK, 12)
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

    private func readResponseEnd(from channel: EmbeddedChannel) throws {
        let part = try XCTUnwrap(channel.readOutbound(as: HTTPServerResponsePart.self))
        guard case .end = part else {
            XCTFail("Expected response end, got \(part)")
            throw TestError.unexpectedResponsePart
        }
    }

    private enum TestError: Error {
        case unexpectedResponsePart
    }
}
