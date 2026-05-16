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

    func testOllamaFormatJsonParsed() throws {
        let req = try ServerParsing.ollamaChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "format": "json",
        ])
        XCTAssertEqual(req.responseFormat, .json)
    }

    func testOllamaFormatSchemaParsed() throws {
        let req = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "m", "prompt": "hi", "stream": false,
            "format": ["type": "object", "properties": ["x": ["type": "number"]]],
        ])
        guard case .schema(let s)? = req.responseFormat else {
            return XCTFail("expected schema")
        }
        XCTAssertTrue(s.contains("properties"))
    }

    func testOpenAIResponseFormatJsonObject() throws {
        let req = try ServerParsing.openAIChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "response_format": ["type": "json_object"],
        ])
        XCTAssertEqual(req.responseFormat, .json)
    }

    func testStructuredOutputExtractsJSONFromProse() {
        let text = "Sure! Here you go:\n```json\n{\"a\": 1, \"b\": [2,3]}\n```\nHope that helps."
        let out = StructuredOutput.coerce(text, format: .json)
        let obj = try? JSONSerialization.jsonObject(with: Data(out.utf8)) as? [String: Any]
        XCTAssertEqual(obj?["a"] as? Int, 1)
    }

    func testStructuredOutputArrayAndPassthrough() {
        XCTAssertEqual(StructuredOutput.extractJSON(from: "x [1, 2, 3] y"), "[1,2,3]")
        // No JSON + format set -> original text preserved (visible refusal).
        XCTAssertEqual(StructuredOutput.coerce("I cannot.", format: .json), "I cannot.")
        // No format -> untouched.
        XCTAssertEqual(StructuredOutput.coerce("plain", format: nil), "plain")
    }

    func testStructuredOutputInjectsSystemTurn() {
        let out = StructuredOutput.injectFormatSystem(
            into: [["role": "user", "content": "hi"]], format: .json)
        XCTAssertEqual(out.first?["role"], "system")
        XCTAssertTrue(out.first?["content"]?.contains("valid JSON") ?? false)
    }

    func testOpenAIChatParsesMinPAndAcceptsPenalties() throws {
        let req = try ServerParsing.openAIChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "min_p": 0.05,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
        ])
        XCTAssertEqual(req.sampling.minP, 0.05, accuracy: 1e-6)
        XCTAssertEqual(req.sampling.samplingParams.minP, 0.05, accuracy: 1e-6)
    }

    func testOllamaNumPredictMinusOneMeansInfinite() throws {
        let req = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "m", "prompt": "hi", "stream": false,
            "options": ["num_predict": -1],
        ])
        XCTAssertGreaterThan(req.maxTokens, 1 << 19)
    }

    func testOllamaChatParsesMinPFromOptions() throws {
        let req = try ServerParsing.ollamaChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "options": ["min_p": 0.1],
        ])
        XCTAssertEqual(req.sampling.minP, 0.1, accuracy: 1e-6)
    }

    func testCorsPreflightReturns204WithAllowOrigin() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        var head = HTTPRequestHead(version: .http1_1, method: .OPTIONS, uri: "/api/chat")
        head.headers.add(name: "Origin", value: "http://localhost")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
        let rh = try readResponseHead(from: channel)
        XCTAssertEqual(rh.status, .noContent)
        XCTAssertEqual(rh.headers.first(name: "Access-Control-Allow-Origin"), "http://localhost")
        try readResponseEnd(from: channel)
    }

    func testCorsHeaderOnJSONResponseForAllowedOrigin() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        var head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/healthz")
        head.headers.add(name: "Origin", value: "http://127.0.0.1")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
        let rh = try readResponseHead(from: channel)
        XCTAssertEqual(rh.headers.first(name: "Access-Control-Allow-Origin"), "http://127.0.0.1")
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testCorsNoGrantForDisallowedOrigin() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        var head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/healthz")
        head.headers.add(name: "Origin", value: "https://evil.example")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
        let rh = try readResponseHead(from: channel)
        XCTAssertNil(rh.headers.first(name: "Access-Control-Allow-Origin"))
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testOpenAIChatRequestParsesMaxCompletionTokensAlias() throws {
        let request = try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "hello"]],
            "max_completion_tokens": 33,
        ])

        XCTAssertEqual(request.maxTokens, 33)
    }

    func testOpenAIChatRequestParsesTools() throws {
        let req = try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": "weather?"]],
            "tools": [[
                "type": "function",
                "function": [
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": ["type": "object",
                                   "properties": ["city": ["type": "string"]]],
                ],
            ]],
        ])
        XCTAssertEqual(req.tools.count, 1)
        XCTAssertEqual(req.tools.first?.name, "get_weather")
        XCTAssertTrue(req.tools.first?.parametersJSON.contains("city") ?? false)
    }

    func testChatRequestNormalizesToolResultTurns() throws {
        // assistant tool_calls + role:tool result must round-trip into the
        // [String:String] message path without a 400.
        let req = try ServerParsing.openAIChatRequest(from: [
            "model": "m",
            "messages": [
                ["role": "user", "content": "weather in NYC?"],
                ["role": "assistant", "content": NSNull(),
                 "tool_calls": [["type": "function",
                                 "function": ["name": "get_weather",
                                               "arguments": "{\"city\":\"NYC\"}"]]]],
                ["role": "tool", "name": "get_weather", "content": "{\"temp\":72}"],
            ],
        ])
        XCTAssertEqual(req.messages.count, 3)
        XCTAssertTrue(req.messages[1]["content"]?.contains("<tool_call>") ?? false)
        XCTAssertTrue(req.messages[2]["content"]?.contains("<tool_response>") ?? false)
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
        // OpenAI chat now accepts string OR an array of content blocks.
        // A bare object (not array) is invalid; verify the new error message.
        XCTAssertThrowsError(try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": ["type": "text"]]],
        ])) { error in
            XCTAssertEqual(
                error as? ServerRequestError,
                .invalidType(field: "messages[0].content", expected: "a string or an array of content blocks")
            )
        }
    }

    func testOpenAIChatRequestRejectsNumericMessageContent() {
        XCTAssertThrowsError(try ServerParsing.openAIChatRequest(from: [
            "model": "local-model",
            "messages": [["role": "user", "content": 42]],
        ])) { error in
            XCTAssertEqual(
                error as? ServerRequestError,
                .invalidType(field: "messages[0].content", expected: "a string or an array of content blocks")
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

    func testOllamaGenerateRequestAcceptsImagesField() throws {
        // `images` is no longer rejected at the parsing layer — it is captured
        // into the media payload and validated by the server handler against
        // the loaded model. Parsing alone should succeed.
        let request = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "local-model",
            "prompt": "describe this",
            "images": ["dGVzdA=="],
        ])
        XCTAssertEqual(request.media.images, ["dGVzdA=="])
        XCTAssertNil(request.media.audio)
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

    func testChatCompletionsToolsAcceptedReachesModelGate() throws {
        // tools[] is now supported: the request must pass parsing and reach
        // the model gate (503, no model loaded) rather than 400-rejecting.
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(
            to: channel,
            method: .POST,
            uri: "/v1/chat/completions",
            body: [
                "model": "local-model",
                "messages": [["role": "user", "content": "weather?"]],
                "tools": [[
                    "type": "function",
                    "function": ["name": "get_weather", "description": "w",
                                 "parameters": ["type": "object"]],
                ]],
            ]
        )

        let responseHead = try readResponseHead(from: channel)
        XCTAssertEqual(responseHead.status, .serviceUnavailable)

        let body = try readJSONResponseBody(from: channel)
        XCTAssertTrue((body["error"] as? String)?.contains("No model loaded") ?? false)

        try readResponseEnd(from: channel)
    }

    func testOllamaGenerateNumPredictMinusOneIsAcceptedAsInfinite() throws {
        // Ollama parity: num_predict=-1 means "generate until EOS", not an
        // error. The request must pass parsing and reach the model gate.
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
        XCTAssertEqual(responseHead.status, .serviceUnavailable)

        let body = try readJSONResponseBody(from: channel)
        XCTAssertTrue((body["error"] as? String)?.contains("No model loaded") ?? false)

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

    // MARK: - Ollama compat endpoints (Phase 1: WS-A)

    func testApiVersionReturnsVersionAndKrillmVersion() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        let head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/api/version")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))

        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let json = try readJSONResponseBody(from: channel)
        XCTAssertNotNil(json["version"] as? String)
        XCTAssertEqual(json["krillm_version"] as? String, OllamaCompat.krillVersion)
        try readResponseEnd(from: channel)
    }

    func testApiPsWithoutModelReturnsEmptyList() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        let head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/api/ps")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))

        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let json = try readJSONResponseBody(from: channel)
        XCTAssertEqual((json["models"] as? [[String: Any]])?.count, 0)
        try readResponseEnd(from: channel)
    }

    func testApiShowUnknownModelReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(to: channel, method: .POST, uri: "/api/show",
                             body: ["model": "does-not-exist"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testApiShowKnownModelReturnsMetadata() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-show-\(UUID().uuidString)")
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        try registry.saveManifest(Self.fixtureManifest(name: "fixture-7b"))
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(to: channel, method: .POST, uri: "/api/show",
                             body: ["model": "fixture-7b"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let json = try readJSONResponseBody(from: channel)
        XCTAssertNotNil(json["modelfile"] as? String)
        XCTAssertNotNil(json["template"] as? String)
        XCTAssertEqual((json["capabilities"] as? [String])?.contains("completion"), true)
        let details = try XCTUnwrap(json["details"] as? [String: Any])
        XCTAssertEqual(details["family"] as? String, "qwen")
        try readResponseEnd(from: channel)
    }

    func testApiDeleteUnknownModelReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(to: channel, method: .DELETE, uri: "/api/delete",
                             body: ["model": "ghost"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testApiCopyRoundTripsManifest() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-copy-\(UUID().uuidString)")
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        try registry.saveManifest(Self.fixtureManifest(name: "src-model"))
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(to: channel, method: .POST, uri: "/api/copy",
                             body: ["source": "src-model", "destination": "dst-model"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
        XCTAssertTrue(registry.hasModel("dst-model"))
        XCTAssertEqual(registry.getModel("dst-model")?.family, .qwen)
    }

    func testApiBlobHeadMissingReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        let head = HTTPRequestHead(version: .http1_1, method: .HEAD,
                                   uri: "/api/blobs/sha256:deadbeef")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testCompatOpenAIModeDisablesOllamaEndpoints() throws {
        let channel = try makeChannel(compat: .openai)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        let head = HTTPRequestHead(version: .http1_1, method: .GET, uri: "/api/version")
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.head(head)))
        XCTAssertNoThrow(try channel.writeInbound(HTTPServerRequestPart.end(nil)))
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testCompatModeParsing() {
        XCTAssertEqual(CompatMode(label: "OLLAMA"), .ollama)
        XCTAssertEqual(CompatMode(label: "both"), .both)
        XCTAssertNil(CompatMode(label: "garbage"))
        XCTAssertTrue(CompatMode.both.ollamaEnabled && CompatMode.both.openAIEnabled)
        XCTAssertFalse(CompatMode.openai.ollamaEnabled)
    }

    func testOllamaCompatShowPayloadShape() {
        let m = Self.fixtureManifest(name: "shape-test")
        let payload = OllamaCompat.showPayload(for: m)
        XCTAssertTrue((payload["modelfile"] as? String)?.contains("FROM") ?? false)
        let details = payload["details"] as? [String: Any]
        XCTAssertEqual(details?["quantization_level"] as? String, "4bit")
        XCTAssertEqual(payload["capabilities"] as? [String], ["completion"])
    }

    // MARK: - Embeddings (WS-B)

    func testEmbedMissingModelReturns400() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/embed",
                             body: ["input": "hello"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testEmbedMissingInputReturns400() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-emb1-\(UUID().uuidString)")
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        try registry.saveManifest(Self.fixtureEmbeddingManifest(name: "bge-x"))
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/embed",
                             body: ["model": "bge-x"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testEmbedUnknownModelReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/embed",
                             body: ["model": "nope", "input": "hi"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        let j = try readJSONResponseBody(from: channel)
        XCTAssertTrue((j["error"] as? String)?.contains("krillm pull") ?? false)
        try readResponseEnd(from: channel)
    }

    func testEmbedRejectsNonEmbeddingModelFamily() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-emb2-\(UUID().uuidString)")
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        try registry.saveManifest(Self.fixtureManifest(name: "qwen-chat"))
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/embed",
                             body: ["model": "qwen-chat", "input": "hi"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        let j = try readJSONResponseBody(from: channel)
        XCTAssertTrue((j["error"] as? String)?.contains("not a sentence-embedding") ?? false)
        try readResponseEnd(from: channel)
    }

    func testOpenAIEmbeddingsUnknownModelReturns404() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/v1/embeddings",
                             body: ["model": "nope", "input": ["a", "b"]])
        XCTAssertEqual(try readResponseHead(from: channel).status, .notFound)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testLegacyEmbeddingsMissingPromptReturns400() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-emb3-\(UUID().uuidString)")
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        try registry.saveManifest(Self.fixtureEmbeddingManifest(name: "bge-y"))
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/embeddings",
                             body: ["model": "bge-y"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    // MARK: - Tool calling (WS-D D1)

    func testToolCallExtractionSentinel() {
        let text = "Sure.\n<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"NYC\"}}</tool_call>"
        let (calls, cleaned) = ToolCalling.extractToolCalls(from: text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertTrue(calls.first?.argumentsJSON.contains("NYC") ?? false)
        XCTAssertEqual(cleaned, "Sure.")
    }

    func testToolCallExtractionToleratesMissingCloseTagAndBackticks() {
        // Real llama-3.2-1b output shape: backticks, no </tool_call>, trailing ;
        let text = "`<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Tokyo\"}};`"
        let (calls, cleaned) = ToolCalling.extractToolCalls(from: text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "get_weather")
        XCTAssertTrue(calls.first?.argumentsJSON.contains("Tokyo") ?? false)
        XCTAssertFalse(cleaned.contains("tool_call"))
    }

    func testToolCallExtractionBalancedNestedBraces() {
        let text = "<tool_call>{\"name\":\"f\",\"arguments\":{\"q\":{\"a\":1},\"s\":\"}}\"}}</tool_call>"
        let (calls, _) = ToolCalling.extractToolCalls(from: text)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls.first?.name, "f")
    }

    func testToolCallExtractionBareJSON() {
        let (calls, _) = ToolCalling.extractToolCalls(
            from: "{\"name\":\"f\",\"arguments\":{\"x\":1}}")
        XCTAssertEqual(calls.first?.name, "f")
    }

    func testToolCallExtractionNoneIsPlainText() {
        let (calls, cleaned) = ToolCalling.extractToolCalls(from: "Just a normal answer.")
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(cleaned, "Just a normal answer.")
    }

    func testToolSystemInjectionMergesIntoExistingSystem() {
        let msgs = [["role": "system", "content": "Be terse."],
                    ["role": "user", "content": "hi"]]
        let spec = ServerToolSpec(name: "t", description: "d", parametersJSON: "{}")
        let out = ToolCalling.injectToolSystem(into: msgs, tools: [spec])
        XCTAssertEqual(out.count, 2)
        XCTAssertEqual(out[0]["role"], "system")
        XCTAssertTrue(out[0]["content"]?.contains("Be terse.") ?? false)
        XCTAssertTrue(out[0]["content"]?.contains("<tool_call>") ?? false)
    }

    func testOpenAIVsOllamaToolCallShapes() {
        let calls = [ToolCalling.ParsedToolCall(name: "f", argumentsJSON: "{\"a\":1}")]
        let oa = ToolCalling.openAIToolCalls(calls)
        XCTAssertEqual((oa[0]["function"] as? [String: Any])?["arguments"] as? String, "{\"a\":1}")
        XCTAssertEqual(oa[0]["type"] as? String, "function")
        let ol = ToolCalling.ollamaToolCalls(calls)
        let olArgs = (ol[0]["function"] as? [String: Any])?["arguments"] as? [String: Any]
        XCTAssertEqual(olArgs?["a"] as? Int, 1)
    }

    static func fixtureEmbeddingManifest(name: String) -> ModelManifest {
        ModelManifest(
            name: name, family: .bert, params: "33M", quant: "fp32",
            source: "BAAI/bge-small-en-v1.5", context: 512,
            files: [], draftPair: nil, chatTemplate: "none",
            sizeBytes: 133_000_000, pulledAt: Date())
    }

    static func fixtureManifest(name: String) -> ModelManifest {
        ModelManifest(
            name: name, family: .qwen, params: "7B", quant: "4bit",
            source: "mlx-community/Qwen2.5-7B-Instruct-4bit", context: 32768,
            files: [], draftPair: nil, chatTemplate: "chatml",
            sizeBytes: 4_200_000_000, pulledAt: Date())
    }

    private func makeChannel(
        baseDir: URL? = nil,
        registry: Registry? = nil,
        compat: CompatMode = .both
    ) throws -> EmbeddedChannel {
        let root = baseDir ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-server-tests-\(UUID().uuidString)")
        let engine = InferenceEngine(modelDirectory: root.appendingPathComponent("model"))
        let reg = registry ?? Registry(baseDir: root.appendingPathComponent("registry"))
        let channel = EmbeddedChannel()
        try channel.pipeline.addHandler(
            KLMServer._makeHTTPHandlerForTesting(engine: engine, registry: reg, compat: compat)
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
