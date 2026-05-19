import XCTest
import NIOCore
import NIOEmbedded
import NIOHTTP1
import KLMEngine
import KLMRegistry
import KLMSampler
import MLX
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

    func testAnthropicParseSystemToolsAndBlocks() {
        let p = AnthropicCompat.parse([
            "model": "claude-x",
            "max_tokens": 256,
            "system": "You are helpful.",
            "messages": [
                ["role": "user", "content": "weather?"],
                ["role": "assistant", "content": [
                    ["type": "tool_use", "id": "t1", "name": "get_weather",
                     "input": ["city": "NYC"]]]],
                ["role": "user", "content": [
                    ["type": "tool_result", "tool_use_id": "t1",
                     "content": "{\"temp\":70}"]]],
            ],
            "tools": [["name": "get_weather", "description": "w",
                       "input_schema": ["type": "object"]]],
            "thinking": ["type": "enabled", "budget_tokens": 1024],
        ])
        XCTAssertEqual(p.messages.first?["role"], "system")
        XCTAssertEqual(p.maxTokens, 256)
        XCTAssertEqual(p.tools.first?.name, "get_weather")
        XCTAssertTrue(p.thinking)
        XCTAssertTrue(p.messages.contains { $0["content"]?.contains("<tool_call>") ?? false })
        XCTAssertTrue(p.messages.contains { $0["content"]?.contains("<tool_response>") ?? false })
    }

    func testAnthropicResponseToolUseShape() {
        let r = AnthropicCompat.response(
            model: "m", text: "",
            toolCalls: [ToolCalling.ParsedToolCall(name: "f", argumentsJSON: "{\"a\":1}")],
            thinking: nil, inputTokens: 3, outputTokens: 4)
        XCTAssertEqual(r["type"] as? String, "message")
        XCTAssertEqual(r["stop_reason"] as? String, "tool_use")
        let content = r["content"] as? [[String: Any]]
        XCTAssertEqual(content?.first?["type"] as? String, "tool_use")
        XCTAssertEqual(content?.first?["name"] as? String, "f")
    }

    func testAnthropicMessagesWithoutModelReturns503() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/v1/messages", body: [
            "model": "claude-x", "max_tokens": 64,
            "messages": [["role": "user", "content": "hi"]],
        ])
        XCTAssertEqual(try readResponseHead(from: channel).status, .serviceUnavailable)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
    }

    func testGenerationQueueSerializesAndCapsAtMaxQueue() async throws {
        // numParallel=1, maxQueue=1: one active + one queued allowed; a
        // third concurrent enter() must throw QueueFull (-> 503).
        let q = GenerationQueue(numParallel: 1, maxQueue: 1)
        try await q.enter()                       // active
        let waiter = Task { try await q.enter() } // queued (depth 1)
        try? await Task.sleep(nanoseconds: 50_000_000)
        let d = await q.depth
        XCTAssertEqual(d, 1)
        do {
            try await q.enter()                   // exceeds maxQueue
            XCTFail("expected QueueFull")
        } catch is GenerationQueue.QueueFull {
            // expected
        }
        await q.leave()                           // hand slot to waiter
        try await waiter.value
        await q.leave()
    }

    func testGenerationQueueWithSlotReleasesOnThrow() async {
        let q = GenerationQueue(numParallel: 1, maxQueue: 0)
        struct E: Error {}
        do {
            try await q.withSlot { throw E() }
        } catch {}
        // Slot must have been released despite the throw: a fresh enter
        // succeeds immediately.
        do { try await q.enter(); await q.leave() }
        catch { XCTFail("slot leaked after throw") }
    }

    func testKeepAliveDurationParsing() {
        XCTAssertEqual(KeepAliveParse.duration("5m"), 300)
        XCTAssertEqual(KeepAliveParse.duration("30s"), 30)
        XCTAssertEqual(KeepAliveParse.duration("1h"), 3600)
        XCTAssertEqual(KeepAliveParse.duration("1h30m"), 5400)
        XCTAssertEqual(KeepAliveParse.duration("90"), 90)
        XCTAssertEqual(KeepAliveParse.seconds(from: -1), -1)
        XCTAssertEqual(KeepAliveParse.seconds(from: 0), 0)
        XCTAssertEqual(KeepAliveParse.seconds(from: "10m"), 600)
        XCTAssertNil(KeepAliveParse.seconds(from: nil))
    }

    func testKeepAliveControllerEvictionSemantics() async {
        let c = KeepAliveController(defaultSeconds: 300)
        await c.touch(override: -1)            // pin: never evict
        var evict = await c.shouldEvict()
        XCTAssertFalse(evict)
        await c.touch(override: 0)             // evict immediately
        evict = await c.shouldEvict()
        XCTAssertTrue(evict)
        await c.touch(override: 300)          // fresh 5m deadline
        evict = await c.shouldEvict()
        XCTAssertFalse(evict)
        let past = await c.shouldEvict(now: Date().addingTimeInterval(400))
        XCTAssertTrue(past)
    }

    /// PR #18 rereview: keep_alive:0 must evict ONLY after the in-flight
    /// request that carried it drains — never before or during it.
    func testKeepAliveZeroDefersEvictionUntilRequestDrains() async {
        let c = KeepAliveController(defaultSeconds: 300)
        await c.beginRequest()                 // a generate is in flight
        await c.touch(override: 0)             // client asked keep_alive:0
        var evict = await c.shouldEvict()
        XCTAssertFalse(evict, "must not evict while the request is in flight")
        await c.endRequest()                   // request completes
        evict = await c.shouldEvict()
        XCTAssertTrue(evict, "evicts once the active request has drained")
    }

    /// PR #20 review P1: a request accepted while the model was loaded but
    /// still queued behind another request must keep the model loaded. With
    /// the hold registered at acceptance (before queueing), inFlight never
    /// hits 0 — and thus the evictor can't unload — until *every* accepted
    /// request, queued ones included, has drained.
    func testKeepAliveZeroWaitsForQueuedRequestNotJustActiveOne() async {
        let c = KeepAliveController(defaultSeconds: 300)
        await c.beginRequest()                 // A: active
        await c.beginRequest()                 // B: accepted, queued behind A
        await c.touch(override: 0)             // client asked keep_alive:0
        await c.endRequest()                   // A drains; B still queued
        var evict = await c.shouldEvict()
        XCTAssertFalse(evict, "queued request B must still hold the model")
        await c.endRequest()                   // B drains too
        evict = await c.shouldEvict()
        XCTAssertTrue(evict, "evicts only once the whole accepted set drains")
    }

    /// An elapsed deadline must also not evict a model out from under an
    /// in-flight request.
    func testKeepAliveDeadlineDoesNotEvictDuringActiveRequest() async {
        let c = KeepAliveController(defaultSeconds: 300)
        await c.beginRequest()
        await c.touch(override: 300)
        let duringRequest = await c.shouldEvict(now: Date().addingTimeInterval(400))
        XCTAssertFalse(duringRequest, "deadline must wait for the request")
        await c.endRequest()
        let afterRequest = await c.shouldEvict(now: Date().addingTimeInterval(400))
        XCTAssertTrue(afterRequest)
    }

    func testSamplerPenaltiesActiveFlagGatesHistory() {
        XCTAssertFalse(SamplingParams(temperature: 0.7).penaltiesActive)
        XCTAssertTrue(SamplingParams(temperature: 0.7, repetitionPenalty: 1.2).penaltiesActive)
        XCTAssertTrue(SamplingParams(temperature: 0.7, presencePenalty: 0.5).penaltiesActive)
        XCTAssertTrue(SamplingParams(temperature: 0.7, mirostat: 2).penaltiesActive)
        XCTAssertFalse(Sampler(params: SamplingParams(temperature: 0.7)).needsHistory)
    }

    func testRepetitionPenaltyDownweightsRecentTokenGreedy() {
        // Greedy: without penalty argmax=index2; a strong repetition penalty
        // on token 2 must change the winner.
        let logits = MLXArray([1.0, 2.0, 5.0, 0.5] as [Float])
        let plain = Sampler(params: SamplingParams(temperature: 0.0))
        XCTAssertEqual(plain.sample(logits, recent: []), 2)
        let penal = Sampler(params: SamplingParams(
            temperature: 0.0, repetitionPenalty: 10.0, repeatLastN: 64))
        XCTAssertNotEqual(penal.sample(logits, recent: [2, 2, 2]), 2)
    }

    func testPresencePenaltyShiftsGreedyWinner() {
        let logits = MLXArray([3.0, 2.9] as [Float])
        let s = Sampler(params: SamplingParams(
            temperature: 0.0, presencePenalty: 1.0, repeatLastN: 64))
        // Token 0 seen -> penalized by 1.0 -> token 1 should win.
        XCTAssertEqual(s.sample(logits, recent: [0]), 1)
    }

    func testMirostatProducesValidToken() {
        let logits = MLXArray([0.1, 0.2, 5.0, 0.3, 0.05] as [Float])
        let s = Sampler(params: SamplingParams(
            temperature: 1.0, mirostat: 2, mirostatTau: 5.0, mirostatEta: 0.1))
        let t = s.sample(logits, recent: [])
        XCTAssertTrue((0..<5).contains(t))
    }

    func testOllamaParsesPenaltyAndMirostatOptions() throws {
        let req = try ServerParsing.ollamaChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "options": [
                "presence_penalty": 0.7, "frequency_penalty": 0.4,
                "repeat_last_n": 128, "mirostat": 2,
                "mirostat_tau": 4.0, "mirostat_eta": 0.2,
            ],
        ])
        let sp = req.sampling.samplingParams
        XCTAssertEqual(sp.presencePenalty, 0.7, accuracy: 1e-6)
        XCTAssertEqual(sp.frequencyPenalty, 0.4, accuracy: 1e-6)
        XCTAssertEqual(sp.repeatLastN, 128)
        XCTAssertEqual(sp.mirostat, 2)
        XCTAssertEqual(sp.mirostatTau, 4.0, accuracy: 1e-6)
        XCTAssertTrue(sp.penaltiesActive)
    }

    func testOpenAIParsesPresenceFrequencyPenalty() throws {
        let req = try ServerParsing.openAIChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "presence_penalty": 0.5, "frequency_penalty": 0.25,
        ])
        XCTAssertEqual(req.sampling.samplingParams.presencePenalty, 0.5, accuracy: 1e-6)
        XCTAssertEqual(req.sampling.samplingParams.frequencyPenalty, 0.25, accuracy: 1e-6)
    }

    func testOllamaParsesNumCtxFromOptionsAndTopLevel() throws {
        let a = try ServerParsing.ollamaChatRequest(from: [
            "model": "m", "messages": [["role": "user", "content": "hi"]],
            "options": ["num_ctx": 2048],
        ])
        XCTAssertEqual(a.contextLimit, 2048)
        let b = try ServerParsing.ollamaGenerateRequest(from: [
            "model": "m", "prompt": "hi", "stream": false, "num_ctx": 4096,
        ])
        XCTAssertEqual(b.contextLimit, 4096)
        let c = try ServerParsing.ollamaChatRequest(from: [
            "model": "m", "messages": [["role": "user", "content": "hi"]],
        ])
        XCTAssertNil(c.contextLimit)
    }

    func testOllamaChatParsesKeepAlive() throws {
        let req = try ServerParsing.ollamaChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "keep_alive": "10m",
        ])
        XCTAssertEqual(req.keepAlive, 600)
        let req2 = try ServerParsing.ollamaChatRequest(from: [
            "model": "m",
            "messages": [["role": "user", "content": "hi"]],
            "keep_alive": -1,
        ])
        XCTAssertEqual(req2.keepAlive, -1)
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

    /// PR #18 rereview: streaming (SSE / ndjson) response heads must carry
    /// the CORS grant too, not only JSON responses, so browser clients can
    /// consume `stream:true`.
    func testStreamingResponseHeadsIncludeCorsHeaders() {
        let cors = [("Access-Control-Allow-Origin", "http://localhost"),
                    ("Vary", "Origin")]
        let ollama = ServerResponseHeads.ollamaStreaming(cors: cors)
        XCTAssertEqual(ollama.headers.first(name: "Content-Type"),
                       "application/x-ndjson")
        XCTAssertEqual(ollama.headers.first(name: "Access-Control-Allow-Origin"),
                       "http://localhost")
        XCTAssertEqual(ollama.headers.first(name: "Vary"), "Origin")

        let openai = ServerResponseHeads.openAIStreaming(cors: cors)
        XCTAssertEqual(openai.headers.first(name: "Content-Type"),
                       "text/event-stream")
        XCTAssertEqual(openai.headers.first(name: "Access-Control-Allow-Origin"),
                       "http://localhost")

        // No grant (disallowed/missing origin) ⇒ no CORS header injected.
        XCTAssertNil(ServerResponseHeads.ollamaStreaming()
            .headers.first(name: "Access-Control-Allow-Origin"))
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
        // Seed a source whose Modelfile overrides must survive the copy
        // (PR #18 rereview: /api/copy previously dropped them).
        let src = Self.fixtureManifest(name: "src-model")
        let withOverrides = ModelManifest(
            name: src.name, family: src.family, params: src.params,
            quant: src.quant, source: src.source, context: src.context,
            files: src.files, draftPair: src.draftPair,
            chatTemplate: src.chatTemplate, sizeBytes: src.sizeBytes,
            pulledAt: src.pulledAt,
            overrides: ModelOverrides(system: "You are a pirate.",
                                      template: "{{ .Prompt }}",
                                      parameters: ["temperature": "0.1"]))
        try registry.saveManifest(withOverrides)
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(to: channel, method: .POST, uri: "/api/copy",
                             body: ["source": "src-model", "destination": "dst-model"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
        XCTAssertTrue(registry.hasModel("dst-model"))
        let dst = registry.getModel("dst-model")
        XCTAssertEqual(dst?.family, .qwen)
        XCTAssertEqual(dst?.overrides?.system, "You are a pirate.")
        XCTAssertEqual(dst?.overrides?.template, "{{ .Prompt }}")
        XCTAssertEqual(dst?.overrides?.parameters["temperature"], "0.1")
    }

    func testApiCreateFromModelfileThenShowReflectsOverrides() throws {
        let baseDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-create-srv-\(UUID().uuidString)")
        let registry = Registry(baseDir: baseDir.appendingPathComponent("registry"))
        try registry.ensureDirectories()
        try registry.saveManifest(Self.fixtureManifest(name: "qwen-base"))
        try FileManager.default.createDirectory(
            at: registry.modelPath("qwen-base"), withIntermediateDirectories: true)
        let channel = try makeChannel(baseDir: baseDir, registry: registry)
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }

        try writeJSONRequest(to: channel, method: .POST, uri: "/api/create", body: [
            "model": "qwen-custom", "stream": false,
            "modelfile": "FROM qwen-base\nPARAMETER temperature 0.1\nSYSTEM You are Krill.",
        ])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
        XCTAssertEqual(registry.getModel("qwen-custom")?.overrides?.system, "You are Krill.")

        // /api/show reflects the overrides.
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/show",
                             body: ["model": "qwen-custom"])
        XCTAssertEqual(try readResponseHead(from: channel).status, .ok)
        let j = try readJSONResponseBody(from: channel)
        XCTAssertEqual(j["system"] as? String, "You are Krill.")
        XCTAssertTrue((j["parameters"] as? String)?.contains("temperature 0.1") ?? false)
        try readResponseEnd(from: channel)
    }

    func testApiCreateMissingBaseReturnsError() throws {
        let channel = try makeChannel()
        defer { _ = try? channel.finish(acceptAlreadyClosed: true) }
        try writeJSONRequest(to: channel, method: .POST, uri: "/api/create", body: [
            "model": "x", "stream": false, "modelfile": "FROM ghost\nSYSTEM hi",
        ])
        XCTAssertEqual(try readResponseHead(from: channel).status, .badRequest)
        _ = try readResponseBody(from: channel)
        try readResponseEnd(from: channel)
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

    // MARK: - Native Gemma 4 tool format (WS1-4)

    func testGemma4ExtractCanonicalCall() {
        let (calls, cleaned) = ToolCalling.extractToolCalls(
            from: "<|tool_call>call:add{a:12, \"b\":30}<tool_call|>",
            format: .gemma4)
        XCTAssertEqual(calls.count, 1)
        XCTAssertEqual(calls[0].name, "add")
        // Bare key `a` normalized to strict JSON so shapers decode it.
        let obj = (try? JSONSerialization.jsonObject(
            with: Data(calls[0].argumentsJSON.utf8)))
            .flatMap { $0 as? [String: Any] }
        XCTAssertEqual(obj?["a"] as? Int, 12)
        XCTAssertEqual(obj?["b"] as? Int, 30)
        XCTAssertEqual(cleaned, "")
    }

    func testGemma4ExtractToleratesSloppyShapes() {
        // Quoted name + `parameters:` label (observed from the 2B model).
        let a = ToolCalling.extractToolCalls(
            from: "<|tool_call>call: \"multiply\", \"parameters\": {\"a\": 7, \"b\": 6}<tool_call|>",
            format: .gemma4).calls
        XCTAssertEqual(a.first?.name, "multiply")
        XCTAssertTrue(a.first?.argumentsJSON.contains("\"a\"") ?? false)
        // Quoted name + colon, and a missing close sentinel.
        let b = ToolCalling.extractToolCalls(
            from: "<|tool_call>call: \"add\": {\"a\": 100, \"b\": 23}",
            format: .gemma4).calls
        XCTAssertEqual(b.first?.name, "add")
    }

    func testGemma4ExtractStripsThoughtChannel() {
        let (calls, cleaned) = ToolCalling.extractToolCalls(
            from: "<|channel>thought\nI should add.<channel|>Sure!",
            format: .gemma4)
        XCTAssertTrue(calls.isEmpty)
        XCTAssertEqual(cleaned, "Sure!")
    }

    func testGemma4NormalizeArgsBareKeysAndPythonLiterals() {
        let j = ToolCalling.normalizeGemma4Args("{a:1, b:'two', c:True, d:None}")
        let obj = (try? JSONSerialization.jsonObject(with: Data(j.utf8)))
            .flatMap { $0 as? [String: Any] }
        XCTAssertEqual(obj?["a"] as? Int, 1)
        XCTAssertEqual(obj?["b"] as? String, "two")
        XCTAssertEqual(obj?["c"] as? Bool, true)
        XCTAssertTrue(obj?["d"] is NSNull)
    }

    func testGemma4InjectionUsesToolsRoleNotHermesPrompt() {
        let spec = ServerToolSpec(name: "add", description: "sum",
                                  parametersJSON: "{\"type\":\"object\"}")
        let out = ToolCalling.injectToolSystem(
            into: [["role": "user", "content": "hi"]],
            tools: [spec], format: .gemma4)
        XCTAssertEqual(out.first?["role"], "tools")
        XCTAssertTrue(out.first?["content"]?.contains("\"name\": \"add\"") ?? false)
        // No foreign Hermes instruction is injected for Gemma 4.
        XCTAssertFalse(out.contains { ($0["content"] ?? "").contains("<tool_call>") })
    }

    func testGemma4InjectionConvertsHermesHistoryToNative() {
        // `normalizeToolTurns` canonical forms -> native roles.
        let msgs = [
            ["role": "user", "content": "<tool_response>name=add 42</tool_response>"],
            ["role": "assistant",
             "content": "<tool_call>{\"name\": \"add\", \"arguments\": {\"a\":1}}</tool_call>"],
        ]
        let spec = ServerToolSpec(name: "add", description: "", parametersJSON: "{}")
        let out = ToolCalling.injectToolSystem(into: msgs, tools: [spec],
                                               format: .gemma4)
        XCTAssertTrue(out.contains { $0["role"] == "tool" && $0["content"] == "42" })
        XCTAssertTrue(out.contains {
            $0["role"] == "assistant"
                && ($0["content"]?.contains("<|tool_call>call:add") ?? false)
        })
    }

    func testLlamaInjectionMirrorsOllamaTemplate() {
        let spec = ServerToolSpec(name: "add", description: "sum",
                                  parametersJSON: "{\"type\":\"object\"}")
        let out = ToolCalling.injectToolSystem(
            into: [["role": "system", "content": "Be nice."],
                   ["role": "user", "content": "do it"]],
            tools: [spec], format: .llama)
        // System guidance is appended (Ollama's llama3.2 tool system msg).
        XCTAssertEqual(out.first?["role"], "system")
        XCTAssertTrue(out.first?["content"]?.contains("tool calling capabilities") ?? false)
        // Tool block is spliced into the LAST user turn, not a new one.
        let lastUser = out.last { $0["role"] == "user" }
        XCTAssertTrue(lastUser?["content"]?.contains("Given the following functions") ?? false)
        XCTAssertTrue(lastUser?["content"]?.hasSuffix("do it") ?? false)
    }

    func testLlamaExtractNativeAndRejectsEchoedSchema() {
        let ok = ToolCalling.extractToolCalls(
            from: "```json\n{\"name\": \"add\", \"parameters\": {\"a\": 1, \"b\": 2}}\n```",
            format: .llama).calls
        XCTAssertEqual(ok.first?.name, "add")
        XCTAssertTrue(ok.first?.argumentsJSON.contains("\"a\"") ?? false)
        // An echoed tool *schema* must not be mistaken for a call.
        let echoed = ToolCalling.extractToolCalls(
            from: "{\"type\": \"function\", \"function\": {\"name\": \"add\"}}",
            format: .llama).calls
        XCTAssertTrue(echoed.isEmpty)
    }

    func testQwenExtractToleratesLeadingJunk() {
        // The Qwen path must recover a call even with a stray prefix.
        let calls = ToolCalling.extractToolCalls(
            from: ">{\"name\": \"add\", \"arguments\": {\"a\": 100, \"b\": 23}}",
            format: .qwen).calls
        XCTAssertEqual(calls.first?.name, "add")
        XCTAssertEqual(calls.count, 1)
        // Qwen injection reuses the (Qwen-native) Hermes prompt.
        let out = ToolCalling.injectToolSystem(
            into: [["role": "user", "content": "hi"]],
            tools: [ServerToolSpec(name: "t", description: "d", parametersJSON: "{}")],
            format: .qwen)
        XCTAssertTrue(out.first?["content"]?.contains("<tool_call>") ?? false)
    }

    func testHermesPathUnchangedByDefault() {
        // Default format stays .hermes so non-Gemma families regress-free.
        let spec = ServerToolSpec(name: "t", description: "d", parametersJSON: "{}")
        let out = ToolCalling.injectToolSystem(
            into: [["role": "user", "content": "hi"]], tools: [spec])
        XCTAssertEqual(out.first?["role"], "system")
        XCTAssertTrue(out.first?["content"]?.contains("<tool_call>") ?? false)
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
