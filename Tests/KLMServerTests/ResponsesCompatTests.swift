import XCTest
@testable import KLMServer

/// Pure translation tests for the OpenAI Responses API compat layer
/// (`POST /v1/responses`). Transport/streaming is covered separately; these
/// exercise `ResponsesCompat.parse` and `.response`/output-item builders.
final class ResponsesCompatTests: XCTestCase {

    func testParseStringInputAndInstructions() {
        let p = ResponsesCompat.parse([
            "model": "gpt-x",
            "instructions": "You are helpful.",
            "input": "weather?",
            "max_output_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.82,
        ])
        XCTAssertEqual(p.model, "gpt-x")
        XCTAssertEqual(p.maxTokens, 256)
        XCTAssertEqual(p.messages.first?["role"], "system")
        XCTAssertEqual(p.messages.first?["content"], "You are helpful.")
        XCTAssertEqual(p.messages.last?["role"], "user")
        XCTAssertEqual(p.messages.last?["content"], "weather?")
        XCTAssertEqual(p.sampling.temperature, 0.7, accuracy: 0.0001)
        XCTAssertEqual(p.sampling.topP, 0.82, accuracy: 0.0001)
    }

    func testParseItemArrayWithToolCallAndOutput() {
        let p = ResponsesCompat.parse([
            "model": "gpt-x",
            "input": [
                ["type": "message", "role": "user",
                 "content": [["type": "input_text", "text": "weather in NYC?"]]],
                ["type": "function_call", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{\"city\":\"NYC\"}"],
                ["type": "function_call_output", "call_id": "call_1",
                 "output": "{\"temp\":70}"],
            ],
            "tools": [["type": "function", "name": "get_weather",
                       "description": "w", "parameters": ["type": "object"]]],
        ])
        XCTAssertEqual(p.tools.first?.name, "get_weather")
        XCTAssertTrue(p.messages.contains { $0["content"]?.contains("weather in NYC?") ?? false })
        XCTAssertTrue(p.messages.contains {
            ($0["content"]?.contains("<tool_call>") ?? false)
                && ($0["content"]?.contains("get_weather") ?? false)
        })
        XCTAssertTrue(p.messages.contains { $0["content"]?.contains("<tool_response>") ?? false })
    }

    func testParseToolCallNormalizesBlankOrInvalidArguments() {
        // Blank arguments -> valid {} so the sentinel stays parseable.
        let blank = ResponsesCompat.parse([
            "input": [["type": "function_call", "name": "f", "arguments": ""]],
        ])
        XCTAssertTrue(blank.messages.contains {
            $0["content"]?.contains("\"arguments\": {}") ?? false
        })
        // Non-JSON arguments -> {} (never splice malformed JSON).
        let bad = ResponsesCompat.parse([
            "input": [["type": "function_call", "name": "f", "arguments": "not json"]],
        ])
        XCTAssertTrue(bad.messages.contains {
            $0["content"]?.contains("\"arguments\": {}") ?? false
        })
        // Object-shaped arguments (tolerated) -> serialized JSON, not dropped.
        let obj = ResponsesCompat.parse([
            "input": [["type": "function_call", "name": "f", "arguments": ["a": 1]]],
        ])
        XCTAssertTrue(obj.messages.contains {
            ($0["content"]?.contains("<tool_call>") ?? false)
                && ($0["content"]?.contains("\"a\"") ?? false)
        })
    }

    func testParseToolOutputStringifiesStructuredResults() {
        // A JSON-object tool result must reach the model, not be dropped.
        let p = ResponsesCompat.parse([
            "input": [["type": "function_call_output", "call_id": "c1",
                       "output": ["temp": 70]]],
        ])
        let resp = p.messages.first { $0["content"]?.contains("<tool_response>") ?? false }
        XCTAssertNotNil(resp)
        XCTAssertTrue(resp?["content"]?.contains("\"temp\"") ?? false)
        XCTAssertFalse(resp?["content"]?.contains("<tool_response></tool_response>") ?? true)

        // An arbitrary JSON array (not content parts) is serialized, not dropped.
        let arr = ResponsesCompat.parse([
            "input": [["type": "function_call_output", "call_id": "c1",
                       "output": [["id": 1], ["id": 2]]]],
        ])
        let aresp = arr.messages.first { $0["content"]?.contains("<tool_response>") ?? false }
        XCTAssertTrue(aresp?["content"]?.contains("\"id\"") ?? false)
    }

    func testParseSkipsNonFunctionTools() {
        let p = ResponsesCompat.parse([
            "input": "hi",
            "tools": [
                ["type": "web_search"],
                ["type": "function", "name": "f", "parameters": ["type": "object"]],
            ],
        ])
        XCTAssertEqual(p.tools.count, 1)
        XCTAssertEqual(p.tools.first?.name, "f")
    }

    func testParseStreamFlagDefaultsFalse() {
        XCTAssertFalse(ResponsesCompat.parse(["input": "hi"]).stream)
        XCTAssertTrue(ResponsesCompat.parse(["input": "hi", "stream": true]).stream)
    }

    func testResponseTextShape() {
        let r = ResponsesCompat.response(
            id: "resp_1", model: "m", text: "hello", toolCalls: [],
            createdAt: 123, inputTokens: 3, outputTokens: 4)
        XCTAssertEqual(r["object"] as? String, "response")
        XCTAssertEqual(r["status"] as? String, "completed")
        let output = r["output"] as? [[String: Any]]
        XCTAssertEqual(output?.first?["type"] as? String, "message")
        let content = output?.first?["content"] as? [[String: Any]]
        XCTAssertEqual(content?.first?["type"] as? String, "output_text")
        XCTAssertEqual(content?.first?["text"] as? String, "hello")
        let usage = r["usage"] as? [String: Any]
        XCTAssertEqual(usage?["total_tokens"] as? Int, 7)
    }

    func testResponseFunctionCallShape() {
        let r = ResponsesCompat.response(
            id: "resp_1", model: "m", text: "",
            toolCalls: [ToolCalling.ParsedToolCall(name: "f", argumentsJSON: "{\"a\":1}")],
            createdAt: 123, inputTokens: 1, outputTokens: 2)
        let output = r["output"] as? [[String: Any]]
        // Empty text + tool call ⇒ only the function_call item.
        XCTAssertEqual(output?.count, 1)
        let item = output?.first
        XCTAssertEqual(item?["type"] as? String, "function_call")
        XCTAssertEqual(item?["name"] as? String, "f")
        XCTAssertEqual(item?["arguments"] as? String, "{\"a\":1}")
        XCTAssertNotNil(item?["call_id"] as? String)
    }
}
