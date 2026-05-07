import XCTest
@testable import KLMCore

final class ToolParserTests: XCTestCase {
    func testExtractsMultipleGemma4ToolCalls() {
        let text = """
        preface
        <|tool_call|>{"name":"search","arguments":{"query":"gemma 4","limit":3}}<tool_call|>
        middle
        <|tool_call|>{"name":"open","arguments":{"id":"doc-1"}}<tool_call|>
        """

        let calls = ToolParser.extractToolCalls(from: text)

        XCTAssertEqual(calls.count, 2)
        XCTAssertEqual(calls[0].name, "search")
        XCTAssertEqual(calls[0].parsedArguments()["query"] as? String, "gemma 4")
        XCTAssertEqual(calls[0].parsedArguments()["limit"] as? Int, 3)
        XCTAssertEqual(calls[1].name, "open")
        XCTAssertEqual(calls[1].parsedArguments()["id"] as? String, "doc-1")
    }

    func testMalformedToolCallIsNotPromotedToExecutableCall() {
        let text = #"<|tool_call|>{"arguments":{"query":"missing name"}}<tool_call|>"#

        XCTAssertTrue(ToolParser.containsToolCall(text))
        XCTAssertTrue(ToolParser.extractToolCalls(from: text).isEmpty)
    }

    func testFormatsToolDefinitionsWithGemma4Delimiters() {
        let tool = ToolParser.ToolDefinition(
            type: "function",
            function: .init(
                name: "search",
                description: "Search docs",
                parameters: .init(
                    type: "object",
                    properties: [
                        "query": .init(type: "string", description: "Search query")
                    ],
                    required: ["query"])))

        let formatted = ToolParser.formatToolDefinitions([tool])

        XCTAssertTrue(formatted.hasPrefix("<|tool|>\n"))
        XCTAssertTrue(formatted.hasSuffix("\n<tool|>\n"))
        XCTAssertTrue(formatted.contains(#""name":"search""#))
        XCTAssertTrue(formatted.contains(#""required":["query"]"#))
    }
}
