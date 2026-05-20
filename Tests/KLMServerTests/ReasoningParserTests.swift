import XCTest
@testable import KLMServer

final class ReasoningParserTests: XCTestCase {

    func testStripsThinkingTagAndReturnsCapturedContent() {
        let raw = "<thinking>Let me consider the question.</thinking>The answer is 42."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "The answer is 42.")
        XCTAssertEqual(thinking, "Let me consider the question.")
    }

    func testStripsThinkTag() {
        // Qwen 3 / DeepSeek-R1 native reasoning tag.
        let raw = "<think>\nbreak it down\n</think>\n\nThe answer is 42."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "The answer is 42.")
        XCTAssertEqual(thinking, "break it down")
    }

    func testReturnsInputUnchangedWhenNoTag() {
        let raw = "Plain answer with no reasoning markup."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, raw)
        XCTAssertNil(thinking)
    }

    func testUnbalancedOpenTagIsLeftIntact() {
        // Half-tag must not be silently truncated: clients see the
        // raw output (and can complain) instead of an invisibly
        // mangled payload.
        let raw = "<think>incomplete reasoning with no close..."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, raw)
        XCTAssertNil(thinking)
    }

    func testThinkingTagTakesPrecedenceOverThink() {
        // If both tags somehow appear (defensive), <thinking> matches
        // first because we look for it first. This keeps the
        // Anthropic-style server-injected path stable.
        let raw = "<thinking>outer</thinking> middle <think>inner</think> tail"
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(thinking, "outer")
        XCTAssertTrue(visible.contains("<think>inner</think>"),
            "Only the first matched tag is stripped per call")
    }

    func testEmptyThinkingTagYieldsNilCapture() {
        let raw = "<think></think>just text"
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "just text")
        XCTAssertNil(thinking, "Empty captured content must not leak as an empty `thinking` field")
    }
}
