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

    func testUnbalancedOpenTagDropsTailAsReasoning() {
        // max_tokens-truncated reasoning: model opened <think> but
        // never reached </think> before the token budget ran out.
        // The whole open-to-end span is reasoning and must not be
        // shown to the client. The pre-tag prefix (typically empty
        // for Qwen 3) is preserved.
        let raw = "<think>incomplete reasoning with no close..."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "")
        XCTAssertEqual(thinking, "incomplete reasoning with no close...")
    }

    func testUnbalancedOpenTagPreservesPretagPrefix() {
        let raw = "preamble text <think>truncated reasoning..."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "preamble text")
        XCTAssertEqual(thinking, "truncated reasoning...")
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

    // MARK: - Gemma 4 reasoning channels

    func testStripsGemmaChannelMarker() {
        // The exact shape observed from Gemma-4-12B on a plain prompt.
        let raw = "<|channel>thought\n<channel|>Brown"
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "Brown")
        XCTAssertEqual(thinking, "thought")
    }

    func testStripsAllGemmaChannels() {
        // The model may open more than one channel; every span is removed.
        let raw = "<|channel>plan<channel|>The answer "
            + "<|channel>double-check<channel|>is 42."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "The answer is 42.")
        XCTAssertTrue(thinking?.contains("plan") == true)
        XCTAssertTrue(thinking?.contains("double-check") == true)
    }

    func testGemmaThinkMarker() {
        let raw = "<|think|>reasoning here<think|>Final answer."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "Final answer.")
        XCTAssertEqual(thinking, "reasoning here")
    }

    func testUnbalancedGemmaChannelDropsToEnd() {
        // Truncated mid-channel (no close): drop from the open marker on.
        let raw = "<|channel>reasoning that never closed"
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, "")
        XCTAssertEqual(thinking, "reasoning that never closed")
    }

    func testGemmaChannelWithNoMarkersUnchanged() {
        let raw = "A plain Gemma answer with no channel."
        let (visible, thinking) = ReasoningParser.strip(raw)
        XCTAssertEqual(visible, raw)
        XCTAssertNil(thinking)
    }

    func testStreamingFilterStripsGemmaChannelAcrossChunks() {
        let f = StreamingReasoningFilter()
        var out = ""
        for chunk in ["<|channel>", "thinking ", "stuff", "<channel|>", "Brown"] {
            out += f.consume(chunk)
        }
        out += f.finish()
        XCTAssertEqual(out, "Brown",
            "Gemma channel reasoning must not reach the streamed output")
    }

    func testStreamingFilterDropsTruncatedGemmaChannel() {
        let f = StreamingReasoningFilter()
        XCTAssertEqual(f.consume("<|channel>"), "")
        XCTAssertEqual(f.consume("partial channel never closed"), "")
        XCTAssertEqual(f.finish(), "")
    }

    func testStreamingFilterPassesNonReasoningAngleTokens() {
        // The repeated-block fix widened scanning; ensure ordinary angle-bracket
        // content (HTML, generics, etc.) still streams through untouched.
        let f = StreamingReasoningFilter()
        var out = ""
        for chunk in ["Here is ", "<html>", "<body>", "hi", "</body>", "</html>"] {
            out += f.consume(chunk)
        }
        out += f.finish()
        XCTAssertEqual(out, "Here is <html><body>hi</body></html>",
            "Non-reasoning angle-bracket content must pass through untouched")
    }

    func testStreamingFilterStripsRepeatedGemmaChannelBlocks() {
        // A degenerate Gemma think-loop emits many channel blocks back to back;
        // the filter must strip every block, not dump everything after the first.
        let f = StreamingReasoningFilter()
        var out = ""
        for chunk in ["<|channel>", "thought", "<channel|>",
                      "<|channel>", "thought", "<channel|>",
                      "<|channel>", "thought", "<channel|>", "Answer"] {
            out += f.consume(chunk)
        }
        out += f.finish()
        XCTAssertEqual(out, "Answer",
            "Every Gemma channel block must be stripped, even a long run of them")
    }

    // MARK: - Streaming filter

    func testStreamingFilterEmitsOnlyPostReasoningTokens() {
        let f = StreamingReasoningFilter()
        var out = ""
        for chunk in ["<think>", "let me reason", "</think>", "\n\n", "The answer ", "is 42."] {
            out += f.consume(chunk)
        }
        out += f.finish()
        XCTAssertEqual(out, "The answer is 42.",
            "Reasoning chunks must not reach the streamed output")
    }

    func testStreamingFilterHoldsAmbiguousPrefixUntilDisambiguated() {
        let f = StreamingReasoningFilter()
        // `<` alone could be the start of a tag. The filter must
        // hold it, NOT emit it immediately.
        XCTAssertEqual(f.consume("<"), "")
        XCTAssertEqual(f.consume("hello>"), "<hello>",
            "Disambiguated to literal text once it cannot be an open tag")
        XCTAssertEqual(f.consume(" world"), " world")
        XCTAssertEqual(f.finish(), "")
    }

    func testStreamingFilterDropsTruncatedReasoning() {
        // Mid-reasoning stream end: no </think> ever arrives.
        // finish() must drop the partial reasoning rather than
        // flushing it.
        let f = StreamingReasoningFilter()
        XCTAssertEqual(f.consume("<think>"), "")
        XCTAssertEqual(f.consume("partial reasoning that never closed"), "")
        XCTAssertEqual(f.finish(), "")
    }

    func testStreamingFilterPassesThroughWhenNoTagEverAppears() {
        let f = StreamingReasoningFilter()
        var out = ""
        for chunk in ["Plain ", "answer", " streamed ", "in pieces."] {
            out += f.consume(chunk)
        }
        out += f.finish()
        XCTAssertEqual(out, "Plain answer streamed in pieces.")
    }

    func testStreamingFilterHandlesClosingTagSplitAcrossChunks() {
        let f = StreamingReasoningFilter()
        XCTAssertEqual(f.consume("<think>reasoning</thi"), "")
        XCTAssertEqual(f.consume("nk>visible"), "visible")
        XCTAssertEqual(f.finish(), "")
    }
}
