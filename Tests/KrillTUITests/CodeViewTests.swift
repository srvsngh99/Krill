import XCTest
@testable import KrillTUI

final class CodeViewTests: XCTestCase {

    func testUserTaskHangingIndent() {
        let lines = CodeView.userTask("change the greeting to hello krill please now", width: 20)
        XCTAssertEqual(lines.first?.style, .user)
        XCTAssertTrue(lines[0].text.hasPrefix("> "), "first line carries the marker")
        if lines.count > 1 {
            XCTAssertTrue(lines[1].text.hasPrefix("  "), "continuation is hanging-indented")
        }
        // Every line stays within the width.
        for l in lines { XCTAssertLessThanOrEqual(l.text.count, 20) }
    }

    func testToolCallChipClipsToWidth() {
        let lines = CodeView.toolCall(
            name: "edit_file",
            argumentsJSON: #"{"path":"a/very/long/path/that/overflows.txt","old":"x","new":"y"}"#,
            width: 30)
        XCTAssertEqual(lines.count, 1)
        XCTAssertEqual(lines[0].style, .toolName)
        XCTAssertEqual(lines[0].text.count, 30, "chip is clipped to exactly the width")
        XCTAssertTrue(lines[0].text.hasSuffix("\u{2026}"), "clip marks the cut with an ellipsis")
        XCTAssertTrue(lines[0].text.hasPrefix("\u{25B8} edit_file"))
    }

    func testToolCallChipSurfacesPath() {
        // The chip surfaces the salient path argument, not the raw JSON.
        let lines = CodeView.toolCall(
            name: "edit_file", argumentsJSON: #"{"path":"a.txt","old_string":"x","new_string":"y"}"#, width: 60)
        XCTAssertEqual(lines[0].text, "\u{25B8} edit_file  a.txt")
    }

    func testToolCallChipFallsBackToJSON() {
        // No salient field: fall back to the compacted JSON after the name.
        let lines = CodeView.toolCall(name: "grep", argumentsJSON: #"{"q":"x"}"#, width: 40)
        XCTAssertEqual(lines[0].text, #"▸ grep  {"q":"x"}"#)
    }

    func testToolResultTagsDiffLines() {
        let content = "Edited g.txt: 1 replacement(s).\n- world\n+ krill"
        let lines = CodeView.toolResult(content: content, isError: false, width: 40, maxLines: 20)
        XCTAssertEqual(lines.count, 3)
        XCTAssertEqual(lines[0].style, .toolOk)
        XCTAssertEqual(lines[1].style, .diffDel)
        XCTAssertEqual(lines[2].style, .diffAdd)
        // All indented.
        for l in lines { XCTAssertTrue(l.text.hasPrefix("    ")) }
    }

    func testToolResultErrorStyle() {
        let lines = CodeView.toolResult(
            content: "Permission denied: plan mode is read-only", isError: true, width: 60, maxLines: 20)
        XCTAssertEqual(lines.first?.style, .toolError)
    }

    func testToolResultTruncatesWithHint() {
        let content = (1...30).map { "line \($0)" }.joined(separator: "\n")
        let lines = CodeView.toolResult(content: content, isError: false, width: 40, maxLines: 10)
        // 10 shown + 1 truncation hint.
        XCTAssertEqual(lines.count, 11)
        XCTAssertEqual(lines.last?.style, .dim)
        XCTAssertTrue(lines.last!.text.contains("20 more lines"))
    }

    func testToolResultSingleHiddenLineGrammar() {
        let content = (1...11).map { "line \($0)" }.joined(separator: "\n")
        let lines = CodeView.toolResult(content: content, isError: false, width: 40, maxLines: 10)
        XCTAssertTrue(lines.last!.text.contains("1 more line)"), "singular grammar for one hidden line")
    }

    func testLongResultLineWraps() {
        let long = String(repeating: "x", count: 100)
        let lines = CodeView.toolResult(content: long, isError: false, width: 24, maxLines: 20)
        XCTAssertGreaterThan(lines.count, 1, "an over-wide observation line wraps")
        for l in lines { XCTAssertLessThanOrEqual(l.text.count, 24) }
    }
}
