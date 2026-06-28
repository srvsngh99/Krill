import XCTest
@testable import KrillHarness

/// The classic `krill code` renderer routes through the shared `foldAgentEvent`
/// seam. These tests pin the exact line output so it can't silently drift.
final class LineAgentRendererTests: XCTestCase {

    /// Collect emitted lines instead of printing.
    private func makeRenderer() -> (LineAgentRenderer, () -> [String]) {
        final class Sink: @unchecked Sendable { var lines: [String] = [] }
        let sink = Sink()
        let r = LineAgentRenderer(emit: { sink.lines.append($0) })
        return (r, { sink.lines })
    }

    func testAssistantPreambleThenToolThenFinal() {
        let (r, lines) = makeRenderer()
        r.handle(.assistantTurn(text: "Let me run it."))
        r.handle(.toolStarted(name: "bash", argumentsJSON: #"{"command":"python hello.py"}"#))
        r.handle(.toolFinished(ToolInvocation(
            name: "bash", argumentsJSON: #"{"command":"python hello.py"}"#,
            result: ToolResult(content: "4", isError: false))))
        r.handle(.finalAnswer("The output is 4."))
        XCTAssertEqual(lines(), [
            "Let me run it.",
            #"  [*] bash({"command":"python hello.py"})"#,
            "      4",
            "The output is 4.",
        ])
    }

    func testErrorMarkerForFailedTool() {
        let (r, lines) = makeRenderer()
        r.handle(.toolStarted(name: "bash", argumentsJSON: #"{"command":"nope"}"#))
        r.handle(.toolFinished(ToolInvocation(
            name: "bash", argumentsJSON: #"{"command":"nope"}"#,
            result: ToolResult(content: "command not found", isError: true))))
        XCTAssertEqual(lines(), [
            #"  [x] bash({"command":"nope"})"#,
            "      command not found",
        ])
    }

    /// A denied/unknown tool skips `toolStarted`; the chip must still print once
    /// (driven by `foldAgentEvent`'s `chipShown` tracking).
    func testDeniedToolWithoutStartStillPrintsChip() {
        let (r, lines) = makeRenderer()
        r.handle(.toolFinished(ToolInvocation(
            name: "write_file", argumentsJSON: #"{"path":"x"}"#,
            result: ToolResult(content: "Error: denied", isError: true))))
        XCTAssertEqual(lines(), [
            #"  [x] write_file({"path":"x"})"#,
            "      Error: denied",
        ])
    }

    func testMultilineResultUnderDefaultCapShowsAllLines() {
        let (r, lines) = makeRenderer()
        r.handle(.toolStarted(name: "read_file", argumentsJSON: "{}"))
        r.handle(.toolFinished(ToolInvocation(
            name: "read_file", argumentsJSON: "{}",
            result: ToolResult(content: "l1\nl2\nl3\nl4", isError: false))))
        // default cap is 20, so all four lines show
        XCTAssertEqual(lines(), [
            "  [*] read_file({})",
            "      l1", "      l2", "      l3", "      l4",
        ])
    }

    func testCapActuallyTruncates() {
        final class Sink: @unchecked Sendable { var lines: [String] = [] }
        let sink = Sink()
        let r = LineAgentRenderer(maxResultLines: 2, emit: { sink.lines.append($0) })
        r.handle(.toolStarted(name: "read_file", argumentsJSON: "{}"))
        r.handle(.toolFinished(ToolInvocation(
            name: "read_file", argumentsJSON: "{}",
            result: ToolResult(content: "a\nb\nc\nd", isError: false))))
        XCTAssertEqual(sink.lines, ["  [*] read_file({})", "      a", "      b"])
    }

    func testIterationLimitAndCancelledUseSharedNotes() {
        let (r1, l1) = makeRenderer()
        r1.handle(.iterationLimitReached)
        XCTAssertEqual(l1(), ["[stopped at the iteration limit without a final answer]"])

        let (r2, l2) = makeRenderer()
        r2.handle(.cancelled)
        XCTAssertEqual(l2(), ["(cancelled)"])
    }

    func testEmptyAssistantTurnEmitsNothing() {
        let (r, lines) = makeRenderer()
        r.handle(.assistantTurn(text: "   \n  "))
        XCTAssertTrue(lines().isEmpty)
    }
}
