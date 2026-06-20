import XCTest
@testable import KrillHarness
import KrillTooling

final class AgentEntryFoldTests: XCTestCase {

    private func fold(_ events: [AgentEvent]) -> [AgentEntry] {
        var entries: [AgentEntry] = []
        var chip = false
        for e in events { foldAgentEvent(e, into: &entries, chipShown: &chip) }
        return entries
    }

    func testAssistantTurnSkippedWhenEmpty() {
        XCTAssertEqual(fold([.assistantTurn(text: "   ")]), [])
        XCTAssertEqual(fold([.assistantTurn(text: "hi")]), [.assistant("hi")])
    }

    func testToolStartedThenFinishedProducesChipThenResult() {
        let inv = ToolInvocation(
            name: "grep", argumentsJSON: #"{"pattern":"x"}"#,
            result: ToolResult(content: "match", isError: false))
        let out = fold([.toolStarted(name: "grep", argumentsJSON: #"{"pattern":"x"}"#), .toolFinished(inv)])
        XCTAssertEqual(out, [
            .toolCall(name: "grep", args: #"{"pattern":"x"}"#),
            .toolResult(content: "match", isError: false),
        ])
    }

    func testDeniedToolWithoutStartStillGetsChip() {
        // A denied/unknown tool never emits toolStarted; the chip is synthesized
        // so the observation is never orphaned.
        let inv = ToolInvocation(
            name: "bash", argumentsJSON: #"{"command":"rm"}"#,
            result: ToolResult(content: "Permission denied", isError: true))
        let out = fold([.toolFinished(inv)])
        XCTAssertEqual(out, [
            .toolCall(name: "bash", args: #"{"command":"rm"}"#),
            .toolResult(content: "Permission denied", isError: true),
        ])
    }

    func testFinalAnswerAndLifecycleNotes() {
        XCTAssertEqual(fold([.finalAnswer("done")]), [.assistant("done")])
        XCTAssertEqual(fold([.cancelled]), [.note("(cancelled)")])
        XCTAssertEqual(fold([.iterationLimitReached]).count, 1)
    }

    func testChipFlagResetsBetweenCalls() {
        // After a finished call resets the flag, the next started call emits its
        // own chip (no leakage that would drop the second chip).
        let inv1 = ToolInvocation(name: "a", argumentsJSON: "{}", result: ToolResult(content: "1"))
        let out = fold([
            .toolStarted(name: "a", argumentsJSON: "{}"), .toolFinished(inv1),
            .toolStarted(name: "b", argumentsJSON: "{}"),
        ])
        XCTAssertEqual(out, [
            .toolCall(name: "a", args: "{}"),
            .toolResult(content: "1", isError: false),
            .toolCall(name: "b", args: "{}"),
        ])
    }
}
