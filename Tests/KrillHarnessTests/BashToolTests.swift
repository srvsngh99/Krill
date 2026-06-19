import XCTest
@testable import KrillHarness

final class BashToolTests: XCTestCase {
    func testEchoRunsAndReturnsStdout() async {
        let r = await BashTool().run(argumentsJSON: #"{"command":"echo hello-krill"}"#)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("hello-krill"), "got: \(r.content)")
    }

    func testMissingCommandIsAnError() async {
        let r = await BashTool().run(argumentsJSON: #"{}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("command"))
    }

    func testMalformedArgsIsAnError() async {
        let r = await BashTool().run(argumentsJSON: "not json")
        XCTAssertTrue(r.isError)
    }

    func testNonZeroExitIsReportedAsError() async {
        let r = await BashTool().run(argumentsJSON: #"{"command":"exit 3"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("Exit code 3"), "got: \(r.content)")
    }

    func testCombinedStderr() async {
        let r = await BashTool().run(argumentsJSON: #"{"command":"echo err 1>&2"}"#)
        XCTAssertFalse(r.isError)
        XCTAssertTrue(r.content.contains("err"))
    }

    func testTimeoutTerminatesLongCommand() async {
        let r = await BashTool(timeout: 1).run(argumentsJSON: #"{"command":"sleep 5"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("timed out"), "got: \(r.content)")
    }

    func testTimeoutKillsTermIgnoringChild() async {
        // The child traps SIGTERM, so only SIGKILL escalation can stop it.
        // Must return well before the child's own sleep would finish.
        let start = Date()
        let r = await BashTool(timeout: 1).run(
            argumentsJSON: #"{"command":"trap '' TERM; sleep 10"}"#)
        XCTAssertTrue(r.isError)
        XCTAssertTrue(r.content.contains("timed out"))
        XCTAssertLessThan(
            Date().timeIntervalSince(start), 6,
            "SIGKILL escalation must stop a TERM-ignoring child")
    }

    func testTruncationKeepsTailOnCharacterBoundary() async {
        // Emit more than the cap of a multibyte char; the tail must decode
        // cleanly (no dropped output, no mojibake) and carry the marker.
        let tool = BashTool(maxOutputBytes: 64)
        let r = await tool.run(argumentsJSON: #"{"command":"for i in $(seq 1 200); do printf 'é'; done"}"#)
        XCTAssertTrue(r.content.contains("truncated"))
        XCTAssertTrue(r.content.contains("é"), "kept tail must be valid UTF-8")
    }
}
