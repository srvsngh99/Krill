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
}
