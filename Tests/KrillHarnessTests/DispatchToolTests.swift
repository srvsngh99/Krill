import XCTest
@testable import KrillHarness

final class DispatchToolTests: XCTestCase {

    func testEnqueuesSpawnRequest() async {
        let queue = SpawnQueue()
        let tool = DispatchTool(queue: queue)
        let result = await tool.run(argumentsJSON: #"{"task":"explore the auth module","title":"auth"}"#)
        XCTAssertFalse(result.isError)
        let pending = queue.drain()
        XCTAssertEqual(pending.count, 1)
        XCTAssertEqual(pending.first?.title, "auth")
        XCTAssertEqual(pending.first?.task, "explore the auth module")
    }

    func testDerivesTitleWhenMissing() async {
        let queue = SpawnQueue()
        let result = await DispatchTool(queue: queue).run(argumentsJSON: #"{"task":"add unit tests for the parser module please"}"#)
        XCTAssertFalse(result.isError)
        let req = queue.drain().first
        XCTAssertEqual(req?.title, "add unit tests for the")   // first 5 words
    }

    func testMissingTaskIsError() async {
        let queue = SpawnQueue()
        let result = await DispatchTool(queue: queue).run(argumentsJSON: #"{"title":"x"}"#)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(queue.drain().isEmpty, "no request enqueued on error")
    }

    func testDispatchToolIsReadOnlyAtParent() {
        // The tool only enqueues; spawning never prompts. The child's own actions
        // are gated in its session.
        XCTAssertTrue(DispatchTool(queue: SpawnQueue()).isReadOnly)
    }
}
