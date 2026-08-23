import XCTest
@testable import KrillHarness

private final class ExecuteQuestionGate: UserQuestionGate, @unchecked Sendable {
    private let lock = NSLock()
    private let answer: UserAnswer?
    private var continuation: CheckedContinuation<UserAnswer, Never>?
    private var count = 0

    init(_ answer: UserAnswer?) { self.answer = answer }

    func ask(_ question: UserQuestion) async -> UserAnswer {
        increment()
        if let answer { return answer }
        return await withCheckedContinuation { continuation in
            lock.lock(); self.continuation = continuation; lock.unlock()
        }
    }

    private func increment() { lock.lock(); count += 1; lock.unlock() }

    func cancelPending() {
        lock.lock(); let pending = continuation; continuation = nil; lock.unlock()
        pending?.resume(returning: .declinedAnswer)
    }

    var questionCount: Int { lock.lock(); defer { lock.unlock() }; return count }
}

final class RequestExecuteToolTests: XCTestCase {
    func testReadOnlyAndAllowedInPlan() {
        let tool = RequestExecuteTool(
            box: PermissionBox(mode: .plan), gate: ExecuteQuestionGate(.declinedAnswer))
        XCTAssertTrue(tool.isReadOnly)
        XCTAssertEqual(
            PermissionPolicy(mode: .plan)
                .decision(toolName: tool.name, isReadOnly: tool.isReadOnly),
            .allow)
    }

    func testFirstOptionPromotesToAcceptEditsWithReminder() async {
        let box = PermissionBox(mode: .plan)
        let result = await RequestExecuteTool(
            box: box, gate: ExecuteQuestionGate(UserAnswer(text: "yes", optionIndex: 0)))
            .run(argumentsJSON: #"{"summary":"Implement it"}"#)
        XCTAssertFalse(result.isError)
        XCTAssertEqual(box.effective, .acceptEdits)
        XCTAssertTrue(result.content.contains("accept-edits"))
        XCTAssertTrue(result.content.contains(
            "Ignore the earlier PLAN MODE instruction in your system prompt — it no longer applies."))
        XCTAssertEqual(result.effect, .permissionMode(.acceptEdits))
    }

    func testSecondOptionPromotesToAsk() async {
        let box = PermissionBox(mode: .plan)
        let result = await RequestExecuteTool(
            box: box, gate: ExecuteQuestionGate(UserAnswer(text: "ask", optionIndex: 1)))
            .run(argumentsJSON: #"{"summary":"Implement"}"#)
        XCTAssertFalse(result.isError)
        XCTAssertEqual(box.effective, .ask)
    }

    func testDeclineThirdOptionAndTimeoutLeaveBoxUnchanged() async {
        for answer in [UserAnswer(text: "no", optionIndex: 2), .declinedAnswer] {
            let box = PermissionBox(mode: .plan)
            let result = await RequestExecuteTool(box: box, gate: ExecuteQuestionGate(answer))
                .run(argumentsJSON: #"{"summary":"Implement"}"#)
            XCTAssertFalse(result.isError)
            XCTAssertEqual(box.effective, .plan)
            XCTAssertFalse(result.content.contains("call request_execute again"))
        }

        let timedBox = PermissionBox(mode: .plan)
        let timed = await RequestExecuteTool(
            box: timedBox, gate: ExecuteQuestionGate(nil), timeout: 0.02)
            .run(argumentsJSON: #"{"summary":"Implement"}"#)
        XCTAssertFalse(timed.isError)
        XCTAssertEqual(timedBox.effective, .plan)
    }

    func testAdaptiveSelfPromotesWithoutQuestion() async {
        let box = PermissionBox(mode: .adaptive)
        let gate = ExecuteQuestionGate(.declinedAnswer)
        let result = await RequestExecuteTool(box: box, gate: gate).run(argumentsJSON: "{}")
        XCTAssertFalse(result.isError)
        XCTAssertEqual(gate.questionCount, 0)
        XCTAssertEqual(box.effective, .acceptEdits)
        XCTAssertEqual(result.effect, .permissionMode(.acceptEdits))
    }

    func testSecondCallAndAlreadyExecutingAreNoOps() async {
        let box = PermissionBox(mode: .plan)
        let gate = ExecuteQuestionGate(UserAnswer(optionIndex: 0))
        let tool = RequestExecuteTool(box: box, gate: gate)
        _ = await tool.run(argumentsJSON: #"{"summary":"One"}"#)
        let second = await tool.run(argumentsJSON: #"{"summary":"Two"}"#)
        XCTAssertFalse(second.isError)
        XCTAssertTrue(second.content.contains("already implementing"))
        XCTAssertEqual(gate.questionCount, 1)
        XCTAssertNil(second.effect)
    }

    func testMissingSummaryDoesNotHardFail() async {
        let box = PermissionBox(mode: .plan)
        let result = await RequestExecuteTool(
            box: box, gate: ExecuteQuestionGate(UserAnswer(optionIndex: 2)))
            .run(argumentsJSON: "{}")
        XCTAssertFalse(result.isError)
        XCTAssertEqual(box.effective, .plan)
    }
}
