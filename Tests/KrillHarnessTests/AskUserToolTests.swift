import XCTest
@testable import KrillHarness
import KrillTooling

private final class StubQuestionGate: UserQuestionGate, @unchecked Sendable {
    private let lock = NSLock()
    private let answer: UserAnswer?
    private var continuation: CheckedContinuation<UserAnswer, Never>?
    private(set) var questions: [UserQuestion] = []
    private(set) var cancellationCount = 0

    init(answer: UserAnswer?) { self.answer = answer }

    func ask(_ question: UserQuestion) async -> UserAnswer {
        if let answer = record(question) { return answer }
        return await withCheckedContinuation { continuation in
            lock.lock()
            self.continuation = continuation
            lock.unlock()
        }
    }

    private func record(_ question: UserQuestion) -> UserAnswer? {
        lock.lock(); defer { lock.unlock() }
        questions.append(question)
        return answer
    }

    func cancelPending() {
        lock.lock()
        cancellationCount += 1
        let pending = continuation
        continuation = nil
        lock.unlock()
        pending?.resume(returning: .declinedAnswer)
    }

    var recordedQuestions: [UserQuestion] {
        lock.lock(); defer { lock.unlock() }
        return questions
    }

    var recordedCancellations: Int {
        lock.lock(); defer { lock.unlock() }
        return cancellationCount
    }
}

final class AskUserToolTests: XCTestCase {
    func testSchemaAndReadOnlyContract() {
        let tool = AskUserTool(gate: StubQuestionGate(answer: .declinedAnswer))
        XCTAssertTrue(tool.isReadOnly)
        XCTAssertNotNil(tool.parametersJSON.data(using: .utf8).flatMap {
            try? JSONSerialization.jsonObject(with: $0)
        })
        XCTAssertTrue(ToolCalling.argsSatisfySchema(
            argumentsJSON: #"{"question":"Which?","options":["A","B"]}"#,
            parametersJSON: tool.parametersJSON))
    }

    func testHappyPathIncludesAnswerAndNumberedOptions() async {
        let gate = StubQuestionGate(answer: UserAnswer(text: "B", optionIndex: 1))
        let result = await AskUserTool(gate: gate).run(
            argumentsJSON: #"{"header":"Choice","question":"Which?","options":["A","B"]}"#)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("User answered: B"))
        XCTAssertTrue(result.content.contains("1. A"))
        XCTAssertTrue(result.content.contains("2. B"))
        XCTAssertNotNil(result.display)
    }

    func testToleratesObjectDelimitedAndEmptyOptionShapes() async {
        let cases = [
            #"{"question":"Q","options":[{"label":"A"},{"text":"B"}]}"#,
            #"{"prompt":"Q","options":"A, B; C|D"}"#,
            #"{"text":"Q","options":[]}"#,
        ]
        for arguments in cases {
            let gate = StubQuestionGate(answer: UserAnswer(text: "answer", wasFreeText: true))
            let result = await AskUserTool(gate: gate).run(argumentsJSON: arguments)
            XCTAssertFalse(result.isError, arguments)
            XCTAssertEqual(gate.recordedQuestions.count, 1)
        }
    }

    func testOptionsAreDedupedAndCapped() async {
        let gate = StubQuestionGate(answer: UserAnswer(text: "A"))
        _ = await AskUserTool(gate: gate).run(
            argumentsJSON: #"{"question":"Q","options":"A,a,B,C,D,E,F,G"}"#)
        XCTAssertEqual(gate.recordedQuestions[0].options, ["A", "B", "C", "D", "E", "F"])
    }

    func testMissingQuestionIsError() async {
        let gate = StubQuestionGate(answer: UserAnswer(text: "x"))
        let result = await AskUserTool(gate: gate).run(argumentsJSON: #"{"options":["A"]}"#)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(gate.recordedQuestions.isEmpty)
    }

    func testDeclineIsNonErrorAndDiscouragesRetry() async {
        let result = await AskUserTool(gate: StubQuestionGate(answer: .declinedAnswer))
            .run(argumentsJSON: #"{"question":"Q"}"#)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("state the assumption you made, and do not ask again"))
    }

    func testTimeoutCancelsPendingQuestion() async {
        let gate = StubQuestionGate(answer: nil)
        let result = await AskUserTool(gate: gate, timeout: 0.05)
            .run(argumentsJSON: #"{"question":"Q"}"#)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("declined"))
        XCTAssertGreaterThanOrEqual(gate.recordedCancellations, 1)
    }
}
