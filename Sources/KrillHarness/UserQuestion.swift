import Foundation

/// One model-initiated question presented to the user.
public struct UserQuestion: Sendable, Equatable, Hashable {
    public let header: String
    public let question: String
    public let body: String
    public let options: [String]

    public init(header: String, question: String, body: String = "", options: [String] = []) {
        self.header = header
        self.question = question
        self.body = body
        self.options = options
    }
}

/// A structured answer returned by a `UserQuestionGate`.
public struct UserAnswer: Sendable, Equatable, Hashable {
    public let text: String
    public let optionIndex: Int?
    public let wasFreeText: Bool
    public let declined: Bool

    public init(
        text: String = "",
        optionIndex: Int? = nil,
        wasFreeText: Bool = false,
        declined: Bool = false
    ) {
        self.text = text
        self.optionIndex = optionIndex
        self.wasFreeText = wasFreeText
        self.declined = declined
    }

    public static let declinedAnswer = UserAnswer(declined: true)
}

/// Interactive seam shared by ordinary questions and plan approval.
/// Implementations support one pending request and must make `cancelPending()`
/// safe to call when no request is waiting.
public protocol UserQuestionGate: Sendable {
    func ask(_ question: UserQuestion) async -> UserAnswer
    func cancelPending()
}

private enum QuestionRace: Sendable {
    case answer(UserAnswer)
    case timeout
    case cancelled
}

/// Shared timeout/cancellation discipline for both interactive tools.
func askQuestion(
    _ question: UserQuestion,
    gate: any UserQuestionGate,
    timeout: TimeInterval
) async -> UserAnswer {
    await withTaskCancellationHandler {
        await withTaskGroup(of: QuestionRace.self, returning: UserAnswer.self) { group in
            group.addTask { .answer(await gate.ask(question)) }
            group.addTask {
                do {
                    try await Task.sleep(for: .seconds(max(0, timeout)))
                    return .timeout
                } catch {
                    return .cancelled
                }
            }

            let first = await group.next() ?? .cancelled
            group.cancelAll()
            switch first {
            case .answer(let answer):
                return answer
            case .timeout, .cancelled:
                gate.cancelPending()
                // The cancellation callback above resumes continuation-backed
                // gates, allowing the task group to drain before this returns.
                return .declinedAnswer
            }
        }
    } onCancel: {
        gate.cancelPending()
    }
}
