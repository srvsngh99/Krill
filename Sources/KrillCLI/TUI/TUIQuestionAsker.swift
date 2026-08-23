import Foundation
import KrillHarness

/// Bridges an agent tool's async question to the main TUI render/key loop. One
/// request may be parked at a time; unlike permission approval there is no
/// session-sticky answer.
final class TUIQuestionAsker: UserQuestionGate, @unchecked Sendable {
    private let lock = NSLock()
    private var question: UserQuestion?
    private var continuation: CheckedContinuation<UserAnswer, Never>?

    // MARK: UserQuestionGate (background task)

    func ask(_ question: UserQuestion) async -> UserAnswer {
        await withCheckedContinuation { (cont: CheckedContinuation<UserAnswer, Never>) in
            register(question, cont: cont)
        }
    }

    private func register(
        _ question: UserQuestion,
        cont: CheckedContinuation<UserAnswer, Never>
    ) {
        lock.lock()
        self.question = question
        continuation = cont
        lock.unlock()
    }

    func cancelPending() {
        resolve(UserAnswer(text: "", optionIndex: nil, wasFreeText: false, declined: true))
    }

    // MARK: Main task

    func pending() -> UserQuestion? {
        lock.lock()
        defer { lock.unlock() }
        return question
    }

    func resolve(option index: Int) {
        guard let question = pending(), question.options.indices.contains(index) else { return }
        resolve(UserAnswer(
            text: question.options[index], optionIndex: index,
            wasFreeText: false, declined: false))
    }

    func resolve(text: String) {
        let answer = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !answer.isEmpty else { cancelPending(); return }
        resolve(UserAnswer(text: answer, optionIndex: nil, wasFreeText: true, declined: false))
    }

    func decline() { cancelPending() }

    private func resolve(_ answer: UserAnswer) {
        lock.lock()
        guard let cont = continuation else { lock.unlock(); return }
        continuation = nil
        question = nil
        lock.unlock()
        cont.resume(returning: answer)
    }
}
