import Foundation
import KrillHarness

/// Interactive question gate for the classic `krill code` surface.
/// Blank input and EOF decline; a number selects an option and any other
/// non-empty line is returned as free text.
final class StdinQuestionAsker: UserQuestionGate, @unchecked Sendable {
    private struct Pending {
        let id: UUID
        let continuation: CheckedContinuation<UserAnswer, Never>
    }

    private let lock = NSLock()
    private var pending: Pending?

    func ask(_ question: UserQuestion) async -> UserAnswer {
        let id = UUID()
        return await withCheckedContinuation { continuation in
            let old: Pending? = lock.withLock {
                let old = pending
                pending = Pending(id: id, continuation: continuation)
                return old
            }
            old?.continuation.resume(returning: .declinedAnswer)

            DispatchQueue.global().async { [weak self] in
                var lines: [String] = ["\n  \(question.header)", "  \(question.question)"]
                if !question.body.isEmpty { lines.append("  \(question.body)") }
                for (index, option) in question.options.enumerated() {
                    lines.append("    \(index + 1). \(option)")
                }
                lines.append(question.options.isEmpty
                    ? "  Answer (blank to skip): "
                    : "  Choose 1-\(question.options.count), type an answer, or blank to skip: ")
                FileHandle.standardOutput.write(Data(lines.joined(separator: "\n").utf8))

                let raw = readLine(strippingNewline: true)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                let answer: UserAnswer
                if raw.isEmpty {
                    answer = .declinedAnswer
                } else if let selected = Int(raw), question.options.indices.contains(selected - 1) {
                    answer = UserAnswer(
                        text: question.options[selected - 1],
                        optionIndex: selected - 1,
                        wasFreeText: false,
                        declined: false)
                } else {
                    answer = UserAnswer(
                        text: raw, optionIndex: nil, wasFreeText: true, declined: false)
                }
                self?.resolve(id: id, answer: answer)
            }
        }
    }

    func cancelPending() {
        let continuation = lock.withLock { () -> CheckedContinuation<UserAnswer, Never>? in
            defer { pending = nil }
            return pending?.continuation
        }
        continuation?.resume(returning: .declinedAnswer)
    }

    private func resolve(id: UUID, answer: UserAnswer) {
        let continuation = lock.withLock { () -> CheckedContinuation<UserAnswer, Never>? in
            guard pending?.id == id else { return nil }
            defer { pending = nil }
            return pending?.continuation
        }
        continuation?.resume(returning: answer)
    }
}
