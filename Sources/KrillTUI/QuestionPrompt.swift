import Foundation

/// Pure selection and free-text state for an interactive agent question.
/// Terminal rendering and async continuation handling live in the CLI target;
/// keeping this state in KrillTUI makes wrapping and key behavior unit-testable.
public struct QuestionPrompt: Equatable, Sendable {
    public let options: [String]
    public private(set) var selected: Int
    public private(set) var isTyping: Bool
    public private(set) var freeText: String

    /// There is always one final row for a custom response, even with no options.
    public var rowCount: Int { options.count + 1 }
    public var freeTextIndex: Int { options.count }
    public var isFreeTextSelected: Bool { selected == freeTextIndex }
    public var selectedOptionIndex: Int? {
        options.indices.contains(selected) ? selected : nil
    }

    public init(options: [String], selected: Int = 0) {
        self.options = options
        self.selected = min(max(0, selected), options.count)
        self.isTyping = false
        self.freeText = ""
    }

    public mutating func selectNext() {
        selected = (selected + 1) % rowCount
    }

    public mutating func selectPrevious() {
        selected = (selected - 1 + rowCount) % rowCount
    }

    /// Maps the displayed 1-based option shortcut to an option row. The custom
    /// response row deliberately has no digit shortcut.
    public func index(forDigit digit: Int) -> Int? {
        let index = digit - 1
        return options.indices.contains(index) ? index : nil
    }

    public mutating func select(index: Int) {
        guard (0..<rowCount).contains(index) else { return }
        selected = index
    }

    public mutating func beginTyping() {
        selected = freeTextIndex
        isTyping = true
    }

    public mutating func append(_ character: Character) {
        guard isTyping else { return }
        freeText.append(character)
    }

    public mutating func backspace() {
        guard isTyping, !freeText.isEmpty else { return }
        freeText.removeLast()
    }

    public mutating func stopTyping() { isTyping = false }
}
