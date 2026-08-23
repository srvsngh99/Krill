import Testing
@testable import KrillTUI

@Suite("Question prompt")
struct QuestionPromptTests {
    @Test func selectionWrapsInBothDirections() {
        var prompt = QuestionPrompt(options: ["Alpha", "Beta"])
        prompt.selectPrevious()
        #expect(prompt.selected == prompt.freeTextIndex)
        prompt.selectNext()
        #expect(prompt.selected == 0)
        prompt.selectNext()
        prompt.selectNext()
        prompt.selectNext()
        #expect(prompt.selected == 0)
    }

    @Test func singleOptionStillIncludesFreeTextRow() {
        var prompt = QuestionPrompt(options: ["Only"])
        #expect(prompt.rowCount == 2)
        #expect(prompt.selectedOptionIndex == 0)
        prompt.selectNext()
        #expect(prompt.isFreeTextSelected)
        #expect(prompt.selectedOptionIndex == nil)
    }

    @Test func emptyOptionsHasOnlyFreeTextRow() {
        var prompt = QuestionPrompt(options: [])
        #expect(prompt.rowCount == 1)
        #expect(prompt.selected == prompt.freeTextIndex)
        #expect(prompt.isFreeTextSelected)
        prompt.selectNext()
        prompt.selectPrevious()
        #expect(prompt.selected == 0)
    }

    @Test func digitMappingRejectsOutOfRangeValues() {
        let prompt = QuestionPrompt(options: ["One", "Two", "Three"])
        #expect(prompt.index(forDigit: 1) == 0)
        #expect(prompt.index(forDigit: 3) == 2)
        #expect(prompt.index(forDigit: 0) == nil)
        #expect(prompt.index(forDigit: 4) == nil)
        #expect(prompt.index(forDigit: 9) == nil)
    }

    @Test func freeTextRowIsAlwaysLast() {
        #expect(QuestionPrompt(options: []).freeTextIndex == 0)
        #expect(QuestionPrompt(options: ["A"]).freeTextIndex == 1)
        #expect(QuestionPrompt(options: ["A", "B"]).freeTextIndex == 2)
    }

    @Test func freeTextEditingIsPureState() {
        var prompt = QuestionPrompt(options: ["A"])
        prompt.beginTyping()
        prompt.append("h")
        prompt.append("i")
        prompt.backspace()
        #expect(prompt.freeText == "h")
        #expect(prompt.isTyping)
        #expect(prompt.isFreeTextSelected)
    }
}
