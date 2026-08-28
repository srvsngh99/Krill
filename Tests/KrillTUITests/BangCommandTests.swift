import Testing
@testable import KrillTUI

@Suite("Bang command")
struct BangCommandTests {
    @Test func singleBangIsANonPrivateShellEscape() {
        let bang = BangCommand.parse("!git status")
        #expect(bang == BangCommand(command: "git status", isPrivate: false))
    }

    @Test func doubleBangIsPrivate() {
        let bang = BangCommand.parse("!!cat ~/.aws/credentials")
        #expect(bang == BangCommand(command: "cat ~/.aws/credentials", isPrivate: true))
    }

    @Test func spaceAfterTheBangIsOptional() {
        #expect(BangCommand.parse("!  ls -la")?.command == "ls -la")
        #expect(BangCommand.parse("!! ls -la")?.command == "ls -la")
    }

    @Test func leadingWhitespaceStillParses() {
        // processSubmit trims before dispatching, but the parser must not depend
        // on that or the classic REPL and the TUI would disagree.
        let bang = BangCommand.parse("   !echo hi")
        #expect(bang?.command == "echo hi")
        #expect(bang?.isPrivate == false)
    }

    @Test func bareBangParsesWithAnEmptyCommand() {
        // Not nil: the caller answers a bare `!` with usage text rather than
        // sending it to the model as a prompt.
        #expect(BangCommand.parse("!") == BangCommand(command: "", isPrivate: false))
        #expect(BangCommand.parse("!!") == BangCommand(command: "", isPrivate: true))
    }

    @Test func nonBangLinesAreNotShellEscapes() {
        #expect(BangCommand.parse("/help") == nil)
        #expect(BangCommand.parse("what does ! mean in bash?") == nil)
        #expect(BangCommand.parse("") == nil)
    }

    @Test func innerBangsBelongToTheCommand() {
        // History expansion, negation, `!!` as a shell idiom: only the leading
        // marker is consumed, the rest is handed to the shell verbatim.
        #expect(BangCommand.parse("!echo !$")?.command == "echo !$")
        #expect(BangCommand.parse("!!sudo !!")?.command == "sudo !!")
        #expect(BangCommand.parse("!!!x") == BangCommand(command: "!x", isPrivate: true))
    }
}
