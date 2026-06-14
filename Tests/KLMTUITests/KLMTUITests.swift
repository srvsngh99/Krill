import XCTest
@testable import KLMTUI

final class KeyDecoderTests: XCTestCase {
    func testPrintableAndControls() {
        XCTAssertEqual(KeyDecoder.decode(Array("ab".utf8)), [.char("a"), .char("b")])
        XCTAssertEqual(KeyDecoder.decode([0x0d]), [.enter])
        XCTAssertEqual(KeyDecoder.decode([0x09]), [.tab])
        XCTAssertEqual(KeyDecoder.decode([0x7f]), [.backspace])
        XCTAssertEqual(KeyDecoder.decode([0x03]), [.ctrlC])
        XCTAssertEqual(KeyDecoder.decode([0x04]), [.ctrlD])
        XCTAssertEqual(KeyDecoder.decode([0x15]), [.ctrlU])
    }

    func testArrowKeys() {
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x41]), [.up])
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x42]), [.down])
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x43]), [.right])
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x44]), [.left])
        // SS3 form (application cursor keys)
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x4f, 0x41]), [.up])
    }

    func testNavAndLoneEscape() {
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x35, 0x7e]), [.pageUp])
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x36, 0x7e]), [.pageDown])
        XCTAssertEqual(KeyDecoder.decode([0x1b, 0x5b, 0x33, 0x7e]), [.delete])
        XCTAssertEqual(KeyDecoder.decode([0x1b]), [.escape])
    }

    func testTypingThenEnterInOneChunk() {
        XCTAssertEqual(KeyDecoder.decode(Array("hi".utf8) + [0x0d]),
                       [.char("h"), .char("i"), .enter])
    }

    func testUTF8Multibyte() {
        // "é" is two bytes; should decode to a single char.
        XCTAssertEqual(KeyDecoder.decode(Array("é".utf8)), [.char("é")])
    }
}

final class LayoutWrapTests: XCTestCase {
    func testWrapsOnWordBoundary() {
        XCTAssertEqual(Layout.wrap("the quick brown fox", width: 10),
                       ["the quick", "brown fox"])
    }

    func testPreservesExplicitNewlines() {
        XCTAssertEqual(Layout.wrap("a\n\nb", width: 10), ["a", "", "b"])
    }

    func testHardBreaksLongWord() {
        XCTAssertEqual(Layout.wrap("abcdefghij", width: 4), ["abcd", "efgh", "ij"])
    }

    func testShortLineUnchanged() {
        XCTAssertEqual(Layout.wrap("hello", width: 80), ["hello"])
    }
}

final class SlashMenuTests: XCTestCase {
    func testActivatesOnSlashPrefix() {
        var m = SlashMenu()
        m.update(for: "/s")
        XCTAssertTrue(m.isActive)
        XCTAssertTrue(m.matches.allSatisfy { $0.name.hasPrefix("/s") })
        XCTAssertTrue(m.matches.contains { $0.name == "/system" })
        XCTAssertTrue(m.matches.contains { $0.name == "/save" })
    }

    func testInactiveOnceArgumentTyped() {
        var m = SlashMenu()
        m.update(for: "/system ")
        XCTAssertFalse(m.isActive)
        m.update(for: "hello")
        XCTAssertFalse(m.isActive)
    }

    func testCycleWrapsAround() {
        var m = SlashMenu()
        m.update(for: "/s")
        let n = m.matches.count
        XCTAssertGreaterThan(n, 1)
        XCTAssertEqual(m.selected, 0)
        m.selectPrevious()                      // wraps to last
        XCTAssertEqual(m.selected, n - 1)
        m.selectNext()                          // back to first
        XCTAssertEqual(m.selected, 0)
    }

    func testHighlightStaysOnItemAcrossKeystrokes() {
        var m = SlashMenu()
        m.update(for: "/s")
        m.selectNext()
        let picked = m.current?.name
        m.update(for: "/s")                     // same query, another keystroke
        XCTAssertEqual(m.current?.name, picked)
    }
}
