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

    func testMouseWheelScroll() {
        // SGR mouse reports: button 64 = wheel up, 65 = wheel down.
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[<64;10;5M".utf8)), [.scrollUp])
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[<65;10;5M".utf8)), [.scrollDown])
        // A normal click (button 0) is recognized but ignored, not leaked as text.
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[<0;3;4M".utf8)), [])
        // Wheel report followed by a typed char still decodes the char.
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[<64;1;1Mx".utf8)), [.scrollUp, .char("x")])
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

final class ChromeInputFieldTests: XCTestCase {
    // The field interior is always exactly textWidth columns, regardless of
    // input length or cursor position, so the box frame never breaks.
    func testWidthInvariant() {
        for textWidth in [1, 2, 5, 20, 80] {
            for len in [0, 1, 5, 50, 200] {
                let text = Array(String(repeating: "x", count: len))
                for cursor in [0, len / 2, len] {
                    let (content, col) = Chrome.inputField(text: text, cursor: cursor, textWidth: textWidth)
                    XCTAssertEqual(content.count, textWidth, "w=\(textWidth) len=\(len) cur=\(cursor)")
                    XCTAssertTrue((0..<textWidth).contains(col), "cursor col out of range")
                }
            }
        }
    }

    func testShortInputNotScrolled() {
        let (content, col) = Chrome.inputField(text: Array("hi"), cursor: 2, textWidth: 10)
        XCTAssertTrue(content.hasPrefix("hi"))
        XCTAssertEqual(col, 2)                       // trailing cursor cell
    }

    func testLongInputScrollsToShowCursorTail() {
        let text = Array("abcdefghij")               // 10 chars
        let (content, col) = Chrome.inputField(text: text, cursor: 10, textWidth: 4)
        XCTAssertEqual(content.count, 4)
        XCTAssertEqual(col, 3)                        // cursor pinned to last column
        XCTAssertTrue(content.contains("j"))         // tail is visible
        XCTAssertFalse(content.contains("a"))        // head scrolled off
    }

    func testCursorInMiddleStaysVisible() {
        let text = Array("abcdefghij")
        let (content, col) = Chrome.inputField(text: text, cursor: 0, textWidth: 4)
        XCTAssertEqual(content.count, 4)
        XCTAssertEqual(col, 0)
        XCTAssertEqual(String(Array(content)[0]), "a")
    }
}

final class ChromeBorderTests: XCTestCase {
    func testBorderExactWidth() {
        for w in [2, 3, 8, 80] {
            let b = Chrome.border(width: w, left: "[", fill: "-", right: "]")
            XCTAssertEqual(b.count, w)
            XCTAssertTrue(b.hasPrefix("["))
            XCTAssertTrue(b.hasSuffix("]"))
        }
    }

    func testBorderClipsBelowTwo() {
        XCTAssertLessThanOrEqual(Chrome.border(width: 1, left: "[", fill: "-", right: "]").count, 1)
        XCTAssertEqual(Chrome.border(width: 0, left: "[", fill: "-", right: "]"), "")
    }

    func testCenterPadNeverNegative() {
        XCTAssertEqual(Chrome.centerPad(visibleWidth: 10, totalWidth: 20), 5)
        XCTAssertEqual(Chrome.centerPad(visibleWidth: 30, totalWidth: 20), 0)   // over-wide
    }

    func testAnchorBlankTop() {
        // Bottom-anchored: short content hugs the bottom (all slack above).
        XCTAssertEqual(Chrome.anchorBlankTop(paneCount: 4, convHeight: 10, centered: false), 6)
        // Centered (splash): slack split in half.
        XCTAssertEqual(Chrome.anchorBlankTop(paneCount: 4, convHeight: 10, centered: true), 3)
        // Overflowing content is not padded (it scrolls).
        XCTAssertEqual(Chrome.anchorBlankTop(paneCount: 20, convHeight: 10, centered: false), 0)
        XCTAssertEqual(Chrome.anchorBlankTop(paneCount: 10, convHeight: 10, centered: false), 0)
    }
}

final class BannerTests: XCTestCase {
    func testRowsEqualWidthAndPureAscii() {
        let rows = Banner.krillm
        let w = Banner.width(rows)
        XCTAssertGreaterThan(w, 0)
        for row in rows {
            XCTAssertEqual(row.count, w, "banner rows must be equal width")
            XCTAssertTrue(row.allSatisfy { $0.isASCII }, "banner must be pure ASCII")
        }
    }
}
