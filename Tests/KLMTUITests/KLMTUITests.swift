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

    func testOSCColorReportIsIgnored() {
        // An OSC 11 background-color report (BEL-terminated) must be consumed and
        // dropped, not decoded as ESC + stray characters that leak into input.
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}]11;rgb:1c1c/1c1c/1c1c\u{07}".utf8)), [])
        // ...and a real key after it still comes through.
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}]11;rgb:ffff/ffff/ffff\u{07}x".utf8)), [.char("x")])
        // ST-terminated form ( ESC \ ).
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}]11;rgb:0/0/0\u{1b}\\y".utf8)), [.char("y")])
    }

    func testMouseClickDecodesToNoKeys() {
        // A plain click (button 0 press/release) is recognized-but-ignored - it
        // must NOT be conflated with EOF by the reader (see KeyReader.read docs).
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[<0;5;7M".utf8)), [])
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[<0;5;7m".utf8)), [])
    }

    func testSplitMouseSequenceBuffered() {
        // A wheel report chopped across a read boundary must NOT decode to stray
        // text; the incomplete tail comes back as the remainder.
        let head = Array("\u{1b}[<65;66;1".utf8)            // no terminating M yet
        let r1 = KeyDecoder.decodeStreaming(head)
        XCTAssertEqual(r1.keys, [])
        XCTAssertEqual(r1.remainder, head)                  // whole thing buffered
        // Next read completes it (plus a real keystroke after).
        let r2 = KeyDecoder.decodeStreaming(r1.remainder + Array("9M".utf8) + Array("x".utf8))
        XCTAssertEqual(r2.keys, [.scrollDown, .char("x")])
        XCTAssertTrue(r2.remainder.isEmpty)
    }

    func testCompleteKeysLeaveNoRemainder() {
        let r = KeyDecoder.decodeStreaming(Array("hi".utf8) + [0x0d])
        XCTAssertEqual(r.keys, [.char("h"), .char("i"), .enter])
        XCTAssertTrue(r.remainder.isEmpty)
    }

    func testFocusEventConsumedNotLeaked() {
        // Terminal focus in/out ( ESC [ I / ESC [ O ) must be consumed, not typed.
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[I".utf8)), [])
        XCTAssertEqual(KeyDecoder.decode(Array("\u{1b}[Ohello".utf8)), Array("hello").map { Key.char($0) })
    }

    func testTruncatedUTF8Buffered() {
        let twoByte = Array("é".utf8)                       // 2 bytes
        let r = KeyDecoder.decodeStreaming([twoByte[0]])    // first byte only
        XCTAssertEqual(r.keys, [])
        XCTAssertEqual(r.remainder, [twoByte[0]])
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

final class CustomCommandTests: XCTestCase {
    func testExpandWholeArgumentTokens() {
        for token in ["$ARGUMENTS", "$ARGS", "$INPUT"] {
            let c = CustomCommand(name: "x", description: "", template: "Review this: \(token)")
            XCTAssertEqual(c.expand(arguments: "  the diff  "), "Review this: the diff")
        }
    }

    func testExpandPositional() {
        let c = CustomCommand(name: "x", description: "", template: "from $1 to $2")
        XCTAssertEqual(c.expand(arguments: "a b c"), "from a to b")
        // Missing positional -> empty string, not the literal token.
        XCTAssertEqual(c.expand(arguments: "only"), "from only to ")
    }

    func testNoPlaceholderAppendsArgs() {
        let c = CustomCommand(name: "x", description: "", template: "Summarize the text.")
        XCTAssertEqual(c.expand(arguments: "hello"), "Summarize the text.\n\nhello")
        // No args -> template unchanged, no trailing blank lines.
        XCTAssertEqual(c.expand(arguments: ""), "Summarize the text.")
    }

    func testPlaceholderPresentMeansNoAppend() {
        let c = CustomCommand(name: "x", description: "", template: "Echo: $ARGS")
        XCTAssertEqual(c.expand(arguments: "hi"), "Echo: hi")
    }

    func testMultiDigitPositionalLeftLiteral() {
        // $10 is unsupported and must not be eaten as $1 + "0".
        let c = CustomCommand(name: "x", description: "", template: "price $10 and $2")
        // No placeholder substituted for $10; $2 is the only real positional.
        XCTAssertEqual(c.expand(arguments: "a b c"), "price $10 and b")
    }

    func testArgumentContainingTokenNotReExpanded() {
        // A user word that looks like a token must be taken literally, not
        // re-expanded by a second pass.
        let c = CustomCommand(name: "x", description: "", template: "[$1]")
        XCTAssertEqual(c.expand(arguments: "$INPUT extra"), "[$INPUT]")
    }

    func testArgsPrefixOfArgumentsNotCorrupted() {
        let c = CustomCommand(name: "x", description: "", template: "<$ARGUMENTS>")
        XCTAssertEqual(c.expand(arguments: "hello"), "<hello>")
    }

    func testLoneDollarLeftLiteral() {
        let c = CustomCommand(name: "x", description: "", template: "cost is $ and $x")
        XCTAssertEqual(c.expand(arguments: ""), "cost is $ and $x")
    }

    func testParseFrontmatterDescription() {
        let src = "---\ndescription: Code review helper\n---\nReview: $ARGS\n"
        let c = CustomCommandStore.parse(name: "Review", contents: src)
        XCTAssertEqual(c.name, "review")          // lowercased
        XCTAssertEqual(c.description, "Code review helper")
        XCTAssertEqual(c.template, "Review: $ARGS")
    }

    func testParseNoFrontmatterUsesFirstLine() {
        let c = CustomCommandStore.parse(name: "tldr", contents: "Make a TLDR.\nMore detail.")
        XCTAssertEqual(c.description, "Make a TLDR.")
        XCTAssertEqual(c.template, "Make a TLDR.\nMore detail.")
    }

    func testLookupWithOrWithoutSlash() {
        let store = CustomCommandStore(commands: [
            CustomCommand(name: "review", description: "", template: "t"),
        ])
        XCTAssertNotNil(store.command(named: "review"))
        XCTAssertNotNil(store.command(named: "/review"))
        XCTAssertNotNil(store.command(named: "/REVIEW"))
        XCTAssertNil(store.command(named: "/nope"))
    }

    func testIsValidName() {
        XCTAssertTrue(CustomCommandStore.isValidName("code-review_2"))
        XCTAssertFalse(CustomCommandStore.isValidName(""))
        XCTAssertFalse(CustomCommandStore.isValidName("has space"))
        XCTAssertFalse(CustomCommandStore.isValidName("dot.name"))
    }

    func testStoreDedupsByName() {
        let store = CustomCommandStore(commands: [
            CustomCommand(name: "review", description: "first", template: "1"),
            CustomCommand(name: "review", description: "second", template: "2"),
        ])
        XCTAssertEqual(store.commands.count, 1)
        XCTAssertEqual(store.command(named: "review")?.description, "first")  // first wins
    }
}

final class ThemeTests: XCTestCase {
    func testOverrideWins() {
        XCTAssertEqual(Theme.resolve(override: "light", colorFGBG: "15;0"), .light)
        XCTAssertEqual(Theme.resolve(override: "dark", colorFGBG: "0;15"), .dark)
        XCTAssertEqual(Theme.resolve(override: "LIGHT", colorFGBG: nil), .light)  // case-insensitive
    }

    func testAutoOrUnknownOverrideFallsThrough() {
        XCTAssertEqual(Theme.resolve(override: "auto", colorFGBG: "0;15"), .light)
        XCTAssertEqual(Theme.resolve(override: "nonsense", colorFGBG: "15;0"), .dark)
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: nil), .unknown)
        XCTAssertEqual(Theme.resolve(override: "auto", colorFGBG: nil), .unknown)
    }

    func testColorFGBGBackgroundIsLastField() {
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: "15;0"), .dark)   // bg 0
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: "0;15"), .light)  // bg 15
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: "7;0;15"), .light) // 3-field, bg 15
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: "0;7"), .light)   // bg 7 = light gray
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: "15;8"), .dark)   // bg 8 = dark gray
        XCTAssertEqual(Theme.resolve(override: nil, colorFGBG: "garbage"), .unknown)
    }

    func testPalettes() {
        XCTAssertEqual(Theme.palette(for: .dark).userSGR, "97")
        XCTAssertEqual(Theme.palette(for: .dark).modelSGR, "90")
        XCTAssertEqual(Theme.palette(for: .light).userSGR, "1")   // bold default, not 97
        XCTAssertEqual(Theme.palette(for: .light).modelSGR, "90")
        XCTAssertEqual(Theme.palette(for: .unknown).userSGR, "1")
        XCTAssertNil(Theme.palette(for: .unknown).modelSGR)        // terminal default fg
        XCTAssertEqual(Theme.palette(for: .unknown).chromeSGR, "2")
    }

    func testLuminanceFromOSC11() {
        // White background -> high luminance -> light.
        let white = Theme.luminance(fromOSC11: "\u{1b}]11;rgb:ffff/ffff/ffff\u{07}")
        XCTAssertNotNil(white)
        XCTAssertEqual(Theme.background(forLuminance: white!), .light)
        // Black background -> low luminance -> dark.
        let black = Theme.luminance(fromOSC11: "\u{1b}]11;rgb:0000/0000/0000\u{07}")
        XCTAssertEqual(Theme.background(forLuminance: black!), .dark)
        // Two-digit channels normalise too.
        XCTAssertEqual(Theme.background(forLuminance: Theme.luminance(fromOSC11: "rgb:ff/ff/ff")!), .light)
        // Garbage -> nil.
        XCTAssertNil(Theme.luminance(fromOSC11: "\u{1b}[0m"))
    }
}

final class ModelPickerTests: XCTestCase {
    private let entries = [
        ModelPicker.Entry(name: "gemma-4-e2b", detail: "2B", downloaded: true),
        ModelPicker.Entry(name: "gemma-4-12b", detail: "12B", downloaded: true),
        ModelPicker.Entry(name: "qwen2.5-3b", detail: "3B", downloaded: false),
    ]

    func testStartsOnCurrentModel() {
        let p = ModelPicker(entries: entries, current: "gemma-4-12b")
        XCTAssertEqual(p.selected, 1)
        XCTAssertEqual(p.current?.name, "gemma-4-12b")
    }

    func testUnknownCurrentDefaultsToFirst() {
        let p = ModelPicker(entries: entries, current: "not-installed")
        XCTAssertEqual(p.selected, 0)
        XCTAssertEqual(p.current?.name, "gemma-4-e2b")
    }

    func testCycleWraps() {
        var p = ModelPicker(entries: entries)        // starts at 0
        p.selectNext(); XCTAssertEqual(p.selected, 1)
        p.selectNext(); XCTAssertEqual(p.selected, 2)
        p.selectNext(); XCTAssertEqual(p.selected, 0)   // wrap forward
        p.selectPrevious(); XCTAssertEqual(p.selected, 2) // wrap backward
    }

    func testEmptyPickerIsSafe() {
        var p = ModelPicker(entries: [])
        XCTAssertTrue(p.isEmpty)
        XCTAssertNil(p.current)
        p.selectNext(); p.selectPrevious()              // no crash, no move
        XCTAssertEqual(p.selected, 0)
    }
}

final class SlashMenuExtraTests: XCTestCase {
    func testExtraCommandsMatch() {
        var m = SlashMenu()
        m.extra = [SlashMenu.Item(name: "/review", summary: "custom")]
        m.update(for: "/rev")
        XCTAssertEqual(m.matches.map { $0.name }, ["/review"])
    }

    func testBuiltinShadowsExtra() {
        var m = SlashMenu()
        // An extra named like a built-in must not appear twice.
        m.extra = [SlashMenu.Item(name: "/help", summary: "dupe")]
        m.update(for: "/help")
        XCTAssertEqual(m.matches.filter { $0.name == "/help" }.count, 1)
    }

    func testBuiltinsStillMatchWithExtras() {
        var m = SlashMenu()
        m.extra = [SlashMenu.Item(name: "/review", summary: "custom")]
        m.update(for: "/mo")
        XCTAssertEqual(m.matches.map { $0.name }, ["/model"])
    }
}
