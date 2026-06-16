import XCTest
@testable import KLMCore

/// Tests for the pure markdown-to-speech cleanup that runs before text-to-speech,
/// so spoken replies don't read code fences, backticks, or emphasis aloud.
final class SpokenTextTests: XCTestCase {

    func testPlainTextUnchanged() {
        XCTAssertEqual(SpokenText.clean("Hello there, how are you?"),
                       "Hello there, how are you?")
    }

    func testStripsFencedCodeBlocks() {
        let s = "Here is code:\n```swift\nlet x = 1\nprint(x)\n```\nDone."
        let out = SpokenText.clean(s)
        XCTAssertFalse(out.contains("let x"))
        XCTAssertFalse(out.contains("```"))
        XCTAssertTrue(out.hasPrefix("Here is code:"))
        XCTAssertTrue(out.hasSuffix("Done."))
    }

    func testInlineCodeKeepsContent() {
        XCTAssertEqual(SpokenText.clean("Call `run()` to start."), "Call run() to start.")
    }

    func testStripsEmphasis() {
        XCTAssertEqual(SpokenText.clean("This is **bold** and *italic* and _under_."),
                       "This is bold and italic and under.")
        XCTAssertEqual(SpokenText.clean("Mark __strong__ here."), "Mark strong here.")
    }

    func testEmphasisDoesNotEatArithmeticOrIdentifiers() {
        // The emphasis pass must not swallow multiplication operators or the
        // underscores inside identifiers (a coding model emits both constantly).
        XCTAssertEqual(SpokenText.clean("Compute 2 * 3 and 4 * 5."),
                       "Compute 2 * 3 and 4 * 5.")
        XCTAssertEqual(SpokenText.clean("Call my_func_name and other_var soon."),
                       "Call my_func_name and other_var soon.")
        XCTAssertEqual(SpokenText.clean("area = w * h * 2"), "area = w * h * 2")
        // Two bare `*` operators on one line (no surrounding spaces) must NOT be
        // treated as an emphasis span.
        XCTAssertEqual(SpokenText.clean("a*b and c*d"), "a*b and c*d")
        XCTAssertEqual(SpokenText.clean("1*2*3*4"), "1*2*3*4")
        // Real emphasis still strips (markers flanked by non-word chars).
        XCTAssertEqual(SpokenText.clean("This is *italic* and **bold** done."),
                       "This is italic and bold done.")
    }

    func testLinksKeepVisibleText() {
        XCTAssertEqual(SpokenText.clean("See [the docs](https://example.com/x) now."),
                       "See the docs now.")
    }

    func testStripsHeadingAndBulletMarkers() {
        let s = "# Title\n- first\n- second\n* third"
        XCTAssertEqual(SpokenText.clean(s), "Title first second third")
    }

    func testCollapsesWhitespace() {
        XCTAssertEqual(SpokenText.clean("a\n\n\nb    c"), "a b c")
    }

    func testEmptyAndCodeOnly() {
        XCTAssertEqual(SpokenText.clean(""), "")
        XCTAssertEqual(SpokenText.clean("```\njust code\n```"), "")
    }
}
