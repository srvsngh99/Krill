import XCTest
@testable import KrillTUI

final class BlockFontTests: XCTestCase {
    func testEveryGlyphIsFiveRows() {
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-." {
            let rows = BlockFont.render(String(c))
            XCTAssertEqual(rows.count, BlockFont.height, "glyph \(c) wrong height")
            let w = rows.map(\.count).max() ?? 0
            for r in rows { XCTAssertEqual(r.count, w, "glyph \(c) ragged rows") }
        }
    }

    func testRenderLaysOutLettersWithGaps() {
        let rows = BlockFont.render("AB")
        XCTAssertEqual(rows.count, 5)
        // Two glyphs joined by a one-column gap -> wider than either alone.
        XCTAssertGreaterThan(BlockFont.width(rows), BlockFont.width(BlockFont.render("A")))
    }

    func testUnknownCharIsBlankSlotNotCrash() {
        let rows = BlockFont.render("~")
        XCTAssertEqual(rows.count, 5)
    }
}
