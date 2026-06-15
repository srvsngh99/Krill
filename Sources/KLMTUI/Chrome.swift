import Foundation

/// Pure geometry for the TUI chrome (input box, banners, centering). Kept free
/// of ANSI styling so it is deterministic and unit-testable; callers in the
/// executable target layer styling (color / inverse / bold) on top.
public enum Chrome {
    /// The windowed, padded interior of the single-line input field, exactly
    /// `textWidth` columns wide, plus the column where the block cursor sits.
    /// The text scrolls horizontally so the cursor is always in view: the frame
    /// width never depends on the input length.
    ///
    /// - Returns: `content` is `textWidth` characters (trailing-space padded);
    ///   `cursorCol` is a valid index into it (`0 ..< textWidth`).
    public static func inputField(text: [Character], cursor: Int, textWidth: Int)
        -> (content: String, cursorCol: Int) {
        let tw = max(1, textWidth)
        let c = min(max(0, cursor), text.count)
        // Keep the cursor on the last visible column once the text overflows.
        let start = c >= tw ? c - tw + 1 : 0
        let end = min(text.count, start + tw)
        var content = String(text[start..<end])
        let cursorCol = min(c - start, tw - 1)
        // Guarantee a cell exists at the cursor (e.g. cursor past the last char),
        // then pad / clip to exactly the field width.
        let needed = max(tw, cursorCol + 1)
        if content.count < needed { content += String(repeating: " ", count: needed - content.count) }
        content = String(content.prefix(tw))
        return (content, cursorCol)
    }

    /// A horizontal box border of exactly `width` columns: `left`, `fill`
    /// repeated, `right`. For `width < 2` the corners are clipped so the result
    /// is still at most `width` wide.
    public static func border(width: Int, left: String, fill: String, right: String) -> String {
        guard width >= 2 else { return String((left + right).prefix(max(0, width))) }
        return left + String(repeating: fill, count: width - 2) + right
    }

    /// Leading-space count to center `visibleWidth` columns inside `totalWidth`.
    /// Never negative (over-wide content starts at column 0).
    public static func centerPad(visibleWidth: Int, totalWidth: Int) -> Int {
        max(0, (totalWidth - visibleWidth) / 2)
    }

    /// Leading blank ROWS to place `paneCount` content lines within `convHeight`
    /// visible rows: bottom-anchored by default (content hugs the bottom, so new
    /// messages appear just above the input), or vertically centered when
    /// `centered` (used for the splash). Returns 0 when content overflows the
    /// viewport (it scrolls instead of being padded).
    public static func anchorBlankTop(paneCount: Int, convHeight: Int, centered: Bool) -> Int {
        guard paneCount < convHeight else { return 0 }
        let slack = convHeight - paneCount
        return centered ? slack / 2 : slack
    }
}

/// ASCII-art wordmark banners for the splash. Pure ASCII (figlet "big" font, the
/// line-drawing `_ | / \` style) so they sit inside the house ASCII rule and
/// render in any terminal.
public enum Banner {
    /// The "KrillLM" wordmark in the classic line-drawing style, scaled up for
    /// presence. All rows are the same width; ~31 columns, so callers should
    /// fall back to a plain wordmark on terminals too narrow to fit it.
    public static let krillm: [String] = [
        " _  __     _ _ _ _      __  __ ",
        "| |/ /    (_) | | |    |  \\/  |",
        "| ' / _ __ _| | | |    | \\  / |",
        "|  < | '__| | | | |    | |\\/| |",
        "| . \\| |  | | | | |____| |  | |",
        "|_|\\_\\_|  |_|_|_|______|_|  |_|",
    ]

    /// A small pixel-art "krill" mascot (block glyphs, written as `\u{}` escapes
    /// so the source bytes stay ASCII). Faces right toward the wordmark it sits
    /// beside, with an eye, antenna, and little swimmeret legs. All rows padded
    /// to the same width for clean side-by-side alignment.
    public static let krillMascot: [String] = [
        "            \u{2597}\u{2584}",
        "  \u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}\u{2584}  \u{2597}\u{259B} ",
        " \u{259F}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2599}",
        " \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{25CF}\u{258C}",
        " \u{259C}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{259B}",
        "  \u{2598} \u{2598} \u{2598} \u{2598} \u{2598} \u{2598} ",
    ]

    /// The pixel width of a banner (its widest row).
    public static func width(_ banner: [String]) -> Int { banner.map(\.count).max() ?? 0 }
}
