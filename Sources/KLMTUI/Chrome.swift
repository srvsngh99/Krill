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

/// ASCII-art wordmark banners for the splash. Pure ASCII (figlet "colossal"
/// font) so they sit inside the house ASCII rule and render in any terminal.
public enum Banner {
    /// The "KrillLM" wordmark, a dense filled block banner. All rows are the
    /// same width; ~49 columns, so callers should fall back to a plain wordmark
    /// on terminals too narrow to fit it.
    public static let krillm: [String] = [
        "888    d8P        d8b888888888     888b     d888 ",
        "888   d8P         Y8P888888888     8888b   d8888 ",
        "888  d8P             888888888     88888b.d88888 ",
        "888d88K    888d888888888888888     888Y88888P888 ",
        "8888888b   888P\"  888888888888     888 Y888P 888 ",
        "888  Y88b  888    888888888888     888  Y8P  888 ",
        "888   Y88b 888    888888888888     888   \"   888 ",
        "888    Y88b888    88888888888888888888       888 ",
    ]

    /// The pixel width of a banner (its widest row).
    public static func width(_ banner: [String]) -> Int { banner.map(\.count).max() ?? 0 }
}
