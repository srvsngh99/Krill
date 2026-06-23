import Foundation

/// A compact 5-row solid-block font for stylized monochrome wordmarks (model
/// family logos). Each glyph is exactly 5 rows; `render` lays them out side by
/// side with a one-column gap. Pure and unit-tested (every glyph is 5 rows).
public enum BlockFont {
    public static let height = 5

    /// Render `text` (uppercased; unknown chars become a blank slot) as 5 rows.
    public static func render(_ text: String) -> [String] {
        let glyphs = text.uppercased().map { glyph(for: $0) }
        guard !glyphs.isEmpty else { return Array(repeating: "", count: height) }
        var rows = [String](repeating: "", count: height)
        for (i, g) in glyphs.enumerated() {
            // Pad every row of this glyph to the glyph's own max width so a row
            // authored a column short never shifts the letters after it.
            let gw = g.map(\.count).max() ?? 0
            let gap = i == 0 ? "" : "  "
            for r in 0..<height {
                let cell = g[r].padding(toLength: gw, withPad: " ", startingAt: 0)
                rows[r] += gap + cell
            }
        }
        return rows
    }

    /// Visible width of a rendered wordmark.
    public static func width(_ rows: [String]) -> Int { rows.map(\.count).max() ?? 0 }

    private static func glyph(for c: Character) -> [String] {
        if let g = table[c] { return g }
        if c == " " { return Array(repeating: "  ", count: height) }
        return Array(repeating: "  ", count: height)   // unknown -> blank slot
    }

    // Solid-block glyphs. Authored on a fixed grid per letter; widths vary but
    // every glyph is exactly `height` rows (gated by a unit test).
    private static let b = "\u{2588}"   // full block
    private static let table: [Character: [String]] = {
        let B = b
        func g(_ rows: [String]) -> [String] { rows }
        return [
            "A": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)", "\(B) \(B)", "\(B) \(B)"]),
            "B": g(["\(B)\(B) ", "\(B) \(B)", "\(B)\(B) ", "\(B) \(B)", "\(B)\(B) "]),
            "C": g(["\(B)\(B)\(B)", "\(B)  ", "\(B)  ", "\(B)  ", "\(B)\(B)\(B)"]),
            "D": g(["\(B)\(B) ", "\(B) \(B)", "\(B) \(B)", "\(B) \(B)", "\(B)\(B) "]),
            "E": g(["\(B)\(B)\(B)", "\(B)  ", "\(B)\(B) ", "\(B)  ", "\(B)\(B)\(B)"]),
            "F": g(["\(B)\(B)\(B)", "\(B)  ", "\(B)\(B) ", "\(B)  ", "\(B)  "]),
            "G": g(["\(B)\(B)\(B)", "\(B)  ", "\(B) \(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "H": g(["\(B) \(B)", "\(B) \(B)", "\(B)\(B)\(B)", "\(B) \(B)", "\(B) \(B)"]),
            "I": g(["\(B)\(B)\(B)", " \(B) ", " \(B) ", " \(B) ", "\(B)\(B)\(B)"]),
            "J": g(["\(B)\(B)\(B)", "  \(B)", "  \(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "K": g(["\(B) \(B)", "\(B)\(B) ", "\(B)  ", "\(B)\(B) ", "\(B) \(B)"]),
            "L": g(["\(B)  ", "\(B)  ", "\(B)  ", "\(B)  ", "\(B)\(B)\(B)"]),
            "M": g(["\(B)   \(B)", "\(B)\(B) \(B)\(B)", "\(B) \(B) \(B)", "\(B)   \(B)", "\(B)   \(B)"]),
            "N": g(["\(B)  \(B)", "\(B)\(B) \(B)", "\(B) \(B)\(B)", "\(B)  \(B)", "\(B)  \(B)"]),
            "O": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B) \(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "P": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)", "\(B)  ", "\(B)  "]),
            "Q": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B) \(B)", "\(B)\(B)\(B)", "  \(B)"]),
            "R": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B) ", "\(B) \(B)", "\(B) \(B)"]),
            "S": g(["\(B)\(B)\(B)", "\(B)  ", "\(B)\(B)\(B)", "  \(B)", "\(B)\(B)\(B)"]),
            "T": g(["\(B)\(B)\(B)", " \(B) ", " \(B) ", " \(B) ", " \(B) "]),
            "U": g(["\(B) \(B)", "\(B) \(B)", "\(B) \(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "V": g(["\(B) \(B)", "\(B) \(B)", "\(B) \(B)", "\(B) \(B)", " \(B) "]),
            "W": g(["\(B)   \(B)", "\(B)   \(B)", "\(B) \(B) \(B)", "\(B)\(B) \(B)\(B)", "\(B)   \(B)"]),
            "X": g(["\(B) \(B)", "\(B) \(B)", " \(B) ", "\(B) \(B)", "\(B) \(B)"]),
            "Y": g(["\(B) \(B)", "\(B) \(B)", " \(B) ", " \(B) ", " \(B) "]),
            "Z": g(["\(B)\(B)\(B)", "  \(B)", " \(B) ", "\(B)  ", "\(B)\(B)\(B)"]),
            "0": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B) \(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "1": g([" \(B)", "\(B)\(B)", " \(B)", " \(B)", "\(B)\(B)\(B)"]),
            "2": g(["\(B)\(B)\(B)", "  \(B)", "\(B)\(B)\(B)", "\(B)  ", "\(B)\(B)\(B)"]),
            "3": g(["\(B)\(B)\(B)", "  \(B)", " \(B)\(B)", "  \(B)", "\(B)\(B)\(B)"]),
            "4": g(["\(B) \(B)", "\(B) \(B)", "\(B)\(B)\(B)", "  \(B)", "  \(B)"]),
            "5": g(["\(B)\(B)\(B)", "\(B)  ", "\(B)\(B)\(B)", "  \(B)", "\(B)\(B)\(B)"]),
            "6": g(["\(B)\(B)\(B)", "\(B)  ", "\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "7": g(["\(B)\(B)\(B)", "  \(B)", " \(B) ", " \(B) ", " \(B) "]),
            "8": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)"]),
            "9": g(["\(B)\(B)\(B)", "\(B) \(B)", "\(B)\(B)\(B)", "  \(B)", "\(B)\(B)\(B)"]),
            "-": g(["   ", "   ", "\(B)\(B)\(B)", "   ", "   "]),
            ".": g([" ", " ", " ", " ", "\(B)"]),
        ]
    }()
}
