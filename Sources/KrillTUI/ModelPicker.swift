import Foundation

/// State for the `/model` picker: a selectable list of every known model, each
/// flagged as downloaded (switch instantly) or not (download first). Pure and
/// unit-tested; the TUI renders it and feeds it key events (Up/Down cycle).
public struct ModelPicker {
    public struct Entry: Equatable, Sendable {
        public let name: String
        public let params: String        // e.g. "12B"
        public let quant: String         // e.g. "nvfp4"
        public let size: String          // e.g. "7.2 GB" / "~8.0 GB"
        public let downloaded: Bool
        /// Legacy combined detail (params . quant . size), kept for callers/tests.
        public var detail: String {
            [params, quant, size].filter { !$0.isEmpty }.joined(separator: " \u{00B7} ")
        }
        public init(name: String, params: String = "", quant: String = "",
                    size: String = "", downloaded: Bool) {
            self.name = name; self.params = params; self.quant = quant
            self.size = size; self.downloaded = downloaded
        }
        /// Legacy init: a pre-joined detail string (used by tests).
        public init(name: String, detail: String, downloaded: Bool) {
            self.name = name; self.params = detail; self.quant = ""
            self.size = ""; self.downloaded = downloaded
        }
    }

    public private(set) var entries: [Entry]
    public private(set) var selected: Int

    /// Build a picker over `entries`, starting the highlight on `current` if it
    /// is present (otherwise the first entry).
    public init(entries: [Entry], current: String? = nil) {
        self.entries = entries
        self.selected = current.flatMap { c in entries.firstIndex { $0.name == c } } ?? 0
    }

    public var isEmpty: Bool { entries.isEmpty }
    public var current: Entry? { entries.indices.contains(selected) ? entries[selected] : nil }

    public mutating func selectNext() {
        guard !entries.isEmpty else { return }
        selected = (selected + 1) % entries.count
    }

    public mutating func selectPrevious() {
        guard !entries.isEmpty else { return }
        selected = (selected - 1 + entries.count) % entries.count
    }
}
