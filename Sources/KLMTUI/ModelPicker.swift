import Foundation

/// State for the `/model` picker: a selectable list of every known model, each
/// flagged as downloaded (switch instantly) or not (download first). Pure and
/// unit-tested; the TUI renders it and feeds it key events (Up/Down cycle).
public struct ModelPicker {
    public struct Entry: Equatable, Sendable {
        public let name: String
        public let detail: String        // e.g. "12B . 4bit"
        public let downloaded: Bool
        public init(name: String, detail: String, downloaded: Bool) {
            self.name = name
            self.detail = detail
            self.downloaded = downloaded
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
