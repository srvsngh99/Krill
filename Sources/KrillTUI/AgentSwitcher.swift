import Foundation

/// State for the `/agents` switcher: a selectable list of the main view plus
/// every background agent session. Pure and unit-tested; the TUI renders it and
/// feeds it key events (Up/Down cycle, Enter attach, Esc close). Mirrors
/// `ModelPicker`.
public struct AgentSwitcher {
    public struct Entry: Equatable, Sendable {
        /// Session id, or nil for the "main" view row.
        public let id: Int?
        public let title: String
        public let status: String
        public init(id: Int?, title: String, status: String) {
            self.id = id; self.title = title; self.status = status
        }
    }

    public private(set) var entries: [Entry]
    public private(set) var selected: Int

    public init(entries: [Entry], current: Int? = nil) {
        self.entries = entries
        // `current` is a session id; nil means the main row (id == nil).
        self.selected = entries.firstIndex { $0.id == current } ?? 0
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
