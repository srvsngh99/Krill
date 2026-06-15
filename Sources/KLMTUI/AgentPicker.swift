import Foundation

/// State for the `krillm launch` roster picker: a selectable list of every
/// launchable coding agent, each flagged as installed (on PATH, launches now) or
/// not (needs an install step first). Pure and unit-tested; the launch command
/// renders it and feeds it key events (Up/Down cycle), exactly like
/// ``ModelPicker``.
public struct AgentPicker {
    public struct Entry: Equatable, Sendable {
        public let id: String            // e.g. "claude"
        public let displayName: String   // e.g. "Claude Code"
        public let summary: String       // one-line description
        public let installed: Bool       // binary found on PATH
        public init(id: String, displayName: String, summary: String, installed: Bool) {
            self.id = id; self.displayName = displayName
            self.summary = summary; self.installed = installed
        }
    }

    public private(set) var entries: [Entry]
    public private(set) var selected: Int

    /// Build a picker over `entries`, starting the highlight on the first
    /// installed agent when there is one (so the default selection is something
    /// the user can launch right away), otherwise the first entry.
    public init(entries: [Entry]) {
        self.entries = entries
        self.selected = entries.firstIndex { $0.installed } ?? 0
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
