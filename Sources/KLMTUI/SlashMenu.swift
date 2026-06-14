import Foundation

/// The slash-command autosuggest popup state: which commands match the partial
/// input and which one is highlighted. Pure and unit-tested; the TUI renders it
/// and feeds it key events (Up/Down cycle, Tab/Enter accept).
public struct SlashMenu {
    public init() {}
    public struct Item: Equatable, Sendable {
        public let name: String
        public let summary: String
        public init(name: String, summary: String) {
            self.name = name
            self.summary = summary
        }
    }

    /// Canonical command list shown in the popup (and accepted by the TUI).
    public static let all: [Item] = [
        Item(name: "/help", summary: "Show keys and commands"),
        Item(name: "/image", summary: "Attach an image to your next message"),
        Item(name: "/audio", summary: "Attach an audio clip"),
        Item(name: "/mic", summary: "Record from the microphone"),
        Item(name: "/attach", summary: "List pending attachments"),
        Item(name: "/remove", summary: "Drop attachment number n"),
        Item(name: "/drop", summary: "Drop all pending attachments"),
        Item(name: "/system", summary: "Set the system prompt"),
        Item(name: "/model", summary: "Switch to another model"),
        Item(name: "/history", summary: "Show the conversation"),
        Item(name: "/compact", summary: "Summarize and shrink the conversation"),
        Item(name: "/save", summary: "Save the transcript"),
        Item(name: "/clear", summary: "Clear the conversation"),
        Item(name: "/quit", summary: "Exit"),
    ]

    /// Extra commands registered at runtime (user-authored custom commands),
    /// merged with `all` for matching. Built-ins win on a name clash.
    public var extra: [Item] = []

    /// All candidate commands: the built-ins followed by any registered extras
    /// that do not shadow a built-in.
    public var candidates: [Item] {
        let builtinNames = Set(Self.all.map { $0.name })
        return Self.all + extra.filter { !builtinNames.contains($0.name) }
    }

    public private(set) var matches: [Item] = []
    public private(set) var selected = 0

    /// True when the popup should be shown.
    public var isActive: Bool { !matches.isEmpty }

    /// The highlighted item, if any.
    public var current: Item? { matches.indices.contains(selected) ? matches[selected] : nil }

    /// Recompute matches for the current input. The popup is active only while
    /// the user is typing the command word: the line starts with `/` and has no
    /// space yet (once an argument is typed, the popup closes).
    public mutating func update(for input: String) {
        guard input.hasPrefix("/"), !input.contains(" ") else {
            matches = []; selected = 0; return
        }
        let q = input.lowercased()
        let previous = current?.name
        matches = candidates.filter { $0.name.hasPrefix(q) }
        // Keep the highlight on the same item across keystrokes when possible.
        if let previous, let idx = matches.firstIndex(where: { $0.name == previous }) {
            selected = idx
        } else if selected >= matches.count {
            selected = max(0, matches.count - 1)
        }
    }

    public mutating func selectNext() {
        guard !matches.isEmpty else { return }
        selected = (selected + 1) % matches.count
    }

    public mutating func selectPrevious() {
        guard !matches.isEmpty else { return }
        selected = (selected - 1 + matches.count) % matches.count
    }

    public mutating func close() {
        matches = []; selected = 0
    }
}
