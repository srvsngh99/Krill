import Foundation

/// How the agent loop treats tools that can change the world (write files, run
/// shell commands). Read-only tools (read_file, grep, list_dir, glob) are always
/// allowed regardless of mode - they cannot damage anything.
public enum PermissionMode: String, Sendable, CaseIterable {
    /// Run every tool without asking. The autonomous default - the proven
    /// hands-off flow the loop shipped with.
    case acceptAll = "accept-all"
    /// Ask for approval before each mutating tool; read-only tools still run
    /// freely. Requires a `PermissionGate` to answer the prompt (the CLI wires
    /// a terminal approver). With no gate, an `.ask` decision is denied.
    case ask
    /// Auto-apply file edits (write/edit), but still ask before running shell
    /// commands. The middle posture between `ask` and `acceptAll` - lets the
    /// agent iterate on files freely while keeping a human in the loop for the
    /// riskier command tools. Like `.ask`, command tools need a `PermissionGate`.
    case acceptEdits = "accept-edits"
    /// Read-only: the agent may explore (read_file/grep/glob/list_dir) but every
    /// mutating tool is denied, so it can only investigate and propose a plan.
    case plan
}

public extension PermissionMode {
    /// Lenient parse of a config / CLI posture string. Accepts the canonical
    /// raw values plus friendly synonyms ("auto" = accept-all, "edits" =
    /// accept-edits). Returns nil for an unrecognized string.
    static func parse(_ s: String) -> PermissionMode? {
        switch s.lowercased().trimmingCharacters(in: .whitespaces) {
        case "auto", "accept-all", "acceptall", "all": return .acceptAll
        case "accept-edits", "acceptedits", "edits", "accept_edits": return .acceptEdits
        case "ask", "confirm": return .ask
        case "plan", "read-only", "readonly": return .plan
        default: return PermissionMode(rawValue: s)
        }
    }

    /// Resolve a persisted permission posture without ever failing open.
    /// Invalid or empty configuration falls back to read-only plan mode.
    static func configuredDefault(_ raw: String?) -> PermissionMode {
        guard let raw,
              !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let parsed = parse(raw) else {
            return .plan
        }
        return parsed
    }

    /// Order Shift+Tab cycles the posture through in the TUI: safest -> freest.
    static let cycleOrder: [PermissionMode] = [.plan, .ask, .acceptEdits, .acceptAll]

    /// The next posture when the user presses Shift+Tab (wraps).
    var next: PermissionMode {
        let order = PermissionMode.cycleOrder
        let i = order.firstIndex(of: self) ?? 0
        return order[(i + 1) % order.count]
    }

    /// Short label for the footer chip / notes (e.g. "auto", "accept-edits").
    var label: String {
        switch self {
        case .acceptAll: return "auto"
        case .acceptEdits: return "accept-edits"
        case .ask: return "ask"
        case .plan: return "plan"
        }
    }

    /// One-line description shown when the posture changes, so the leash state
    /// is never a hidden surprise.
    var postureNote: String {
        switch self {
        case .plan: return "plan - read-only; the agent investigates and proposes a plan (no edits, no commands)"
        case .ask: return "ask - confirm every file edit and command before it runs"
        case .acceptEdits: return "accept-edits - file edits apply automatically; commands still ask"
        case .acceptAll: return "auto - every tool runs without asking"
        }
    }
}

/// The outcome of consulting the policy for one tool call.
public enum PermissionDecision: Sendable, Equatable {
    /// Run the tool.
    case allow
    /// Do not run the tool; feed `reason` back so the model can adapt.
    case deny(reason: String)
    /// Defer to the interactive `PermissionGate` (only mutating tools in
    /// `.ask` mode reach this).
    case ask
}

/// Pure, value-type permission policy: a mode plus explicit allow/deny
/// tool-name lists. Decides each tool call with no side effects (so it is
/// trivially testable); the interactive prompt for `.ask` lives behind
/// `PermissionGate`, not here.
public struct PermissionPolicy: Sendable {
    public let mode: PermissionMode
    /// Tool names always allowed - overrides the mode, but NOT the deny list.
    public let allow: Set<String>
    /// Tool names always denied - highest precedence.
    public let deny: Set<String>

    public init(
        mode: PermissionMode = .acceptAll,
        allow: Set<String> = [],
        deny: Set<String> = []
    ) {
        self.mode = mode
        self.allow = allow
        self.deny = deny
    }

    /// Decide a tool call from its name and how it mutates the workspace.
    /// Precedence (highest first):
    ///   1. deny list  -> deny
    ///   2. allow list -> allow
    ///   3. read-only  -> allow (a read can never damage the workspace)
    ///   4. mode       -> accept-all: allow / accept-edits: edits allow, commands ask
    ///                    / ask: ask / plan: deny
    ///
    /// `isFileEdit` only matters in `.acceptEdits`: a file edit auto-applies
    /// while a command (bash) still defers to the gate.
    public func decision(
        toolName: String, isReadOnly: Bool, isFileEdit: Bool = false
    ) -> PermissionDecision {
        if deny.contains(toolName) {
            return .deny(reason: "tool '\(toolName)' is on the deny list")
        }
        if allow.contains(toolName) { return .allow }
        if isReadOnly { return .allow }
        switch mode {
        case .acceptAll:
            return .allow
        case .acceptEdits:
            return isFileEdit ? .allow : .ask
        case .ask:
            return .ask
        case .plan:
            return .deny(reason:
                "plan mode is read-only; '\(toolName)' would change files or run commands. "
                + "Do not call it - investigate with the read-only tools and present a plan instead.")
        }
    }
}

/// Seam the loop calls when the policy returns `.ask`. The CLI provides a
/// terminal implementation; tests inject a canned one. Return `true` to allow
/// the call, `false` to deny it (the loop then feeds a denial back to the model).
public protocol PermissionGate: Sendable {
    func approve(toolName: String, argumentsJSON: String) async -> Bool
}
