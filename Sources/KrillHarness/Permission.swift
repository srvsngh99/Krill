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
    /// Read-only: the agent may explore (read_file/grep/glob/list_dir) but every
    /// mutating tool is denied, so it can only investigate and propose a plan.
    case plan
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

    /// Decide a tool call from its name and whether the tool is read-only.
    /// Precedence (highest first):
    ///   1. deny list  -> deny
    ///   2. allow list -> allow
    ///   3. read-only  -> allow (a read can never damage the workspace)
    ///   4. mode       -> accept-all: allow / ask: ask / plan: deny
    public func decision(toolName: String, isReadOnly: Bool) -> PermissionDecision {
        if deny.contains(toolName) {
            return .deny(reason: "tool '\(toolName)' is on the deny list")
        }
        if allow.contains(toolName) { return .allow }
        if isReadOnly { return .allow }
        switch mode {
        case .acceptAll:
            return .allow
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
