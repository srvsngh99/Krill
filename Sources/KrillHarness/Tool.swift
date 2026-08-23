import Foundation
import KrillTooling

/// A line in a UI-only unified diff. This deliberately lives outside the
/// model-facing `content`, so a future renderer can show full diffs without
/// spending context tokens on them.
public struct ToolDiffLine: Sendable, Equatable {
    public enum Kind: String, Sendable, Equatable {
        case context, addition, deletion
    }

    public let oldLine: Int?
    public let newLine: Int?
    public let kind: Kind
    public let text: String

    public init(oldLine: Int?, newLine: Int?, kind: Kind, text: String) {
        self.oldLine = oldLine
        self.newLine = newLine
        self.kind = kind
        self.text = text
    }
}

/// A contiguous UI-only diff hunk.
public struct ToolDiffHunk: Sendable, Equatable {
    public let oldStart: Int
    public let oldCount: Int
    public let newStart: Int
    public let newCount: Int
    public let lines: [ToolDiffLine]

    public init(
        oldStart: Int, oldCount: Int, newStart: Int, newCount: Int,
        lines: [ToolDiffLine]
    ) {
        self.oldStart = oldStart
        self.oldCount = oldCount
        self.newStart = newStart
        self.newCount = newCount
        self.lines = lines
    }
}

/// Rich presentation metadata for human-facing surfaces. It is never appended
/// to the model transcript. The diff case is defined now so file tools can adopt
/// the same seam later without changing `ToolResult` again.
public enum ToolDisplay: Sendable, Equatable {
    case question(UserQuestion, answer: UserAnswer)
    case diff(path: String, hunks: [ToolDiffHunk])
}

/// A declared state transition performed by a tool. The loop only relays this
/// for observability; the authoritative mutation belongs to `PermissionBox`.
public enum ToolEffect: Sendable, Equatable {
    case permissionMode(PermissionMode)
}

/// Result of running a tool. `content` is fed back to the model verbatim;
/// `display` is UI-only and `effect` declares an already-applied state change.
public struct ToolResult: Sendable, Equatable {
    public let content: String
    public let isError: Bool
    public let display: ToolDisplay?
    public let effect: ToolEffect?

    public init(
        content: String,
        isError: Bool = false,
        display: ToolDisplay? = nil,
        effect: ToolEffect? = nil
    ) {
        self.content = content
        self.isError = isError
        self.display = display
        self.effect = effect
    }
}

/// A capability the agent can invoke. Kept deliberately small: a name, a
/// JSON-schema describing its arguments (the same `parametersJSON` string the
/// server's tool path uses, so `ToolCalling` can render/parse it family-aware),
/// and an async `run`. Concrete tools (Bash, Read, Edit, ...) conform to this.
public protocol Tool: Sendable {
    var name: String { get }
    var description: String { get }
    /// JSON-schema object (as a string) for the tool's arguments.
    var parametersJSON: String { get }
    /// Whether the tool only observes (never writes files or runs commands).
    /// Read-only tools are always allowed by the permission layer, even in plan
    /// mode. Defaults to `false` - a tool must opt in to being trusted as safe.
    var isReadOnly: Bool { get }
    /// Whether the tool's mutation is a file edit (write/edit a file) as opposed
    /// to running an arbitrary command (bash). Lets the `accept-edits` posture
    /// auto-apply edits while still gating shell commands. Defaults to `false`
    /// (a mutating tool is treated as a command unless it opts in).
    var isFileEdit: Bool { get }
    /// Execute with the model-provided arguments (a JSON object string) and
    /// return the observation to feed back. Implementations must not throw -
    /// surface failures as a `ToolResult(isError: true)` so the loop can keep
    /// going and let the model recover.
    func run(argumentsJSON: String) async -> ToolResult
}

public extension Tool {
    /// Conservative default: a tool is treated as mutating unless it declares
    /// otherwise, so an unaudited tool requires approval rather than running free.
    var isReadOnly: Bool { false }
    /// Conservative default: a mutating tool is treated as a command (the most
    /// gated category) unless it declares itself a file edit.
    var isFileEdit: Bool { false }
}

/// Ordered, name-indexed set of the tools offered to the model for a run.
public struct ToolRegistry: Sendable {
    private let byName: [String: any Tool]
    private let order: [String]

    public init(_ tools: [any Tool]) {
        var byName: [String: any Tool] = [:]
        var order: [String] = []
        for tool in tools where byName[tool.name] == nil {
            byName[tool.name] = tool
            order.append(tool.name)
        }
        self.byName = byName
        self.order = order
    }

    /// The tool specs, in registration order, for `ToolCalling.injectToolSystem`.
    public func specs() -> [ServerToolSpec] {
        order.compactMap { byName[$0] }.map {
            ServerToolSpec(name: $0.name, description: $0.description, parametersJSON: $0.parametersJSON)
        }
    }

    public func tool(named name: String) -> (any Tool)? { byName[name] }

    /// The `ServerToolSpec` for one tool (for schema checks / arg-repair prompts).
    public func spec(named name: String) -> ServerToolSpec? {
        byName[name].map {
            ServerToolSpec(name: $0.name, description: $0.description, parametersJSON: $0.parametersJSON)
        }
    }

    public var isEmpty: Bool { order.isEmpty }

    /// Tool names in registration order (for unknown-tool error feedback).
    public var names: [String] { order }
}
