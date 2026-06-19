import Foundation
import KrillTooling

/// Result of running a tool. `content` is fed back to the model verbatim as
/// the observation for the next turn; `isError` lets the loop (and a future
/// renderer) flag failures without changing the feedback contract.
public struct ToolResult: Sendable, Equatable {
    public let content: String
    public let isError: Bool
    public init(content: String, isError: Bool = false) {
        self.content = content
        self.isError = isError
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
    /// Execute with the model-provided arguments (a JSON object string) and
    /// return the observation to feed back. Implementations must not throw -
    /// surface failures as a `ToolResult(isError: true)` so the loop can keep
    /// going and let the model recover.
    func run(argumentsJSON: String) async -> ToolResult
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

    public var isEmpty: Bool { order.isEmpty }
}
