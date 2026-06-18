import Foundation

/// A single operator-agent tool.
///
/// The operator agent's tool surface is **deliberately scoped to Krill
/// itself** - no shell, no file_read, no file_write, no web_fetch. Every
/// conforming tool wraps an existing CLI command or HTTP endpoint (the
/// registry, the catalog, the running server). New tools that would touch
/// the user's filesystem or shell do NOT belong here; users who want a
/// coding agent point Aider / gptme / OpenHands at Krill's `/v1/*`
/// endpoints instead.
public protocol OperatorTool: Sendable {
    /// Stable tool name the model emits in `<tool_call>` JSON. Must be
    /// `[a-z][a-z0-9_]*` and unique within an `OperatorToolRegistry`.
    var name: String { get }

    /// One-sentence description shown to the model in the tool schema.
    var description: String { get }

    /// JSON Schema for the tool's arguments, serialized as a string.
    /// Empty schemas pass `"{}"`; a typed shape passes a full
    /// `{"type":"object","properties":{...},"required":[...]}` blob.
    var parametersJSON: String { get }

    /// Execute the tool. The returned string is fed back to the model as
    /// the tool result. Throw to signal a hard failure that the loop
    /// should surface as an error event (the loop still continues so the
    /// model can recover).
    func execute(arguments: [String: Any]) async throws -> String
}

/// An ordered collection of tools the loop offers the model.
///
/// Lookup is by tool `name`. Construction is order-preserving so the
/// system-prompt schema list and the `/agent/tools` introspection
/// surface emit tools in the order they were registered (helps stable
/// snapshot tests).
public struct OperatorToolRegistry: Sendable {
    public let tools: [any OperatorTool]
    private let index: [String: Int]

    public init(_ tools: [any OperatorTool]) {
        self.tools = tools
        var index: [String: Int] = [:]
        for (i, t) in tools.enumerated() {
            index[t.name] = i
        }
        self.index = index
    }

    public func tool(named name: String) -> (any OperatorTool)? {
        guard let i = index[name] else { return nil }
        return tools[i]
    }

    public var names: [String] { tools.map(\.name) }
}

/// A tool call the model emitted, post-parse.
public struct OperatorToolCall: Equatable, Sendable {
    public let name: String
    /// Raw JSON string of the arguments object (as emitted by the model).
    public let argumentsJSON: String

    public init(name: String, argumentsJSON: String) {
        self.name = name
        self.argumentsJSON = argumentsJSON
    }

    /// Decode the arguments JSON to a dictionary. Returns an empty
    /// dictionary when the payload is missing or unparseable; tool
    /// implementations validate fields themselves.
    public func arguments() -> [String: Any] {
        guard let data = argumentsJSON.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return [:]
        }
        return obj
    }
}

/// The result of executing one tool call.
public struct OperatorToolResult: Equatable, Sendable {
    /// The tool that ran.
    public let toolName: String
    /// String content the loop feeds back to the model as a `tool`
    /// turn. On success this is the tool's own return value; on failure
    /// it is `"Error: <message>"` so the model can recover instead of
    /// the loop bailing.
    public let content: String
    /// True iff the tool threw.
    public let isError: Bool

    public init(toolName: String, content: String, isError: Bool = false) {
        self.toolName = toolName
        self.content = content
        self.isError = isError
    }
}
