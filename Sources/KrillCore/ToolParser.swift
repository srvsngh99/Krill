import Foundation

// MARK: - Tool/Function Calling Parser for Gemma 4

/// Parses Gemma 4's structured tool calling output.
///
/// Gemma 4 uses special tokens to delimit tool interactions:
/// - `<|tool|>...<tool|>` - tool definitions
/// - `<|tool_call|>...<tool_call|>` - function call requests
/// - `<|tool_response|>...<tool_response|>` - execution results
public struct ToolParser: Sendable {

    /// A parsed tool call from model output.
    public struct ToolCall: Sendable {
        public let name: String
        public let rawArguments: String

        public init(name: String, rawArguments: String) {
            self.name = name
            self.rawArguments = rawArguments
        }

        /// Parse arguments as a dictionary (convenience, not Sendable-safe for storage).
        public func parsedArguments() -> [String: Any] {
            (try? JSONSerialization.jsonObject(
                with: rawArguments.data(using: .utf8) ?? Data()
            ) as? [String: Any]) ?? [:]
        }
    }

    /// A tool definition to inject into the prompt.
    public struct ToolDefinition: Codable, Sendable {
        public let type: String
        public let function: FunctionDef

        public struct FunctionDef: Codable, Sendable {
            public let name: String
            public let description: String
            public let parameters: ParametersDef
        }

        public struct ParametersDef: Codable, Sendable {
            public let type: String
            public let properties: [String: PropertyDef]
            public let required: [String]?
        }

        public struct PropertyDef: Codable, Sendable {
            public let type: String
            public let description: String?
        }
    }

    // MARK: - Parsing

    /// Check if generated text contains a tool call.
    public static func containsToolCall(_ text: String) -> Bool {
        text.contains("<|tool_call|>")
    }

    /// Extract tool calls from generated text.
    ///
    /// Looks for pattern: `<|tool_call|> { "name": "...", "arguments": {...} } <tool_call|>`
    public static func extractToolCalls(from text: String) -> [ToolCall] {
        var calls: [ToolCall] = []

        let pattern = "<|tool_call|>"
        let endPattern = "<tool_call|>"

        var remaining = text
        while let startRange = remaining.range(of: pattern) {
            remaining = String(remaining[startRange.upperBound...])

            guard let endRange = remaining.range(of: endPattern) else { break }
            let jsonStr = String(remaining[..<endRange.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines)

            if let call = parseToolCallJSON(jsonStr) {
                calls.append(call)
            }

            remaining = String(remaining[endRange.upperBound...])
        }

        return calls
    }

    /// Format tool definitions for prompt injection.
    ///
    /// Wraps definitions in `<|tool|>` tokens as Gemma 4 expects.
    public static func formatToolDefinitions(_ tools: [ToolDefinition]) -> String {
        var result = ""
        for tool in tools {
            guard let data = try? JSONEncoder().encode(tool),
                  let json = String(data: data, encoding: .utf8) else { continue }
            result += "<|tool|>\n\(json)\n<tool|>\n"
        }
        return result
    }

    /// Format a tool response for feeding back to the model.
    public static func formatToolResponse(name: String, result: String) -> String {
        "<|tool_response|>\n{\"name\": \"\(name)\", \"result\": \(result)}\n<tool_response|>"
    }

    /// Load tool definitions from a JSON file.
    public static func loadTools(from url: URL) throws -> [ToolDefinition] {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode([ToolDefinition].self, from: data)
    }

    // MARK: - Internal

    private static func parseToolCallJSON(_ json: String) -> ToolCall? {
        guard let data = json.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let name = obj["name"] as? String else {
            return nil
        }

        let args: String
        if let argsObj = obj["arguments"] {
            if let argsData = try? JSONSerialization.data(withJSONObject: argsObj) {
                args = String(data: argsData, encoding: .utf8) ?? "{}"
            } else {
                args = "{}"
            }
        } else {
            args = "{}"
        }

        return ToolCall(name: name, rawArguments: args)
    }
}
