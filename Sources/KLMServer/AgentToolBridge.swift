import Foundation
import KLMRegistry

/// Public surface that KLMAgent uses to convert between assistant text
/// and parsed tool calls in the family-appropriate wire format.
///
/// The full tool-call adapter (`ToolCalling`) is intentionally internal
/// to KLMServer because its types and helpers are used across many
/// server-only call sites and a wide public surface would lock down the
/// implementation. This bridge keeps those internals internal and
/// exposes only what the operator agent actually needs: extract +
/// inject, keyed on a `ModelFamily` (or nil → generic Hermes).
public enum AgentToolBridge {

    /// One parsed tool call. Mirrors `OperatorToolCall` but lives in
    /// KLMServer so the bridge can return it without KLMServer
    /// importing KLMAgent (which would cycle).
    public struct ParsedToolCall: Sendable, Equatable {
        public let name: String
        public let argumentsJSON: String

        public init(name: String, argumentsJSON: String) {
            self.name = name
            self.argumentsJSON = argumentsJSON
        }
    }

    /// One tool's schema in the form the prompt injector expects.
    /// `parametersJSON` is the raw JSON-schema object serialized to a
    /// string (matching `ServerToolSpec.parametersJSON` exactly).
    public struct ToolSpec: Sendable, Equatable {
        public let name: String
        public let description: String
        public let parametersJSON: String

        public init(name: String, description: String, parametersJSON: String) {
            self.name = name
            self.description = description
            self.parametersJSON = parametersJSON
        }
    }

    /// Extract tool calls from a raw assistant turn using the wire
    /// format the given family was fine-tuned on. `nil` family falls
    /// through to the generic Hermes parser.
    public static func extract(from text: String, family: ModelFamily?)
        -> (calls: [ParsedToolCall], cleanedText: String)
    {
        let format = ToolCalling.ToolFormat.forFamily(family?.rawValue)
        let result = ToolCalling.extractToolCalls(from: text, format: format)
        let calls = result.calls.map {
            ParsedToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
        }
        return (calls, result.cleanedText)
    }

    /// Inject the tool-schema instructions into `messages` in the
    /// family-appropriate format. `nil` family uses generic Hermes.
    public static func injectToolSystem(
        into messages: [[String: String]],
        tools: [ToolSpec],
        family: ModelFamily?
    ) -> [[String: String]] {
        let format = ToolCalling.ToolFormat.forFamily(family?.rawValue)
        let serverSpecs = tools.map {
            ServerToolSpec(
                name: $0.name,
                description: $0.description,
                parametersJSON: $0.parametersJSON)
        }
        return ToolCalling.injectToolSystem(
            into: messages, tools: serverSpecs, format: format)
    }
}
