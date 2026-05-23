import Foundation
import KLMRegistry
import KLMServer

/// Adapts `KLMServer.AgentToolBridge` to the loop's `OperatorToolCall`
/// shape. Keeps `OperatorLoop` agnostic of which parser is in use.
internal enum FamilyAwareToolCallExtractor {
    static func extract(from text: String, family: String?)
        -> (calls: [OperatorToolCall], cleanedText: String)
    {
        let modelFamily = family.flatMap { ModelFamily(rawValue: $0) }
        let result = AgentToolBridge.extract(from: text, family: modelFamily)
        let calls = result.calls.map {
            OperatorToolCall(name: $0.name, argumentsJSON: $0.argumentsJSON)
        }
        return (calls, result.cleanedText)
    }
}
