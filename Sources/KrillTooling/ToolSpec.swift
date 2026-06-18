import Foundation

/// A normalized tool/function definition (Sendable). `parametersJSON` is
/// the raw JSON-schema object serialized to a string so it can flow through
/// Sendable boundaries and be embedded verbatim into the tool prompt.
public struct ServerToolSpec: Equatable, Sendable {
    public let name: String
    public let description: String
    public let parametersJSON: String

    public init(name: String, description: String, parametersJSON: String) {
        self.name = name
        self.description = description
        self.parametersJSON = parametersJSON
    }
}

/// OpenAI `tool_choice`: how the model may use the offered tools.
/// `.auto` (default) = model decides; `.none` = never call a tool;
/// `.required` = must call some tool; `.function(name)` = must call exactly
/// that tool. The forced variants let the server grammar-CONSTRAIN the output
/// to a valid tool-call JSON (best-effort; fails open), where `.auto` leaves
/// decoding unconstrained so the model can also answer in prose.
public enum ServerToolChoice: Equatable, Sendable {
    case auto
    case none
    case required
    case function(String)
    /// True when the model MUST emit a tool call (safe to constrain decoding).
    public var forcesCall: Bool {
        switch self { case .required, .function: return true; case .auto, .none: return false }
    }
}
