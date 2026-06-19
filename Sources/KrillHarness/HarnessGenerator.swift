import Foundation
import KrillTooling

/// The one seam between the agent loop and a model. The loop owns tool
/// injection and parsing (via `KrillTooling`); the generator only turns a
/// message history into the model's raw output text.
///
/// In production this is backed in-process by `InferenceEngine` (see
/// `EngineGenerator` in KrillCLI). In tests it is a deterministic mock, so the
/// loop is fully unit-testable with no model and no MLX. This is the only
/// abstraction we keep - justified by testability, not portability (Krill is
/// Mac-only, in-process).
public protocol HarnessGenerator: Sendable {
    /// Wire format for the loaded model family, so the loop can render the
    /// tool system turn and parse tool calls the way this model was trained.
    var toolFormat: ToolCalling.ToolFormat { get }

    /// Produce the model's raw completion for the given chat history.
    func complete(messages: [[String: String]]) async -> String
}
