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

    /// Generate constrained so the output is a single JSON object conforming to
    /// `jsonSchema`. Used to REPAIR a tool call whose free-form args were empty
    /// or invalid - the small-local-model tool-arg fix: the engine grammar-
    /// constrains decoding so required fields cannot be omitted. The default
    /// falls back to unconstrained generation for backends that cannot
    /// constrain (e.g. the test mock unless it overrides this).
    func completeConstrained(messages: [[String: String]], jsonSchema: String) async -> String

    /// Free-form generation with the tool-name slot constrained to offered names.
    /// This must be a protocol requirement so calls through `any HarnessGenerator`
    /// dynamically dispatch to engine-backed implementations.
    func complete(
        messages: [[String: String]], constrainingToolNames toolNames: [String]
    ) async -> String
}

extension HarnessGenerator {
    public func completeConstrained(messages: [[String: String]], jsonSchema: String) async -> String {
        await complete(messages: messages)
    }

    /// Free-form generation with the tool-NAME slot constrained to `toolNames`.
    ///
    /// This is the primary defence against a model naming a tool this harness
    /// does not offer: inside a tool call the sampler can only spell one of the
    /// offered names, so an unknown name is never generated in the first place.
    /// Prose, reasoning, and argument values decode normally.
    ///
    /// The default ignores the constraint, so a backend that cannot mask logits
    /// (or a test mock) keeps working unchanged and falls back to the recovery
    /// layers in `AgentLoop`.
    public func complete(
        messages: [[String: String]], constrainingToolNames toolNames: [String]
    ) async -> String {
        await complete(messages: messages)
    }
}
