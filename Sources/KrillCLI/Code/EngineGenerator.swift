import Foundation
import KrillEngine
import KrillHarness
import KrillTooling

/// In-process `HarnessGenerator` backed by the loaded `InferenceEngine`. This
/// is the production binding of the loop's one seam: it picks the tool wire
/// format from the model family and accumulates the streamed completion. Greedy
/// decoding - the right default for tool-call reliability on small local models.
struct EngineGenerator: HarnessGenerator {
    let engine: InferenceEngine
    let maxTokens: Int
    /// Reports each completed generation's stats (tok/s, token counts) so a
    /// hosting surface can keep its footer live during agent runs. Called on
    /// the loop's task, after the stream finishes.
    var onStats: (@Sendable (GenerationStats) -> Void)? = nil

    var toolFormat: ToolCalling.ToolFormat {
        ToolCalling.ToolFormat.forFamily(engine.family)
    }

    func complete(messages: [[String: String]]) async -> String {
        // Serialize against every other in-process generation (foreground chat +
        // other background agents) so decodes never overlap on the single GPU.
        await GenerationGate.shared.acquire()
        defer { GenerationGate.shared.release() }
        let (stream, stats) = engine.generate(messages: messages, params: .greedy, maxTokens: maxTokens)
        let text = await collect(stream)
        if let s = stats() { onStats?(s) }
        return text
    }

    /// Free generation with the tool-name slot constrained (see
    /// `OutputFormat.toolNames`). Families with no unambiguous tool-call
    /// sentinel resolve to an empty sentinel list, and the engine then decodes
    /// unconstrained - identical to `complete(messages:)`.
    func complete(
        messages: [[String: String]], constrainingToolNames toolNames: [String]
    ) async -> String {
        let sentinels = ToolCallSentinels.sentinels(for: toolFormat)
        guard !sentinels.isEmpty, !toolNames.isEmpty else {
            return await complete(messages: messages)
        }
        await GenerationGate.shared.acquire()
        defer { GenerationGate.shared.release() }
        let (stream, stats) = engine.generate(
            messages: messages, params: .greedy, maxTokens: maxTokens,
            format: .toolNames(
                sentinels: sentinels,
                nameKey: ToolCallSentinels.nameKey(for: toolFormat),
                names: toolNames))
        let text = await collect(stream)
        if let s = stats() { onStats?(s) }
        return text
    }

    func completeConstrained(messages: [[String: String]], jsonSchema: String) async -> String {
        // Grammar-constrain decoding to a JSON object matching the tool's
        // parameter schema, so a small model cannot omit required fields.
        await GenerationGate.shared.acquire()
        defer { GenerationGate.shared.release() }
        let (stream, stats) = engine.generate(
            messages: messages, params: .greedy, maxTokens: 256,
            format: .jsonSchemaCompact(jsonSchema))
        let text = await collect(stream)
        if let s = stats() { onStats?(s) }
        return text
    }

    private func collect(_ stream: AsyncStream<TokenEvent>) async -> String {
        var out = ""
        for await event in stream {
            // Honor Ctrl-C from the TUI: leaving the for-await drops the stream,
            // whose onTermination cancels the engine's decode loop on the GPU.
            if Task.isCancelled { break }
            if event.isEnd { break }
            out += event.text
        }
        return out
    }
}
