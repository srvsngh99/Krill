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

    var toolFormat: ToolCalling.ToolFormat {
        ToolCalling.ToolFormat.forFamily(engine.family)
    }

    func complete(messages: [[String: String]]) async -> String {
        await collect(engine.generate(messages: messages, params: .greedy, maxTokens: maxTokens).stream)
    }

    func completeConstrained(messages: [[String: String]], jsonSchema: String) async -> String {
        // Grammar-constrain decoding to a JSON object matching the tool's
        // parameter schema, so a small model cannot omit required fields.
        let (stream, _) = engine.generate(
            messages: messages, params: .greedy, maxTokens: 256,
            format: .jsonSchemaCompact(jsonSchema))
        return await collect(stream)
    }

    private func collect(_ stream: AsyncStream<TokenEvent>) async -> String {
        var out = ""
        for await event in stream {
            if event.isEnd { break }
            out += event.text
        }
        return out
    }
}
