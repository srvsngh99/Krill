import XCTest
import Foundation
@testable import KLMEngine

/// Live regression for Gemma 4 long-context generation. gemma-4-e2b uses a
/// 512-token sliding window on most layers; before the windowed mask landed,
/// KrillLM applied a plain full-causal mask to every layer, so beyond ~2x the
/// window those layers attended out-of-distribution context and the model
/// emitted its stop token immediately - generating NOTHING. Every prior test
/// used short prompts, so this was invisible. This test pins it: a long prompt
/// must still produce non-empty, on-topic output.
///
/// Set `KLM_GEMMA4_MODEL_PATH` to a Gemma 4 checkpoint (e.g. gemma-4-e2b).
final class Gemma4LongContextLiveTests: XCTestCase {

    private func modelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"],
              !path.isEmpty else { throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set") }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue
        else { throw XCTSkip("KLM_GEMMA4_MODEL_PATH is not a directory: \(path)") }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func generate(_ engine: InferenceEngine, _ prompt: String) async -> String {
        let (stream, _) = engine.generate(
            messages: [["role": "user", "content": prompt]], params: .greedy,
            maxTokens: 32, useSpeculative: false, usePrefixCache: false)
        var out = ""
        for await ev in stream where !ev.isEnd { out += ev.text }
        return out
    }

    func testLongContextStillGenerates() async throws {
        let engine = InferenceEngine(modelDirectory: try modelDir())
        try await engine.load()

        // VARIED content (NOT one sentence repeated). A repeated context masks
        // long-context attention bugs: the answer saturates the residual stream
        // so even broken attention echoes it (this is exactly how the earlier
        // KV-shared decode RoPE-offset bug slipped past the original repeated
        // version of this test). Distinct sentences + a single NEEDLE fact force
        // real long-range retrieval. ~12 words/sentence * 180 = well past 2048
        // tokens, comfortably beyond the ~1024 (2x sliding-window) cliff.
        let filler = [
            "The continuous batcher serves many concurrent decode rows per weight read.",
            "Prefix KV cache is shared across requests to avoid re-prefilling context.",
            "Native Swift pipelines handle vision and voice without a Python bridge.",
            "Tool calling uses per-family adapters that emit the native call format.",
            "Grammar-constrained decoding can force schema-valid JSON output.",
            "Cold model load and total request latency are measured wins over Ollama.",
        ]
        var sentences: [String] = []
        for i in 0 ..< 180 { sentences.append(filler[i % filler.count]) }
        // Drop the needle in the MIDDLE so it sits outside the final 512 window
        // and can only be retrieved by the full-attention layers.
        sentences.insert("The internal project codename for this release is Marlin-Seven.", at: 90)
        let longContext = sentences.joined(separator: " ")
            + "\n\nQuestion: What is the internal project codename for this release?\nAnswer:"

        let long = await generate(engine, longContext)
        let trimmed = long.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertFalse(trimmed.isEmpty,
            "long-context generation must not be empty (the sliding-window bug); got \(long.count) chars")
        // Degeneration guard: the KV-shared decode bug produced loops like
        // `{"{"{"...` / ` ```json ` repeated. Assert output is not dominated by a
        // single repeated short fragment.
        let words = trimmed.split(separator: " ")
        if words.count >= 6 {
            let unique = Set(words.map(String.init)).count
            XCTAssertGreaterThan(unique, 2,
                "long-context output degenerated into a repetition loop: \(long)")
        }
        // Needle retrieval: only correct long-range attention recovers this.
        XCTAssertTrue(long.lowercased().contains("marlin"),
            "long-context answer must retrieve the mid-context needle; got: \(long)")

        // Sanity: a short prompt (within the window) still works.
        let short = await generate(engine, "In one word, what language is Swift?")
        XCTAssertFalse(short.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "short-context generation regressed")
    }
}
