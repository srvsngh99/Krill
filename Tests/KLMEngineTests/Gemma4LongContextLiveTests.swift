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

        // ~40 words per unit; 60 units is well over 2000 tokens, comfortably past
        // the 512 window (and the ~1024-token cliff where the bug appeared).
        let unit = "KrillLM is a native Swift and MLX inference engine for Apple "
            + "Silicon. It serves text, vision, audio, embeddings, and tool calling. "
        let longContext = String(repeating: unit, count: 60)
            + "\n\nQuestion: In one sentence, what is KrillLM written in?\nAnswer:"

        let long = await generate(engine, longContext)
        XCTAssertFalse(long.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "long-context generation must not be empty (the sliding-window bug); got \(long.count) chars")
        XCTAssertTrue(long.lowercased().contains("swift"),
            "long-context answer should mention Swift; got: \(long)")

        // Sanity: a short prompt (within the window) still works.
        let short = await generate(engine, "In one word, what language is Swift?")
        XCTAssertFalse(short.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            "short-context generation regressed")
    }
}
