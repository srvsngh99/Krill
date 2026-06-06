import XCTest
import Foundation
@testable import KLMEngine

/// Live regression: the CONCURRENT batched path (`submitBatched` ->
/// `ContinuousBatcher` -> `Gemma4Model.batchedDecode`) must still generate
/// non-empty output past the 512-token sliding window. The solo path got the
/// per-layer windowed mask first; this pins the batched-decode step, where the
/// window is applied as a row-independent "too-old" column mask on the stacked
/// cache. Before it, a long-context concurrent request decoded its sliding
/// layers over the full out-of-distribution context and stopped immediately.
///
/// Set `KLM_BATCH_MODEL_PATH` to a Gemma 4 checkpoint (e.g. gemma-4-e2b).
final class Gemma4BatchedLongContextLiveTests: XCTestCase {
    func testBatchedLongContextStillGenerates() async throws {
        guard let p = ProcessInfo.processInfo.environment["KLM_BATCH_MODEL_PATH"], !p.isEmpty
        else { throw XCTSkip("KLM_BATCH_MODEL_PATH not set") }
        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: p, isDirectory: true))
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }

        // ~16 words/unit * 60 ~= 2000+ tokens, well past the 512 window.
        let unit = "KrillLM is a native Swift and MLX inference engine for Apple "
            + "Silicon. It serves text, vision, audio, embeddings, and tool calling. "
        let longCtx = String(repeating: unit, count: 60)
            + "\n\nQuestion: what language is KrillLM written in?\nAnswer:"

        func run(_ text: String) async -> [Int] {
            guard let r = engine.submitBatched(
                BatchGenRequest(messages: [["role": "user", "content": text]],
                                params: .greedy, maxTokens: 16, usePrefixCache: false),
                maxRows: 4, windowMs: 0) else { return [] }
            var toks: [Int] = []
            for await ev in r.stream { if ev.isEnd { break }; toks.append(ev.tokenId) }
            return toks
        }

        let short = await run("In one word, what language is Swift?")
        XCTAssertGreaterThan(short.count, 0, "batched short-context produced no tokens")
        let long = await run(longCtx)
        XCTAssertGreaterThan(long.count, 0,
            "batched LONG-context produced no tokens - the sliding window is not "
            + "applied on the batched-decode step")
    }
}
