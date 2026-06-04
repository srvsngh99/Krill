import XCTest
import Foundation
@testable import KLMEngine

/// Live, env-gated correctness gate for partial-prefix (shared-prefix) KV reuse
/// (BENCHMARK_ISSUES #0). A request that shares a long prefix with a recent
/// prefill but diverges in the tail must:
///   1. produce output BYTE-IDENTICAL to a cold full prefill (greedy), and
///   2. prefill materially faster (only the diverging suffix is forwarded).
///
/// Set `KLM_TEXT_MODEL_PATH` to a full-attention text checkpoint (e.g.
/// qwen2.5-3b). Sliding-window families (Gemma 4) are intentionally excluded
/// from partial reuse, so do NOT point this at one.
final class PrefixCachePartialReuseLiveTests: XCTestCase {

    private func modelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_TEXT_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_TEXT_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_TEXT_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// A long shared scaffold (system prompt + reused context) followed by a
    /// short varying question — the agentic/RAG shape.
    private let sharedPrefix =
        "You are an expert assistant. "
        + String(repeating: "Here is reference context reused across many queries. ", count: 40)

    private func userMessage(_ q: String) -> [[String: String]] {
        [["role": "user", "content": sharedPrefix + q]]
    }

    private func run(
        _ engine: InferenceEngine, _ q: String, maxTokens: Int = 48
    ) async -> (text: String, prefill: Double) {
        let (stream, getStats) = engine.generate(
            messages: userMessage(q), params: .greedy, maxTokens: maxTokens,
            useSpeculative: false, usePrefixCache: true)
        var out = ""
        for await ev in stream { if ev.isEnd { break }; out += ev.text }
        return (out, getStats()?.prefillTime ?? 0)
    }

    func testPartialReuseIsBitExactAndFaster() async throws {
        let dir = try modelDir()

        // Cold engine: never sees the shared prefix before this question, so it
        // full-prefills. This is the ground truth.
        let cold = InferenceEngine(modelDirectory: dir)
        try await cold.load()
        let baseline = await run(cold, "Question: Explain in one sentence why the sky is blue.")

        // Warm engine: prime the shared prefix with a DIFFERENT question, then
        // ask the same question as the cold engine. The second call shares the
        // whole scaffold and must reuse it (suffix-only prefill).
        let warm = InferenceEngine(modelDirectory: dir)
        try await warm.load()
        _ = await run(warm, "Question: Say hello.")
        let reused = await run(warm, "Question: Explain in one sentence why the sky is blue.")

        XCTAssertFalse(baseline.text.isEmpty, "baseline produced no output")
        XCTAssertEqual(
            reused.text, baseline.text,
            "partial-prefix reuse must be byte-identical to a cold full prefill "
            + "(greedy); a mismatch means the restored prefix / suffix mask is wrong")
        // The reused prefill forwards only the short suffix, so it is far cheaper
        // than the cold full prefill of the ~500-token scaffold. Generous bound
        // (half) to stay robust across machines while still proving reuse engaged.
        XCTAssertLessThan(
            reused.prefill, baseline.prefill * 0.5,
            "reused prefill (\(reused.prefill)s) should be far below the cold "
            + "full prefill (\(baseline.prefill)s); near-equal means reuse did not engage")
    }
}
