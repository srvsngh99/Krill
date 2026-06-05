import XCTest
import Foundation
@testable import KLMEngine

/// Live, env-gated correctness gate for partial-prefix (shared-prefix) KV reuse
/// on **Gemma 4** - the case the dense `PrefixCachePartialReuseLiveTests` does
/// not cover.
///
/// Gemma 4 uses cross-layer KV sharing: a trailing run of layers reuses a donor
/// layer's accumulated K/V and keeps an empty own cache, so a naive suffix
/// forward rotates the shared layers' Q at offset 0 (correct only for a cold
/// full prefill that starts at position 0). For a partial-prefix RESUME the
/// suffix must be rotated at its TRUE positions [LCP, count); `Gemma4Attention`
/// derives that base from the donor's post-update length. This test pins the
/// invariant: a shared-prefix request must decode BYTE-IDENTICALLY to a cold
/// full prefill, and prefill far faster.
///
/// Set `KLM_GEMMA4_MODEL_PATH` to a Gemma 4 checkpoint (e.g. gemma-4-e2b).
/// Runs on the default fp16 serial KV path (do not set KRILL_KV_CACHE_DTYPE).
final class Gemma4PartialReuseLiveTests: XCTestCase {

    private func modelDir() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// A long shared scaffold (system prompt + reused context) followed by a
    /// short varying question - the agentic/RAG shape that drives a long
    /// shared-prefix suffix forward through the KV-shared layers.
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

    func testGemma4PartialReuseIsBitExactAndFaster() async throws {
        let dir = try modelDir()

        // Cold engine: never sees the shared prefix before this question, so it
        // full-prefills. This is the ground truth.
        let cold = InferenceEngine(modelDirectory: dir)
        try await cold.load()
        let baseline = await run(cold, "Question: Explain in one sentence why the sky is blue.")

        // Warm engine: prime the shared prefix with a DIFFERENT question, then
        // ask the same question as the cold engine. The second call shares the
        // whole scaffold and must reuse it (suffix-only prefill across the
        // KV-shared layers).
        let warm = InferenceEngine(modelDirectory: dir)
        try await warm.load()
        _ = await run(warm, "Question: Say hello.")
        let reused = await run(warm, "Question: Explain in one sentence why the sky is blue.")

        XCTAssertFalse(baseline.text.isEmpty, "baseline produced no output")
        XCTAssertEqual(
            reused.text, baseline.text,
            "Gemma 4 partial-prefix reuse must be byte-identical to a cold full "
            + "prefill (greedy); a mismatch means the shared-layer suffix-Q RoPE "
            + "offset is wrong (the divergence the offset-0 path produced)")
        XCTAssertLessThan(
            reused.prefill, baseline.prefill * 0.5,
            "reused prefill (\(reused.prefill)s) should be far below the cold "
            + "full prefill (\(baseline.prefill)s); near-equal means reuse did not engage")
    }

    /// Same invariant on the int8-KV serial path (`KRILL_KV_CACHE_DTYPE=int8`).
    /// Gemma 4 is the only family that uses int8 KV; the partial reuse there
    /// restores quantized per-layer snapshots, truncates them to the shared
    /// prefix length, and forwards the suffix. The shared-layer suffix-Q RoPE
    /// fix is dtype-agnostic (it reads the donor cache's sequence length), so
    /// the int8 resume must be byte-identical to a cold int8 prefill too.
    func testGemma4PartialReuseIsBitExactAndFasterInt8() async throws {
        let dir = try modelDir()

        let cold = InferenceEngine(modelDirectory: dir, kvCacheDtype: "int8")
        try await cold.load()
        let baseline = await run(cold, "Question: Explain in one sentence why the sky is blue.")

        let warm = InferenceEngine(modelDirectory: dir, kvCacheDtype: "int8")
        try await warm.load()
        _ = await run(warm, "Question: Say hello.")
        let reused = await run(warm, "Question: Explain in one sentence why the sky is blue.")

        XCTAssertFalse(baseline.text.isEmpty, "int8 baseline produced no output")
        XCTAssertEqual(
            reused.text, baseline.text,
            "Gemma 4 int8 partial-prefix reuse must be byte-identical to a cold "
            + "int8 full prefill (greedy)")
        XCTAssertLessThan(
            reused.prefill, baseline.prefill * 0.5,
            "int8 reused prefill (\(reused.prefill)s) should be far below the cold "
            + "full prefill (\(baseline.prefill)s); near-equal means reuse did not engage")
    }
}
