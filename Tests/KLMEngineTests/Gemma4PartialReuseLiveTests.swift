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
/// derives that base from the donor's post-update length.
///
/// CORRECTNESS STANDARD (why not strict greedy-token equality here). The dense
/// families decode byte-identically on a reused prefix and are gated that way.
/// Gemma 4 computes in **bf16**: a partial reuse forwards only the suffix, whose
/// shorter GEMM rounds a few percent differently than the cold full forward.
/// The reused cache is still numerically correct (see
/// `testCacheMatchesColdWithinBf16`), but that rounding can flip a downstream
/// greedy near-tie into an equally-valid different continuation - the same bf16
/// behavior the batched-decode gate already documents. So Gemma 4 is gated on
/// the robust, prompt-independent invariants instead, which split cleanly by
/// what they protect:
///   1. RESTORE/TRUNCATE/SUFFIX-FORWARD: the reused KV cache (what the DONOR,
///      non-shared layers write) matches a cold prefill within bf16 GEMM noise.
///      A bad restore or wrong truncate length corrupts it at O(1) relative.
///      This does NOT cover the shared-layer Q offset (Q is never cached).
///   2. SHARED-LAYER Q OFFSET + engine wiring: the first decoded token - which
///      flows through the KV-shared layers and which the pre-fix offset-0 path
///      grossly corrupted - matches cold, and prefill is far faster (reuse
///      actually engaged). A bf16 tie-flip only diverges LATER in the sequence,
///      so a first-token match is a sound proxy for the offset being right.
///
/// Set `KLM_GEMMA4_MODEL_PATH` to a Gemma 4 checkpoint (e.g. gemma-4-e2b).
final class Gemma4PartialReuseLiveTests: XCTestCase {

    /// Relative bound on the reused-vs-cold cache diff. Gemma 4's bf16 suffix
    /// GEMM rounds at a few percent (measured ~0.05 on e2b); a bad restore or a
    /// wrong truncate length corrupts the donor cache at O(1) relative, so this
    /// generously separates "correct within rounding" from "wrong". (The
    /// shared-layer Q offset is gated by the first-token check, not this bound -
    /// Q is never cached.)
    private let bf16CacheRelBound: Float = 0.2

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
    ) async -> (tokens: [Int], prefill: Double) {
        let (stream, getStats) = engine.generate(
            messages: userMessage(q), params: .greedy, maxTokens: maxTokens,
            useSpeculative: false, usePrefixCache: true)
        // Drain to completion rather than breaking on the end event: the engine
        // writes `getStats()` just before it finishes the stream, so breaking
        // early can read stats before they are populated (prefillTime == 0).
        var toks: [Int] = []
        for await ev in stream where !ev.isEnd { toks.append(ev.tokenId) }
        return (toks, getStats()?.prefillTime ?? 0)
    }

    /// Invariant 1 (rigorous, prompt-independent): the reused cache matches a
    /// cold prefill within bf16 GEMM rounding. Restore-truncate-suffix-forward
    /// must reproduce the cold full-prefill donor K/V; a bad restore or a wrong
    /// truncate length corrupts it at O(1) relative. (Scope: donor caches only -
    /// the shared-layer Q offset is not cached and is gated by Invariant 2.)
    func testCacheMatchesColdWithinBf16() async throws {
        let dir = try modelDir()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard let toks = engine.encodeForBatchTest(
            sharedPrefix + "Question: name a primary color."), toks.count > 32 else {
            throw XCTSkip("tokenizer produced too few tokens")
        }
        let split = toks.count - 12
        let r = try XCTUnwrap(engine.partialPrefillCacheMaxDiff(
            prefix: Array(toks[0..<split]), suffix: Array(toks[split...])))
        // Guard against a vacuous pass: at least one populated layer must have
        // been compared (maxCold stays 0 only if every layer was skipped).
        XCTAssertGreaterThan(r.maxCold, 0,
            "no populated layers were compared - the probe measured nothing")
        let rel = r.maxDiff / r.maxCold
        XCTAssertLessThan(rel, bf16CacheRelBound,
            "Gemma 4 reused donor cache diverged from cold by \(rel) relative "
            + "(maxDiff \(r.maxDiff), maxCold \(r.maxCold)); above bf16 GEMM noise "
            + "this means the restore or truncate length is wrong")
    }

    /// Invariant 2 (fp16 engine end-to-end): partial reuse engages - the first
    /// decoded token matches cold (a broken shared-layer offset would corrupt
    /// it, as the pre-fix offset-0 path did) and prefill is far faster.
    func testPartialReuseEngagesAndIsFaster() async throws {
        try await assertEngagesAndIsFaster(kvCacheDtype: nil)
    }

    /// Invariant 2 on the int8-KV serial path (`KRILL_KV_CACHE_DTYPE=int8`).
    /// Gemma 4 is the only family that uses int8 KV; partial reuse there restores
    /// quantized per-layer snapshots, truncates to the shared length, and
    /// forwards the suffix (per-token quantization, so the prefix is restored
    /// bit-identically).
    func testPartialReuseEngagesAndIsFasterInt8() async throws {
        try await assertEngagesAndIsFaster(kvCacheDtype: "int8")
    }

    private func assertEngagesAndIsFaster(kvCacheDtype: String?) async throws {
        let dir = try modelDir()
        let question = "Question: Explain in one sentence why the sky is blue."

        // Cold engine: never sees the shared prefix before this question.
        let cold = InferenceEngine(modelDirectory: dir, kvCacheDtype: kvCacheDtype)
        try await cold.load()
        let baseline = await run(cold, question)

        // Warm engine: prime the shared prefix with a DIFFERENT question, then
        // ask the same question - the second call shares the whole scaffold and
        // must reuse it (suffix-only prefill across the KV-shared layers).
        let warm = InferenceEngine(modelDirectory: dir, kvCacheDtype: kvCacheDtype)
        try await warm.load()
        _ = await run(warm, "Question: Say hello.")
        let reused = await run(warm, question)

        XCTAssertFalse(baseline.tokens.isEmpty, "baseline produced no output")
        XCTAssertEqual(reused.tokens.first, baseline.tokens.first,
            "Gemma 4 partial-reuse first token \(String(describing: reused.tokens.first)) "
            + "!= cold \(String(describing: baseline.tokens.first)); a mismatch means the "
            + "shared-layer suffix-Q offset is wrong (gross divergence, not a bf16 tie-flip)")
        XCTAssertLessThan(reused.prefill, baseline.prefill * 0.5,
            "reused prefill (\(reused.prefill)s) should be far below the cold full "
            + "prefill (\(baseline.prefill)s); near-equal means reuse did not engage")
    }
}
