import XCTest
import MLX
@testable import KLMEngine

/// Verifies engine-level chunked prefill on the Gemma 4 unified (12B) model:
///   1. CORRECTNESS - chunked prefill is numerically the single-shot prefill
///      (greedy token prefixes match) at a length that fits both ways.
///   2. CAPABILITY - chunked prefill RUNS at 16k+ context where single-shot
///      OOMs on MLX's materialized score matrix, and the model still answers a
///      needle planted in the long context.
///
///   KLM_GEMMA4_UNIFIED_MODEL_PATH=~/.krillm/models/blobs/gemma-4-12b \
///     swift test --filter Gemma4ChunkedPrefillTests
final class Gemma4ChunkedPrefillTests: XCTestCase {

    private func requireModelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_UNIFIED_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_UNIFIED_MODEL_PATH not set")
        }
        let expanded = (path as NSString).expandingTildeInPath
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("not a directory: \(expanded)")
        }
        return URL(fileURLWithPath: expanded, isDirectory: true)
    }

    /// Long context with a planted needle the model must echo back.
    private func longPrompt(sections: Int) -> String {
        let para = """
        The KrillLM engine runs large language models natively on Apple Silicon \
        using MLX. It supports continuous batching, prefix-cache reuse, grammar \
        constrained decoding, native tool calling, and multimodal input. The \
        secret pass phrase for this document is "tangerine satellite".
        """
        var blocks: [String] = []
        for i in 0..<sections { blocks.append("Section \(i): " + para) }
        blocks.append("\nQuestion: What is the secret pass phrase mentioned above? Answer:")
        return blocks.joined(separator: "\n")
    }

    private func runGreedy(
        modelDir: URL, prefillChunkSize: Int, prompt: String, maxTokens: Int
    ) async throws -> (tokens: [Int], text: String, decodeTokPerSec: Double, peakGB: Double) {
        let engine = InferenceEngine(
            modelDirectory: modelDir, kvCacheDtype: "fp16", prefillChunkSize: prefillChunkSize)
        try await engine.load()
        defer { engine.unload() }

        GPU.resetPeakMemory()
        let basePeak = Memory.snapshot().peakMemory
        let (stream, _) = engine.generate(
            prompt: prompt, params: .greedy, maxTokens: maxTokens, usePrefixCache: false)

        var tokens: [Int] = []
        var text = ""
        var firstAt: Date?
        var lastAt = Date()
        for await event in stream {
            if event.isEnd { break }
            if firstAt == nil { firstAt = Date() }
            lastAt = Date()
            tokens.append(event.tokenId)
            text += event.text
            if tokens.count >= maxTokens { break }
        }
        let rate: Double = {
            guard let s = firstAt, tokens.count > 1 else { return 0 }
            let dt = lastAt.timeIntervalSince(s)
            return dt > 0 ? Double(tokens.count - 1) / dt : 0
        }()
        let peakGB = Double(Memory.snapshot().peakMemory - basePeak) / 1_073_741_824
        return (tokens, text, rate, peakGB)
    }

    /// CORRECTNESS: chunked (2048) must equal single-shot (0) at ~6.6k ctx.
    func testChunkedMatchesSingleShot() async throws {
        let modelDir = try requireModelDirectory()
        let prompt = longPrompt(sections: 120)  // ~6.6k tokens, fits both ways

        let single = try await runGreedy(
            modelDir: modelDir, prefillChunkSize: 0, prompt: prompt, maxTokens: 16)
        let chunked = try await runGreedy(
            modelDir: modelDir, prefillChunkSize: 2048, prompt: prompt, maxTokens: 16)

        let n = min(single.tokens.count, chunked.tokens.count)
        XCTAssertGreaterThan(n, 0, "no tokens produced")

        // Robust invariant, NOT strict byte-equality: chunking changes the
        // attention GEMM shape ([chunk, ctx] vs [L, L]), so bf16 rounding can
        // flip a greedy tie at a chunk boundary - the same non-determinism that
        // made the partial-prefix-reuse work gate on a within-bf16 invariant
        // rather than exact tokens (see Gemma4PartialReuseLiveTests). Require the
        // first (prefill-sampled) token to match and a high overall match ratio.
        let firstMatch = chunked.tokens.first == single.tokens.first
        let matches = (0..<n).filter { chunked.tokens[$0] == single.tokens[$0] }.count
        let ratio = Double(matches) / Double(n)
        let firstDiverge = (0..<n).first { chunked.tokens[$0] != single.tokens[$0] } ?? n
        print("""

        [chunked-prefill correctness] ~6.6k ctx
          exact tokens match: \(chunked.tokens == single.tokens)  match ratio: \(String(format: "%.2f", ratio))  first diverge @ \(firstDiverge)/\(n)
          single-shot peak=\(String(format: "%.2f", single.peakGB))GB  chunked peak=\(String(format: "%.2f", chunked.peakGB))GB

        """)
        XCTAssertTrue(firstMatch, "chunked prefill changed the first sampled token")
        XCTAssertGreaterThanOrEqual(ratio, 0.85,
            "chunked prefill diverged from single-shot beyond bf16 tie noise (ratio \(ratio))")
    }

    /// CAPABILITY: chunked prefill runs at 16k+ where single-shot hard-OOMs, and
    /// the model answers the planted needle. Reports tok/s and prefill peak.
    func testChunkedRunsAtLongContext() async throws {
        let modelDir = try requireModelDirectory()
        for sections in [330, 580] {  // ~20k, ~35k tokens
            let prompt = longPrompt(sections: sections)
            let r = try await runGreedy(
                modelDir: modelDir, prefillChunkSize: 2048, prompt: prompt, maxTokens: 24)
            XCTAssertGreaterThan(r.tokens.count, 0,
                "chunked prefill produced no tokens at sections=\(sections)")
            let found = r.text.lowercased().contains("tangerine")
            print("""

            [chunked-prefill capability] sections=\(sections)
              ran OK, \(r.tokens.count) tokens, decode=\(String(format: "%.1f", r.decodeTokPerSec)) tok/s
              prefill peak=\(String(format: "%.2f", r.peakGB))GB  needle-found=\(found)
              answer: \(r.text.prefix(80))

            """)
            XCTAssertTrue(found,
                "needle 'tangerine satellite' not echoed at sections=\(sections); answer=\(r.text.prefix(120))")
        }
    }

    /// A/B baseline: single-shot prefill (chunkSize=0) on the SAME ~35k prompt is
    /// EXPECTED to hard-OOM (fatal, aborts the process) - the bf16 score matrix
    /// [16,~35k,~35k] needs a ~40GB single buffer (measured 39.86GB), far past
    /// MLX's 14.3GB limit. (At ~20k single-shot still fits, ~13GB scores; the
    /// wall is ~21k.) Opt-in only so it never crashes CI: KLM_RUN_OOM_BASELINE=1.
    func testSingleShotOOMsAtLongContext() async throws {
        guard ProcessInfo.processInfo.environment["KLM_RUN_OOM_BASELINE"] == "1" else {
            throw XCTSkip("KLM_RUN_OOM_BASELINE != 1 (this test intentionally crashes)")
        }
        let modelDir = try requireModelDirectory()
        let prompt = longPrompt(sections: 580)  // ~35k; single-shot scores ~40GB
        _ = try await runGreedy(
            modelDir: modelDir, prefillChunkSize: 0, prompt: prompt, maxTokens: 4)
        XCTFail("single-shot prefill did NOT OOM at ~32k - the score matrix limit moved")
    }
}
