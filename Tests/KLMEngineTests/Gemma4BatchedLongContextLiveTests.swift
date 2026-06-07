import XCTest
import Foundation
@testable import KLMEngine

/// Live regression for the CONCURRENT batched path (`submitBatched` ->
/// `ContinuousBatcher` -> `Gemma4Model.batchedDecode`) on long context.
///
/// Two distinct bugs are pinned here, both invisible to short prompts:
///  1. Sliding window on the batched-decode step: before it, a long-context
///     concurrent request decoded its sliding layers over the full
///     out-of-distribution context and stopped immediately (empty output).
///  2. KV-shared decode RoPE offset: shared layers (e2b/e4b layers 15-34) reuse
///     a donor's K (rotated at the row's TRUE position), so the shared-layer Q
///     must rotate at that same per-row position (`rowOffsets`), not 0. With the
///     old `zeroOffsets` the relative positions were wrong past ~2x the window
///     and long-context decode degenerated into repetition (non-empty garbage,
///     so a bare "is it empty?" check missed it).
///
/// This uses VARIED content (a repeated context masks both bugs by saturating
/// the residual stream) with a per-row NEEDLE placed mid-context (outside the
/// final 512-token window, so only correct long-range attention + the right
/// shared-layer offset can retrieve it). Each request goes through
/// `batchedDecode`; the shared-layer offset bug bites at B == 1 too (a single
/// shared row got `zeroOffsets = [0]` instead of its true position), so the rows
/// run sequentially and distinct needles also guard against row/KV bleed.
///
/// Set `KLM_BATCH_MODEL_PATH` to a Gemma 4 checkpoint (e.g. gemma-4-e2b).
final class Gemma4BatchedLongContextLiveTests: XCTestCase {

    /// Long varied context with `needle` dropped in the middle.
    private func haystack(needle: String) -> String {
        let filler = [
            "The continuous batcher serves many concurrent decode rows per weight read.",
            "Prefix KV cache is shared across requests to avoid re-prefilling context.",
            "Native Swift pipelines handle vision and voice without a Python bridge.",
            "Tool calling uses per-family adapters that emit the native call format.",
            "Grammar-constrained decoding can force schema-valid JSON output.",
            "Cold model load and total request latency are measured wins over Ollama.",
        ]
        var s: [String] = []
        for i in 0 ..< 180 { s.append(filler[i % filler.count]) }
        s.insert("The internal project codename is \(needle).", at: 90)
        return s.joined(separator: " ")
            + "\n\nQuestion: What is the internal project codename?\nAnswer:"
    }

    func testBatchedLongContextRetrievesNeedle() async throws {
        guard let p = ProcessInfo.processInfo.environment["KLM_BATCH_MODEL_PATH"], !p.isEmpty
        else { throw XCTSkip("KLM_BATCH_MODEL_PATH not set") }
        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: p, isDirectory: true))
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }

        func run(_ text: String) async -> String {
            guard let r = engine.submitBatched(
                BatchGenRequest(messages: [["role": "user", "content": text]],
                                params: .greedy, maxTokens: 16, usePrefixCache: false),
                maxRows: 4, windowMs: 0) else { return "" }
            var out = ""
            for await ev in r.stream { if ev.isEnd { break }; out += ev.text }
            return out
        }

        // Each long-context row goes through `batchedDecode` (the fixed path);
        // the shared-layer offset bug bites at B == 1 too (a single shared row
        // got `zeroOffsets = [0]` instead of its true position). Distinct needles
        // also guard against any row/KV bleed across requests.
        let needles = ["Marlin-Seven", "Beluga-Three"]
        for needle in needles {
            let out = await run(haystack(needle: needle))
            XCTAssertFalse(out.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "batched long-context produced no tokens for needle \(needle) "
                + "(sliding window not applied on the batched-decode step)")
            // Degeneration guard: the KV-shared offset bug produced repetition loops.
            let words = out.split(separator: " ")
            if words.count >= 6 {
                XCTAssertGreaterThan(Set(words.map(String.init)).count, 2,
                    "batched long-context degenerated into a repetition loop: \(out)")
            }
            // Needle retrieval: only the correct per-row shared-layer offset +
            // long-range attention recovers the mid-context needle. The leading
            // word of the codename is what the row must surface.
            let lead = needle.split(separator: "-").first.map(String.init) ?? needle
            XCTAssertTrue(out.lowercased().contains(lead.lowercased()),
                "batched row did not retrieve its mid-context needle \(needle); got: \(out)")
        }
    }
}
