import XCTest
@testable import KLMEngine

/// Stage B correctness gate: a batched (B>1) decode of several DIFFERENT,
/// ragged-length prompts must produce, for every row, exactly the tokens that
/// prompt produces when decoded ALONE. This is the test that catches cross-row
/// attention bleed, wrong per-row RoPE positions, and left-pad mask leaks.
///
/// Skipped unless `KLM_BATCH_MODEL_PATH` points at a real plain-causal
/// checkpoint (Llama 3.x or Qwen 2.5/3 dense), matching the live-test gating
/// convention used elsewhere in this target.
final class BatchedDecodeLiveTests: XCTestCase {

    private func requireModelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_BATCH_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_BATCH_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_BATCH_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    /// The correctness gate: a batched decode of several ragged-length
    /// prompts must, for EVERY row, reproduce that prompt's solo logits at
    /// every decode step (teacher-forced, so greedy fp tie-flips do not
    /// confound). A near-zero per-row diff proves there is no cross-row
    /// attention bleed, the per-row RoPE positions are right, and the
    /// left-pad mask isolates each row. (Exact greedy-token equality is NOT
    /// asserted: fp16 batched-GEMM rounding of ~1 ULP can flip a greedy tie
    /// into an equally valid continuation - the same accepted behavior other
    /// batched-inference engines exhibit.)
    func testBatchedDecodeMatchesSerializedPerRow() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible (not plain-causal)")
        }

        // Three prompts of deliberately DIFFERENT lengths (ragged batch).
        let texts = [
            "The capital of France is",
            "Write one short sentence about the ocean and why it matters to the planet.",
            "List three colors:",
        ]
        let prompts = try texts.map { text -> [Int] in
            guard let toks = engine.encodeForBatchTest(text), !toks.isEmpty else {
                throw XCTSkip("tokenizer produced no tokens")
            }
            return toks
        }
        XCTAssertGreaterThan(Set(prompts.map(\.count)).count, 1, "prompts should differ in length")

        let diffs = try XCTUnwrap(
            engine.teacherForcedBatchedVsSerialMaxDiff(promptTokens: prompts, steps: 16))
        // fp16 batched-GEMM rounding is ~1 ULP (observed ~0.03 at logit
        // magnitudes ~20). A real cross-row/position bug corrupts logits by
        // O(1) or more, so 0.5 cleanly separates "fp noise" from "bug".
        for (r, d) in diffs.enumerated() {
            XCTAssertLessThan(d, 0.5,
                "row \(r) (len \(prompts[r].count)) batched logits diverged from solo by \(d)")
        }
    }

    /// R=1 batched decode must equal the canonical decoder (proves the batched
    /// forward itself is correct, independent of cross-row effects).
    func testBatchedDecodeRowOfOneMatchesSerial() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }
        let prompt = try XCTUnwrap(engine.encodeForBatchTest("Once upon a time"))
        let serial = try XCTUnwrap(engine.serialGreedyDecode(promptTokens: prompt, maxTokens: 24))
        let batched = try XCTUnwrap(engine.batchedGreedyDecode(promptTokens: [prompt], maxTokens: 24))
        XCTAssertEqual(batched[0], serial)
    }

    /// End-to-end test of the live streaming entry `generateBatched(_:)` (the
    /// path the BatchScheduler drives): three different chat prompts batched,
    /// each streamed to its own `AsyncStream`. Every row must terminate, emit
    /// at least one token, and — critically — produce the SAME first token as
    /// its serial `generate(...)` run. The first token comes from per-row B=1
    /// prefill (identical math in both paths), so it must match exactly; this
    /// is the strongest non-flaky cross-row-isolation signal. Later tokens can
    /// tie-flip under fp16 batched GEMM (documented in #91, asserted rigorously
    /// by the teacher-forced logit test above), so they are not compared here.
    func testBatchedStreamingMatchesSerialFirstToken() async throws {
        let dir = try requireModelDirectory()
        let engine = try await makeEngine(dir)
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }

        let prompts = [
            "Write one short sentence about the ocean.",
            "List three primary colors.",
            "What is two plus two?"
        ]

        // Serial greedy reference (no spec / no prefix cache, exactly what the
        // batched path uses), capturing each prompt's first generated token.
        var serialFirst: [Int] = []
        for p in prompts {
            let (s, _) = engine.generate(
                messages: [["role": "user", "content": p]],
                params: .greedy, maxTokens: 24,
                useSpeculative: false, usePrefixCache: false)
            var first: Int?
            for await ev in s where !ev.isEnd { first = ev.tokenId; break }
            serialFirst.append(try XCTUnwrap(first))
        }

        let reqs = prompts.map {
            BatchGenRequest(messages: [["role": "user", "content": $0]],
                            params: .greedy, maxTokens: 24)
        }
        let results = engine.generateBatched(reqs)
        XCTAssertEqual(results.count, reqs.count)

        for (i, res) in results.enumerated() {
            var first: Int?
            var count = 0
            var sawEnd = false
            for await ev in res.stream {
                if ev.isEnd { sawEnd = true; break }
                if first == nil { first = ev.tokenId }
                count += 1
            }
            XCTAssertTrue(sawEnd, "row \(i) stream never terminated")
            XCTAssertGreaterThan(count, 0, "row \(i) produced no tokens")
            XCTAssertEqual(first, serialFirst[i],
                           "row \(i) first token diverged from its serial run")
        }
    }
}
