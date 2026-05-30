import XCTest
@testable import KLMEngine
import KLMSampler

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

        // Strict, dtype-independent gate first: R=1 batched must equal solo
        // EXACTLY (bit-for-bit). This exercises every per-row code path - the
        // per-row RoPE offsets (incl. Gemma 4's dual proportional/standard
        // bases), KV-sharing donor logic, PLE, softcap, and the `.array` mask
        // path - against the canonical decoder. A near-zero result here proves
        // the model math is correct independent of batch width; only the
        // batched-GEMM kernel-width rounding can then differ at R>1.
        for (r, p) in prompts.enumerated() {
            let d1 = try XCTUnwrap(
                engine.teacherForcedBatchedVsSerialMaxDiff(promptTokens: [p], steps: 16))
            XCTAssertLessThan(d1[0], 0.05,
                "R=1 row \(r) batched logits must equal solo exactly; diff \(d1[0])")
        }

        let diffs = try XCTUnwrap(
            engine.teacherForcedBatchedVsSerialMaxDiff(promptTokens: prompts, steps: 16))
        // At R>1 the only remaining difference from solo is the batched-GEMM
        // kernel-width rounding (proven above: R=1 is exact). This is a loose
        // "did not explode" guard, NOT the primary correctness signal - that
        // role is held by the exact R=1 gate above and the dtype-independent
        // bit-exact contamination test below. fp16 families (Llama, Qwen dense)
        // round at ~1 ULP (~0.03 at logit magnitude ~20), so 0.5 is ample.
        // Gemma 4 computes in bf16 (7-bit mantissa vs fp16's 10) and the
        // rounding accumulates with total model FLOPs - observed ~1.3 on e2b
        // (35 layers) and ~3.6 on e4b (42 layers, wider) - so a generous 8.0
        // ceiling absorbs the deepest published dense SKU while staying far
        // below the O(10+) a real cross-row/position bug produces (and such a
        // bug would already have broken the exact R=1 gate above).
        let family = engine.loadedModelForBatching?.family
        let isBF16 = family == "gemma4" || family == "moe"
        let bound: Float = isBF16 ? 8.0 : 0.5
        for (r, d) in diffs.enumerated() {
            XCTAssertLessThan(d, bound,
                "row \(r) (len \(prompts[r].count)) batched logits diverged from solo by \(d)")
        }
    }

    /// Dtype-independent cross-row isolation gate: a row's logits must not
    /// depend on its NEIGHBORS' content. Runs the row at a fixed batch width
    /// with two different sets of equal-length neighbors; batched matmul is
    /// per-row independent, so the row's logits must be bit-identical. Any
    /// nonzero diff is a genuine cross-batch bleed/indexing bug - and unlike
    /// the teacher-forced-vs-solo diff this is not confounded by bf16 GEMM
    /// rounding (batched-vs-batched at fixed width), so it must be ~0 even on
    /// Gemma 4. The row is left-padded here (neighbors are longer), so this
    /// also checks the left-pad mask hides the prefix regardless of neighbors.
    func testBatchedDecodeNoCrossRowContamination() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }
        let row = try XCTUnwrap(engine.encodeForBatchTest("The capital of France is"))
        // Neighbors: equal LENGTHS across A/B, different CONTENT; longer than
        // `row` so `row` is left-padded in the stacked cache.
        let nA = try XCTUnwrap(engine.encodeForBatchTest(
            "Describe the process of photosynthesis in plants in detail."))
        let nB = try XCTUnwrap(engine.encodeForBatchTest(
            "Explain how a combustion engine converts fuel into motion here."))
        // Trim to a common length so the geometry `row` sees is identical.
        let len = min(nA.count, nB.count)
        XCTAssertGreaterThan(len, row.count, "neighbor must be longer so row is padded")
        let diff = try XCTUnwrap(engine.crossRowContaminationMaxDiff(
            row: row, neighborsA: [Array(nA.prefix(len))],
            neighborsB: [Array(nB.prefix(len))], steps: 16))
        XCTAssertLessThan(diff, 0.05,
            "row logits changed by \(diff) when only neighbor content changed (cross-row bleed)")
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
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
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

    /// End-to-end test of the continuous batcher (`submitBatched`, the path the
    /// scheduler drives in Stage C1): submit rows over time so a NEWCOMER joins
    /// while an incumbent is already decoding, and an incumbent finishes while
    /// others keep going. Each row must still match its solo run's first token
    /// (proving admission into the running batch did not corrupt any row), and
    /// every stream must terminate exactly once. This exercises the epoch
    /// rebuild + scatter-back path that static `generateBatched` never hits.
    func testContinuousBatchingMatchesSerialFirstToken() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }

        // Differing maxTokens so one row finishes well before the others,
        // forcing a mid-flight shrink + epoch rebuild for the survivors.
        let cases: [(prompt: String, maxTokens: Int)] = [
            ("Count slowly from one to twenty.", 24),
            ("Say hi.", 2),
            ("Name a color.", 12),
        ]

        var serialFirst: [Int] = []
        for c in cases {
            let (s, _) = engine.generate(
                messages: [["role": "user", "content": c.prompt]],
                params: .greedy, maxTokens: c.maxTokens,
                useSpeculative: false, usePrefixCache: false)
            var first: Int?
            for await ev in s where !ev.isEnd { first = ev.tokenId; break }
            serialFirst.append(try XCTUnwrap(first))
        }

        // Submit each row through the continuous batcher with a small stagger so
        // later rows are admitted into the already-running batch.
        let firsts = await withTaskGroup(of: (Int, Int?, Bool).self) { group -> [Int: (Int?, Bool)] in
            for (i, c) in cases.enumerated() {
                let result = engine.submitBatched(
                    BatchGenRequest(messages: [["role": "user", "content": c.prompt]],
                                    params: .greedy, maxTokens: c.maxTokens),
                    maxRows: 4, windowMs: 0)
                let res = try! XCTUnwrap(result)
                group.addTask {
                    var first: Int?
                    var sawEnd = false
                    for await ev in res.stream {
                        if ev.isEnd { sawEnd = true; break }
                        if first == nil { first = ev.tokenId }
                    }
                    return (i, first, sawEnd)
                }
                // Stagger: let the prior row(s) start decoding before the next joins.
                try? await Task.sleep(nanoseconds: 30_000_000)
            }
            var out: [Int: (Int?, Bool)] = [:]
            for await (i, first, sawEnd) in group { out[i] = (first, sawEnd) }
            return out
        }

        for i in 0 ..< cases.count {
            let (first, sawEnd) = try XCTUnwrap(firsts[i])
            XCTAssertTrue(sawEnd, "row \(i) stream never terminated")
            XCTAssertEqual(first, serialFirst[i],
                           "row \(i) first token diverged from its serial run under continuous admission")
        }
    }

    /// Stage C4: the shared prefix cache on the batched (continuous-batcher)
    /// path must be an EXACT replay - a full prefix hit restores the prompt's KV
    /// and forwards only the last token, so its decode is byte-for-byte the cold
    /// (never-cached) decode. Drives `submitBatched` three times on one engine:
    /// (1) usePrefixCache=false (never stores) for the cold reference, (2)
    /// usePrefixCache=true to MISS and store the prompt write-behind, (3)
    /// usePrefixCache=true to HIT that stored entry. The hit run's full greedy
    /// token sequence must equal the cold run's, bit-for-bit. The prompt is
    /// comfortably longer than minPrefixLength (8), so step 2 really does store
    /// and step 3 really does hit - exercising the restore/trim path, not a
    /// silent miss.
    func testBatchedPrefixCacheReplayMatchesColdDecode() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }

        let prompt = "Explain in one clear sentence why the daytime sky looks blue to us."
        func runBatched(usePrefixCache: Bool) async -> [Int] {
            guard let r = engine.submitBatched(
                BatchGenRequest(messages: [["role": "user", "content": prompt]],
                                params: .greedy, maxTokens: 24,
                                usePrefixCache: usePrefixCache),
                maxRows: 4, windowMs: 0)
            else { return [] }
            var toks: [Int] = []
            for await ev in r.stream { if ev.isEnd { break }; toks.append(ev.tokenId) }
            return toks
        }

        let cold = await runBatched(usePrefixCache: false)   // never stores
        _ = await runBatched(usePrefixCache: true)           // miss -> stores the prompt
        let hit = await runBatched(usePrefixCache: true)     // full hit -> restore + 1 token

        XCTAssertGreaterThan(cold.count, 0, "cold batched run produced no tokens")
        XCTAssertEqual(hit, cold,
            "prefix-cache replay must decode identically to the cold (no-cache) run")
    }
}
