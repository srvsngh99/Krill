import XCTest
@testable import KLMEngine

/// Foundational correctness gate for batched speculative decode: a single
/// `[R, L]` batched verify forward (per-row block-causal mask + per-row RoPE)
/// must reproduce, for every row and every position, the logits of L sequential
/// single-token steps. If this holds, a batched speculative verify can commit
/// multiple tokens per row in one forward — the prerequisite for batched n-gram
/// spec (filling the R=4-8 GPU-occupancy gap, see docs/CONCURRENT_THROUGHPUT.md).
///
/// Gated on `KLM_BATCH_MODEL_PATH` (a plain-causal checkpoint), matching the
/// live-test convention in `BatchedDecodeLiveTests`.
final class BatchedVerifyTests: XCTestCase {

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

    func testBatchedVerifyMatchesSequentialPerRow() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible (not plain-causal)")
        }

        // Ragged-length prompts so per-row left-pad + per-row RoPE offsets are all
        // exercised (the same cases the single-token batched gate covers).
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

        // L = 8 new tokens verified in ONE [R, 8] forward vs 8 sequential steps.
        let diffs = try XCTUnwrap(
            engine.batchedVerifyVsSerialMaxDiff(promptTokens: prompts, steps: 8))
        XCTAssertEqual(diffs.count, prompts.count)
        for (r, d) in diffs.enumerated() {
            XCTAssertLessThan(d, 0.05,
                "row \(r): [R,L] batched verify logits must match L sequential steps "
                + "(max abs diff \(d)) — block-causal mask + per-row RoPE correct")
        }
    }

    /// The full batched n-gram spec driver: each row's speculative output must
    /// match that row's batched GREEDY output up to the shorter length (prefix
    /// consistency — fp16 verify-vs-decode tie-flips can fork the tail, the same
    /// guarantee the single-stream n-gram path gives). Uses an echo-heavy prompt
    /// so the proposer actually fires (multi-token accept path exercised) and
    /// distinct prompts so per-row isolation is tested.
    func testBatchedNgramSpecMatchesBatchedGreedyPerRow() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else {
            throw XCTSkip("loaded model is not batched-eligible")
        }

        let texts = [
            "Repeat exactly three times: red green blue, red green blue,",
            "The capital of France is",
            "def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return a + b\n\ndef add(a, b):",
        ]
        let prompts = try texts.map { t -> [Int] in
            guard let toks = engine.encodeForBatchTest(t), !toks.isEmpty else {
                throw XCTSkip("tokenizer produced no tokens")
            }
            return toks
        }

        let greedy = try XCTUnwrap(engine.batchedGreedyDecode(promptTokens: prompts, maxTokens: 48))
        let spec = try XCTUnwrap(engine.batchedNgramSpecDecode(promptTokens: prompts, maxTokens: 48))
        XCTAssertEqual(spec.count, greedy.count)

        for r in 0 ..< prompts.count {
            var lcp = 0
            while lcp < greedy[r].count && lcp < spec[r].count && greedy[r][lcp] == spec[r][lcp] {
                lcp += 1
            }
            XCTAssertEqual(lcp, min(greedy[r].count, spec[r].count),
                "row \(r): batched spec must match batched greedy up to the shorter "
                + "length (LCP=\(lcp), greedy=\(greedy[r].count), spec=\(spec[r].count))")
        }
    }

    /// Production-path parity: the continuous batcher WITH n-gram spec enabled
    /// must produce, for every concurrent row, the same tokens as the plain
    /// batched path up to the shorter length (prefix consistency; both use the
    /// batched fp16 GEMM, so this isolates the spec round from fp16 batched
    /// tie-flips). Drives the real `submitBatched` -> `ContinuousBatcher` path.
    func testContinuousBatcherSpecMatchesPlainBatchedPerRow() async throws {
        let dir = try requireModelDirectory()
        let texts = [
            "Repeat exactly three times: alpha beta gamma, alpha beta gamma,",
            "The capital of Japan is",
            "List three fruits:",
        ]

        func runBatched(ngram: Bool) async throws -> [[Int]] {
            let engine = InferenceEngine(modelDirectory: dir)
            try await engine.load()
            guard engine.supportsBatchedDecode else { throw XCTSkip("not batch-eligible") }
            if ngram { engine.setNgramSpec(true) }
            let results = texts.map { t in
                engine.submitBatched(
                    BatchGenRequest(messages: [["role": "user", "content": t]],
                                    params: .greedy, maxTokens: 48),
                    maxRows: 4, windowMs: 30)
            }
            var out = [[Int]](repeating: [], count: texts.count)
            await withTaskGroup(of: (Int, [Int]).self) { group in
                for (i, r) in results.enumerated() {
                    guard let res = r else { continue }
                    group.addTask {
                        var toks: [Int] = []
                        for await ev in res.stream where !ev.isEnd { toks.append(ev.tokenId) }
                        return (i, toks)
                    }
                }
                for await (i, toks) in group { out[i] = toks }
            }
            return out
        }

        let plain = try await runBatched(ngram: false)
        let spec = try await runBatched(ngram: true)
        for r in 0 ..< texts.count {
            var lcp = 0
            while lcp < plain[r].count && lcp < spec[r].count && plain[r][lcp] == spec[r][lcp] {
                lcp += 1
            }
            XCTAssertGreaterThan(spec[r].count, 0, "row \(r) produced no tokens under spec")
            XCTAssertEqual(lcp, min(plain[r].count, spec[r].count),
                "row \(r): batcher spec must match plain batched up to the shorter length "
                + "(LCP=\(lcp), plain=\(plain[r].count), spec=\(spec[r].count))")
        }
    }

    /// Wall-clock observation (gated): batched n-gram spec vs batched greedy on an
    /// echo-heavy batch at R=4 and R=8 — the regime where the wider verify forward
    /// should fill the under-occupied GPU and commit multiple tokens/round. Prints
    /// only; proves (or refutes) the occupancy thesis for the batched path.
    func testBatchedNgramSpecThroughputObservation() async throws {
        guard ProcessInfo.processInfo.environment["KLM_BATCH_SPEC_PERF"] != nil else {
            throw XCTSkip("Set KLM_BATCH_SPEC_PERF to run the batched-spec throughput observation")
        }
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        guard engine.supportsBatchedDecode else { throw XCTSkip("not batched-eligible") }

        let echo = "Repeat the following sentence verbatim five times: the swift brown "
            + "fox leaps over the lazy dog by the river."
        guard let toks = engine.encodeForBatchTest(echo), !toks.isEmpty else {
            throw XCTSkip("tokenizer produced no tokens")
        }

        func time(_ body: () -> [[Int]]?) -> (s: Double, tok: Int) {
            _ = body()                              // warm
            let t0 = CFAbsoluteTimeGetCurrent()
            let out = body() ?? []
            return (CFAbsoluteTimeGetCurrent() - t0, out.reduce(0) { $0 + $1.count })
        }

        for R in [4, 8] {
            let prompts = Array(repeating: toks, count: R)
            let g = time { engine.batchedGreedyDecode(promptTokens: prompts, maxTokens: 96) }
            let s = time { engine.batchedNgramSpecDecode(promptTokens: prompts, maxTokens: 96) }
            let gtps = g.s > 0 ? Double(g.tok) / g.s : 0
            let stps = s.s > 0 ? Double(s.tok) / s.s : 0
            print(String(format:
                "[batched-spec R=%d] greedy %.1f tok/s (%.2fs)  |  spec %.1f tok/s (%.2fs)  |  %.2fx",
                R, gtps, g.s, stps, s.s, gtps > 0 ? stps / gtps : 0))
        }
    }
}
