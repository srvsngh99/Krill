import XCTest
@testable import KLMEngine
import KLMSampler

/// Live greedy-parity gate for n-gram (prompt-lookup) speculative decode.
///
/// The contract: with the SAME engine, greedy decode with `useNgramSpeculative:
/// true` reproduces standard decode token-for-token up to the first position
/// where the width-K verify forward's argmax disagrees with the width-1 decode
/// forward's argmax — an fp16 tie-flip. Every accepted token is the target
/// model's own greedy argmax at that position, so output is bit-identical except
/// at near-ties, exactly the nondeterminism Krill's batched decoder already
/// exhibits (see `BatchedDecodeLiveTests`). We therefore assert PREFIX
/// consistency (one sequence is a prefix of the other — `LCP == min length`),
/// which a structural bug would break (early mid-stream disagreement) while a
/// boundary tie-flip does not.
///
/// Skipped unless `KLM_NGRAM_MODEL_PATH` points at a plain-causal checkpoint
/// (e.g. `~/.krill/models/blobs/llama-3.2-3b`), matching the live-test gating
/// convention used elsewhere in this target.
final class NgramLiveParityTests: XCTestCase {

    private func requireModelDirectory() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["KLM_NGRAM_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_NGRAM_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_NGRAM_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    private func collect(
        _ engine: InferenceEngine, prompt: String, ngram: Bool, maxTokens: Int
    ) async -> (tokens: [Int], stats: GenerationStats?) {
        let (s, stats) = engine.generate(
            messages: [["role": "user", "content": prompt]],
            params: .greedy, maxTokens: maxTokens,
            useSpeculative: false, usePrefixCache: false,
            useNgramSpeculative: ngram)
        var toks: [Int] = []
        var sawEnd = false
        // Drain to natural completion (not `break` on isEnd): the producer sets
        // stats and only THEN calls finish(), so reading stats() after the loop
        // ends avoids racing the producer. isEnd carries no token, so skip it.
        // Invariant: isEnd must be terminal — no content event may follow it.
        // (Regression guard for the EOS-inside-a-fully-accepted-run bug, where a
        // bonus token was emitted past the stop id.)
        for await ev in s {
            if ev.isEnd { sawEnd = true; continue }
            XCTAssertFalse(sawEnd, "a content token was emitted after isEnd (post-stop emission)")
            toks.append(ev.tokenId)
        }
        return (toks, stats())
    }

    func testNgramGreedyOutputMatchesStandardDecode() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()

        // Repetitive prompts maximize n-gram hits so the accept path is exercised.
        let prompts = [
            "Repeat this list exactly three times: apple, banana, cherry, apple, banana, cherry,",
            "def add(a, b):\n    return a + b\n\ndef add(a, b):\n    return a + b\n\ndef add(a, b):",
            "Count: one two three four five one two three four five one two three four",
        ]

        for p in prompts {
            let ref = await collect(engine, prompt: p, ngram: false, maxTokens: 64)
            let ng = await collect(engine, prompt: p, ngram: true, maxTokens: 64)

            // Prefix consistency: identical up to the shorter length. A structural
            // bug (wrong cache rollback, position, or proposal alignment) would
            // disagree mid-stream (LCP < min); an fp16 tie-flip would not.
            var lcp = 0
            while lcp < ref.tokens.count && lcp < ng.tokens.count
                && ref.tokens[lcp] == ng.tokens[lcp] { lcp += 1 }
            XCTAssertEqual(lcp, min(ref.tokens.count, ng.tokens.count),
                "n-gram output must match standard decode up to the shorter length "
                + "(LCP=\(lcp), ref=\(ref.tokens.count), ngram=\(ng.tokens.count))")

            // The proposer should engage at least once on a repetitive prompt.
            let spec = try XCTUnwrap(ng.stats?.speculative,
                "n-gram run must report speculative stats")
            XCTAssertGreaterThan(spec.rounds, 0,
                "n-gram path should have run verify rounds on a repetitive prompt")
            XCTAssertGreaterThanOrEqual(spec.acceptedTokens, spec.rounds)
        }
    }

    /// The stall-monitor handoff must be byte-transparent: a long non-echo
    /// generation drives the proposer's rolling acceptance below threshold, it
    /// latches `stalled`, and the engine hands off mid-stream to the plain
    /// pipeline loop. The bridge forward (advance cache, sample next, don't
    /// re-emit) must leave NO gap or duplicate at the seam, so the n-gram output
    /// stays prefix-consistent with standard decode across the handoff boundary.
    func testNgramHandoffMatchesStandardDecode() async throws {
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()

        // Non-echo prose: the model paraphrases, so the lookup rarely matches and
        // the monitor (window 48) trips well within this length, forcing the
        // handoff partway through the generation.
        let prompt = "Write a thorough original explanation of how photosynthesis "
            + "works, then describe three distinct real-world consequences if it "
            + "stopped, using fresh wording throughout."

        let ref = await collect(engine, prompt: prompt, ngram: false, maxTokens: 220)
        let ng = await collect(engine, prompt: prompt, ngram: true, maxTokens: 220)

        var lcp = 0
        while lcp < ref.tokens.count && lcp < ng.tokens.count
            && ref.tokens[lcp] == ng.tokens[lcp] { lcp += 1 }
        XCTAssertEqual(lcp, min(ref.tokens.count, ng.tokens.count),
            "handoff output must match standard decode up to the shorter length "
            + "(LCP=\(lcp), ref=\(ref.tokens.count), ngram=\(ng.tokens.count)) — "
            + "a seam gap/duplicate would break this before the shorter length")

        // Sanity: the generation was long enough to exercise the monitor window.
        XCTAssertGreaterThan(ng.tokens.count, 48,
            "test prompt must generate past the monitor window to exercise handoff")
    }

    /// Wall-clock observation (gated): decode tok/s standard vs n-gram on a
    /// highly repetitive prompt. Prints only; proves the single-stream win.
    func testNgramDecodeSpeedupObservation() async throws {
        guard ProcessInfo.processInfo.environment["KLM_NGRAM_PERF"] != nil else {
            throw XCTSkip("Set KLM_NGRAM_PERF to run the n-gram decode-speedup observation")
        }
        let dir = try requireModelDirectory()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()

        func run(_ prompt: String, ngram: Bool, maxTokens: Int) async
            -> (tps: Double, accept: Double, toks: Int, decode: Double, gen: Int, rounds: Int)
        {
            let r = await collect(engine, prompt: prompt, ngram: ngram, maxTokens: maxTokens)
            let st = r.stats
            let tps = (st?.decodeTime ?? 0) > 0 ? Double(st!.generatedTokens) / st!.decodeTime : 0
            return (tps, st?.speculative?.acceptanceRate ?? 0, r.tokens.count,
                    st?.decodeTime ?? -1, st?.generatedTokens ?? -1, st?.speculative?.rounds ?? -1)
        }

        // Two regimes:
        //  - non-echo (the model paraphrases): expect ~1.0x (floor),
        //  - echo-heavy (the model reproduces a verbatim block): expect > 1x.
        let nonEcho = """
        Explain in your own words why the sky appears blue during the day and \
        red at sunset. Be thorough.
        """
        let echo = """
        Repeat the following text exactly, three times in a row:

        The quick brown fox jumps over the lazy dog while the sleepy cat watches \
        from the warm windowsill and dreams of summer fields.
        """

        for (label, prompt, maxTok) in [("non-echo", nonEcho, 160), ("echo", echo, 160)] {
            // Token sequences for divergence diagnosis.
            let baseTok = await collect(engine, prompt: prompt, ngram: false, maxTokens: maxTok)
            let ngTok = await collect(engine, prompt: prompt, ngram: true, maxTokens: maxTok)
            var lcp = 0
            while lcp < baseTok.tokens.count && lcp < ngTok.tokens.count
                && baseTok.tokens[lcp] == ngTok.tokens[lcp] { lcp += 1 }

            _ = await run(prompt, ngram: false, maxTokens: maxTok)   // warm
            let base = await run(prompt, ngram: false, maxTokens: maxTok)
            let ng = await run(prompt, ngram: true, maxTokens: maxTok)
            print(String(format:
                "[ngram-perf:%@] standard %.1f tok/s (%.3fs gen=%d)  |  n-gram %.1f tok/s (%.3fs gen=%d rounds=%d accept=%.2f)  |  %.2fx | toks %d/%d | LCP=%d/%d",
                label, base.tps, base.decode, base.gen,
                ng.tps, ng.decode, ng.gen, ng.rounds, ng.accept,
                base.tps > 0 ? ng.tps / base.tps : 0, ng.toks, base.toks,
                lcp, min(baseTok.tokens.count, ngTok.tokens.count)))
        }
    }
}
