import XCTest
import MLX
import MLXNN
@testable import KrillEngine
@testable import KrillCache
@testable import KrillSampler
@testable import KrillCore
import KrillRuntime

/// Phase 2 (docs/LOGPROBS_PLAN.md §5.3, §7): proves `SpeculativeDecoder.step`
/// (draft-model spec) and `.ngramStep` (n-gram/prompt-lookup spec) derive
/// CORRECT raw logprobs for accepted tokens from the verify step's own
/// `targetLogits`, matching what the plain decode path would report at the
/// same (target model, position) — not just "populates a non-nil field".
///
/// The "target model" here is a tiny, fully deterministic, CONTEXT-FREE
/// synthetic `LoadedModel`: its forward pass ignores the KV caches entirely
/// and maps each input token id to a fixed logits row from a hardcoded
/// table (`nextTokenTable`). This is deliberately NOT a real transformer —
/// it exists only to give an independently-computable ground truth for
/// "what would log-softmax(targetLogits)[chosen] be at this token", so the
/// test can check `SpeculativeDecoder`'s reported logprob against a
/// from-scratch reference implementation (plain Swift, no `Sampler`
/// involved) rather than against Sampler's own math (which would just be
/// testing the code against itself).
final class SpeculativeLogprobsTests: XCTestCase {
    private func requireMLX() throws {
        #if os(macOS) && arch(arm64) && canImport(MLX)
        if ProcessInfo.processInfo.environment["KRILL_SKIP_MLX_TESTS"] == "1" {
            throw XCTSkip("MLX tests skipped by KRILL_SKIP_MLX_TESTS")
        }
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is not available to this test process")
        }
        #else
        throw XCTSkip("MLX-backed tests require macOS arm64 with MLX/Metal")
        #endif
    }

    private func withMLX<T>(_ body: () throws -> T) throws -> T {
        try requireMLX()
        return try Device.withDefaultDevice(.cpu) { try body() }
    }

    // MARK: - Synthetic deterministic "model"

    /// vocab = 6. Row t's logits are a spike at `(t+1) % vocab` (the "next"
    /// token under greedy decode) with smaller, DISTINCT values elsewhere so
    /// log-softmax and top-N ordering are well defined and stable.
    private static let vocab = 6
    private static func targetRow(_ t: Int) -> [Float] {
        var row = (0 ..< vocab).map { Float($0) * 0.1 }   // distinct small values
        row[(t + 1) % vocab] = 5.0                        // clear greedy winner
        return row
    }

    /// Draft "model": agrees with target everywhere EXCEPT token 2, where it
    /// proposes 0 instead of the target's greedy 3 — guarantees a rejection
    /// exercises the mismatch path whenever the draft run passes through 2.
    private static func draftRow(_ t: Int) -> [Float] {
        var row = (0 ..< vocab).map { Float($0) * 0.1 }
        let next = t == 2 ? 0 : (t + 1) % vocab
        row[next] = 5.0
        return row
    }

    /// A reference (independent of `Sampler`) raw log-softmax + top-N,
    /// computed in plain Swift `Double` arithmetic.
    private func referenceLogSoftmax(_ row: [Float]) -> [Double] {
        let d = row.map { Double($0) }
        let m = d.max()!
        let sumExp = d.reduce(0.0) { $0 + Foundation.exp($1 - m) }
        let logSumExp = m + Foundation.log(sumExp)
        return d.map { $0 - logSumExp }
    }

    private func makeLoadedModel(table: @escaping (Int) -> [Float]) -> LoadedModel {
        LoadedModel(
            module: Module(),
            numLayers: 1,
            family: "synthetic-test",
            forward: { tokens, _ in
                // tokens: [1, seqLen] int32. Output: [1, seqLen, vocab],
                // row i built purely from tokens[0, i] (context-free by
                // design — see class doc).
                let ids = tokens.asArray(Int32.self)
                let flat = ids.flatMap { table(Int($0)) }
                return MLXArray(flat, [1, ids.count, Self.vocab])
            },
            vocabSize: Self.vocab)
    }

    // MARK: - Draft-model spec path (`step`)

    func testStepAllAcceptedLogprobsMatchReferenceAndPlainPathDefinition() throws {
        try withMLX {
            let target = makeLoadedModel(table: Self.targetRow)
            // Draft agrees with target for every token in this run (starts
            // at 4, never touches the divergent token 2), so all K proposals
            // are accepted and a bonus token is produced.
            let draft = makeLoadedModel(table: Self.draftRow)
            let decoder = SpeculativeDecoder(
                targetModel: target, draftModel: draft, initialK: 3, temperature: 0)

            let targetCaches: [RestorableKVCache] = [KVCache()]
            let draftCaches: [KVCache] = [KVCache()]
            let topN = 3

            let (tokens, logprobs) = decoder.step(
                lastToken: 4, targetCaches: targetCaches, draftCaches: draftCaches,
                wantLogprobs: true, topLogprobs: topN)

            // Greedy target sequence from 4: 4->5->0->1 (+ bonus ->2).
            XCTAssertEqual(tokens, [5, 0, 1, 2])
            let logprobsArr = try XCTUnwrap(logprobs)
            XCTAssertEqual(logprobsArr.count, tokens.count)

            // Each position i's context token is `([4] + tokens.dropLast())[i]`
            // — the sequence the verify forward actually conditioned on.
            let contexts = [4] + tokens.dropLast()
            for (i, ctx) in contexts.enumerated() {
                let ref = referenceLogSoftmax(Self.targetRow(ctx))
                let chosen = tokens[i]
                XCTAssertEqual(
                    Double(logprobsArr[i].logprob), ref[chosen], accuracy: 1e-4,
                    "position \(i): chosen logprob must match raw log-softmax of the target's own logits")
                XCTAssertEqual(logprobsArr[i].topAlternates.count, topN)
                // Top-1 alternate is the sampled token itself (it IS the argmax here).
                XCTAssertEqual(logprobsArr[i].topAlternates.first?.tokenId, chosen)
                let sortedRef = ref.enumerated().sorted { $0.element > $1.element }
                for j in 0 ..< topN {
                    XCTAssertEqual(logprobsArr[i].topAlternates[j].tokenId, sortedRef[j].offset)
                    XCTAssertEqual(
                        Double(logprobsArr[i].topAlternates[j].logprob), sortedRef[j].element,
                        accuracy: 1e-4)
                }
            }
        }
    }

    func testStepRejectionLogprobsCoverOnlyAcceptedPrefixNoneForRejectedDraft() throws {
        try withMLX {
            let target = makeLoadedModel(table: Self.targetRow)
            let draft = makeLoadedModel(table: Self.draftRow)
            let decoder = SpeculativeDecoder(
                targetModel: target, draftModel: draft, initialK: 3, temperature: 0)

            let targetCaches: [RestorableKVCache] = [KVCache()]
            let draftCaches: [KVCache] = [KVCache()]

            // Start at 1: target greedy is 1->2->3->4; draft diverges at
            // token 2 (proposes 0 instead of 3) — draft proposes [2, 0, 1]
            // (from context 1, then drafts from ITS OWN proposal 2, i.e.
            // draftRow(2) -> 0, then draftRow(0) -> 1). Verify forwards
            // [1, 2, 0] and checks against target's greedy at each: target
            // says 2 (matches draft's 2 -> accept), then target says 3
            // (draft said 0 -> REJECT, target's 3 replaces it).
            let (tokens, logprobs) = decoder.step(
                lastToken: 1, targetCaches: targetCaches, draftCaches: draftCaches,
                wantLogprobs: true, topLogprobs: 0)

            XCTAssertEqual(tokens, [2, 3])   // 1 accepted draft + 1 target replacement, no bonus
            let logprobsArr = try XCTUnwrap(logprobs)
            XCTAssertEqual(logprobsArr.count, 2)
            let ref0 = referenceLogSoftmax(Self.targetRow(1))[2]
            let ref1 = referenceLogSoftmax(Self.targetRow(2))[3]
            XCTAssertEqual(Double(logprobsArr[0].logprob), ref0, accuracy: 1e-4)
            XCTAssertEqual(Double(logprobsArr[1].logprob), ref1, accuracy: 1e-4)
            // topLogprobs: 0 -> no alternates computed, matching the plain path's convention.
            XCTAssertEqual(logprobsArr[0].topAlternates.count, 0)
        }
    }

    func testStepWantLogprobsFalseReturnsNilLogprobsArray() throws {
        try withMLX {
            let target = makeLoadedModel(table: Self.targetRow)
            let draft = makeLoadedModel(table: Self.targetRow)   // fully agreeing
            let decoder = SpeculativeDecoder(
                targetModel: target, draftModel: draft, initialK: 2, temperature: 0)
            let (tokens, logprobs) = decoder.step(
                lastToken: 0, targetCaches: [KVCache()], draftCaches: [KVCache()])
            XCTAssertFalse(tokens.isEmpty)
            XCTAssertNil(logprobs, "default wantLogprobs:false must not compute or return logprobs")
        }
    }

    // MARK: - N-gram spec path (`ngramStep`)

    func testNgramStepNoMatchLogprobMatchesReference() throws {
        try withMLX {
            let target = makeLoadedModel(table: Self.targetRow)
            let decoder = SpeculativeDecoder.ngram(targetModel: target, temperature: 0)
            let proposer = NgramProposer(config: .init(), eosIds: [])
            proposer.reset(prompt: [9, 9, 9])   // no repeated n-gram to match -> k == 0

            let (tokens, logprobs) = decoder.ngramStep(
                lastToken: 3, targetCaches: [KVCache()], proposer: proposer,
                wantLogprobs: true, topLogprobs: 2)

            XCTAssertEqual(tokens, [4])   // greedy(3) == 4
            let logprobsArr = try XCTUnwrap(logprobs)
            XCTAssertEqual(logprobsArr.count, 1)
            let ref = referenceLogSoftmax(Self.targetRow(3))[4]
            XCTAssertEqual(Double(logprobsArr[0].logprob), ref, accuracy: 1e-4)
            XCTAssertEqual(logprobsArr[0].topAlternates.count, 2)
        }
    }

    func testNgramStepAcceptedRunLogprobsMatchReferencePerPosition() throws {
        try withMLX {
            let target = makeLoadedModel(table: Self.targetRow)
            let decoder = SpeculativeDecoder.ngram(targetModel: target, temperature: 0)
            // Seed history so the proposer's lookup matches the upcoming
            // greedy run exactly (an "echo" - prompt-lookup's ideal case):
            // history ends in ...0,1 and earlier the same context 0,1
            // was followed by 2,3 - so proposing after 0,1 again proposes [2,3].
            let proposer = NgramProposer(config: .init(), eosIds: [])
            proposer.reset(prompt: [0, 1, 2, 3, 4, 5, 0, 1])

            let (tokens, logprobs) = decoder.ngramStep(
                lastToken: 1, targetCaches: [KVCache()], proposer: proposer,
                wantLogprobs: true, topLogprobs: 1)

            // Greedy target from 1: 1->2->3->4 (+ bonus if all accepted).
            XCTAssertFalse(tokens.isEmpty)
            let logprobsArr = try XCTUnwrap(logprobs)
            XCTAssertEqual(logprobsArr.count, tokens.count)
            let contexts = [1] + tokens.dropLast()
            for (i, ctx) in contexts.enumerated() {
                let ref = referenceLogSoftmax(Self.targetRow(ctx))[tokens[i]]
                XCTAssertEqual(Double(logprobsArr[i].logprob), ref, accuracy: 1e-4)
            }
        }
    }
}
