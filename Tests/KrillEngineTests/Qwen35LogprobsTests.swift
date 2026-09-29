import XCTest
import Foundation
import KrillSampler
@testable import KrillEngine

/// Regression test for docs/LOGPROBS_QWEN35_EMPTY.md: `logprobs.content` was
/// always `[]` for every qwen3_5-family model (qwen3.5-4b, Ornith-9B,
/// Qwythos-9B, Qwen3.8-27B) because ALL of them - including text-only usage
/// of qwen3.5-4b - load through `ArchitectureDetection`'s `hasVision` branch
/// into `Qwen35VLForConditionalGeneration` (their config.json carries
/// `vision_config`/`image_token_id` even for a purely-text checkpoint) and
/// `InferenceEngine.generate(messages:)` routes EVERY such request through
/// the dedicated `generateQwen35VL` / `Qwen35VLRuntime.generate` native
/// runtime - a completely separate decode loop from the generic dense path
/// that Phase 1 logprobs (docs/LOGPROBS_PLAN.md) was wired into. Neither
/// `wantLogprobs` nor `topLogprobs` was threaded into that runtime before
/// this fix, and its `onToken` callback never carried a `TokenLogprobInfo`,
/// so `TokenEvent.logprob` was `nil` for every token regardless of request
/// shape - this is exactly the bug this test would have caught (every
/// assertion below failed against the pre-fix code).
///
/// Gated on `KRILL_QWEN35_MODEL_PATH` (or the older Ornith-specific
/// `KRILL_ORNITH_MODEL_PATH`), same as `Qwen35VLSmokeTests`; skipped when
/// unset since it needs a real checkpoint to exercise the native runtime.
final class Qwen35LogprobsTests: XCTestCase {

    private func requireModel() throws -> URL {
        let env = ProcessInfo.processInfo.environment
        guard let path = [env["KRILL_QWEN35_MODEL_PATH"], env["KRILL_ORNITH_MODEL_PATH"]]
            .compactMap({ $0 }).first(where: { !$0.isEmpty }) else {
            throw XCTSkip("KRILL_QWEN35_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(
            atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KRILL_QWEN35_MODEL_PATH is not a directory: \(path)")
        }
        return URL(fileURLWithPath: path)
    }

    /// Every generated (non-`isEnd`) token must carry a `TokenLogprobInfo`
    /// with the requested number of top-N alternates, and - since this is a
    /// greedy (temperature 0), unmasked, penalty-free request - the sampled
    /// token must equal the raw distribution's own top-1 alternate exactly
    /// (docs/LOGPROBS_PLAN.md §4.1: greedy picks argmax of the same raw
    /// logits the reported logprob is computed from). This second check
    /// guards against a shallow fix that merely populates SOME non-nil
    /// value without it being numerically the right one.
    func testLogprobsPopulatedAndConsistentForHybridSSMModel() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()
        XCTAssertEqual(engine.family, "qwen3_5",
            "the checkpoint must load through the native qwen3_5 loader")

        let topN = 5
        let (stream, _) = engine.generate(
            messages: [["role": "user", "content": "Say hello in one short sentence."]],
            params: .greedy, maxTokens: 8, usePrefixCache: false,
            wantLogprobs: true, topLogprobs: topN)

        var sawContentToken = false
        for await event in stream {
            if event.isEnd { break }
            sawContentToken = true
            guard let info = event.logprob else {
                XCTFail("token \(event.tokenId) (text: \(event.text.debugDescription)) has no "
                    + "logprob info - this is the qwen3_5 empty-content regression")
                continue
            }
            XCTAssertEqual(info.topAlternates.count, topN,
                "expected \(topN) top alternates, got \(info.topAlternates.count)")
            XCTAssertFalse(info.logprob.isNaN, "logprob must not be NaN")
            XCTAssertLessThanOrEqual(info.logprob, 0.0001,
                "a log-probability must be <= 0 (allowing tiny float slack)")
            // Greedy + no mask/penalties: the sampled token IS the raw
            // distribution's argmax, so it must be alternate #0 with an
            // identical logprob.
            guard let top0 = info.topAlternates.first else {
                XCTFail("topAlternates unexpectedly empty")
                continue
            }
            XCTAssertEqual(top0.tokenId, event.tokenId,
                "greedy-sampled token must be the raw top-1 alternate")
            XCTAssertEqual(top0.logprob, info.logprob, accuracy: 1e-5,
                "the sampled token's own logprob must match its own top-N entry")
        }
        XCTAssertTrue(sawContentToken, "generation must produce at least one content token")
    }

    /// Same check with thinking explicitly disabled - the plan's other
    /// documented repro axis (`KRILL_ENABLE_THINKING=0`) - and with
    /// `top_logprobs` at 0, matching the "any top_logprobs" repro note in
    /// docs/LOGPROBS_QWEN35_EMPTY.md.
    func testLogprobsPopulatedWithThinkingDisabledAndZeroTopLogprobs() async throws {
        let dir = try requireModel()
        let engine = InferenceEngine(modelDirectory: dir)
        try await engine.load()

        let (stream, _) = engine.generate(
            messages: [["role": "user", "content": "Reply with just the word yes."]],
            params: .greedy, maxTokens: 4, usePrefixCache: false,
            enableThinking: false, wantLogprobs: true, topLogprobs: 0)

        var count = 0
        for await event in stream {
            if event.isEnd { break }
            count += 1
            XCTAssertNotNil(event.logprob, "token \(event.tokenId) missing logprob info")
            XCTAssertEqual(event.logprob?.topAlternates.count ?? -1, 0,
                "top_logprobs: 0 must report zero alternates")
        }
        XCTAssertGreaterThan(count, 0, "generation must produce at least one content token")
    }
}
