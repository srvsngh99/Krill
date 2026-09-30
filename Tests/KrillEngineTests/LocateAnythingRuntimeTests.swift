import XCTest
import MLX
import KrillSampler
@testable import KrillCore
@testable import KrillEngine

/// Checkpoint-free tests for the native LocateAnything-3B decode driver
/// (`LocateAnythingRuntime`), on a tiny synthetic fp32 model - the same
/// "no real checkpoint needed" pattern `Qwen25VLRuntimeTests` uses for the
/// plumbing (config + weights are random; the point is exercising the real
/// Swift+MLX forward + `Sampler` code path, not model quality).
///
/// Task B (2026-09-30, docs/LOGPROBS_PLAN.md "Engine follow-ups"):
/// `LocateAnythingRuntime` never threaded `wantLogprobs`/`topLogprobs` at all
/// - its `onToken` callback only ever carried a bare token id. NVIDIA
/// LocateAnything-3B (7.8 GB) and its MLX re-releases were all over this
/// task's download budget (<=3 GB/download, <=6 GB total already spent on
/// the Qwen 2.5-VL real-model check), so this is a synthetic-model
/// verification, NOT a real-checkpoint one - see the PR description for what
/// that means for confidence here.
final class LocateAnythingRuntimeTests: XCTestCase {

    // MARK: - Synthetic model (text-only decode; LocateAnything's decode is
    // plain 1D RoPE - identical to a standard Qwen2.5 decoder - so a
    // text-only synthetic run exercises the exact code path `onToken` and
    // `Sampler.sampleWithLogprobs` run on, without needing to also drive the
    // MoonViT vision tower.)

    private func config() throws -> LocateAnythingConfig {
        let json: [String: Any] = [
            "image_token_index": 151_665,
            "text_config": [
                "model_type": "qwen2",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_hidden_layers": 2,
                "vocab_size": 256,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000.0,
                "max_position_embeddings": 4096,
            ],
            "vision_config": [String: Any](),
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(LocateAnythingConfig.self, from: data)
    }

    /// `LocateAnythingRuntime.generate` with `wantLogprobs: true` must: (a)
    /// populate `onToken`'s second argument with a `TokenLogprobInfo` whose
    /// `topAlternates.count` matches the requested N on every token, (b)
    /// report a top-1 alternate matching the greedy-sampled token (the raw
    /// distribution's own argmax), and (c) sample the IDENTICAL token
    /// sequence as the logprobs-OFF path - the reporting must never change
    /// what gets sampled.
    func testRuntimeLogprobsPopulatedAndOffPathTokensUnchanged() throws {
        let cfg = try config()
        let model = LocateAnythingForConditionalGeneration(cfg)
        let prompt: [Int] = [11, 12, 13, 14, 15, 16, 17, 18]

        let topN = 4
        var seenLogprobs: [TokenLogprobInfo?] = []
        let onOutput = LocateAnythingRuntime.generate(
            model: model, promptTokens: prompt, pixelValues: nil, grid: nil,
            maxTokens: 6, stopIds: [], params: .greedy,
            wantLogprobs: true, topLogprobs: topN,
            onToken: { _, info in seenLogprobs.append(info) })

        XCTAssertEqual(onOutput.tokens.count, 6)
        XCTAssertEqual(seenLogprobs.count, onOutput.tokens.count)
        for info in seenLogprobs {
            guard let info else { XCTFail("missing logprob info"); continue }
            XCTAssertEqual(info.topAlternates.count, topN)
            XCTAssertFalse(info.logprob.isNaN)
            XCTAssertLessThanOrEqual(info.logprob, 0.0001)
            guard let top0 = info.topAlternates.first else {
                XCTFail("topAlternates unexpectedly empty"); continue
            }
            XCTAssertEqual(top0.logprob, info.logprob, accuracy: 1e-5)
        }
        // Greedy: every reported top-1 alternate must equal the token that
        // was ACTUALLY sampled and yielded for that same step.
        for (tok, info) in zip(onOutput.tokens, seenLogprobs) {
            XCTAssertEqual(info?.topAlternates.first?.tokenId, tok,
                "greedy-sampled token must be the raw top-1 alternate")
        }

        var offSeen: [Int] = []
        let offOutput = LocateAnythingRuntime.generate(
            model: model, promptTokens: prompt, pixelValues: nil, grid: nil,
            maxTokens: 6, stopIds: [], params: .greedy,
            onToken: { tok, _ in offSeen.append(tok) })
        XCTAssertEqual(offOutput.tokens, onOutput.tokens,
            "requesting logprobs must never change which tokens are sampled")
        XCTAssertEqual(offSeen, onOutput.tokens)
    }

    /// `top_logprobs: 0` must still populate `logprob` with zero alternates.
    func testRuntimeZeroTopLogprobsStillPopulatesLogprob() throws {
        let cfg = try config()
        let model = LocateAnythingForConditionalGeneration(cfg)
        let prompt: [Int] = [21, 22, 23, 24]

        var seenLogprobs: [TokenLogprobInfo?] = []
        _ = LocateAnythingRuntime.generate(
            model: model, promptTokens: prompt, pixelValues: nil, grid: nil,
            maxTokens: 3, stopIds: [], params: .greedy,
            wantLogprobs: true, topLogprobs: 0,
            onToken: { _, info in seenLogprobs.append(info) })

        XCTAssertEqual(seenLogprobs.count, 3)
        for info in seenLogprobs {
            XCTAssertNotNil(info)
            XCTAssertEqual(info?.topAlternates.count, 0)
        }
    }
}
