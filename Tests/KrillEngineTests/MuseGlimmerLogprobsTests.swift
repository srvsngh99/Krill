import XCTest
import MLX
import KrillCache
import KrillSampler
@testable import KrillCore
@testable import KrillEngine

/// Checkpoint-free logprobs test for the native Muse Glimmer IMAGE driver
/// (`MuseGlimmerRuntime`), on a tiny synthetic fp32 model - same pattern as
/// `Tests/KrillCoreTests/MuseGlimmerNativeTests.swift` (whose config JSON
/// this borrows). The real Muse Glimmer is 30B and the smallest published
/// MLX build is 19.4 GB - far over this task's download budget (<=3 GB per
/// download) - so there is no real-checkpoint gate for this family anywhere
/// in the repo (see that file's header comment); this is the same
/// synthetic-model standard applied to the new logprobs plumbing.
///
/// Task B (2026-09-30, docs/LOGPROBS_PLAN.md "Engine follow-ups"): the Muse
/// Glimmer IMAGE path (`generateMuseGlimmer` / `MuseGlimmerRuntime`) never
/// threaded `wantLogprobs`/`topLogprobs` at all - its `onToken` callback only
/// ever carried a bare token id.
final class MuseGlimmerLogprobsTests: XCTestCase {

    // MARK: - Config (mirrors MuseGlimmerNativeTests' configJSON(withVision:))

    private func configJSON() -> Data {
        let layers = 4, hidden = 32, heads = 4, kvHeads = 2, headDim = 8, vocab = 64
        let types = (0 ..< layers).map { ($0 + 1) % 4 == 0 ? "full_attention" : "sliding_attention" }
        let thetas = (0 ..< layers).map { ($0 + 1) % 4 == 0 ? "0" : "500000.0" }
        let json = """
        {
          "architectures": ["MuseGlimmerForConditionalGeneration"],
          "model_type": "muse_glimmer",
          "image_token_id": 61, "video_token_id": 60,
          "out_hidden_size": 64, "projector_hidden_size": 24,
          "projector_hidden_act": "gelu",
          "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": \(hidden), "intermediate_size": \(hidden * 2),
            "num_hidden_layers": \(layers),
            "num_attention_heads": \(heads), "num_key_value_heads": \(kvHeads),
            "head_dim": \(headDim), "vocab_size": \(vocab),
            "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
            "sliding_window": 4,
            "layer_types": [\(types.map { "\"\($0)\"" }.joined(separator: ", "))],
            "layer_rope_theta": [\(thetas.joined(separator: ", "))],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.19611613513818404,
            "final_logit_softcapping": 20.0,
            "attention_bias": false, "tie_word_embeddings": false,
            "max_position_embeddings": 131072, "hidden_activation": "silu"
          },
          "vision_config": {
            "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
            "num_attention_heads": 2, "patch_size": 14, "patch_temporal": 2,
            "merge_size": 2, "pos_emb_height": 32, "pos_emb_width": 32,
            "layer_norm_eps": 1e-5, "hidden_act": "gelu",
            "rope_parameters": {"rope_theta": 10000.0},
            "layer_types": ["window_attention", "full_attention"]
          }
        }
        """
        return Data(json.utf8)
    }

    private func makeConfig() throws -> MuseGlimmerConfig {
        try JSONDecoder().decode(MuseGlimmerConfig.self, from: configJSON())
    }

    /// `MuseGlimmerRuntime.generate` on an IMAGE request with `wantLogprobs:
    /// true` must populate `onToken`'s second argument on every token, report
    /// a top-1 alternate matching the greedy-sampled token, and sample the
    /// IDENTICAL sequence as the logprobs-OFF path.
    func testImagePathLogprobsPopulatedAndOffPathTokensUnchanged() throws {
        let c = try makeConfig()
        let vision = try XCTUnwrap(c.visionConfig)
        let model = MuseGlimmerForConditionalGeneration(c)

        // 4x4 grid at merge 2 -> 4 `<|patch|>` placeholders (mirrors
        // MuseGlimmerNativeTests.testImageFeaturesProjectToTextHidden and
        // MuseGlimmerRuntime.placeholderCount).
        let grid = (t: 1, h: 4, w: 4)
        let placeholderCount = MuseGlimmerRuntime.placeholderCount(
            grid: grid, mergeSize: vision.mergeSize)
        XCTAssertEqual(placeholderCount, 4)
        let pixels = MLXArray.zeros([grid.h * grid.w, vision.patchDim])
        let imgTok = Int(c.imageTokenId)
        let prompt: [Int] = [1, 2] + Array(repeating: imgTok, count: placeholderCount) + [3, 4]

        let topN = 3
        var seenLogprobs: [TokenLogprobInfo?] = []
        let onOutput = MuseGlimmerRuntime.generate(
            model: model, promptTokens: prompt, pixelValues: pixels, grid: grid,
            maxTokens: 5, stopIds: [], params: .greedy,
            wantLogprobs: true, topLogprobs: topN,
            onToken: { _, info in seenLogprobs.append(info) })

        XCTAssertEqual(onOutput.tokens.count, 5)
        XCTAssertEqual(seenLogprobs.count, 5)
        for (tok, info) in zip(onOutput.tokens, seenLogprobs) {
            guard let info else { XCTFail("missing logprob info"); continue }
            XCTAssertEqual(info.topAlternates.count, topN)
            XCTAssertFalse(info.logprob.isNaN)
            XCTAssertLessThanOrEqual(info.logprob, 0.0001)
            XCTAssertEqual(info.topAlternates.first?.tokenId, tok,
                "greedy-sampled token must be the raw top-1 alternate")
            XCTAssertEqual(info.topAlternates.first?.logprob ?? .nan, info.logprob, accuracy: 1e-5)
        }

        var offSeen: [Int] = []
        let offOutput = MuseGlimmerRuntime.generate(
            model: model, promptTokens: prompt, pixelValues: pixels, grid: grid,
            maxTokens: 5, stopIds: [], params: .greedy,
            onToken: { tok, _ in offSeen.append(tok) })
        XCTAssertEqual(offOutput.tokens, onOutput.tokens,
            "requesting logprobs must never change which tokens are sampled")
        XCTAssertEqual(offSeen, onOutput.tokens)
    }

    /// `top_logprobs: 0` must still populate `logprob` with zero alternates
    /// on the image path.
    func testImagePathZeroTopLogprobsStillPopulatesLogprob() throws {
        let c = try makeConfig()
        let vision = try XCTUnwrap(c.visionConfig)
        let model = MuseGlimmerForConditionalGeneration(c)
        let grid = (t: 1, h: 4, w: 4)
        let placeholderCount = MuseGlimmerRuntime.placeholderCount(
            grid: grid, mergeSize: vision.mergeSize)
        let pixels = MLXArray.zeros([grid.h * grid.w, vision.patchDim])
        let imgTok = Int(c.imageTokenId)
        let prompt: [Int] = [1] + Array(repeating: imgTok, count: placeholderCount) + [2]

        var seenLogprobs: [TokenLogprobInfo?] = []
        _ = MuseGlimmerRuntime.generate(
            model: model, promptTokens: prompt, pixelValues: pixels, grid: grid,
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
