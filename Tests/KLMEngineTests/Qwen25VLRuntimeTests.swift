import XCTest
import MLX
import KLMCache
import KLMSampler
@testable import KLMCore
@testable import KLMEngine

/// WS5 native Qwen 2.5-VL runtime: decode-correctness tests.
///
/// The gold standard here is an equivalence test. An incremental
/// prefill + KV-cached decode loop MUST produce the same logits at
/// each generated position as a single full forward over the
/// concatenated `prompt + generated` sequence. For a 3D-mRoPE model
/// this is exactly where the decode-step positional offset matters:
/// after an image span the KV-cache length is NOT the next mRoPE
/// position, so a wrong (or missing) offset diverges from the full
/// forward. The test runs on a small synthetic fp32 model, so it
/// validates the *plumbing* (positions + masking + cache) without
/// needing a real checkpoint.
final class Qwen25VLRuntimeTests: XCTestCase {

    // MARK: - Synthetic model

    private func config() throws -> Qwen25VLConfig {
        let json: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 2,
            "vocab_size": 256,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1_000_000.0,
            "max_position_embeddings": 4096,
            "head_dim": 16,
            "image_token_id": 151_655,
            "video_token_id": 151_656,
            "vision_start_token_id": 151_652,
            "vision_end_token_id": 151_653,
            "rope_scaling": ["type": "mrope", "mrope_section": [4, 6, 6]],
            "vision_config": [
                "depth": 2, "hidden_size": 32, "intermediate_size": 128,
                "num_heads": 4, "patch_size": 14, "temporal_patch_size": 2,
                "in_chans": 3, "spatial_merge_size": 2,
                "fullatt_block_indexes": [1], "window_size": 56,
                "out_hidden_size": 64,
            ],
            "tie_word_embeddings": false,
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder().decode(Qwen25VLConfig.self, from: data)
    }

    /// A 56x84 image -> 4x6 patch grid -> 24 patches -> 2x3 merged
    /// grid -> 6 `<|image_pad|>` tokens. Deliberately NON-square so
    /// `gridH * gridW (6) != max(gridH, gridW) (3)`: the KV-cache
    /// length after this prompt does not equal the mRoPE frontier,
    /// which is precisely what the decode offset must correct.
    private func imagePixels(cfg: Qwen25VLConfig, fill: Float) -> MLXArray {
        let pixels = MLXArray.ones([56, 84, 3]).asType(.float32) * fill
        return Qwen25VLImagePreprocessor.toConv3DInput(
            Qwen25VLImagePreprocessor.normalize(pixels),
            patchSize: cfg.vision.patchSize,
            temporalPatchSize: cfg.vision.temporalPatchSize,
            spatialMergeSize: cfg.vision.spatialMergeSize)
    }

    // MARK: - Equivalence: incremental decode == full forward

    /// Run the incremental prefill+decode loop and a single full
    /// forward over `prompt + generated`, assert per-position logits
    /// match. `image` toggles a non-square image span in the prompt.
    private func assertDecodeEquivalence(image: Bool) throws {
        let cfg = try config()
        let model = Qwen25VLForConditionalGeneration(cfg)
        let imgPad = Int32(cfg.imageTokenId)

        // Prompt. With an image: 2 text + 6 image_pad + 2 text = 10.
        let prompt32: [Int32]
        let pixelValues: MLXArray?
        let grid: (Int, Int)?
        if image {
            prompt32 = [11, 12] + Array(repeating: imgPad, count: 6) + [13, 14]
            pixelValues = imagePixels(cfg: cfg, fill: 0.6)
            grid = (2, 3)
        } else {
            prompt32 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            pixelValues = nil
            grid = nil
        }
        let Lp = prompt32.count
        let T = 12  // > 9 so the KV cache crosses its compaction threshold

        // --- Incremental: prefill + decode, capturing the logits
        //     that select each generated token ---
        let caches = makeKVCaches(numLayers: cfg.numHiddenLayers)
        let promptArr = MLXArray(prompt32).reshaped(1, Lp)
        let prefill = model(
            promptArr, pixelValues: pixelValues, imageGridMerged: grid,
            caches: caches, mropePositionOffset: nil)
        let frontier = Int(Qwen25VLPositions.compute(
            tokenIds: prompt32, imageTokenId: cfg.imageTokenId,
            gridHMerged: grid?.0 ?? 0, gridWMerged: grid?.1 ?? 0).nextPos)

        func argmaxLast(_ logits: MLXArray) -> Int {
            let last = logits[0, logits.dim(1) - 1, 0...]
            return argMax(last).item(Int.self)
        }

        var stepLogits: [MLXArray] = [prefill[0, Lp - 1, 0...]]
        var generated: [Int32] = [Int32(argmaxLast(prefill))]
        for k in 0 ..< (T - 1) {
            let tokArr = MLXArray([generated[k]]).reshaped(1, 1)
            let logits = model(
                tokArr, pixelValues: nil, imageGridMerged: nil,
                caches: caches, mropePositionOffset: frontier + k)
            stepLogits.append(logits[0, 0, 0...])
            generated.append(Int32(argMax(logits[0, 0, 0...]).item(Int.self)))
        }

        // --- Reference: one full forward over prompt + generated ---
        // The full sequence omits the last generated token (it has
        // no logits to compare against).
        let fullTokens = prompt32 + Array(generated.prefix(T - 1))
        let full = model(
            MLXArray(fullTokens).reshaped(1, fullTokens.count),
            pixelValues: pixelValues, imageGridMerged: grid,
            caches: nil, mropePositionOffset: nil)
        eval(full)
        for s in stepLogits { eval(s) }

        // logits that selected generated[k] must match full forward
        // logits at index Lp-1+k (which predicts token Lp+k).
        for k in 0 ..< T {
            let inc = stepLogits[k]
            let ref = full[0, Lp - 1 + k, 0...]
            let diff = abs(inc - ref).max().item(Float.self)
            XCTAssertLessThan(diff, 2e-3,
                "Incremental decode logits at generated position \(k) "
                + "diverge from the full forward (max abs diff \(diff)). "
                + "image=\(image) - a wrong decode mRoPE offset is the "
                + "usual cause.")
        }
    }

    func testDecodeEquivalenceTextOnly() throws {
        try assertDecodeEquivalence(image: false)
    }

    func testDecodeEquivalenceImagePrompt() throws {
        // The load-bearing case: a non-square image span makes the
        // KV-cache length disagree with the mRoPE frontier.
        try assertDecodeEquivalence(image: true)
    }

    // MARK: - Driver

    func testRuntimeGreedyMatchesInlineLoop() throws {
        // Qwen25VLRuntime.generate must produce the same greedy
        // tokens as a hand-rolled inline loop using the same offset.
        let cfg = try config()
        let model = Qwen25VLForConditionalGeneration(cfg)
        let imgPad = Int32(cfg.imageTokenId)
        let prompt: [Int] = [11, 12] + Array(repeating: Int(imgPad), count: 6) + [13, 14]
        let pixels = imagePixels(cfg: cfg, fill: 0.4)

        let out = Qwen25VLRuntime.generate(
            model: model, promptTokens: prompt,
            pixelValues: pixels, imageGridMerged: (2, 3),
            maxTokens: 10, stopIds: [], params: .greedy)
        XCTAssertEqual(out.tokens.count, 10,
            "Greedy generation with no stop id must run to maxTokens")
        XCTAssertGreaterThan(out.prefillSeconds, 0)

        // Re-derive greedily inline and compare.
        let caches = makeKVCaches(numLayers: cfg.numHiddenLayers)
        let p32 = prompt.map { Int32($0) }
        let prefill = model(
            MLXArray(p32).reshaped(1, p32.count),
            pixelValues: pixels, imageGridMerged: (2, 3),
            caches: caches, mropePositionOffset: nil)
        let frontier = Int(Qwen25VLPositions.compute(
            tokenIds: p32, imageTokenId: cfg.imageTokenId,
            gridHMerged: 2, gridWMerged: 3).nextPos)
        var tok = argMax(prefill[0, p32.count - 1, 0...]).item(Int.self)
        var inline: [Int] = [tok]
        for k in 0 ..< 9 {
            let logits = model(
                MLXArray([Int32(tok)]).reshaped(1, 1),
                pixelValues: nil, imageGridMerged: nil,
                caches: caches, mropePositionOffset: frontier + k)
            tok = argMax(logits[0, 0, 0...]).item(Int.self)
            inline.append(tok)
        }
        XCTAssertEqual(out.tokens, inline,
            "Qwen25VLRuntime greedy output must match the inline "
            + "prefill+decode loop with the same mRoPE offset")
    }

    func testRuntimeStopsOnStopId() throws {
        let cfg = try config()
        let model = Qwen25VLForConditionalGeneration(cfg)
        // First greedy token, used as the stop id so generation
        // halts on step 1.
        let prompt: [Int] = [5, 6, 7, 8, 9, 10, 11, 12]
        let probe = Qwen25VLRuntime.generate(
            model: model, promptTokens: prompt,
            pixelValues: nil, imageGridMerged: nil,
            maxTokens: 1, stopIds: [], params: .greedy)
        let firstToken = try XCTUnwrap(probe.tokens.first)
        let out = Qwen25VLRuntime.generate(
            model: model, promptTokens: prompt,
            pixelValues: nil, imageGridMerged: nil,
            maxTokens: 50, stopIds: [firstToken], params: .greedy)
        XCTAssertEqual(out.tokens, [firstToken],
            "Generation must stop as soon as a stop id is produced")
    }
}
