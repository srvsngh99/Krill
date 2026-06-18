import XCTest
import MLX
@testable import KrillCore
import KrillCache

/// Image-serving correctness for the native Llama-3.2-Vision (mllama) runtime:
/// the full multi-image cross-attention-mask path and the cross-KV decode cache.
/// Gated on `KRILL_MLLAMA_PARITY_DIR` (see `tools/verify_mllama_parity.py`).
///
/// `MllamaParityTests` already covers the single-image and text-only forwards;
/// these add (1) a TWO-image forward with a sparse `cross_attention_mask` built
/// by `MllamaProcessing`, checked against mlx-vlm, and (2) a decode
/// self-consistency check that proves reusing the cached vision K/V across
/// decode steps equals re-prefilling the whole sequence every step.
final class MllamaImageServingTests: XCTestCase {

    private struct Reference: Decodable {
        let tokens: [Int]
        let vocab_size: Int
        let image_token_id: Int
        let num_tiles: [Int]
        let last_token_logits: [Float]
        let argmax: Int
    }

    private func parityDir() throws -> URL {
        guard let dirPath = ProcessInfo.processInfo.environment["KRILL_MLLAMA_PARITY_DIR"] else {
            throw XCTSkip("Set KRILL_MLLAMA_PARITY_DIR (see tools/verify_mllama_parity.py)")
        }
        return URL(fileURLWithPath: dirPath)
    }

    private func loadVisionModel(_ dir: URL) throws -> Llama32VisionForCausalLM {
        let loaded = try loadModel(from: dir)
        guard let model = loaded.module as? Llama32VisionForCausalLM else {
            throw XCTSkip("expected Llama32VisionForCausalLM, got \(type(of: loaded.module))")
        }
        return model
    }

    /// Two images + a sparse cross-attention mask: the native forward (mask built
    /// by `MllamaProcessing`) must match mlx-vlm's, which prepares the same mask
    /// from the dense int form internally.
    func testMultiImageCrossMaskMatchesMLXVLMLogits() throws {
        let dir = try parityDir()
        let refData = try Data(
            contentsOf: dir.appendingPathComponent("reference_multiimage_logits.json"))
        let ref = try JSONDecoder().decode(Reference.self, from: refData)
        let model = try loadVisionModel(dir)

        let inputs = try MLX.loadArrays(
            url: dir.appendingPathComponent("inputs/multiimage_inputs.safetensors"))
        let pixelValues = inputs["pixel_values"]!
        let aspectRatioIds = inputs["aspect_ratio_ids"]!
        let aspectRatioMask = inputs["aspect_ratio_mask"]!
        let inputIds = MLXArray(ref.tokens.map { Int32($0) }).reshaped([1, ref.tokens.count])

        // Build the additive cross mask the same way the serving driver will.
        let length = ref.tokens.count
        let maxImages = ref.num_tiles.count
        let maxTiles = model.config.visionConfig.maxNumTiles
        let numVisionTokens = model.config.visionConfig.numPatches
        let tokenMask = MllamaProcessing.crossAttentionTokenMask(
            inputIds: ref.tokens, imageTokenId: ref.image_token_id)
        let dense = MllamaProcessing.denseCrossMask(
            tokenMask: tokenMask, numTiles: ref.num_tiles,
            maxImages: maxImages, maxTiles: maxTiles, length: length)
        let (crossMask, fullRowMask) = MllamaProcessing.prepareCrossMask(
            dense: dense, length: length, maxImages: maxImages,
            maxTiles: maxTiles, numVisionTokens: numVisionTokens)

        let logits = model(
            inputIds, pixelValues: pixelValues, aspectRatioIds: aspectRatioIds,
            aspectRatioMask: aspectRatioMask, crossMask: crossMask, fullRowMask: fullRowMask)
        let last = logits[0, length - 1, 0...]
        eval(last)
        let got = last.asArray(Float.self)
        XCTAssertEqual(got.count, ref.last_token_logits.count)

        var maxIdx = 0
        for i in 1 ..< got.count where got[i] > got[maxIdx] { maxIdx = i }
        XCTAssertEqual(maxIdx, ref.argmax,
            "multi-image native argmax \(maxIdx) != mlx-vlm argmax \(ref.argmax)")

        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< got.count {
            let a = Double(got[i]), b = Double(ref.last_token_logits[i])
            dot += a * b; na += a * a; nb += b * b
            maxAbs = max(maxAbs, abs(a - b))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999, "multi-image logits cosine \(cosine) too low")
        XCTAssertLessThan(maxAbs, 1e-2, "multi-image max abs logit diff \(maxAbs) too large")
    }

    /// Greedy-decode the single-image fixture two ways: (A) prefill once into a
    /// self-attention KV cache + a cross-KV cache, then step with the cached
    /// vision K/V; (B) re-run a full image prefill over the growing sequence each
    /// step. The two token streams must agree -- proving the cross-KV reuse is
    /// equivalent to recomputing from the vision tower every step.
    func testCrossKVDecodeMatchesFullRecompute() throws {
        struct SingleRef: Decodable { let tokens: [Int] }
        let dir = try parityDir()
        let refData = try Data(contentsOf: dir.appendingPathComponent("reference_logits.json"))
        let ref = try JSONDecoder().decode(SingleRef.self, from: refData)
        let model = try loadVisionModel(dir)

        let inputs = try MLX.loadArrays(
            url: dir.appendingPathComponent("inputs/vision_inputs.safetensors"))
        let pixelValues = inputs["pixel_values"]!
        let aspectRatioIds = inputs["aspect_ratio_ids"]!
        let aspectRatioMask = inputs["aspect_ratio_mask"]!
        let prompt = ref.tokens
        let steps = 6

        func argmaxLast(_ logits: MLXArray, row: Int) -> Int {
            let v = logits[0, row, 0...]
            eval(v)
            let arr = v.asArray(Float.self)
            var m = 0
            for i in 1 ..< arr.count where arr[i] > arr[m] { m = i }
            return m
        }

        // (A) cached decode.
        let numLayers = model.config.textConfig.numHiddenLayers
        var caches: [KVCache] = (0 ..< numLayers).map { _ in KVCache() }
        let crossKV = MllamaCrossKVCache()
        let prefillIds = MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count])
        var logits = model(
            prefillIds, pixelValues: pixelValues, aspectRatioIds: aspectRatioIds,
            aspectRatioMask: aspectRatioMask, caches: caches, crossKV: crossKV)
        var cachedTokens: [Int] = []
        var tok = argmaxLast(logits, row: prompt.count - 1)
        cachedTokens.append(tok)
        for _ in 1 ..< steps {
            let step = MLXArray([Int32(tok)]).reshaped([1, 1])
            logits = model(step, caches: caches, crossKV: crossKV)
            tok = argmaxLast(logits, row: 0)
            cachedTokens.append(tok)
        }

        // (B) iterative full recompute (fresh image prefill each step).
        var seq = prompt
        var fullTokens: [Int] = []
        for _ in 0 ..< steps {
            let ids = MLXArray(seq.map { Int32($0) }).reshaped([1, seq.count])
            let l = model(
                ids, pixelValues: pixelValues, aspectRatioIds: aspectRatioIds,
                aspectRatioMask: aspectRatioMask)
            let next = argmaxLast(l, row: seq.count - 1)
            fullTokens.append(next)
            seq.append(next)
        }

        XCTAssertEqual(cachedTokens, fullTokens,
            "cached cross-KV decode \(cachedTokens) != full-recompute \(fullTokens)")
    }
}
