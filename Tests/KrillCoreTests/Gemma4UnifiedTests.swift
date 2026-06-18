import XCTest
import MLX
import MLXNN
@testable import KrillCore

/// Tests for the Gemma 4 12B "unified" (encoder-free) multimodal model.
///
/// These gate the NEW surface relative to the existing Gemma 4 family:
///   * config parsing for `gemma4_unified` (nested text_config + the
///     encoder-free vision/audio configs),
///   * the encoder-free `Gemma4UnifiedVisionEmbedder` forward math against
///     an independent step-by-step recomputation,
///   * the `unpackImage` seam round-trip,
///   * the audio projector (raw-frame -> text-hidden) shape.
///
/// The full 12B checkpoint run is verified separately (it is an ~11 GB
/// download); here the math is gated on small deterministic fixtures so the
/// suite stays CI-runnable without the weights.
final class Gemma4UnifiedTests: XCTestCase {

    // MARK: - Config parsing

    func testUnifiedVisionConfigParsesReleasedFields() {
        let dict: [String: Any] = [
            "model_type": "gemma4_unified_vision",
            "patch_size": 16, "pooling_kernel_size": 3, "model_patch_size": 48,
            "mm_embed_dim": 3840, "mm_posemb_size": 1120, "num_soft_tokens": 280,
            "output_proj_dims": 3840, "rms_norm_eps": 1e-6,
        ]
        let vc = Gemma4UnifiedVisionConfig(from: dict)
        XCTAssertEqual(vc.modelPatchSize, 48)
        XCTAssertEqual(vc.mmEmbedDim, 3840)
        XCTAssertEqual(vc.mmPosembSize, 1120)
        XCTAssertEqual(vc.numSoftTokens, 280)
        XCTAssertEqual(vc.outputProjDims, 3840)
        // patch_dim = model_patch_size^2 * 3 = 48*48*3 = 6912
        XCTAssertEqual(vc.patchDim, 6912)
    }

    func testUnifiedVisionConfigDerivesModelPatchSize() {
        // When model_patch_size is absent, derive patch_size * pooling_kernel.
        let vc = Gemma4UnifiedVisionConfig(from: ["patch_size": 16, "pooling_kernel_size": 3])
        XCTAssertEqual(vc.modelPatchSize, 48)
        XCTAssertEqual(vc.patchDim, 6912)
    }

    func testUnifiedAudioConfigParsesRawFrameWidth() {
        let vc = Gemma4UnifiedAudioConfig(from: [
            "model_type": "gemma4_unified_audio",
            "audio_samples_per_token": 640, "audio_embed_dim": 640,
            "output_proj_dims": 640, "rms_norm_eps": 1e-6,
        ])
        XCTAssertEqual(vc.audioSamplesPerToken, 640)
        XCTAssertEqual(vc.outputProjDims, 640)
    }

    /// The text backbone parses from the nested `text_config`, including the
    /// 12B-specific shape and the global K-eq-V flags. Crucially: K-eq-V is
    /// full-attention-only, and the 12B disables PLE / KV-sharing / MoE.
    func testUnifiedTextConfigParsesFromNestedTextConfig() throws {
        let json = """
        {
          "architectures": ["Gemma4UnifiedForConditionalGeneration"],
          "model_type": "gemma4_unified",
          "image_token_id": 258880, "audio_token_id": 258881,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 3840, "num_hidden_layers": 48,
            "intermediate_size": 15360, "num_attention_heads": 16,
            "head_dim": 256, "global_head_dim": 512,
            "num_key_value_heads": 8, "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
            "sliding_window": 1024, "attention_k_eq_v": true,
            "use_double_wide_mlp": false, "final_logit_softcapping": 30.0,
            "vocab_size": 262144, "enable_moe_block": false,
            "layer_types": ["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]
          }
        }
        """
        let config = try JSONDecoder().decode(Gemma4Config.self, from: Data(json.utf8))
        XCTAssertEqual(config.hiddenSize, 3840)
        XCTAssertEqual(config.numHiddenLayers, 48)
        XCTAssertEqual(config.numAttentionHeads, 16)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.headDim, 256)
        XCTAssertEqual(config.globalHeadDim, 512)
        XCTAssertEqual(config.slidingWindow, 1024)
        XCTAssertEqual(config.hiddenSizePerLayerInput, 0)
        XCTAssertFalse(config.hasPerLayerInputs, "12B has no PLE")
        XCTAssertEqual(config.numKVSharedLayers, 0)
        XCTAssertFalse(config.enableMoeBlock, "12B is dense, no MoE")
        XCTAssertEqual(config.numGlobalKeyValueHeads, 1)
        XCTAssertEqual(config.finalLogitSoftcapping, 30.0, accuracy: 1e-6)

        // K-eq-V applies to FULL-attention layers only (layer 5 is full per
        // the layer_types above; layers 0-4 are sliding). This is exactly
        // the checkpoint shape: sliding layers ship v_proj, full layers do not.
        XCTAssertTrue(config.isFullAttention(layerIdx: 5))
        XCTAssertFalse(config.isFullAttention(layerIdx: 0))
        XCTAssertTrue(config.useKEqV(layerIdx: 5), "full layer reuses K as V")
        XCTAssertFalse(config.useKEqV(layerIdx: 0), "sliding layer keeps v_proj")
        // Full layers use num_global_key_value_heads (1); sliding use 8.
        XCTAssertEqual(config.kvHeads(layerIdx: 5), 1)
        XCTAssertEqual(config.kvHeads(layerIdx: 0), 8)
    }

    // MARK: - Vision embedder forward (independent recomputation)

    /// Build a tiny `Gemma4UnifiedVisionEmbedder`, set deterministic weights,
    /// and check its output equals an independent step-by-step recomputation
    /// (manual LayerNorm via mean/var, manual matmul, manual pos lookup). This
    /// catches axis / bias / norm-order transcription bugs without needing the
    /// real checkpoint or mlx-vlm.
    func testVisionEmbedderMatchesManualRecompute() throws {
        let patchDim = 12      // tiny stand-in for 6912
        let embedDim = 8       // tiny stand-in for 3840
        let posSize = 5
        let cfg = Gemma4UnifiedVisionConfig(
            patchSize: 2, poolingKernelSize: 2, modelPatchSize: 4,  // patchDim override below
            mmEmbedDim: embedDim, mmPosembSize: posSize, numSoftTokens: 4,
            outputProjDims: embedDim, rmsNormEps: 1e-6)
        // modelPatchSize^2*3 would be 48, but we drive patchDim directly via
        // the embedder init using a config whose patchDim we control: build a
        // config with modelPatchSize=2 so patchDim = 2*2*3 = 12.
        let cfg2 = Gemma4UnifiedVisionConfig(
            patchSize: 1, poolingKernelSize: 1, modelPatchSize: 2,
            mmEmbedDim: embedDim, mmPosembSize: posSize, numSoftTokens: 4,
            outputProjDims: embedDim, rmsNormEps: 1e-6)
        XCTAssertEqual(cfg2.patchDim, patchDim)
        _ = cfg

        let embedder = Gemma4UnifiedVisionEmbedder(cfg2)

        // Deterministic weights.
        MLXRandom.seed(0)
        let ln1w = MLXRandom.normal([patchDim]) * 0.1 + 1.0
        let ln1b = MLXRandom.normal([patchDim]) * 0.1
        let denseW = MLXRandom.normal([embedDim, patchDim]) * 0.1   // [out, in]
        let denseB = MLXRandom.normal([embedDim]) * 0.1
        let ln2w = MLXRandom.normal([embedDim]) * 0.1 + 1.0
        let ln2b = MLXRandom.normal([embedDim]) * 0.1
        let posEmb = MLXRandom.normal([posSize, 2, embedDim]) * 0.1
        let pnw = MLXRandom.normal([embedDim]) * 0.1 + 1.0
        let pnb = MLXRandom.normal([embedDim]) * 0.1

        let params: [(String, MLXArray)] = [
            ("patch_ln1.weight", ln1w), ("patch_ln1.bias", ln1b),
            ("patch_dense.weight", denseW), ("patch_dense.bias", denseB),
            ("patch_ln2.weight", ln2w), ("patch_ln2.bias", ln2b),
            ("pos_embedding", posEmb),
            ("pos_norm.weight", pnw), ("pos_norm.bias", pnb),
        ]
        try embedder.update(parameters: ModuleParameters.unflattened(params), verify: [])
        eval(embedder)

        // Input: 3 patches, one of which is padded (position -1).
        let x = MLXRandom.normal([1, 3, patchDim])
        let positions = MLXArray([Int32(0), 0, 1, 1, -1, -1]).reshaped(1, 3, 2)

        let got = embedder(x, positionIds: positions)

        // Independent recompute.
        func manualLayerNorm(_ v: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
            let mean = MLX.mean(v, axis: -1, keepDims: true)
            let centered = v - mean
            let variance = MLX.mean(centered * centered, axis: -1, keepDims: true)
            let normed = centered / MLX.sqrt(variance + MLXArray(Float(1e-6)))
            return normed * w + b
        }
        var h = manualLayerNorm(x, ln1w, ln1b)
        // dense: x @ W^T + b
        h = MLX.matmul(h, denseW.transposed(1, 0)) + denseB
        h = manualLayerNorm(h, ln2w, ln2b)
        // pos: per patch, add posEmb[x,0]+posEmb[y,1], zeroed for -1.
        var posAdd = MLXArray.zeros([1, 3, embedDim])
        let posArr: [[Int]] = [[0, 0], [1, 1], [-1, -1]]
        var rows: [MLXArray] = []
        for (i, p) in posArr.enumerated() {
            _ = i
            if p[0] == -1 && p[1] == -1 {
                rows.append(MLXArray.zeros([embedDim]))
            } else {
                rows.append(posEmb[p[0], 0] + posEmb[p[1], 1])
            }
        }
        posAdd = concatenated(rows.map { $0.reshaped(1, 1, embedDim) }, axis: 1)
        h = h + posAdd
        let expected = manualLayerNorm(h, pnw, pnb)

        let diff = MLX.max(MLX.abs(got - expected)).item(Float.self)
        XCTAssertLessThan(diff, 2e-4, "vision embedder must match manual recompute, diff=\(diff)")
    }

    /// A patch at a padded (-1) position must receive ZERO position
    /// contribution: its output equals what it would be with positions=nil.
    func testVisionEmbedderZeroesPaddedPositions() throws {
        let cfg = Gemma4UnifiedVisionConfig(
            patchSize: 1, poolingKernelSize: 1, modelPatchSize: 2,
            mmEmbedDim: 8, mmPosembSize: 5, numSoftTokens: 4,
            outputProjDims: 8, rmsNormEps: 1e-6)
        let embedder = Gemma4UnifiedVisionEmbedder(cfg)
        MLXRandom.seed(1)
        try embedder.update(parameters: ModuleParameters.unflattened([
            ("pos_embedding", MLXRandom.normal([5, 2, 8]))]), verify: [])
        eval(embedder)

        let x = MLXRandom.normal([1, 1, cfg.patchDim])
        let padded = MLXArray([Int32(-1), -1]).reshaped(1, 1, 2)
        let withPad = embedder(x, positionIds: padded)
        let noPos = embedder(x, positionIds: nil)
        let diff = MLX.max(MLX.abs(withPad - noPos)).item(Float.self)
        XCTAssertLessThan(diff, 1e-5, "padded position must add zero, diff=\(diff)")
    }

    // MARK: - unpackImage seam

    func testUnpackImageRoundTrip() {
        let patchDim = 12
        let n = 3
        let pixels = MLXRandom.normal([1, n, patchDim])
        let positions = MLXArray((0 ..< (n * 2)).map { Float($0 % 4) }).reshaped(1, n, 2)
        let packed = concatenated([pixels, positions], axis: -1)
        XCTAssertEqual(packed.dim(-1), patchDim + 2)

        let (gotPixels, gotPos) = Gemma4UnifiedModel.unpackImage(packed, patchDim: patchDim)
        XCTAssertNotNil(gotPixels)
        XCTAssertNotNil(gotPos)
        XCTAssertEqual(gotPixels!.shape, [1, n, patchDim])
        XCTAssertEqual(gotPos!.shape, [1, n, 2])
        XCTAssertEqual(gotPos!.dtype, .int32)

        let pixDiff = MLX.max(MLX.abs(gotPixels! - pixels)).item(Float.self)
        XCTAssertLessThan(pixDiff, 1e-5)
        let posDiff = MLX.max(MLX.abs(gotPos!.asType(.float32) - positions)).item(Float.self)
        XCTAssertLessThan(posDiff, 1e-4)
    }

    func testUnpackImageNilAndAlreadyUnpacked() {
        let (p1, q1) = Gemma4UnifiedModel.unpackImage(nil, patchDim: 12)
        XCTAssertNil(p1); XCTAssertNil(q1)

        // A tensor already at patchDim width (no positions) passes through.
        let bare = MLXRandom.normal([1, 2, 12])
        let (p2, q2) = Gemma4UnifiedModel.unpackImage(bare, patchDim: 12)
        XCTAssertNotNil(p2)
        XCTAssertNil(q2)
        XCTAssertEqual(p2!.shape, [1, 2, 12])
    }

    // MARK: - Audio framing + projector shape

    func testAudioFramingShapeAndTokenCount() {
        // 1601 samples at 640/token -> 3 frames (ceil), zero-padded.
        let wave = [Float](repeating: 0.1, count: 1601)
        let frames = preprocessGemma4UnifiedAudio(wave, samplesPerToken: 640)
        XCTAssertEqual(frames.shape, [1, 3, 640])
        XCTAssertEqual(gemma4UnifiedAudioTokenCount(sampleCount: 1601, samplesPerToken: 640), 3)
        // Exact multiple -> no extra frame.
        XCTAssertEqual(gemma4UnifiedAudioTokenCount(sampleCount: 1280, samplesPerToken: 640), 2)
        // Empty waveform must not trap: yields a single zero frame.
        let empty = preprocessGemma4UnifiedAudio([], samplesPerToken: 640)
        XCTAssertEqual(empty.shape, [1, 1, 640])
    }

    func testAudioProjectorProjectsRawFramesToHidden() {
        // embed_audio is a MultimodalEmbedder: RMSNormNoScale -> Linear.
        // Input is raw 640-sample frames; output is text hidden width.
        let frameWidth = 640
        let hidden = 3840
        let embedder = MultimodalEmbedder(
            embeddingDim: frameWidth, textHiddenSize: hidden, eps: 1e-6)
        eval(embedder)
        let frames = MLXRandom.normal([1, 5, frameWidth])
        let out = embedder(frames)
        XCTAssertEqual(out.shape, [1, 5, hidden])
    }
}
