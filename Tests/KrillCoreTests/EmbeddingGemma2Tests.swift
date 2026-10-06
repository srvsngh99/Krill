import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// EmbeddingGemma 2 text path. Everything here runs without the real
/// checkpoint: a tiny synthetic model exercises the forward, pooling, strict
/// weight binding and config decoding; the real-weights parity test lives in
/// `Tests/KrillEngineTests/EmbeddingGemma2ParityTests.swift` (skipped when the
/// weights are absent).
final class EmbeddingGemma2Tests: XCTestCase {

    // The real config.json's text_config (verbatim values; vision/audio omitted).
    private let realConfig = """
    {
      "architectures": ["EmbeddingGemma2Model"],
      "model_type": "embedding_gemma2",
      "text_config": {
        "embedding_dim": 768, "head_dim": 256, "hidden_size": 512,
        "hidden_size_per_layer_input": 512, "intermediate_size": 2048,
        "layer_types": [
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
          "sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],
        "num_attention_heads": 4, "num_hidden_layers": 24, "num_key_value_heads": 2,
        "per_layer_config": {
          "05": {"head_dim": 512, "num_key_value_heads": 1}, "11": {"head_dim": 512, "num_key_value_heads": 1},
          "17": {"head_dim": 512, "num_key_value_heads": 1}, "23": {"head_dim": 512, "num_key_value_heads": 1}},
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
          "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
          "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        "sliding_window": 512, "vocab_size": 262144
      }
    }
    """

    func testRealConfigDecoding() throws {
        let c = try JSONDecoder().decode(EmbeddingGemma2Config.self, from: Data(realConfig.utf8))
        XCTAssertEqual(c.numLayers, 24)
        XCTAssertEqual(c.hiddenSize, 512)
        XCTAssertEqual(c.embeddingDim, 768)
        XCTAssertEqual(c.slidingWindow, 512)
        XCTAssertEqual(c.ropeThetaFull, 1_000_000)
        XCTAssertEqual(c.ropeThetaSliding, 10_000)
        for l in 0 ..< 24 {
            let full = (l % 6 == 5)
            XCTAssertEqual(c.isFull(l), full, "layer \(l)")
            XCTAssertEqual(c.headDim(l), full ? 512 : 256)
            XCTAssertEqual(c.kvHeads(l), full ? 1 : 2)
        }
    }

    func testMRLDimensionsAndContext() {
        XCTAssertEqual(EmbeddingGemma2Config.mrlDimensions, [768, 512, 256, 128])
        XCTAssertEqual(EmbeddingGemma2Config.maxContext, 8192)
    }

    // MARK: - Pure math

    func testL2NormalizeAndZeroVector() {
        let v = EmbeddingMath.l2Normalize([3, 4])
        XCTAssertEqual(v[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(v[1], 0.8, accuracy: 1e-6)
        XCTAssertEqual(EmbeddingMath.l2Normalize([0, 0]), [0, 0])
    }

    func testMRLTruncationRenormalises() {
        let full = EmbeddingMath.l2Normalize([1, 2, 3, 4, 5, 6, 7, 8])
        let t = EmbeddingMath.truncate(full, to: 4)
        XCTAssertEqual(t.count, 4)
        XCTAssertEqual(t.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-5)
        // direction preserved: ratio of kept components unchanged
        XCTAssertEqual(t[1] / t[0], full[1] / full[0], accuracy: 1e-5)
        // the truncated unit vector is NOT just the prefix of the full vector
        XCTAssertNotEqual(t[0], full[0], accuracy: 1e-4)
    }

    func testMaskedMeanIgnoresPadding() {
        let states: [[Float]] = [[1, 2], [3, 4], [100, 100]]
        XCTAssertEqual(EmbeddingMath.maskedMean(states, length: 2), [2, 3])
        XCTAssertEqual(EmbeddingMath.maskedMean(states, length: 0), [])
    }

    // MARK: - Tiny synthetic model

    private func tinyConfig(layers: Int = 3, window: Int = 4) throws -> EmbeddingGemma2Config {
        let types = (0 ..< layers).map { $0 == layers - 1 ? "full_attention" : "sliding_attention" }
        let json = """
        {"text_config": {"embedding_dim": 12, "head_dim": 8, "hidden_size": 16,
          "hidden_size_per_layer_input": 8, "intermediate_size": 32,
          "layer_types": \(String(data: try JSONEncoder().encode(types), encoding: .utf8)!),
          "num_attention_heads": 2, "num_hidden_layers": \(layers), "num_key_value_heads": 1,
          "per_layer_config": {"\(layers - 1)": {"head_dim": 16, "num_key_value_heads": 1}},
          "sliding_window": \(window), "vocab_size": 50}}
        """
        return try JSONDecoder().decode(EmbeddingGemma2Config.self, from: Data(json.utf8))
    }

    private func randomised(_ m: EmbeddingGemma2Model, seed: UInt64 = 7) {
        MLXRandom.seed(seed)
        let p = m.parameters().flattened().map { (k, v) -> (String, MLXArray) in
            // keep layer_scalar / norm weights near 1 so activations stay sane
            if k.hasSuffix("layer_scalar") { return (k, MLXArray([Float(0.5)])) }
            return (k, MLXRandom.normal(v.shape) * 0.1)
        }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.setComputeDtype(.float32)
        eval(m)
    }

    func testTinyForwardShapeAndFiniteness() throws {
        let m = EmbeddingGemma2Model(try tinyConfig())
        randomised(m)
        let tokens = MLXArray([Int32](1 ... 6)).reshaped(1, 6)
        let h = m.lastHiddenState(tokens)
        XCTAssertEqual(h.shape, [1, 6, 12])
        let flat = h.asArray(Float.self)
        XCTAssertTrue(flat.allSatisfy { $0.isFinite })
        XCTAssertGreaterThan(flat.map { abs($0) }.max()!, 0)
    }

    func testAttentionIsBidirectional() throws {
        // A causal model would leave position 0 unchanged when a LATER token
        // changes. The EmbeddingGemma 2 encoder must not.
        let m = EmbeddingGemma2Model(try tinyConfig(window: 64))
        randomised(m)
        let a = MLXArray([Int32]([3, 4, 5, 6])).reshaped(1, 4)
        let b = MLXArray([Int32]([3, 4, 5, 9])).reshaped(1, 4)
        let ha = m.lastHiddenState(a)[0, 0, 0...].asArray(Float.self)
        let hb = m.lastHiddenState(b)[0, 0, 0...].asArray(Float.self)
        let diff = zip(ha, hb).map { abs($0 - $1) }.max()!
        XCTAssertGreaterThan(diff, 1e-4, "position 0 ignored a later token: attention is causal")
    }

    func testPaddedBatchMatchesSolo() throws {
        let m = EmbeddingGemma2Model(try tinyConfig(window: 3))
        randomised(m)
        let s1: [Int32] = [5, 6, 7, 8, 9, 10, 11, 12]  // longer than the window
        let s2: [Int32] = [20, 21, 22]
        func solo(_ s: [Int32]) -> [Float] {
            m.pooled(MLXArray(s).reshaped(1, s.count), lengths: [s.count]).asArray(Float.self)
        }
        let padded = MLXArray(s1 + s2 + [0, 0, 0, 0, 0]).reshaped(2, 8)
        let batch = m.pooled(padded, lengths: [8, 3]).asArray(Float.self)
        let a = solo(s1), b = solo(s2)
        for k in 0 ..< 12 {
            XCTAssertEqual(batch[k], a[k], accuracy: 1e-4, "row 0 dim \(k)")
            XCTAssertEqual(batch[12 + k], b[k], accuracy: 1e-4, "row 1 dim \(k) (padding leaked)")
        }
    }

    func testSlidingWindowLimitsReach() throws {
        // 2 sliding layers + final full layer is not isolable, so test the
        // mask directly through a 1-layer-sliding model by forcing layer_types
        // after decode is impossible; instead assert the band changes output:
        // same tokens, window 1 vs window 64 must differ once T > window + 1.
        let toks = MLXArray([Int32]([1, 2, 3, 4, 5, 6, 7, 8])).reshaped(1, 8)
        let narrow = EmbeddingGemma2Model(try tinyConfig(window: 1))
        randomised(narrow, seed: 11)
        let wide = EmbeddingGemma2Model(try tinyConfig(window: 64))
        randomised(wide, seed: 11)
        let a = narrow.lastHiddenState(toks).asArray(Float.self)
        let b = wide.lastHiddenState(toks).asArray(Float.self)
        XCTAssertGreaterThan(zip(a, b).map { abs($0 - $1) }.max()!, 1e-4)
    }

    func testComputeDtypeIsFloat32OrBFloat16Only() throws {
        let m = EmbeddingGemma2Model(try tinyConfig())
        m.setComputeDtype(.bfloat16)
        XCTAssertEqual(m.computeDtype, .bfloat16)
        XCTAssertTrue(m.parameters().flattened().allSatisfy { $0.1.dtype == .bfloat16 })
        m.setComputeDtype(.float32)
        XCTAssertTrue(m.parameters().flattened().allSatisfy { $0.1.dtype == .float32 })
    }

    // MARK: - Strict weight binding

    func testPartitionKeysCountsAndRejectsUnknown() throws {
        let keys = ["language_model.norm.weight", "language_model.layers.0.mlp.up_proj.weight",
                    "vision_tower.encoder.layers.0.input_layernorm.weight",
                    "embed_vision.embedding_projection.weight",
                    "audio_tower.output_proj.bias", "embed_audio.embedding_projection.weight"]
        let p = try partitionEmbeddingGemma2Keys(keys)
        XCTAssertEqual(p.text.count, 2)
        XCTAssertEqual(p.text["language_model.norm.weight"], "norm.weight")
        XCTAssertEqual(p.vision, 2)
        XCTAssertEqual(p.audio, 2)
        XCTAssertThrowsError(try partitionEmbeddingGemma2Keys(keys + ["mystery.weight"]))
    }

    private func writeTinyCheckpoint(
        dropping: String? = nil, adding extra: [String: MLXArray] = [:]
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("eg2-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let layers = 3
        let types = ["sliding_attention", "sliding_attention", "full_attention"]
        let cfg = """
        {"architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2",
         "text_config": {"embedding_dim": 12, "head_dim": 8, "hidden_size": 16,
          "hidden_size_per_layer_input": 8, "intermediate_size": 32,
          "layer_types": \(String(data: try JSONEncoder().encode(types), encoding: .utf8)!),
          "num_attention_heads": 2,
          "num_hidden_layers": \(layers), "num_key_value_heads": 1,
          "per_layer_config": {"2": {"head_dim": 16, "num_key_value_heads": 1}},
          "sliding_window": 4, "vocab_size": 50}}
        """
        try cfg.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        let m = EmbeddingGemma2Model(try JSONDecoder().decode(
            EmbeddingGemma2Config.self, from: Data(cfg.utf8)))
        var arrays: [String: MLXArray] = [:]
        for (k, v) in m.parameters().flattened() { arrays["language_model." + k] = v }
        if let dropping { arrays.removeValue(forKey: dropping) }
        arrays["vision_tower.patch_embedder.input_proj.weight"] = MLXArray.zeros([4, 4])
        arrays["audio_tower.output_proj.bias"] = MLXArray.zeros([4])
        arrays["embed_audio.embedding_projection.weight"] = MLXArray.zeros([4, 4])
        for (k, v) in extra { arrays[k] = v }
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    func testStrictLoaderBindsEveryTextKeyAndCountsSkips() throws {
        let dir = try writeTinyCheckpoint()
        defer { try? FileManager.default.removeItem(at: dir) }
        let (model, report) = try loadEmbeddingGemma2(directory: dir, dtype: .float32)
        XCTAssertEqual(report.boundText, model.parameters().flattened().count)
        XCTAssertEqual(report.skippedVision, 1)
        XCTAssertEqual(report.skippedAudio, 2)
    }

    func testStrictLoaderFailsOnMissingTextWeight() throws {
        let dir = try writeTinyCheckpoint(dropping: "language_model.layers.1.mlp.up_proj.weight")
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir),
                             "an unbound weight must fail the load, not load as random init")
    }

    func testStrictLoaderFailsOnUnexpectedTextWeight() throws {
        let dir = try writeTinyCheckpoint(
            adding: ["language_model.layers.0.self_attn.mystery.weight": MLXArray.zeros([2])])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testStrictLoaderFailsOnUnrecognisedTopLevelKey() throws {
        let dir = try writeTinyCheckpoint(adding: ["mystery_tower.weight": MLXArray.zeros([2])])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }
}
