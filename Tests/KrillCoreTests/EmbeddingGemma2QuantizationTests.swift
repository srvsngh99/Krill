import XCTest
import Foundation
import MLX
import MLXNN
@testable import KrillCore

/// Quantized EmbeddingGemma 2 checkpoints (mxfp8 / nvfp4): key handling, config
/// cross-checks and strict binding, on a tiny synthetic model (no real weights).
final class EmbeddingGemma2QuantizationTests: XCTestCase {

    private let cfgJSON = """
    {"architectures": ["EmbeddingGemma2Model"], "model_type": "embedding_gemma2",
     "text_config": {"embedding_dim": 24, "head_dim": 16, "hidden_size": 32,
      "hidden_size_per_layer_input": 32, "intermediate_size": 64,
      "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
      "num_attention_heads": 2, "num_hidden_layers": 3, "num_key_value_heads": 1,
      "per_layer_config": {"2": {"head_dim": 32, "num_key_value_heads": 1}},
      "sliding_window": 4, "vocab_size": 64}}
    """

    private func denseModel() throws -> EmbeddingGemma2Model {
        let m = EmbeddingGemma2Model(try JSONDecoder().decode(
            EmbeddingGemma2Config.self, from: Data(cfgJSON.utf8)))
        MLXRandom.seed(11)
        let p = m.parameters().flattened().map { (k, v) -> (String, MLXArray) in
            if k.hasSuffix("layer_scalar") { return (k, MLXArray([Float(0.5)])) }
            return (k, MLXRandom.normal(v.shape) * 0.1)
        }
        m.update(parameters: ModuleParameters.unflattened(p))
        m.setComputeDtype(.float32)
        eval(m)
        return m
    }

    /// `language_model.`-prefixed dense tensors of `m` plus tiny vision / audio stubs.
    private func denseArrays(_ m: EmbeddingGemma2Model) -> [String: MLXArray] {
        var a: [String: MLXArray] = [:]
        for (k, v) in m.parameters().flattened() { a["language_model." + k] = v }
        a["vision_tower.patch_embedder.input_proj.weight"] = MLXArray.zeros([32, 32])
        a["audio_tower.output_proj.bias"] = MLXArray.zeros([4])
        return a
    }

    /// Quantize every 2-D `.weight` whose module path contains none of `skip`.
    private func quantizedArrays(
        _ dense: [String: MLXArray], mode: QuantizationMode, group: Int, bits: Int,
        skip: [String] = [], overrides: [String: (QuantizationMode, Int, Int)] = [:]
    ) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (k, v) in dense {
            let isW = k.hasSuffix(".weight") && v.ndim == 2
            let mod = String(k.dropLast(".weight".count))
            if isW, !skip.contains(where: { mod.contains($0) }) {
                let (md, g, b) = overrides[mod] ?? (mode, group, bits)
                let (wq, s, bi) = MLX.quantized(v, groupSize: g, bits: b, mode: md)
                out[k] = wq; out[mod + ".scales"] = s
                if let bi { out[mod + ".biases"] = bi }
            } else {
                out[k] = v
            }
        }
        return out
    }

    private func write(_ arrays: [String: MLXArray], config: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("eg2q-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try config.write(to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    private func configWith(quant: String?) -> String {
        guard let quant else { return cfgJSON }
        return String(cfgJSON.dropLast()) + ", \"quantization\": \(quant)}"
    }

    private let tokens = MLXArray([Int32(1), 5, 9, 33, 2]).reshaped(1, 5)

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Float {
        let x = a.flattened(), y = b.flattened()
        return (MLX.sum(x * y) / (MLX.sqrt(MLX.sum(x * x)) * MLX.sqrt(MLX.sum(y * y)))).item(Float.self)
    }

    // MARK: key handling

    func testQuantizedModulesFromKeys() throws {
        let mods = try eg2QuantizedModules(in: [
            "language_model.a.weight", "language_model.a.scales",
            "language_model.b.weight", "language_model.norm.weight",
        ])
        XCTAssertEqual(mods, ["language_model.a"])
        XCTAssertThrowsError(try eg2QuantizedModules(in: ["x.scales"])) {
            XCTAssertEqual($0 as? EG2QuantizationError, .scalesWithoutWeight("x"))
        }
    }

    func testValidateQuantizationCrossChecks() throws {
        let q = QuantizationConfig(
            groupSize: 32, bits: 8, mode: "mxfp8",
            moduleOverrides: ["language_model.a": .init(groupSize: 16, bits: 4, mode: "nvfp4")])
        XCTAssertNoThrow(try eg2ValidateQuantization(modules: ["language_model.a"], config: q))
        XCTAssertNoThrow(try eg2ValidateQuantization(modules: [], config: nil))
        XCTAssertThrowsError(try eg2ValidateQuantization(modules: ["m"], config: nil))
        XCTAssertThrowsError(try eg2ValidateQuantization(modules: [], config: q))
        XCTAssertThrowsError(try eg2ValidateQuantization(modules: ["language_model.b"], config: q)) {
            XCTAssertEqual($0 as? EG2QuantizationError, .staleOverride("language_model.a"))
        }
    }

    func testConfigBlockParsing() throws {
        let c = try eg2QuantizationConfig(configData: Data(configWith(quant:
            #"{"group_size": 16, "bits": 4, "mode": "nvfp4", "language_model.x": {"group_size": 32, "bits": 8, "mode": "mxfp8"}}"#).utf8))
        XCTAssertEqual(c?.mode, "nvfp4")
        XCTAssertEqual(c?.effective(for: "language_model.x").mode, "mxfp8")
        XCTAssertEqual(c?.effective(for: "language_model.y").groupSize, 16)
        XCTAssertNil(try eg2QuantizationConfig(configData: Data(cfgJSON.utf8)))
        XCTAssertThrowsError(try eg2QuantizationConfig(configData: Data(configWith(quant: "7").utf8)))
    }

    func testCastKeepsQuantizedDtypes() throws {
        let m = try denseModel()
        let dir = try write(
            quantizedArrays(denseArrays(m), mode: .mxfp8, group: 32, bits: 8),
            config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8"}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        let (q, _) = try loadEmbeddingGemma2(directory: dir, dtype: .bfloat16)
        let flat = Dictionary(uniqueKeysWithValues: q.parameters().flattened())
        XCTAssertEqual(flat["layers.0.self_attn.q_proj.weight"]?.dtype, .uint32)
        XCTAssertEqual(flat["layers.0.self_attn.q_proj.scales"]?.dtype, .uint8)
        XCTAssertEqual(flat["norm.weight"]?.dtype, .bfloat16)
    }

    // MARK: strict loading

    func testMxfp8LoadsAndMatchesDense() throws {
        let m = try denseModel()
        let dir = try write(
            quantizedArrays(denseArrays(m), mode: .mxfp8, group: 32, bits: 8),
            config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8"}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        let (q, report) = try loadEmbeddingGemma2(directory: dir, dtype: .float32)
        XCTAssertGreaterThan(report.quantizedText, 20)
        XCTAssertEqual(report.quantizedText, report.quantizedTotal - 1)  // the vision stub is quantized too
        let ref = m.pooled(tokens, lengths: [5])
        let got = q.pooled(tokens, lengths: [5])
        XCTAssertGreaterThan(cosine(ref, got), 0.99)
    }

    func testNvfp4WithProtectedAndDenseModules() throws {
        let m = try denseModel()
        // nvfp4 base, down_proj at mxfp8 (a per-module override), embedding table dense.
        var ov: [String: (QuantizationMode, Int, Int)] = [:]
        for i in 0 ..< 3 { ov["language_model.layers.\(i).mlp.down_proj"] = (.mxfp8, 32, 8) }
        let arrays = quantizedArrays(
            denseArrays(m), mode: .nvfp4, group: 16, bits: 4, skip: ["embed_tokens"], overrides: ov)
        var block = #"{"group_size": 16, "bits": 4, "mode": "nvfp4""#
        for i in 0 ..< 3 {
            block += #", "language_model.layers.\#(i).mlp.down_proj": {"group_size": 32, "bits": 8, "mode": "mxfp8"}"#
        }
        block += "}"
        let dir = try write(arrays, config: configWith(quant: block))
        defer { try? FileManager.default.removeItem(at: dir) }
        let (q, _) = try loadEmbeddingGemma2(directory: dir, dtype: .float32)
        XCTAssertGreaterThan(cosine(m.pooled(tokens, lengths: [5]), q.pooled(tokens, lengths: [5])), 0.9)
    }

    func testQuantizedTensorsWithoutConfigThrow() throws {
        let m = try denseModel()
        let dir = try write(quantizedArrays(denseArrays(m), mode: .mxfp8, group: 32, bits: 8), config: cfgJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testConfigWithDenseWeightsThrows() throws {
        let m = try denseModel()
        let dir = try write(
            denseArrays(m), config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8"}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testWrongModeInConfigThrows() throws {
        // Weights are nvfp4 (group 16) but the config claims mxfp8 (group 32): scales shapes differ.
        let m = try denseModel()
        let dir = try write(
            quantizedArrays(denseArrays(m), mode: .nvfp4, group: 16, bits: 4),
            config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8"}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testMissingScalesOfOneModuleThrows() throws {
        let m = try denseModel()
        var a = quantizedArrays(denseArrays(m), mode: .mxfp8, group: 32, bits: 8)
        a.removeValue(forKey: "language_model.layers.1.mlp.up_proj.scales")
        let dir = try write(a, config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8"}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testStrayBiasesThrow() throws {
        let m = try denseModel()
        var a = quantizedArrays(denseArrays(m), mode: .mxfp8, group: 32, bits: 8)
        a["language_model.layers.0.mlp.up_proj.biases"] = MLXArray.zeros([64, 1])
        let dir = try write(a, config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8"}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testStaleOverrideThrows() throws {
        let m = try denseModel()
        let dir = try write(
            quantizedArrays(denseArrays(m), mode: .mxfp8, group: 32, bits: 8),
            config: configWith(quant: #"{"group_size": 32, "bits": 8, "mode": "mxfp8", "language_model.nope": {"group_size": 16, "bits": 4, "mode": "nvfp4"}}"#))
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertThrowsError(try loadEmbeddingGemma2(directory: dir))
    }

    func testDenseCheckpointStillLoadsUnquantized() throws {
        let m = try denseModel()
        let dir = try write(denseArrays(m), config: cfgJSON)
        defer { try? FileManager.default.removeItem(at: dir) }
        let (q, report) = try loadEmbeddingGemma2(directory: dir, dtype: .float32)
        XCTAssertEqual(report.quantizedTotal, 0)
        XCTAssertGreaterThan(cosine(m.pooled(tokens, lengths: [5]), q.pooled(tokens, lengths: [5])), 0.99999)
    }

    // MARK: the quantizer itself

    func testCheckpointQuantizerProducesALoadableEmbeddingGemma2() throws {
        let m = try denseModel()
        let src = try write(denseArrays(m), config: cfgJSON)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("eg2q-out-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: src); try? FileManager.default.removeItem(at: out) }
        let n = try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, mode: "nvfp4", dtype: "bf16",
            protect: ["down_proj"], protectBits: 8, protectGroupSize: 32, protectMode: "mxfp8",
            skip: ["embed_tokens"])
        XCTAssertGreaterThan(n, 20)
        let keys = Set(try loadArrays(url: out.appendingPathComponent("model.safetensors")).keys)
        XCTAssertFalse(keys.contains("language_model.embed_tokens.scales"))
        XCTAssertTrue(keys.contains("language_model.layers.0.self_attn.q_proj.scales"))
        XCTAssertTrue(keys.contains("language_model.layers.0.mlp.down_proj.scales"))
        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: out.appendingPathComponent("config.json"))) as! [String: Any]
        let q = cfg["quantization"] as! [String: Any]
        XCTAssertEqual(q["mode"] as? String, "nvfp4")
        XCTAssertEqual((q["language_model.layers.0.mlp.down_proj"] as? [String: Any])?["mode"] as? String, "mxfp8")
        // The produced folder loads through the strict loader.
        let (loaded, report) = try loadEmbeddingGemma2(directory: out, dtype: .float32)
        XCTAssertEqual(report.quantizedTotal, n)
        XCTAssertGreaterThan(cosine(m.pooled(tokens, lengths: [5]), loaded.pooled(tokens, lengths: [5])), 0.9)
    }

    /// A skip that hits one layer's module but not the others would crash the
    /// loader (mlx-swift cannot swap leaves in only some layer-array elements),
    /// so the quantizer refuses it up front.
    func testSkipThatSplitsLayersIsRefused() throws {
        let src = try write(denseArrays(try denseModel()), config: cfgJSON)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("eg2q-out-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: src); try? FileManager.default.removeItem(at: out) }
        XCTAssertThrowsError(try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 8, groupSize: 32, mode: "mxfp8", dtype: "bf16",
            skip: ["language_model.layers.0."])) { e in
            XCTAssertTrue("\(e)".contains("every layer"), "\(e)")
        }
    }

    // MARK: the affine ladder (3 / 5 / 6 bit) and regex patterns

    private func quantizeSynthetic(
        bits: Int, group: Int = 32, mode: String = "affine", protect: [String] = [],
        protectBits: Int = 8, skip: [String] = []
    ) throws -> (model: EmbeddingGemma2Model, out: URL, n: Int) {
        let m = try denseModel()
        let src = try write(denseArrays(m), config: cfgJSON)
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("eg2q-out-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: src) }
        let n = try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: bits, groupSize: group, mode: mode, dtype: "bf16",
            protect: protect, protectBits: protectBits, protectGroupSize: group, skip: skip)
        return (m, out, n)
    }

    func testAffine3_5_6BitCheckpointsLoadStrictAndTrackBits() throws {
        var last: Float = 0
        for bits in [3, 5, 6, 8] {
            let (m, out, n) = try quantizeSynthetic(bits: bits)
            defer { try? FileManager.default.removeItem(at: out) }
            let cfg = try JSONSerialization.jsonObject(
                with: Data(contentsOf: out.appendingPathComponent("config.json"))) as! [String: Any]
            XCTAssertEqual((cfg["quantization"] as? [String: Any])?["bits"] as? Int, bits)
            let (loaded, report) = try loadEmbeddingGemma2(directory: out, dtype: .float32)
            XCTAssertEqual(report.quantizedTotal, n)
            let c = cosine(m.pooled(tokens, lengths: [5]), loaded.pooled(tokens, lengths: [5]))
            XCTAssertGreaterThan(c, bits == 3 ? 0.5 : 0.9, "bits \(bits)")
            XCTAssertGreaterThanOrEqual(c + 0.02, last, "more bits must not be worse (\(bits))")
            last = c
        }
    }

    func testUnsupportedAffineBitsAreRefused() throws {
        for bits in [1, 7, 9] {
            XCTAssertThrowsError(try quantizeSynthetic(bits: bits)) { e in
                XCTAssertTrue("\(e)".contains("2, 3, 4, 5, 6 or 8"), "\(e)")
            }
        }
    }

    func testRegexProtectTargetsOneLayerAndStillLoads() throws {
        // 4-bit base, layer 1 of the text tower at 8-bit: a per-module override on that layer only.
        let (m, out, _) = try quantizeSynthetic(bits: 4, protect: ["re:^language_model\\.layers\\.1\\."])
        defer { try? FileManager.default.removeItem(at: out) }
        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: out.appendingPathComponent("config.json"))) as! [String: Any]
        let q = cfg["quantization"] as! [String: Any]
        let overrides = q.keys.filter { $0.hasPrefix("language_model.") }
        XCTAssertFalse(overrides.isEmpty)
        XCTAssertTrue(overrides.allSatisfy { $0.hasPrefix("language_model.layers.1.") }, "\(overrides)")
        XCTAssertEqual((q[overrides[0]] as? [String: Any])?["bits"] as? Int, 8)
        let (loaded, _) = try loadEmbeddingGemma2(directory: out, dtype: .float32)
        XCTAssertGreaterThan(cosine(m.pooled(tokens, lengths: [5]), loaded.pooled(tokens, lengths: [5])), 0.9)
    }

    func testRegexSkipAnchoredToTheTextTowerKeepsItDense() throws {
        // `re:` anchors tell the text projection from any other `*embedding_projection`.
        let (_, out, _) = try quantizeSynthetic(bits: 6, skip: ["re:^language_model\\.embedding_projection$"])
        defer { try? FileManager.default.removeItem(at: out) }
        let keys = Set(try loadArrays(url: out.appendingPathComponent("model.safetensors")).keys)
        XCTAssertFalse(keys.contains("language_model.embedding_projection.scales"))
        XCTAssertTrue(keys.contains("language_model.embed_tokens.scales"))
        _ = try loadEmbeddingGemma2(directory: out, dtype: .float32)
    }
}
