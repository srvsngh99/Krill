import XCTest
import MLX
import MLXRandom
@testable import KLMCore

/// Unit contracts for the native Swift+MLX checkpoint quantizer. The byte-exact
/// parity gate against `mlx_lm.convert` runs against a real checkpoint
/// (`tools/verify_native_quantize_parity.sh`); these are the deterministic,
/// in-process invariants.
final class CheckpointQuantizerTests: XCTestCase {

    /// Write a synthetic dense source dir (config.json + one safetensors) and
    /// return its URL.
    private func makeSource(_ arrays: [String: MLXArray],
                            config: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-src-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testQuantizesDivisibleWeightsAndPassesTheRestThrough() throws {
        let src = try makeSource([
            // 2-D, inner dim divisible by 64 -> quantized
            "model.embed_tokens.weight": MLXRandom.normal([128, 256]),
            "model.layers.0.mlp.down_proj.weight": MLXRandom.normal([256, 128]),
            // 1-D norm -> passes through, NOT quantized
            "model.layers.0.input_layernorm.weight": MLXArray.ones([128]),
            // 1-D bias -> passes through
            "model.layers.0.self_attn.q_proj.bias": MLXArray.zeros([128]),
        ], config: ["model_type": "llama", "architectures": ["LlamaForCausalLM"]])
        defer { try? FileManager.default.removeItem(at: src) }

        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-out-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }

        let n = try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, dtype: "fp16")
        XCTAssertEqual(n, 2, "exactly the two divisible 2-D weights are quantized")

        let w = try loadArrays(url: out.appendingPathComponent("model.safetensors"))

        // Quantized layers: packed uint32 weight + fp16 scales + fp16 biases.
        for stem in ["model.embed_tokens", "model.layers.0.mlp.down_proj"] {
            XCTAssertEqual(w["\(stem).weight"]?.dtype, .uint32, "\(stem) packed")
            XCTAssertEqual(w["\(stem).scales"]?.dtype, .float16)
            XCTAssertEqual(w["\(stem).biases"]?.dtype, .float16)
        }
        // Pass-through tensors: unchanged shape, fp16 (cast to storage dtype),
        // and NO scales/biases emitted for them.
        XCTAssertEqual(w["model.layers.0.input_layernorm.weight"]?.shape, [128])
        XCTAssertNil(w["model.layers.0.input_layernorm.scales"])
        XCTAssertNotNil(w["model.layers.0.self_attn.q_proj.bias"])

        // config.json carries the quantization block the loader reads.
        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: out.appendingPathComponent("config.json")))
            as? [String: Any]
        let q = cfg?["quantization"] as? [String: Any]
        XCTAssertEqual(q?["group_size"] as? Int, 64)
        XCTAssertEqual(q?["bits"] as? Int, 4)
    }

    func testDequantizeRoundTripsWithinTolerance() throws {
        let original = MLXRandom.normal([128, 256])
        let src = try makeSource(
            ["model.layers.0.mlp.down_proj.weight": original],
            config: ["model_type": "llama"])
        defer { try? FileManager.default.removeItem(at: src) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-rt-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }

        try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, dtype: "fp16")
        let w = try loadArrays(url: out.appendingPathComponent("model.safetensors"))

        let deq = dequantized(
            w["model.layers.0.mlp.down_proj.weight"]!,
            scales: w["model.layers.0.mlp.down_proj.scales"]!,
            biases: w["model.layers.0.mlp.down_proj.biases"]!,
            groupSize: 64, bits: 4)
        // 4-bit affine round-trip: bounded error, not exact.
        let err = MLX.max(MLX.abs(deq.asType(.float32) - original.asType(.float32))).item(Float.self)
        XCTAssertLessThan(err, 0.5, "4-bit dequant stays within a sane bound of the original")
    }

    func testNvfp4ProducesScalesNoBiasesAndOverridesGroup() throws {
        // nvfp4 is 4-bit / group-16 with uint8 block scales and NO biases. Passing
        // group 64 must be overridden to 16, and the config must record nvfp4/16/4.
        let src = try makeSource([
            "model.layers.0.mlp.down_proj.weight": MLXRandom.normal([256, 128]),
        ], config: ["model_type": "llama"])
        defer { try? FileManager.default.removeItem(at: src) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-nvfp4-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }

        // Pass group 64 on purpose; nvfp4 must override to 16.
        try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, mode: "nvfp4")
        let w = try loadArrays(url: out.appendingPathComponent("model.safetensors"))

        XCTAssertEqual(w["model.layers.0.mlp.down_proj.weight"]?.dtype, .uint32, "packed")
        XCTAssertNotNil(w["model.layers.0.mlp.down_proj.scales"], "nvfp4 has scales")
        XCTAssertNil(w["model.layers.0.mlp.down_proj.biases"], "nvfp4 has no biases")

        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: out.appendingPathComponent("config.json"))) as? [String: Any]
        let q = cfg?["quantization"] as? [String: Any]
        XCTAssertEqual(q?["mode"] as? String, "nvfp4")
        XCTAssertEqual(q?["group_size"] as? Int, 16, "nvfp4 group overridden to 16")
        XCTAssertEqual(q?["bits"] as? Int, 4)
    }

    func testThrowsOnNonDivisibleWeight() throws {
        // A 2-D weight whose inner dim is not group-divisible cannot be uniformly
        // loaded by KrillLM, so the quantizer must fail loudly rather than leave it
        // dense.
        let src = try makeSource(
            ["model.layers.0.mlp.down_proj.weight": MLXRandom.normal([8, 30])],
            config: ["model_type": "llama"])
        defer { try? FileManager.default.removeItem(at: src) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-nd-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }
        XCTAssertThrowsError(
            try CheckpointQuantizer.quantize(
                sourceDir: src, outputDir: out, bits: 4, groupSize: 64))
    }

    func testRejectsUnsupportedFamilies() throws {
        for badConfig in [
            ["model_type": "qwen2_moe", "num_experts": 60] as [String: Any],
            ["model_type": "qwen2_5_vl", "vision_config": ["x": 1]] as [String: Any],
            ["model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"]] as [String: Any],
        ] {
            let src = try makeSource(
                ["model.layers.0.mlp.down_proj.weight": MLXArray.ones([64, 64])],
                config: badConfig)
            defer { try? FileManager.default.removeItem(at: src) }
            let out = FileManager.default.temporaryDirectory
                .appendingPathComponent("klm-quant-bad-\(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: out) }
            XCTAssertThrowsError(
                try CheckpointQuantizer.quantize(
                    sourceDir: src, outputDir: out, bits: 4, groupSize: 64),
                "must reject \(badConfig["model_type"] ?? "?")")
        }
    }
}
