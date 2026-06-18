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

    /// Write a synthetic "reference" 4-bit checkpoint: just the `.scales` tensors
    /// for the modules we want the learner to pick up (it only reads names).
    private func makeReference(scalesFor modules: [String]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-ref-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var arrays: [String: MLXArray] = [:]
        for m in modules { arrays["\(m).scales"] = MLXArray.ones([1]) }
        try save(arrays: arrays, url: dir.appendingPathComponent("model.safetensors"))
        return dir
    }

    func testReferenceSetSelectsExactlyTheReferencedModules() throws {
        // A "VL" config that the dense pass would REJECT loads fine with a reference,
        // and only the referenced modules are quantized (the vision tower passes
        // through as float, mirroring Qwen2.5-VL's loader).
        let src = try makeSource([
            "language_model.model.layers.0.mlp.down_proj.weight": MLXRandom.normal([256, 128]),
            "language_model.model.embed_tokens.weight": MLXRandom.normal([128, 256]),
            "visual.blocks.0.attn.qkv.weight": MLXRandom.normal([192, 64]),   // vision tower: float
            "model.norm.weight": MLXArray.ones([256]),
        ], config: ["model_type": "qwen2_5_vl", "vision_config": ["x": 1]])
        defer { try? FileManager.default.removeItem(at: src) }
        let ref = try makeReference(scalesFor: [
            "language_model.model.layers.0.mlp.down_proj",
            "language_model.model.embed_tokens",
        ])
        defer { try? FileManager.default.removeItem(at: ref) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-ref-out-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }

        let n = try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, referenceDir: ref)
        XCTAssertEqual(n, 2, "exactly the two referenced modules are quantized")

        let w = try loadArrays(url: out.appendingPathComponent("model.safetensors"))
        XCTAssertEqual(w["language_model.model.layers.0.mlp.down_proj.weight"]?.dtype, .uint32)
        XCTAssertNotNil(w["language_model.model.embed_tokens.scales"])
        // Vision tower stays float (not in the reference set), no scales emitted.
        XCTAssertEqual(w["visual.blocks.0.attn.qkv.weight"]?.dtype, .float16)
        XCTAssertNil(w["visual.blocks.0.attn.qkv.scales"])
    }

    func testProtectedModuleGetsHigherPrecisionOverride() throws {
        // Top-level nvfp4; a protected module is quantized at 8-bit affine and
        // recorded as a per-module override the loader resolves via effective(for:).
        let src = try makeSource([
            "language_model.model.layers.0.mlp.down_proj.weight": MLXRandom.normal([256, 128]),
            "vision_embedder.patch_dense.weight": MLXRandom.normal([128, 256]),
        ], config: ["model_type": "gemma4"])
        defer { try? FileManager.default.removeItem(at: src) }
        let ref = try makeReference(scalesFor: [
            "language_model.model.layers.0.mlp.down_proj",
            "vision_embedder.patch_dense",
        ])
        defer { try? FileManager.default.removeItem(at: ref) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-prot-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }

        // patch_dense is auto-protected by default (vision projector).
        try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, mode: "nvfp4",
            referenceDir: ref)
        let w = try loadArrays(url: out.appendingPathComponent("model.safetensors"))

        // The dense LM module is nvfp4 (no biases); the protected projector is
        // 8-bit affine (has biases).
        XCTAssertNil(w["language_model.model.layers.0.mlp.down_proj.biases"], "nvfp4: no biases")
        XCTAssertNotNil(w["vision_embedder.patch_dense.biases"], "8-bit affine: biases")

        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: out.appendingPathComponent("config.json"))) as? [String: Any]
        let q = cfg?["quantization"] as? [String: Any]
        XCTAssertEqual(q?["mode"] as? String, "nvfp4", "top-level stays nvfp4")
        let ov = q?["vision_embedder.patch_dense"] as? [String: Any]
        XCTAssertEqual(ov?["bits"] as? Int, 8, "protected module override present")
        XCTAssertEqual(ov?["mode"] as? String, "affine")
        XCTAssertEqual(ov?["group_size"] as? Int, 64)
    }

    func test3DExpertForcedAffineUnderFloatTopLevel() throws {
        // Stacked 3-D experts must be quantized affine (the MoE runtime is
        // affine-only) even when the top-level format is nvfp4.
        let src = try makeSource([
            "model.layers.0.mlp.switch_mlp.gate_proj.weight": MLXRandom.normal([4, 64, 128]),
            "model.layers.0.self_attn.q_proj.weight": MLXRandom.normal([128, 128]),
        ], config: ["model_type": "qwen3_moe", "num_experts": 4])
        defer { try? FileManager.default.removeItem(at: src) }
        let ref = try makeReference(scalesFor: [
            "model.layers.0.mlp.switch_mlp.gate_proj",
            "model.layers.0.self_attn.q_proj",
        ])
        defer { try? FileManager.default.removeItem(at: ref) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-moe-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }

        try CheckpointQuantizer.quantize(
            sourceDir: src, outputDir: out, bits: 4, groupSize: 64, mode: "nvfp4",
            referenceDir: ref)
        let w = try loadArrays(url: out.appendingPathComponent("model.safetensors"))

        // Expert: 3-D packed weight + 3-D scales + biases (affine).
        let eg = "model.layers.0.mlp.switch_mlp.gate_proj"
        XCTAssertEqual(w["\(eg).weight"]?.ndim, 3, "experts stay 3-D")
        XCTAssertNotNil(w["\(eg).biases"], "expert is affine -> has biases")
        // Attn q_proj follows the top-level nvfp4 (no biases).
        XCTAssertNil(w["model.layers.0.self_attn.q_proj.biases"], "nvfp4: no biases")

        let cfg = try JSONSerialization.jsonObject(
            with: Data(contentsOf: out.appendingPathComponent("config.json"))) as? [String: Any]
        let ov = (cfg?["quantization"] as? [String: Any])?[eg] as? [String: Any]
        XCTAssertEqual(ov?["mode"] as? String, "affine", "experts overridden to affine")
        XCTAssertEqual(ov?["bits"] as? Int, 4)
    }

    func testReferenceWithoutScalesThrows() throws {
        let src = try makeSource(
            ["model.layers.0.mlp.down_proj.weight": MLXRandom.normal([256, 128])],
            config: ["model_type": "llama"])
        defer { try? FileManager.default.removeItem(at: src) }
        // A reference dir with no `.scales` tensors is not a quantized build.
        let ref = try makeSource(
            ["model.layers.0.mlp.down_proj.weight": MLXRandom.normal([256, 128])],
            config: ["model_type": "llama"])
        defer { try? FileManager.default.removeItem(at: ref) }
        let out = FileManager.default.temporaryDirectory
            .appendingPathComponent("klm-quant-noref-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: out) }
        XCTAssertThrowsError(
            try CheckpointQuantizer.quantize(
                sourceDir: src, outputDir: out, bits: 4, groupSize: 64, referenceDir: ref))
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
