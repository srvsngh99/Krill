import XCTest
import MLX
import MLXRandom
@testable import KrillCore

/// Unit coverage for the Prism Hadamard qwen35 port (`prism_hadamard_qwen35`,
/// Ternary-Bonsai-2-27B): the `fwht` activation transform round-trips, the
/// `hadamard.json` sign-table parser accepts a valid contract and rejects
/// every malformed one the task ground truth calls out, and the full
/// `PrismHadamardConfig` cross-validates config.json's `modules` manifest
/// against `hadamard.json`. No real checkpoint is needed - everything here
/// runs against small synthetic fixtures.
final class PrismHadamardQwen35Tests: XCTestCase {

    // MARK: - fwht round-trip

    /// `fwht(fwht(x, inverse: false), inverse: true)` must return `x` within
    /// fp tolerance - the core correctness property the whole pack depends
    /// on (this is literally what lets the embedding's inverse transform
    /// undo the linear layers' forward transform end to end).
    func testFWHTRoundTrip() {
        let block = 1024
        let signValues = (0 ..< block).map { $0 % 3 == 0 ? Float(-1) : Float(1) }
        let signs = MLXArray(signValues)
        let x = MLXRandom.normal([2, 5, block])

        let rotated = fwht(x, block: block, signs: signs, inverse: false)
        let restored = fwht(rotated, block: block, signs: signs, inverse: true)

        let a = x.asType(.float32).flattened().asArray(Float.self)
        let b = restored.asType(.float32).flattened().asArray(Float.self)
        XCTAssertEqual(a.count, b.count)
        for i in 0 ..< a.count {
            XCTAssertEqual(a[i], b[i], accuracy: 1e-3, "mismatch at flat index \(i)")
        }
    }

    /// The transform must not silently widen the storage dtype - the module
    /// forward passes fp16 activations through it.
    func testFWHTPreservesInputDType() {
        let block = 512
        let signs = MLXArray([Float](repeating: 1, count: block))
        let x = MLXRandom.normal([1, block]).asType(.float16)
        let y = fwht(x, block: block, signs: signs)
        XCTAssertEqual(y.dtype, .float16)
        XCTAssertEqual(y.shape, x.shape)
    }

    // MARK: - Sign-table parsing (`buildPrismSignTable`)

    func testSignTableSplitsSequentiallyByWidth() throws {
        let table = try buildPrismSignTable(widths: [2, 3], values: [1, -1, -1, 1, 1])
        XCTAssertEqual(table[2]?.asArray(Float.self), [1, -1])
        XCTAssertEqual(table[3]?.asArray(Float.self), [-1, 1, 1])
    }

    func testSignTableRejectsNonUnitValues() {
        XCTAssertThrowsError(try buildPrismSignTable(widths: [2], values: [1, 0.5]))
    }

    func testSignTableRejectsTrailingValues() {
        // 3 values but widths only account for 2 -> "Trailing sign values".
        XCTAssertThrowsError(try buildPrismSignTable(widths: [2], values: [1, -1, 1]))
    }

    func testSignTableRejectsShortValues() {
        // Declares width 4 but only 2 values are available.
        XCTAssertThrowsError(try buildPrismSignTable(widths: [4], values: [1, -1]))
    }

    func testSignTableRejectsZeroValue() {
        // Zero is neither +1 nor -1 - must be rejected like any other non-unit value.
        XCTAssertThrowsError(try buildPrismSignTable(widths: [3], values: [1, 0, -1]))
    }

    // MARK: - PrismHadamardConfig: full config.json + hadamard.json contract

    /// A minimal but structurally complete `prism_hadamard_qwen35` fixture:
    /// one packed `Linear` (`lm_head`, width == hidden_size == block) and one
    /// packed `Embedding` (`model.embed_tokens`). Small enough to construct
    /// and mutate inline per test; `hadamardOverrides`/`moduleOverrides` let
    /// each rejection test corrupt exactly one field.
    private func makeFixture(
        block: Int = 512,
        hadamardJSON: String? = nil,
        configJSON: String? = nil
    ) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("prism-hadamard-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")

        let defaultHadamard = """
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": \(block),
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["language_model.lm_head.weight"],
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": true
        }
        """

        let defaultConfig = """
        {
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "text_config": {
                "hidden_size": \(block),
                "intermediate_size": \(block),
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": \(block / 4),
                "vocab_size": 32,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 4,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
                "full_attention_interval": 4,
                "tie_word_embeddings": false
            },
            "modules": [
                {"path": "lm_head", "block": \(block), "embedding": false, "dtype": "float16"},
                {"path": "model.embed_tokens", "block": \(block), "embedding": true, "dtype": "float16"}
            ],
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "tie_word_embeddings": false
        }
        """

        try (configJSON ?? defaultConfig).write(
            to: dir.appendingPathComponent("config.json"), atomically: true, encoding: .utf8)
        try (hadamardJSON ?? defaultHadamard).write(
            to: dir.appendingPathComponent("hadamard.json"), atomically: true, encoding: .utf8)
        return dir
    }

    private func loadConfig(from dir: URL) throws -> PrismHadamardConfig {
        let configData = try Data(contentsOf: dir.appendingPathComponent("config.json"))
        return try PrismHadamardConfig(configData: configData, directory: dir)
    }

    func testValidFixtureConstructsAndPicksSignsByWidth() throws {
        let dir = try makeFixture()
        let config = try loadConfig(from: dir)
        XCTAssertEqual(config.modules.count, 2)
        XCTAssertEqual(config.blockSize, 512)
        XCTAssertNotNil(config.signs[512])
        XCTAssertEqual(config.signs[512]?.shape, [512])
    }

    func testRejectsUnsupportedBlockSize() throws {
        // 900 is not in {512, 1024, 2048, 4096}.
        let dir = try makeFixture(
            hadamardJSON: try invalidBlockSizeHadamard())
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    private func invalidBlockSizeHadamard() throws -> String {
        let block = 512
        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")
        return """
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": 900,
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["language_model.lm_head.weight"],
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": true
        }
        """
    }

    func testRejectsImplicitSignMode() throws {
        let dir = try makeFixture(hadamardJSON: hadamardWithSignMode("implicit"))
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    func testRejectsUnsupportedVersion() throws {
        let dir = try makeFixture(hadamardJSON: hadamardWithVersion(2))
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    func testRejectsUngroupedGDNV() throws {
        let dir = try makeFixture(hadamardJSON: hadamardWithGDNVGrouped(false))
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    func testRejectsInverseManifestNotExactlyEmbedTokens() throws {
        // inverse_weight_names must be EXACTLY {embed_tokens}; here it wrongly
        // also (or instead) claims lm_head, which must be forward-only.
        let block = 512
        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")
        let badHadamard = """
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": \(block),
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": [],
            "prism.hadamard.inverse_weight_names": ["language_model.lm_head.weight", "language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": true
        }
        """
        let dir = try makeFixture(hadamardJSON: badHadamard)
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    func testRejectsManifestMismatchWithConfigModules() throws {
        // hadamard.json's weight_names names a tensor config.json's `modules`
        // array does not (config still lists "lm_head"), so the two
        // manifests must fail cross-validation.
        let block = 512
        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")
        let badHadamard = """
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": \(block),
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["language_model.some_other_module.weight"],
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": true
        }
        """
        let dir = try makeFixture(hadamardJSON: badHadamard)
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    func testRejectsTiedWordEmbeddings() throws {
        let block = 512
        let badConfig = """
        {
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "text_config": {
                "hidden_size": \(block), "intermediate_size": \(block), "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": \(block / 4),
                "vocab_size": 32, "linear_num_value_heads": 4, "linear_num_key_heads": 4,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "full_attention_interval": 4, "tie_word_embeddings": false
            },
            "modules": [
                {"path": "lm_head", "block": \(block), "embedding": false, "dtype": "float16"},
                {"path": "model.embed_tokens", "block": \(block), "embedding": true, "dtype": "float16"}
            ],
            "quantization": {"bits": 2, "group_size": 128, "mode": "affine"},
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "tie_word_embeddings": true
        }
        """
        let dir = try makeFixture(configJSON: badConfig)
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    func testRejectsUnsupportedQuantization() throws {
        let block = 512
        let badConfig = """
        {
            "schema_version": 2,
            "model_type": "prism_hadamard_qwen35",
            "text_config": {
                "hidden_size": \(block), "intermediate_size": \(block), "num_hidden_layers": 1,
                "num_attention_heads": 4, "num_key_value_heads": 4, "head_dim": \(block / 4),
                "vocab_size": 32, "linear_num_value_heads": 4, "linear_num_key_heads": 4,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "full_attention_interval": 4, "tie_word_embeddings": false
            },
            "modules": [
                {"path": "lm_head", "block": \(block), "embedding": false, "dtype": "float16"},
                {"path": "model.embed_tokens", "block": \(block), "embedding": true, "dtype": "float16"}
            ],
            "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
            "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "tie_word_embeddings": false
        }
        """
        let dir = try makeFixture(configJSON: badConfig)
        XCTAssertThrowsError(try loadConfig(from: dir))
    }

    private func hadamardWithSignMode(_ mode: String) -> String {
        let block = 512
        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")
        return """
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": \(block),
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "\(mode)",
            "prism.hadamard.weight_names": ["language_model.lm_head.weight"],
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": true
        }
        """
    }

    private func hadamardWithVersion(_ version: Int) -> String {
        let block = 512
        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")
        return """
        {
            "prism.hadamard.version": \(version),
            "prism.hadamard.block_size": \(block),
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["language_model.lm_head.weight"],
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": true
        }
        """
    }

    private func hadamardWithGDNVGrouped(_ grouped: Bool) -> String {
        let block = 512
        let signValues = (0 ..< block).map { $0 % 2 == 0 ? "1" : "-1" }.joined(separator: ",")
        return """
        {
            "prism.hadamard.version": 1,
            "prism.hadamard.block_size": \(block),
            "prism.hadamard.transform": "normalized-sylvester-walsh-hadamard",
            "prism.hadamard.axis": "input-last-dimension",
            "prism.hadamard.sign_mode": "explicit",
            "prism.hadamard.weight_names": ["language_model.lm_head.weight"],
            "prism.hadamard.inverse_weight_names": ["language_model.model.embed_tokens.weight"],
            "prism.hadamard.sign_widths": [\(block)],
            "prism.hadamard.sign_values": [\(signValues)],
            "prism.hadamard.gdn_v_grouped": \(grouped)
        }
        """
    }

    // MARK: - Architecture-detection routing

    /// The real pack's config.json carries BOTH a `model_type` of
    /// `prism_hadamard_qwen35` AND a `vision_config` (333 vision tensors sit
    /// in the safetensors, outside the Hadamard manifest - see
    /// `loadPrismHadamardQwen35`'s doc comment). Detection must still select
    /// the always-text `prism_hadamard_qwen35` rule, not fall through to
    /// `qwen3_5` (whose action branches to the VL loader when a
    /// `vision_config` is present) or the generic `qwen` rule.
    func testPrismHadamardRoutesToTextRuleEvenWithVisionConfig() {
        // `detectedArchitectureID` only takes (architectures, model_type) -
        // exactly what `loadModel` extracts from config.json before ever
        // looking at `vision_config` - so a config carrying a `vision_config`
        // resolves identically to one without it: the rule match is on
        // model_type alone, and (unlike the `qwen3_5` rule) its action never
        // branches on `vision_config`.
        let id = detectedArchitectureID(architectures: [], modelType: "prism_hadamard_qwen35")
        XCTAssertEqual(id, "prism_hadamard_qwen35")
        XCTAssertNotEqual(id, "qwen3_5")
        XCTAssertNotEqual(id, "qwen")
        XCTAssertNotEqual(id, "fallback")
    }

    func testPrismHadamardNotStolenByGenericQwenRule() {
        // "prism_hadamard_qwen35" contains "qwen" as a substring but does not
        // start with it, so the generic `qwen` rule's `mt.hasPrefix("qwen")`
        // must not match - this pins that the dedicated rule (not the
        // generic fallback-ish qwen rule) claims it, and that it precedes
        // `qwen3_5` in table order regardless.
        XCTAssertEqual(
            detectedArchitectureID(architectures: ["PrismHadamardQwen35ForCausalLM"], modelType: "prism_hadamard_qwen35"),
            "prism_hadamard_qwen35")
    }

    /// Real config.json from the downloaded pack, if present: confirms the
    /// fixture assumption above (`vision_config` really is in the real
    /// file) without requiring the weights themselves.
    func testRealConfigCarriesVisionConfig() throws {
        let path = "/Users/sourav/.cache/huggingface/bonsai2-mlx-2bit/config.json"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Prism Hadamard pack config.json not present")
        }
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(json?["model_type"] as? String, "prism_hadamard_qwen35")
        XCTAssertNotNil(json?["vision_config"], "fixture assumption: real pack config carries vision_config")
        XCTAssertEqual(
            detectedArchitectureID(architectures: [], modelType: json?["model_type"] as? String ?? ""),
            "prism_hadamard_qwen35")
    }

    // MARK: - Real checkpoint: loader wiring only (no forward pass)

    /// Constructs the full model and loads every weight (including all 402
    /// packed modules) against the real 27B pack. Deliberately does NOT run
    /// a forward pass - this is a "the loader is wired correctly" smoke
    /// test, not a generation test.
    func testRealCheckpointLoads() throws {
        let path = "/Users/sourav/.cache/huggingface/bonsai2-mlx-2bit"
        guard FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Prism Hadamard pack not present")
        }
        let model = try loadModel(from: URL(fileURLWithPath: path))
        XCTAssertEqual(model.family, "prism_hadamard_qwen35")
        XCTAssertEqual(model.numLayers, 64)
        XCTAssertEqual(model.vocabSize, 248320)
        XCTAssertEqual(model.cacheSpec?.count, 64)
    }
}
