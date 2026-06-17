import XCTest
@testable import KLMRegistry

final class KLMRegistryTests: XCTestCase {
    func testConfigTOMLParsesServingKnobs() {
        // The serving knobs (max_loaded_models, num_parallel, keep_alive,
        // max_queue) were env-only; config.toml must now read them too, so a
        // deployment can keep e.g. an embedding model and a generation model
        // both resident by default (max_loaded_models = 2).
        var cfg = KrillConfig()
        XCTAssertEqual(cfg.maxLoadedModels, 1)   // default
        cfg.mergeFromTOML("""
            max_loaded_models = 2
            num_parallel = 4
            keep_alive = "-1"
            max_queue = 256
            """)
        XCTAssertEqual(cfg.maxLoadedModels, 2)
        XCTAssertEqual(cfg.numParallel, 4)
        XCTAssertEqual(cfg.keepAlive, "-1")
        XCTAssertEqual(cfg.maxQueue, 256)
    }

    func testThinkingConfigDefaultsOnAndParses() {
        // Reasoning channel is ON by default (no-op for models without one), and
        // the `thinking` config key sets the per-session default.
        var cfg = KrillConfig()
        XCTAssertTrue(cfg.thinking)            // default ON
        cfg.mergeFromTOML("thinking = false")
        XCTAssertFalse(cfg.thinking)
        var cfg2 = KrillConfig()
        cfg2.mergeFromTOML("enable_thinking = true")  // alias accepted
        XCTAssertTrue(cfg2.thinking)
    }

    func testModelNameValidationRejectsTraversal() {
        XCTAssertTrue(Registry.isValidModelName("llama-3.2-1b"))
        XCTAssertTrue(Registry.isValidModelName("my_model.v2"))
        for bad in ["../etc/passwd", "a/b", "..", ".hidden", "/abs",
                    "~/x", "a\\b", "with\u{0}null", ""] {
            XCTAssertFalse(Registry.isValidModelName(bad), "should reject \(bad)")
        }
    }

    func testCreateModelRejectsTraversalName() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-trav-\(UUID().uuidString)")
        let reg = Registry(baseDir: base)
        try reg.ensureDirectories()
        try reg.saveManifest(ModelManifest(
            name: "safe-base", family: .llama, params: "1B", quant: "4bit",
            source: "x", context: 4096, files: [], chatTemplate: "t",
            sizeBytes: 1))
        let mf = try ModelfileParser.parse("FROM safe-base\nSYSTEM hi")
        XCTAssertThrowsError(try reg.createModel(name: "../escape", from: mf))
        XCTAssertThrowsError(try reg.removeModel("../safe-base"))
    }

    func testModelfileParserDirectives() throws {
        let mf = try ModelfileParser.parse("""
        # a comment
        FROM llama-3.2-1b
        PARAMETER temperature 0.4
        PARAMETER top_p 0.9
        SYSTEM \"\"\"
        You are a terse assistant.
        Always answer in one line.
        \"\"\"
        MESSAGE user Hi
        MESSAGE assistant Hello.
        ADAPTER ./lora
        """)
        XCTAssertEqual(mf.from, "llama-3.2-1b")
        XCTAssertEqual(mf.parameters["temperature"], "0.4")
        XCTAssertEqual(mf.parameters["top_p"], "0.9")
        XCTAssertTrue(mf.system?.contains("terse assistant") ?? false)
        XCTAssertTrue(mf.system?.contains("one line") ?? false)
        XCTAssertEqual(mf.messages.count, 2)
        XCTAssertNotNil(mf.adapterWarning)
    }

    func testModelfileParserRequiresFROM() {
        XCTAssertThrowsError(try ModelfileParser.parse("SYSTEM hi"))
    }

    func testCreateModelReferencesBaseAndStoresOverrides() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-create-\(UUID().uuidString)")
        let reg = Registry(baseDir: base)
        try reg.ensureDirectories()
        // Seed a base model + a blob dir to symlink.
        let baseManifest = ModelManifest(
            name: "base-llama", family: .llama, params: "1B", quant: "4bit",
            source: "mlx-community/x", context: 4096, files: [],
            chatTemplate: "llama3", sizeBytes: 1000)
        try reg.saveManifest(baseManifest)
        try FileManager.default.createDirectory(
            at: reg.modelPath("base-llama"), withIntermediateDirectories: true)

        let mf = try ModelfileParser.parse("""
        FROM base-llama
        PARAMETER temperature 0.2
        SYSTEM You are Krill.
        """)
        let created = try reg.createModel(name: "my-krill", from: mf)
        XCTAssertEqual(created.overrides?.system, "You are Krill.")
        XCTAssertEqual(created.overrides?.parameters["temperature"], "0.2")
        XCTAssertEqual(created.family, .llama)
        XCTAssertTrue(reg.hasModel("my-krill"))
        // Blob dir is a real directory whose entries symlink into the base
        // (no weight copy). Seed a base file to verify the link.
        try "w".write(to: reg.modelPath("base-llama")
            .appendingPathComponent("model.safetensors"),
            atomically: true, encoding: .utf8)
        _ = try reg.createModel(name: "my-krill2", from: mf)
        let linked = reg.modelPath("my-krill2")
            .appendingPathComponent("model.safetensors")
        let attrs = try FileManager.default
            .attributesOfItem(atPath: linked.path)
        XCTAssertEqual(attrs[.type] as? FileAttributeType, .typeSymbolicLink)
    }

    func testCreateModelRejectsMissingBase() {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-create2-\(UUID().uuidString)")
        let reg = Registry(baseDir: base)
        let mf = try! ModelfileParser.parse("FROM not-installed\nSYSTEM hi")
        XCTAssertThrowsError(try reg.createModel(name: "x", from: mf))
    }

    func testOllamaEnvAliasesWithKrillPrecedence() {
        setenv("OLLAMA_CONTEXT_LENGTH", "4096", 1)
        setenv("OLLAMA_KEEP_ALIVE", "10m", 1)
        setenv("OLLAMA_NUM_PARALLEL", "3", 1)
        setenv("OLLAMA_ORIGINS", "https://a.com, https://b.com", 1)
        setenv("OLLAMA_HOST", "http://0.0.0.0:12345", 1)
        setenv("OLLAMA_MODELS", "/tmp/ollama-models", 1)
        setenv("KRILL_CONTEXT_LENGTH", "8192", 1)  // KRILL_* must win
        defer {
            for k in ["OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE",
                      "OLLAMA_NUM_PARALLEL", "OLLAMA_ORIGINS", "OLLAMA_HOST",
                      "OLLAMA_MODELS", "KRILL_CONTEXT_LENGTH"] { unsetenv(k) }
        }
        let cfg = KrillConfig.load()
        XCTAssertEqual(cfg.contextLength, 8192)          // KRILL_ over OLLAMA_
        XCTAssertEqual(cfg.keepAlive, "10m")
        XCTAssertEqual(cfg.numParallel, 3)
        XCTAssertEqual(cfg.origins, ["https://a.com", "https://b.com"])
        XCTAssertEqual(cfg.serverHost, "0.0.0.0")
        XCTAssertEqual(cfg.serverPort, 12345)
        XCTAssertEqual(cfg.modelsDir, "/tmp/ollama-models")  // OLLAMA_MODELS
    }

    func testRegistryModelsDirOverrideRootsManifestsAndBlobs() {
        // OLLAMA_MODELS / KRILL_MODELS_DIR points directly at the models dir;
        // manifests/blobs must root there exactly (Ollama drop-in layout),
        // not under `<dir>/models`.
        let md = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-md-\(UUID().uuidString)")
        let reg = Registry(modelsDir: md)
        XCTAssertEqual(reg.modelsDir, md)
        XCTAssertEqual(reg.manifestsDir, md.appendingPathComponent("manifests"))
        XCTAssertEqual(reg.blobsDir, md.appendingPathComponent("blobs"))
        // Default path (no override) still derives `<base>/models`.
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-base-\(UUID().uuidString)")
        let def = Registry(baseDir: base)
        XCTAssertEqual(def.modelsDir, base.appendingPathComponent("models"))
    }

    func testAliasMapResolvesKnownModel() {
        let resolved = AliasMap.resolve("llama-3.2-3b")
        XCTAssertNotNil(resolved)
        XCTAssertEqual(resolved?.name, "llama-3.2-3b")
        XCTAssertEqual(resolved?.family, .llama)
        XCTAssertEqual(resolved?.params, "3B")
        XCTAssertEqual(resolved?.quant, "4bit")
        XCTAssert(resolved!.repo.contains("mlx-community"))
    }

    func testAliasMapResolvesHFPath() {
        let resolved = AliasMap.resolve("mlx-community/some-model-4bit")
        XCTAssertNotNil(resolved)
        XCTAssertEqual(resolved?.repo, "mlx-community/some-model-4bit")
    }

    func testAliasMapReturnsNilForUnknown() {
        let resolved = AliasMap.resolve("nonexistent-model")
        XCTAssertNil(resolved)
    }

    func testAliasMapCaseInsensitive() {
        let resolved = AliasMap.resolve("Llama-3.2-3B")
        XCTAssertNotNil(resolved)
        XCTAssertEqual(resolved?.name, "llama-3.2-3b")
    }

    func testModelFamilyDetection() {
        let llamaConfig: [String: Any] = ["architectures": ["LlamaForCausalLM"]]
        XCTAssertEqual(ModelFamily.detect(from: llamaConfig), .llama)

        let qwenConfig: [String: Any] = ["architectures": ["Qwen2ForCausalLM"]]
        XCTAssertEqual(ModelFamily.detect(from: qwenConfig), .qwen)

        let gemmaConfig: [String: Any] = ["model_type": "gemma2"]
        XCTAssertEqual(ModelFamily.detect(from: gemmaConfig), .gemma)
    }

    func testRegistryEmptyList() {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-test-\(UUID().uuidString)")
        let registry = Registry(baseDir: tempDir)
        XCTAssertTrue(registry.listModels().isEmpty)
        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    func testManifestSaveAndLoad() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-test-\(UUID().uuidString)")
        let registry = Registry(baseDir: tempDir)

        let manifest = ModelManifest(
            name: "test-model",
            family: .llama,
            params: "3B",
            quant: "4bit",
            source: "org/test",
            context: 4096,
            files: [ModelFile(path: "model.safetensors", sha256: "abc123", sizeBytes: 1000)],
            chatTemplate: "llama",
            sizeBytes: 1000
        )

        try registry.saveManifest(manifest)
        XCTAssertTrue(registry.hasModel("test-model"))

        let loaded = registry.getModel("test-model")
        XCTAssertNotNil(loaded)
        XCTAssertEqual(loaded?.name, "test-model")
        XCTAssertEqual(loaded?.family, .llama)
        XCTAssertEqual(loaded?.sizeBytes, 1000)

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }

    func testRegistryRemoveModel() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-test-\(UUID().uuidString)")
        let registry = Registry(baseDir: tempDir)

        let manifest = ModelManifest(
            name: "removable",
            family: .qwen,
            params: "7B",
            quant: "4bit",
            source: "org/test",
            context: 32768,
            files: [],
            chatTemplate: "qwen",
            sizeBytes: 5000
        )
        try registry.saveManifest(manifest)
        XCTAssertTrue(registry.hasModel("removable"))

        try registry.removeModel("removable")
        XCTAssertFalse(registry.hasModel("removable"))

        // Cleanup
        try? FileManager.default.removeItem(at: tempDir)
    }
}
