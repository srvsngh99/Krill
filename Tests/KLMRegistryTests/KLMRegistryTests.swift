import XCTest
@testable import KLMRegistry

final class KLMRegistryTests: XCTestCase {
    func testOllamaEnvAliasesWithKrillPrecedence() {
        setenv("OLLAMA_CONTEXT_LENGTH", "4096", 1)
        setenv("OLLAMA_KEEP_ALIVE", "10m", 1)
        setenv("OLLAMA_NUM_PARALLEL", "3", 1)
        setenv("OLLAMA_ORIGINS", "https://a.com, https://b.com", 1)
        setenv("OLLAMA_HOST", "http://0.0.0.0:12345", 1)
        setenv("KRILL_CONTEXT_LENGTH", "8192", 1)  // KRILL_* must win
        defer {
            for k in ["OLLAMA_CONTEXT_LENGTH", "OLLAMA_KEEP_ALIVE",
                      "OLLAMA_NUM_PARALLEL", "OLLAMA_ORIGINS", "OLLAMA_HOST",
                      "KRILL_CONTEXT_LENGTH"] { unsetenv(k) }
        }
        let cfg = KrillConfig.load()
        XCTAssertEqual(cfg.contextLength, 8192)          // KRILL_ over OLLAMA_
        XCTAssertEqual(cfg.keepAlive, "10m")
        XCTAssertEqual(cfg.numParallel, 3)
        XCTAssertEqual(cfg.origins, ["https://a.com", "https://b.com"])
        XCTAssertEqual(cfg.serverHost, "0.0.0.0")
        XCTAssertEqual(cfg.serverPort, 12345)
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
