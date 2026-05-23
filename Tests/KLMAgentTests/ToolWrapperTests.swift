import XCTest
import Foundation
import KLMRegistry
@testable import KLMAgent

final class ToolWrapperTests: XCTestCase {

    // MARK: - Shared fixtures

    static let staticFixtureHardware: HardwareInfo = .init(
        arch: "arm64", chip: "Apple M4 Pro",
        totalRAMBytes: 32 * 1_073_741_824,
        freeRAMBytes: 16 * 1_073_741_824,
        cpuCores: 12, gpuCores: 16,
        freeDiskBytes: 500 * 1_073_741_824,
        metalAvailable: true)

    private var fixtureHardware: HardwareInfo { Self.staticFixtureHardware }

    private func makeTempRegistry() -> Registry {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("klmagent-test-\(UUID().uuidString)")
        let reg = Registry(baseDir: dir)
        try? reg.ensureDirectories()
        return reg
    }

    private func seedFakeModel(
        _ reg: Registry, name: String, family: ModelFamily,
        params: String, sizeBytes: Int64
    ) {
        let manifest = ModelManifest(
            name: name, family: family, params: params, quant: "4bit",
            source: "fixture/\(name)", context: 8192,
            files: [],
            chatTemplate: "hermes",
            sizeBytes: sizeBytes,
            pulledAt: Date(timeIntervalSince1970: 1_700_000_000))
        try? reg.saveManifest(manifest)
    }

    private func parseJSON(_ s: String) -> [String: Any] {
        guard let data = s.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return [:] }
        return obj
    }

    // MARK: - hardware_info

    func testHardwareInfoToolReturnsCompactJSON() async throws {
        let hw = fixtureHardware
        let tool = HardwareInfoTool(snapshot: { hw })
        let result = try await tool.execute(arguments: [:])
        let parsed = parseJSON(result)
        XCTAssertEqual(parsed["arch"] as? String, "arm64")
        XCTAssertEqual(parsed["chip"] as? String, "Apple M4 Pro")
        XCTAssertEqual(parsed["total_ram_gb"] as? Double, 32.0)
        XCTAssertEqual(parsed["metal_available"] as? Bool, true)
        XCTAssertEqual(parsed["gpu_cores"] as? Int, 16)
    }

    // MARK: - list_local_models

    func testListLocalModelsToolReturnsInstalledManifests() async throws {
        let reg = makeTempRegistry()
        seedFakeModel(reg, name: "llama-3.2-3b", family: .llama,
                      params: "3B", sizeBytes: 1_900_000_000)
        seedFakeModel(reg, name: "qwen2.5-1.5b", family: .qwen,
                      params: "1.5B", sizeBytes: 950_000_000)
        let tool = ListLocalModelsTool(registry: reg)
        let result = try await tool.execute(arguments: [:])
        let parsed = parseJSON(result)
        XCTAssertEqual(parsed["count"] as? Int, 2)
        let names = (parsed["models"] as? [[String: Any]])?
            .compactMap { $0["name"] as? String } ?? []
        XCTAssertEqual(Set(names), ["llama-3.2-3b", "qwen2.5-1.5b"])
    }

    func testListLocalModelsOnEmptyRegistry() async throws {
        let reg = makeTempRegistry()
        let tool = ListLocalModelsTool(registry: reg)
        let result = try await tool.execute(arguments: [:])
        let parsed = parseJSON(result)
        XCTAssertEqual(parsed["count"] as? Int, 0)
    }

    // MARK: - model_info

    func testModelInfoIncludesCapabilitiesAndFit() async throws {
        let reg = makeTempRegistry()
        seedFakeModel(reg, name: "llama-3.2-3b", family: .llama,
                      params: "3B", sizeBytes: 1_900_000_000)
        let hw = fixtureHardware
        let tool = ModelInfoTool(registry: reg, hardware: { hw })
        let result = try await tool.execute(
            arguments: ["name": "llama-3.2-3b"])
        let parsed = parseJSON(result)
        XCTAssertEqual(parsed["family"] as? String, "llama")
        XCTAssertEqual(parsed["fit"] as? String, "comfortable")
        let caps = (parsed["capabilities"] as? [String]) ?? []
        XCTAssertTrue(caps.contains("textGeneration"))
        XCTAssertTrue(caps.contains("tools"))
    }

    func testModelInfoThrowsOnUnknownModel() async {
        let reg = makeTempRegistry()
        let tool = ModelInfoTool(registry: reg, hardware: { Self.staticFixtureHardware })
        do {
            _ = try await tool.execute(arguments: ["name": "ghost"])
            XCTFail("expected throw")
        } catch let err as RegistryToolError {
            XCTAssertEqual(err, .notInstalled("ghost"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - remove_model

    func testRemoveModelDeletesManifest() async throws {
        let reg = makeTempRegistry()
        seedFakeModel(reg, name: "tiny", family: .qwen,
                      params: "0.5B", sizeBytes: 500_000_000)
        XCTAssertTrue(reg.hasModel("tiny"))
        let tool = RemoveModelTool(registry: reg)
        let result = try await tool.execute(arguments: ["name": "tiny"])
        XCTAssertTrue(result.contains("Removed tiny"))
        XCTAssertFalse(reg.hasModel("tiny"))
    }

    // MARK: - list_remote_catalog

    func testListRemoteCatalogReturnsBuiltInAliasMap() async throws {
        let tool = ListRemoteCatalogTool()
        let result = try await tool.execute(arguments: [:])
        let parsed = parseJSON(result)
        XCTAssertGreaterThan((parsed["count"] as? Int) ?? 0, 10)
    }

    func testListRemoteCatalogFiltersByCapability() async throws {
        let tool = ListRemoteCatalogTool()
        let result = try await tool.execute(
            arguments: ["capability": "audio_input"])
        let parsed = parseJSON(result)
        let entries = (parsed["entries"] as? [[String: Any]]) ?? []
        // Audio is only declared on gemma4 today.
        for e in entries {
            XCTAssertEqual(e["family"] as? String, "gemma4")
        }
        XCTAssertFalse(entries.isEmpty)
    }

    // MARK: - recommend_model

    func testRecommendModelReturnsShortlist() async throws {
        let hw = fixtureHardware
        let tool = RecommendModelTool(hardware: { hw })
        let result = try await tool.execute(
            arguments: ["capability": "text_generation", "limit": 3])
        let parsed = parseJSON(result)
        XCTAssertEqual(parsed["capability"] as? String, "text_generation")
        let candidates = (parsed["candidates"] as? [[String: Any]]) ?? []
        XCTAssertGreaterThan(candidates.count, 0)
        XCTAssertLessThanOrEqual(candidates.count, 3)
        for c in candidates {
            XCTAssertNotNil(c["fit"])
            XCTAssertNotNil(c["alias"])
        }
    }

    func testRecommendModelRejectsUnknownCapability() async {
        let hw = fixtureHardware
        let tool = RecommendModelTool(hardware: { hw })
        do {
            _ = try await tool.execute(
                arguments: ["capability": "telepathy"])
            XCTFail("expected throw")
        } catch let err as RecommendModelToolError {
            XCTAssertEqual(err, .unknownCapability("telepathy"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - disk_usage

    func testDiskUsageSummarizesInstalledModels() async throws {
        let reg = makeTempRegistry()
        seedFakeModel(reg, name: "a", family: .llama,
                      params: "3B", sizeBytes: 2_000_000_000)
        seedFakeModel(reg, name: "b", family: .qwen,
                      params: "1.5B", sizeBytes: 1_000_000_000)
        let hw = fixtureHardware
        let tool = DiskUsageTool(registry: reg, hardware: { hw })
        let result = try await tool.execute(arguments: [:])
        let parsed = parseJSON(result)
        XCTAssertEqual(parsed["model_count"] as? Int, 2)
        let totalGB = parsed["total_used_gb"] as? Double ?? 0
        XCTAssertGreaterThan(totalGB, 2.7)  // 3 GB raw
        XCTAssertLessThan(totalGB, 3.0)
        let bm = (parsed["by_model"] as? [[String: Any]]) ?? []
        XCTAssertEqual(bm.first?["name"] as? String, "a",
                       "largest model first")
    }

    // MARK: - daemon tools with fixture ops

    struct FakeDaemonOps: AgentDaemonOps {
        let canned: AgentDaemonStatus?
        let loadFails: Bool
        func status() async -> AgentDaemonStatus? { canned }
        func loadModel(_ name: String) async throws -> String {
            if loadFails { throw AgentDaemonError.transport("offline") }
            return "Loaded \(name)."
        }
        func unloadModel() async throws -> String {
            return "Unloaded."
        }
    }

    func testServerStatusReportsRunningFalseWhenProbeFails() async throws {
        let tool = ServerStatusTool(
            ops: FakeDaemonOps(canned: nil, loadFails: false))
        let parsed = parseJSON(try await tool.execute(arguments: [:]))
        XCTAssertEqual(parsed["running"] as? Bool, false)
        XCTAssertNotNil(parsed["hint"])
    }

    func testServerStatusReportsLoadedModelFields() async throws {
        let canned = AgentDaemonStatus(
            modelLoaded: true, model: "qwen2.5-1.5b",
            memoryMB: 1024, modelUptimeSeconds: 99)
        let tool = ServerStatusTool(
            ops: FakeDaemonOps(canned: canned, loadFails: false))
        let parsed = parseJSON(try await tool.execute(arguments: [:]))
        XCTAssertEqual(parsed["running"] as? Bool, true)
        XCTAssertEqual(parsed["model"] as? String, "qwen2.5-1.5b")
        XCTAssertEqual(parsed["memory_mb"] as? Int, 1024)
    }

    func testLoadModelToolForwardsToOps() async throws {
        let tool = LoadModelTool(
            ops: FakeDaemonOps(canned: nil, loadFails: false))
        let result = try await tool.execute(
            arguments: ["name": "llama-3.2-1b"])
        XCTAssertEqual(result, "Loaded llama-3.2-1b.")
    }

    func testLoadModelToolThrowsOnTransportError() async {
        let tool = LoadModelTool(
            ops: FakeDaemonOps(canned: nil, loadFails: true))
        do {
            _ = try await tool.execute(arguments: ["name": "x"])
            XCTFail("expected throw")
        } catch {
            XCTAssertTrue("\(error)".contains("offline"))
        }
    }

    func testUnloadModelToolForwardsToOps() async throws {
        let tool = UnloadModelTool(
            ops: FakeDaemonOps(canned: nil, loadFails: false))
        let result = try await tool.execute(arguments: [:])
        XCTAssertEqual(result, "Unloaded.")
    }

    // MARK: - pull_model with fixture puller

    struct FakePuller: AgentPuller {
        let resultSubstring: String
        let shouldFail: Bool
        func pull(alias: String) async throws -> String {
            if shouldFail {
                throw AgentPullerError.unknownModel(alias)
            }
            return "Pulled \(alias). \(resultSubstring)"
        }
    }

    func testPullModelToolHappyPath() async throws {
        let tool = PullModelTool(
            puller: FakePuller(resultSubstring: "1.0 GB", shouldFail: false))
        let result = try await tool.execute(
            arguments: ["alias": "llama-3.2-1b"])
        XCTAssertTrue(result.contains("Pulled llama-3.2-1b"))
    }

    func testPullModelToolThrowsOnMissingAlias() async {
        let tool = PullModelTool(
            puller: FakePuller(resultSubstring: "", shouldFail: false))
        do {
            _ = try await tool.execute(arguments: [:])
            XCTFail("expected throw")
        } catch let err as RegistryToolError {
            XCTAssertEqual(err, .missingArgument("alias"))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - start_server / stop_server (instructions-only default)

    func testStartServerReturnsCommandWithModelFlag() async throws {
        let tool = StartServerTool()
        let result = try await tool.execute(
            arguments: ["model": "llama-3.2-1b"])
        XCTAssertTrue(result.contains("krillm serve"))
        XCTAssertTrue(result.contains("--model llama-3.2-1b"))
    }

    func testStartServerOmitsModelFlagWhenAbsent() async throws {
        let tool = StartServerTool()
        let result = try await tool.execute(arguments: [:])
        XCTAssertTrue(result.contains("krillm serve"))
        XCTAssertFalse(result.contains("--model"))
    }

    func testStopServerReturnsCommand() async throws {
        let tool = StopServerTool()
        let result = try await tool.execute(arguments: [:])
        XCTAssertTrue(result.contains("krillm stop"))
    }

    // MARK: - DefaultToolset aggregator

    func testDefaultToolsetWiresAllThirteen() {
        let reg = makeTempRegistry()
        let toolset = DefaultOperatorToolset.make(
            registry: reg,
            daemonOps: FakeDaemonOps(canned: nil, loadFails: false),
            puller: FakePuller(resultSubstring: "", shouldFail: false))
        XCTAssertEqual(toolset.tools.count, 13)
        let names = Set(toolset.names)
        for expected in [
            "hardware_info", "list_local_models", "list_remote_catalog",
            "model_info", "recommend_model", "pull_model", "remove_model",
            "server_status", "load_model", "unload_model",
            "start_server", "stop_server", "disk_usage",
        ] {
            XCTAssertTrue(names.contains(expected),
                          "default toolset missing \(expected)")
        }
    }
}
