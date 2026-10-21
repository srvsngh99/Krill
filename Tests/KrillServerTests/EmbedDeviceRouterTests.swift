import XCTest
import Foundation
@testable import KrillServer
@testable import KrillEngine
import KrillRegistry

/// Where an embedding request runs, including every fallback.
final class EmbedDeviceRouterTests: XCTestCase {
    let mlx = URL(fileURLWithPath: "/m/blobs/embeddinggemma-2-mlx-4bit-dyn-text")
    let bf16 = URL(fileURLWithPath: "/m/blobs/embeddinggemma-2")
    let pkg = URL(fileURLWithPath: "/m/blobs/embeddinggemma-2-ane-lin8")

    private func route(_ name: String = "embeddinggemma-2-mlx-4bit-dyn-text", dir: URL? = nil,
                       setting: EmbedDevice, override: EmbedDevice? = nil,
                       installed: Bool = true, coreML: Bool = true) -> EmbedDeviceRouter.Decision {
        EmbedDeviceRouter.route(requested: name, requestedDir: dir ?? mlx, setting: setting, override: override,
                                installedPackage: installed ? pkg : nil, coreML: coreML)
    }

    func testGPUSettingNeverUsesThePackage() {
        let d = route(setting: .gpu)
        XCTAssertEqual(d.device, .gpu)
        XCTAssertEqual(d.directory, mlx)
        XCTAssertNil(d.fallbackDirectory)
    }

    func testANEUsesPackageAndFallsBackToTheRequestedBuild() {
        let d = route("embeddinggemma-2-8bit", setting: .ane)
        XCTAssertEqual(d.device, .ane)
        XCTAssertEqual(d.directory, pkg)
        XCTAssertEqual(d.fallbackDirectory, mlx, "long text / media go to the build the client asked for")
    }

    func testEveryQuantizedEG2TextAliasIsRouted() {
        for name in ["embeddinggemma-2-mlx-4bit-dyn-text", "embeddinggemma-2-4bit-dyn-text",
                     "embeddinggemma-2-8bit", "embeddinggemma-2-6bit-dyn"] {
            XCTAssertEqual(route(name, setting: .ane).device, .ane, name)
            XCTAssertEqual(route(name, setting: .auto).device, .ane, name)
        }
    }

    func testFullPrecisionBuildIgnoresTheServerSetting() {
        for setting in [EmbedDevice.ane, .auto] {
            let d = route("embeddinggemma-2", dir: bf16, setting: setting)
            XCTAssertEqual(d.device, .gpu, "corpus/reference build stays on the GPU under \(setting)")
            XCTAssertEqual(d.directory, bf16)
            XCTAssertNil(d.warning)
        }
    }

    func testFullPrecisionBuildMovesOnlyOnAnExplicitRequest() {
        let d = route("embeddinggemma-2", dir: bf16, setting: .gpu, override: .ane)
        XCTAssertEqual(d.device, .ane)
        XCTAssertEqual(d.directory, pkg)
        XCTAssertEqual(d.fallbackDirectory, bf16)
    }

    func testAutoUsesANEWhenAvailable() {
        XCTAssertEqual(route(setting: .auto).device, .ane)
    }

    func testAutoFallsBackSilently() {
        let noPkg = route(setting: .auto, installed: false)
        XCTAssertEqual(noPkg.device, .gpu); XCTAssertNil(noPkg.warning)
        let noML = route(setting: .auto, coreML: false)
        XCTAssertEqual(noML.device, .gpu); XCTAssertNil(noML.warning)
    }

    func testANEWithoutPackageFallsBackLoudly() {
        let d = route(setting: .ane, installed: false)
        XCTAssertEqual(d.device, .gpu)
        XCTAssertEqual(d.directory, mlx)
        XCTAssertTrue(d.warning?.contains("no ANE package is installed for embeddinggemma-2-mlx-4bit-dyn-text") == true)
    }

    func testANEWithoutCoreMLFallsBackLoudly() {
        let d = route(setting: .ane, coreML: false)
        XCTAssertEqual(d.device, .gpu)
        XCTAssertTrue(d.warning?.contains("Core ML") == true)
    }

    func testRequestOverrideBeatsSetting() {
        XCTAssertEqual(route(setting: .ane, override: .gpu).device, .gpu, "a corpus job can pin the GPU build")
        XCTAssertEqual(route(setting: .gpu, override: .ane).device, .ane)
    }

    // MARK: per-model packages (any embedder or reranker)

    private func routeModel(_ name: String, setting: EmbedDevice, override: EmbedDevice? = nil,
                            package: URL?, coreML: Bool = true) -> EmbedDeviceRouter.Decision {
        EmbedDeviceRouter.route(requested: name, requestedDir: URL(fileURLWithPath: "/m/blobs/\(name)"),
                                setting: setting, override: override, installedPackage: package, coreML: coreML)
    }

    func testAnyModelWithAPackageIsRouted() {
        let p = URL(fileURLWithPath: "/m/blobs/bge-base-en-ane-lin8")
        for name in ["bge-base-en", "all-minilm-l6-v2", "gte-modernbert-base", "nomic-embed-text", "bge-reranker-base"] {
            let d = routeModel(name, setting: .ane, package: p)
            XCTAssertEqual(d.device, .ane, name)
            XCTAssertEqual(d.directory, p)
            XCTAssertEqual(d.fallbackDirectory, URL(fileURLWithPath: "/m/blobs/\(name)"), "the GPU build of the same model")
            XCTAssertEqual(routeModel(name, setting: .auto, package: p).device, .ane, name)
            XCTAssertEqual(routeModel(name, setting: .gpu, package: p).device, .gpu, name)
        }
    }

    func testModelsWithoutAPackageStayOnTheGPU() {
        for name in ["bge-base-en", "gte-qwen2-7b", "bge-reranker-base"] {
            let a = routeModel(name, setting: .auto, package: nil)
            XCTAssertEqual(a.device, .gpu); XCTAssertNil(a.warning, "auto is silent")
            let d = routeModel(name, setting: .ane, package: nil)
            XCTAssertEqual(d.device, .gpu)
            XCTAssertEqual(d.directory, URL(fileURLWithPath: "/m/blobs/\(name)"))
            XCTAssertNil(d.fallbackDirectory)
            XCTAssertTrue(d.warning?.contains("no ANE package is installed for \(name)") == true)
            XCTAssertTrue(d.warning?.contains("make ane-package MODEL=\(name)") == true)
        }
    }

    func testRequestDeviceOverrideAppliesToEveryModel() {
        let p = URL(fileURLWithPath: "/m/blobs/bge-reranker-base-ane-lin8")
        XCTAssertEqual(routeModel("bge-reranker-base", setting: .gpu, override: .ane, package: p).device, .ane)
        XCTAssertEqual(routeModel("bge-reranker-base", setting: .ane, override: .gpu, package: p).device, .gpu)
    }

    func testPackageLookupNames() {
        XCTAssertEqual(EmbedDeviceRouter.packageNames(for: "bge-base-en"), ["bge-base-en-ane-lin8", "bge-base-en-ane-fp16"])
        XCTAssertEqual(EmbedDeviceRouter.packageNames(for: "BGE-Reranker-Base"), ["bge-reranker-base-ane-lin8", "bge-reranker-base-ane-fp16"])
        // Every quantized EmbeddingGemma 2 alias shares the one package; a package name is never its own base.
        XCTAssertEqual(EmbedDeviceRouter.packageBase(for: "embeddinggemma-2-8bit"), "embeddinggemma-2")
        XCTAssertEqual(EmbedDeviceRouter.packageBase(for: "embeddinggemma-2"), "embeddinggemma-2")
        XCTAssertEqual(EmbedDeviceRouter.packageBase(for: "nomic-embed-text-v1"), "nomic-embed-text-v1")
    }

    func testInstalledPackageIsLookedUpPerModel() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("route-\(UUID().uuidString)")
        let registry = Registry(modelsDir: root)
        for name in ["bge-base-en-ane-fp16", "bge-small-en-ane-lin8", "bge-small-en-ane-fp16"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent("manifests"), withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: root.appendingPathComponent("manifests/\(name).json"))
            let dir = root.appendingPathComponent("blobs/\(name)")
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: dir.appendingPathComponent(ANEPackageManifest.fileName))
        }
        XCTAssertEqual(EmbedDeviceRouter.installedPackage(for: "bge-base-en", in: registry)?.lastPathComponent, "bge-base-en-ane-fp16")
        XCTAssertEqual(EmbedDeviceRouter.installedPackage(for: "bge-small-en", in: registry)?.lastPathComponent,
                       "bge-small-en-ane-lin8", "int8 is preferred when both exist")
        XCTAssertNil(EmbedDeviceRouter.installedPackage(for: "all-mpnet-base-v2", in: registry))
        XCTAssertNil(EmbedDeviceRouter.installedPackage(for: "embeddinggemma-2-8bit", in: registry))
    }

    func testWarningIsLoggedOncePerModel() {
        let w = EmbedDeviceRouter.WarnOnce()
        XCTAssertTrue(w.first("bge-base-en"))
        XCTAssertFalse(w.first("bge-base-en"))
        XCTAssertFalse(w.first("BGE-Base-EN"))
        XCTAssertTrue(w.first("all-minilm-l6-v2"))
    }

    func testNamingThePackageRunsItRegardlessOfSetting() {
        let d = route("embeddinggemma-2-ane-lin8", dir: pkg, setting: .gpu)
        XCTAssertEqual(d.device, .ane)
        XCTAssertEqual(d.directory, pkg)
    }

    func testEffectiveDevice() {
        XCTAssertEqual(EmbedDeviceRouter.effective(setting: .gpu, packageInstalled: true, coreML: true), .gpu)
        XCTAssertEqual(EmbedDeviceRouter.effective(setting: .ane, packageInstalled: true, coreML: true), .ane)
        XCTAssertEqual(EmbedDeviceRouter.effective(setting: .auto, packageInstalled: false, coreML: true), .gpu)
        XCTAssertEqual(EmbedDeviceRouter.effective(setting: .ane, packageInstalled: true, coreML: false), .gpu)
    }

    func testPackagePreferenceOrderIsInt8First() {
        XCTAssertEqual(EmbedDeviceRouter.packageNames.first, "embeddinggemma-2-ane-lin8")
    }
}
