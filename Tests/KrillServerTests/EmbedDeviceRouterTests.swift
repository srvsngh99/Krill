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
        XCTAssertTrue(d.warning?.contains("not installed") == true)
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

    func testOtherEmbeddersAreNotRouted() {
        let bge = URL(fileURLWithPath: "/m/blobs/bge-base-en")
        let d = route("bge-base-en", dir: bge, setting: .ane)
        XCTAssertEqual(d.device, .gpu)
        XCTAssertEqual(d.directory, bge)
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
