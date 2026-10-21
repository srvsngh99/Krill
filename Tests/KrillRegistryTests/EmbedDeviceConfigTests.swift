import XCTest
@testable import KrillRegistry

/// `embed_device` / `KRILL_EMBED_DEVICE`: parsing and precedence.
/// Precedence (highest first): request `device` field > env > config.toml > default `gpu`.
/// Krill has no per-model config section, so there is no per-model layer between env and file.
final class EmbedDeviceConfigTests: XCTestCase {

    func testParseIsCaseAndSpaceInsensitive() {
        XCTAssertEqual(EmbedDevice.parse("gpu"), .gpu)
        XCTAssertEqual(EmbedDevice.parse(" ANE "), .ane)
        XCTAssertEqual(EmbedDevice.parse("Auto"), .auto)
        XCTAssertNil(EmbedDevice.parse("npu"))
        XCTAssertNil(EmbedDevice.parse(""))
        XCTAssertNil(EmbedDevice.parse(nil))
    }

    func testDefaultIsGPU() {
        XCTAssertEqual(KrillConfig().embedDevice, .gpu)
    }

    func testTOMLKeySetsDevice() {
        var c = KrillConfig()
        c.mergeFromTOML("keep_alive = \"-1\"\nembed_device = \"ane\"\n")
        XCTAssertEqual(c.embedDevice, .ane)
    }

    func testInvalidTOMLValueIsIgnored() {
        var c = KrillConfig()
        c.mergeFromTOML("embed_device = \"tpu\"\n")
        XCTAssertEqual(c.embedDevice, .gpu, "bad value keeps the default")
    }

    func testEnvOverridesFile() {
        var c = KrillConfig()
        c.mergeFromTOML("embed_device = \"ane\"\n")
        c.mergeFromEnvironment(["KRILL_EMBED_DEVICE": "gpu"])
        XCTAssertEqual(c.embedDevice, .gpu)
        c.mergeFromEnvironment(["KRILL_EMBED_DEVICE": "auto"])
        XCTAssertEqual(c.embedDevice, .auto)
    }

    func testInvalidEnvKeepsFileValue() {
        var c = KrillConfig()
        c.mergeFromTOML("embed_device = \"ane\"\n")
        c.mergeFromEnvironment(["KRILL_EMBED_DEVICE": "nope"])
        XCTAssertEqual(c.embedDevice, .ane)
    }

    func testUnsetEnvLeavesFileValue() {
        var c = KrillConfig()
        c.mergeFromTOML("embed_device = \"auto\"\n")
        c.mergeFromEnvironment([:])
        XCTAssertEqual(c.embedDevice, .auto)
    }

    func testIsWritableAndDisplayed() {
        XCTAssertTrue(KrillConfig.writableKeys.contains("embed_device"))
        var c = KrillConfig()
        c.embedDevice = .ane
        XCTAssertEqual(c.displayPairs().first { $0.key == "embed_device" }?.value, "ane")
    }

    func testSetRejectsInvalidValue() {
        XCTAssertThrowsError(try KrillConfig.set(key: "embed_device", value: "tpu"))
    }

    func testUpsertWritesTopLevelKey() {
        let out = KrillConfig.upsertTOML("keep_alive = \"5m\"\n", key: "embed_device", value: "ane")
        XCTAssertTrue(out.contains("embed_device = \"ane\""))
        var c = KrillConfig()
        c.mergeFromTOML(out)
        XCTAssertEqual(c.embedDevice, .ane, "what set() writes round-trips through the parser")
    }
}
