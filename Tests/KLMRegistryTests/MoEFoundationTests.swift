import XCTest
@testable import KLMRegistry

/// WS6 foundation: MoE family detection + capability metadata +
/// experimental tier + alias entries. The native router/expert
/// runtime is NOT in this PR; these tests pin the foundation
/// contract so the follow-up runtime PR cannot silently drop the
/// rejection path or the capability surface.
final class MoEFoundationTests: XCTestCase {

    // MARK: - Family detection

    func testDetectMixtralFromArchitectures() {
        let cfg: [String: Any] = ["architectures": ["MixtralForCausalLM"]]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .moe)
    }

    func testDetectMixtralFromModelType() {
        let cfg: [String: Any] = ["model_type": "mixtral"]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .moe)
    }

    func testDetectQwen3MoEFromArchitectures() {
        let cfg: [String: Any] = ["architectures": ["Qwen3MoeForCausalLM"]]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .moe,
            "Qwen 3 MoE must NOT route to .qwen (the dense text loader would crash on router/expert keys)")
    }

    func testDetectQwen3MoEFromModelType() {
        let cfg: [String: Any] = ["model_type": "qwen3_moe"]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .moe)
    }

    func testDetectionPrefersMoEOverGenericQwen() {
        // The Qwen 3 MoE arch string contains "qwen"; detection
        // must match MoE first so the dense Qwen loader does not
        // claim it.
        let cfg: [String: Any] = [
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .moe)
    }

    func testDetectionPrefersMoEOverGenericMistral() {
        // Same shape: "mixtral" contains "mistral" prefix; detection
        // must match MoE first.
        let cfg: [String: Any] = [
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .moe)
    }

    // MARK: - Capability declaration

    func testMoEDeclaresTextAndMoEAndTools() {
        let caps = ModelCapabilities.capabilities(for: .moe)
        XCTAssertTrue(caps.contains(.textGeneration))
        XCTAssertTrue(caps.contains(.moe),
            "MoE family must declare the moe capability so clients can opt-out at the request layer")
        XCTAssertTrue(caps.contains(.tools),
            "Initial MoE targets (Mixtral, Qwen 3 MoE) inherit a parity-tested tool template")
        XCTAssertFalse(caps.contains(.visionInput))
        XCTAssertFalse(caps.contains(.audioInput))
    }

    // MARK: - Support tier

    func testMoEIsCompatibleFallback() {
        // The WS6 runtime PR ships the Python sidecar / mlx-lm
        // path. Promotion from compatibleFallback to
        // productionNative requires native Swift+MLX router +
        // expert FFN dispatch.
        XCTAssertEqual(
            ModelCapabilities.supportTier(for: .moe),
            .compatibleFallback)
    }

    // MARK: - Stable raw value

    func testRawValueMatchesEnumName() {
        // .moe carries no explicit raw value; the Codable raw is
        // the enum case name. /api/show emits this string, so
        // clients can pin it.
        XCTAssertEqual(ModelFamily.moe.rawValue, "moe")
    }

    // MARK: - Native MoE dispatch helper (WS6 runtime)

    /// `nativeMoEDispatchSupported(at:)` decides at request time
    /// whether an MoE manifest can go through the native
    /// Swift+MLX runtime or must use the Python sidecar bridge.
    /// The server's MoE dispatch uses this; the tests pin the
    /// contract so the bridge-fallback path cannot silently take
    /// over for Qwen 3 MoE, and the native path cannot silently
    /// claim Mixtral.
    private func writeConfig(_ json: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-moe-native-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testNativeDispatchSupportedForQwen3MoE() throws {
        let dir = try writeConfig([
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
            "Qwen 3 MoE has a native runtime in this build and must "
            + "skip the bridge dispatch")
    }

    func testNativeDispatchNotSupportedForMixtral() throws {
        let dir = try writeConfig([
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
            "Mixtral has no native runtime yet; must use the bridge")
    }

    func testNativeDispatchNotSupportedForOLMoE() throws {
        let dir = try writeConfig([
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
            "OLMoE has no native runtime yet; must use the bridge")
    }

    func testNativeDispatchReturnsFalseForMissingConfig() {
        // A directory with no config.json must NOT claim native
        // support — the caller (server) will fall back to the
        // bridge, which emits a clearer error than a dense loader
        // crashing on missing safetensors.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-moe-missing-\(UUID().uuidString)")
        XCTAssertFalse(nativeMoEDispatchSupported(at: dir))
    }
}
