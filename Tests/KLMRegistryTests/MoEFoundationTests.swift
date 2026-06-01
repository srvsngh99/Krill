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
    /// over for the native families (Qwen 3 MoE, Mixtral), and the
    /// native path cannot silently claim a not-yet-ported family
    /// (Qwen2-MoE / OLMoE / DeepSeek).
    private func writeConfig(_ json: [String: Any]) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-moe-native-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    /// Scoped env-var setter so the test process restores the prior
    /// value after the assertion block, even on failure.
    private func withEnv(_ key: String, _ value: String?, _ body: () throws -> Void) rethrows {
        let prior = ProcessInfo.processInfo.environment[key]
        if let value {
            setenv(key, value, 1)
        } else {
            unsetenv(key)
        }
        defer {
            if let prior {
                setenv(key, prior, 1)
            } else {
                unsetenv(key)
            }
        }
        try body()
    }

    func testNativeDispatchSupportedForQwen3MoEByDefault() throws {
        let dir = try writeConfig([
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        // Native is the default now: an unset KRILL_NATIVE_MOE and an
        // explicit "1" both claim native support for a Qwen 3 MoE
        // checkpoint.
        try withEnv("KRILL_NATIVE_MOE", nil) {
            XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
                "Qwen 3 MoE native runtime is the default; an unset "
                + "KRILL_NATIVE_MOE must claim native support")
        }
        try withEnv("KRILL_NATIVE_MOE", "1") {
            XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
                "KRILL_NATIVE_MOE=1 must also claim native support")
        }
    }

    func testNativeDispatchOptOutGateIsHonored() throws {
        // KRILL_NATIVE_MOE=0 is the opt-out: even for Qwen 3 MoE the
        // helper returns false so the server routes to the legacy
        // bridge. Pins the opt-out so it cannot silently disappear.
        let dir = try writeConfig([
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_MOE", "0") {
            XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
                "KRILL_NATIVE_MOE=0 must opt out of the native runtime "
                + "and route Qwen 3 MoE through the bridge")
        }
    }

    func testQwen3MoECheckpointPromotesToProductionNative() throws {
        // The family-only tier stays the conservative floor (the .moe
        // family spans bridge-only members too), but a Qwen 3 MoE
        // checkpoint the native runtime serves promotes to
        // productionNative; the opt-out drops it back to the bridge
        // tier; a non-native MoE family stays on the floor regardless.
        let qwen3 = try writeConfig([
            "architectures": ["Qwen3MoeForCausalLM"],
            "model_type": "qwen3_moe",
        ])
        defer { try? FileManager.default.removeItem(at: qwen3) }
        let deepseek = try writeConfig([
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
        ])
        defer { try? FileManager.default.removeItem(at: deepseek) }

        XCTAssertEqual(ModelCapabilities.supportTier(for: .moe), .compatibleFallback)
        try withEnv("KRILL_NATIVE_MOE", nil) {
            XCTAssertEqual(
                ModelCapabilities.supportTier(for: .moe, at: qwen3),
                .productionNative,
                "A Qwen 3 MoE checkpoint on the native default must "
                + "report productionNative")
            XCTAssertEqual(
                ModelCapabilities.supportTier(for: .moe, at: deepseek),
                .compatibleFallback,
                "A not-yet-ported MoE family (DeepSeek) stays on the bridge floor")
        }
        try withEnv("KRILL_NATIVE_MOE", "0") {
            XCTAssertEqual(
                ModelCapabilities.supportTier(for: .moe, at: qwen3),
                .compatibleFallback,
                "The opt-out drops a Qwen 3 MoE checkpoint back to the "
                + "bridge tier")
        }
    }

    func testNativeDispatchSupportedForQwen2MoEByDefault() throws {
        // Qwen 2 MoE now has a native runtime; the helper claims native
        // support by default and the KRILL_NATIVE_MOE=0 opt-out routes it to
        // the bridge for one transitional release.
        let dir = try writeConfig([
            "architectures": ["Qwen2MoeForCausalLM"],
            "model_type": "qwen2_moe",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_MOE", nil) {
            XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
                "Qwen 2 MoE native runtime is the default")
        }
        try withEnv("KRILL_NATIVE_MOE", "0") {
            XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
                "KRILL_NATIVE_MOE=0 opts Qwen 2 MoE out to the bridge")
        }
    }

    func testNativeDispatchSupportedForMixtralByDefault() throws {
        // Mixtral now has a native runtime, so the helper claims native
        // support by default; the KRILL_NATIVE_MOE=0 opt-out still routes
        // it through the bridge for one transitional release.
        let dir = try writeConfig([
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_MOE", nil) {
            XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
                "Mixtral native runtime is the default")
        }
        try withEnv("KRILL_NATIVE_MOE", "1") {
            XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
                "KRILL_NATIVE_MOE=1 must also claim native support for Mixtral")
        }
        try withEnv("KRILL_NATIVE_MOE", "0") {
            XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
                "KRILL_NATIVE_MOE=0 opts Mixtral out to the bridge")
        }
    }

    func testNativeDispatchSupportedForOLMoEByDefault() throws {
        // OLMoE now has a native runtime.
        let dir = try writeConfig([
            "architectures": ["OlmoeForCausalLM"],
            "model_type": "olmoe",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_MOE", nil) {
            XCTAssertTrue(nativeMoEDispatchSupported(at: dir),
                "OLMoE native runtime is the default")
        }
        try withEnv("KRILL_NATIVE_MOE", "0") {
            XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
                "KRILL_NATIVE_MOE=0 opts OLMoE out to the bridge")
        }
    }

    func testNativeDispatchNotSupportedForDeepSeek() throws {
        let dir = try writeConfig([
            "architectures": ["DeepseekV3ForCausalLM"],
            "model_type": "deepseek_v3",
        ])
        defer { try? FileManager.default.removeItem(at: dir) }
        try withEnv("KRILL_NATIVE_MOE", "1") {
            XCTAssertFalse(nativeMoEDispatchSupported(at: dir),
                "DeepSeek has no native runtime yet; must use the bridge")
        }
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
