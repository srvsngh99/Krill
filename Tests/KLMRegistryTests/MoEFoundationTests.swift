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

    func testMoEIsExperimental() {
        XCTAssertEqual(ModelCapabilities.supportTier(for: .moe), .experimental)
    }

    // MARK: - Stable raw value

    func testRawValueMatchesEnumName() {
        // .moe carries no explicit raw value; the Codable raw is
        // the enum case name. /api/show emits this string, so
        // clients can pin it.
        XCTAssertEqual(ModelFamily.moe.rawValue, "moe")
    }
}
