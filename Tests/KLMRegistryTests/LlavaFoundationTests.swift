import XCTest
@testable import KLMRegistry

/// LLaVA-1.5 registry contract: family detection + capability metadata +
/// support tier. The native runtime (CLIP + projector + Llama) landed in PR
/// #129; the engine image-serving wiring (this PR) registers the family so a
/// `LlavaForConditionalGeneration` checkpoint is detected, advertises vision,
/// and routes through the dense engine. These pin that contract.
final class LlavaFoundationTests: XCTestCase {

    // MARK: - Family detection

    func testDetectFromArchitectures() {
        let cfg: [String: Any] = [
            "architectures": ["LlavaForConditionalGeneration"]
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .llava)
    }

    func testDetectFromModelType() {
        let cfg: [String: Any] = ["model_type": "llava"]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .llava)
    }

    func testDetectionPrefersLlavaOverGenericLlama() {
        // LLaVA's text backbone is Llama, but the arch string is
        // `LlavaForConditionalGeneration` (it does NOT contain "llama"), so
        // the llava arm must claim it before the generic llama arm and the
        // checkpoint must not silently load as a text-only Llama (dropping the
        // vision tower + projector weights).
        let cfg: [String: Any] = [
            "architectures": ["LlavaForConditionalGeneration"],
            "model_type": "llava",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .llava)
    }

    // MARK: - Capability declaration

    func testLlavaDeclaresTextAndVision() {
        let caps = ModelCapabilities.capabilities(for: .llava)
        XCTAssertTrue(caps.contains(.textGeneration))
        XCTAssertTrue(caps.contains(.visionInput))
        XCTAssertFalse(caps.contains(.audioInput),
            "LLaVA-1.5 is image-only; do not promise audioInput")
        XCTAssertFalse(caps.contains(.tools),
            "LLaVA does not ship a parity-tested tool template")
    }

    // MARK: - Support tier

    func testLlavaIsExperimental() {
        // Native + mlx-vlm logit-parity gated, but no serving benchmark gate
        // yet, so `.experimental` (not productionNative).
        XCTAssertEqual(ModelCapabilities.supportTier(for: .llava), .experimental)
    }

    // MARK: - Raw value stability (used by /api/tags, /api/show)

    func testRawValueIsLlava() {
        XCTAssertEqual(ModelFamily.llava.rawValue, "llava")
    }
}
