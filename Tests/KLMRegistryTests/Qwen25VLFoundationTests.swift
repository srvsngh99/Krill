import XCTest
@testable import KLMRegistry

/// WS5 foundation: family detection + capability metadata + support
/// tier exist for Qwen 2.5-VL, but the loader still refuses to
/// instantiate. These tests pin the foundation contract so the
/// follow-up PR that lands the native runtime does not silently drop
/// the rejection path.
final class Qwen25VLFoundationTests: XCTestCase {

    // MARK: - Family detection

    func testDetectFromArchitectures() {
        let configFromArch: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"]
        ]
        XCTAssertEqual(ModelFamily.detect(from: configFromArch), .qwen25vl)
    }

    func testDetectFromModelType() {
        let configFromModelType: [String: Any] = ["model_type": "qwen2_5_vl"]
        XCTAssertEqual(ModelFamily.detect(from: configFromModelType), .qwen25vl)
    }

    func testDetectQwen2VL() {
        // The older Qwen2-VL family routes to the same MLX adapter
        // path. Detection accepts both architectures so a future
        // shared loader does not need a second registry case.
        let cfg: [String: Any] = ["model_type": "qwen2_vl"]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .qwen25vl)
    }

    func testDetectionPrefersVLOverGenericQwen() {
        // The VL architecture string contains "qwen"; the detection
        // order must match VL FIRST so the text loader does not
        // claim the multimodal checkpoint and silently drop the
        // vision tower weights.
        let cfg: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .qwen25vl)
    }

    // MARK: - Capability declaration

    func testQwen25VLDeclaresTextAndVision() {
        let caps = ModelCapabilities.capabilities(for: .qwen25vl)
        XCTAssertTrue(caps.contains(.textGeneration))
        XCTAssertTrue(caps.contains(.visionInput))
        XCTAssertFalse(caps.contains(.audioInput),
            "WS5 scope is image-only; do not promise audioInput")
        XCTAssertTrue(caps.contains(.tools),
            "Qwen 2.5-VL inherits the qwen-family parity-tested tool template")
    }

    // MARK: - Support tier

    func testQwen25VLIsExperimental() {
        // The vision tower + multimodal forward have not landed yet.
        // Tier MUST be experimental until the follow-up PR ships
        // them and adds a fixture-changes-output smoke + benchmark.
        XCTAssertEqual(ModelCapabilities.supportTier(for: .qwen25vl), .experimental)
    }

    // MARK: - Raw value stability (used by /api/tags, /api/show)

    func testRawValueMatchesOllamaSnakeCase() {
        // Ollama clients tag families with snake_case identifiers.
        // Matching this lets the discovery endpoints round-trip
        // without a translation table.
        XCTAssertEqual(ModelFamily.qwen25vl.rawValue, "qwen2_5_vl")
    }
}
