import XCTest
@testable import KLMRegistry

/// Llama-3.2-Vision (mllama) registry contract: family detection + capability
/// metadata + support tier. The native runtime (tiled vision tower + gated
/// cross-attention text decoder + projector) is mlx-vlm logit-parity verified;
/// image-serving wiring (tile preprocessing + cross-KV decode driver) is a
/// follow-up, so vision input is NOT advertised yet.
final class MllamaFoundationTests: XCTestCase {

    func testDetectFromArchitectures() {
        let cfg: [String: Any] = ["architectures": ["MllamaForConditionalGeneration"]]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .llamaVision)
    }

    func testDetectFromModelType() {
        XCTAssertEqual(ModelFamily.detect(from: ["model_type": "mllama"]), .llamaVision)
    }

    func testDetectionPrefersMllamaOverGenericLlama() {
        let cfg: [String: Any] = [
            "architectures": ["MllamaForConditionalGeneration"],
            "model_type": "mllama",
        ]
        XCTAssertEqual(ModelFamily.detect(from: cfg), .llamaVision)
    }

    func testDeclaresTextOnlyForNow() {
        // Runtime + parity have landed; serving (tile preprocessing + cross-KV
        // decode driver) is a follow-up, so vision input is intentionally NOT
        // advertised yet (an image would otherwise be silently dropped).
        let caps = ModelCapabilities.capabilities(for: .llamaVision)
        XCTAssertTrue(caps.contains(.textGeneration))
        XCTAssertFalse(caps.contains(.visionInput),
            "vision input must wait for the image-serving wiring follow-up")
    }

    func testIsExperimental() {
        XCTAssertEqual(ModelCapabilities.supportTier(for: .llamaVision), .experimental)
    }

    func testRawValueIsLlamaVision() {
        XCTAssertEqual(ModelFamily.llamaVision.rawValue, "llama_vision")
    }
}
