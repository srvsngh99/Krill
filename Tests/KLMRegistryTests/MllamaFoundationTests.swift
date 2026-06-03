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

    func testDeclaresVisionInput() {
        // Image serving has landed (tile / aspect-ratio preprocessing, sparse
        // cross-attention mask, cross-KV decode driver), so the family now
        // advertises vision input alongside text generation.
        let caps = ModelCapabilities.capabilities(for: .llamaVision)
        XCTAssertTrue(caps.contains(.textGeneration))
        XCTAssertTrue(caps.contains(.visionInput),
            "vision input is advertised now that mllama image serving is wired")
    }

    func testIsExperimental() {
        XCTAssertEqual(ModelCapabilities.supportTier(for: .llamaVision), .experimental)
    }

    func testRawValueIsLlamaVision() {
        XCTAssertEqual(ModelFamily.llamaVision.rawValue, "llama_vision")
    }
}
