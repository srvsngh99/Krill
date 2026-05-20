import XCTest
@testable import KLMCore

/// WS5 foundation: the loader recognizes Qwen 2.5-VL architecture
/// AND refuses to instantiate it (no silent text-only fallback that
/// would drop the vision tower weights). This test pins both halves
/// of the contract.
final class Qwen25VLLoaderRejectionTests: XCTestCase {

    func testQwen25VLConfigIsRejectedWithDocumentedError() throws {
        // Write the same architectures / model_type strings the
        // mlx-community Qwen2.5-VL repos ship to a temp dir, point
        // the loader at it, and confirm the explicit
        // `unsupportedArchitecture` error fires before any weights
        // are touched.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-qwen25vl-rejection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let configJSON: [String: Any] = [
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "model_type": "qwen2_5_vl",
            "hidden_size": 2048,
            "vocab_size": 151936,
        ]
        let configData = try JSONSerialization.data(withJSONObject: configJSON)
        try configData.write(to: dir.appendingPathComponent("config.json"))

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture(let msg) = modelError else {
                XCTFail("Expected ModelLoadError.unsupportedArchitecture, got \(error)")
                return
            }
            XCTAssertTrue(msg.contains("Qwen 2.5-VL"),
                "Error must name the family for users debugging the rejection")
            XCTAssertTrue(msg.contains("WS5"),
                "Error must point at the workstream doc for the tracking PR")
            XCTAssertTrue(msg.contains("qwen2.5") || msg.contains("text-only"),
                "Error must suggest a working text-only alternative")
        }
    }

    func testQwen2VLAlsoRejectedWithSameMessage() throws {
        // The older Qwen 2 VL ships an architecture string the
        // detection arm catches too. Confirm it routes through the
        // same rejection rather than silently falling through to
        // the text loader.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-qwen2vl-rejection-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let configJSON: [String: Any] = [
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
            "hidden_size": 2048,
            "vocab_size": 151936,
        ]
        let configData = try JSONSerialization.data(withJSONObject: configJSON)
        try configData.write(to: dir.appendingPathComponent("config.json"))

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .unsupportedArchitecture = modelError else {
                XCTFail("Expected ModelLoadError.unsupportedArchitecture, got \(error)")
                return
            }
        }
    }
}
