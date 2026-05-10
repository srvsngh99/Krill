import XCTest
@testable import KLMEngine

final class PythonFallbackTests: XCTestCase {
    func testAvailabilityCheckReportsMLXVLMStatus() {
        let availability = PythonFallback.checkAvailability()

        XCTAssertFalse(availability.pythonCommand.isEmpty)
        XCTAssertFalse(availability.detail.isEmpty)
        if availability.isAvailable {
            XCTAssertEqual(availability.detail, "mlx-vlm available")
        } else {
            XCTAssertTrue(
                availability.detail.contains("mlx")
                    || availability.detail.contains("Python")
                    || availability.detail.contains("No module named"),
                "Unexpected availability detail: \(availability.detail)")
        }
    }

    func testGeneratedPythonScriptEscapesPromptAndMediaPaths() {
        let script = PythonFallback.buildScript(
            modelPath: #"/tmp/model "quoted""#,
            prompt: #"say "hi"\nthen continue"#,
            maxTokens: 12,
            imagePath: #"/tmp/image "one".png"#,
            audioPath: #"/tmp/audio \ sample.wav"#)

        XCTAssertTrue(script.contains(#"model, processor = load("/tmp/model \"quoted\"")"#))
        XCTAssertTrue(script.contains(#"user_prompt = "say \"hi\"\\nthen continue""#))
        XCTAssertTrue(script.contains(#"media_prefix = "<|image|><|audio|>""#))
        XCTAssertTrue(script.contains(#"kwargs["image"] = ["/tmp/image \"one\".png"]"#))
        XCTAssertTrue(script.contains(#"kwargs["audio"] = "/tmp/audio \\ sample.wav""#))
        XCTAssertTrue(script.contains("max_tokens=12"))
    }
}
