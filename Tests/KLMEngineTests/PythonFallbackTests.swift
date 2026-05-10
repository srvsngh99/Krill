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

    // The previous `testGeneratedPythonScriptEscapesPromptAndMediaPaths` test
    // asserted on a Python script string built per-request inside Swift. With
    // the persistent sidecar, prompt/media handling lives in
    // `tools/mlx_vlm_sidecar.py` and is exercised through the JSON protocol.
    // The new sidecar tests in `PythonFallbackSidecarTests.swift` cover that
    // path end-to-end, so this string-shape test no longer applies.
}
