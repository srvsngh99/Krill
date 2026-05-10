import XCTest
@testable import KLMEngine

final class Gemma4SmokeTests: XCTestCase {

    private func requireModel() throws -> String {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"], !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir), isDir.boolValue else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH is not a directory: \(path)")
        }
        return path
    }

    private func requireMLXVLM() throws {
        let availability = PythonFallback.checkAvailability()
        if !availability.isAvailable {
            throw XCTSkip("mlx-vlm not available: \(availability.detail)")
        }
    }

    private func assetsDir() -> URL {
        if let override = ProcessInfo.processInfo.environment["KLM_BENCH_ASSETS_DIR"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return cwd
            .appendingPathComponent(".build", isDirectory: true)
            .appendingPathComponent("benchmarks", isDirectory: true)
            .appendingPathComponent("assets", isDirectory: true)
    }

    private func requireAsset(named name: String) throws -> String {
        let url = assetsDir().appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: url.path) {
            return url.path
        }
        throw XCTSkip("Required asset missing at \(url.path)")
    }

    private func assertContainsAny(
        _ output: String,
        _ candidates: [String],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let lower = output.lowercased()
        let hit = candidates.contains { lower.contains($0.lowercased()) }
        XCTAssertTrue(hit, "Expected output to contain one of \(candidates); got: \(output)", file: file, line: line)
    }

    private func assertContainsNone(
        _ output: String,
        _ forbidden: [String],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let lower = output.lowercased()
        for term in forbidden {
            XCTAssertFalse(lower.contains(term.lowercased()),
                "Output should not contain forbidden term \"\(term)\"; got: \(output)",
                file: file, line: line)
        }
    }

    private func runFallback(
        modelPath: String,
        prompt: String,
        imagePath: String? = nil,
        audioPath: String? = nil,
        maxTokens: Int = 64
    ) async throws -> String {
        let fallback = PythonFallback(modelPath: modelPath)
        return try await fallback.generate(
            prompt: prompt,
            maxTokens: maxTokens,
            imagePath: imagePath,
            audioPath: audioPath)
    }

    func testGemma4TextSmoke() async throws {
        try requireMLXVLM()
        let modelPath = try requireModel()
        let output = try await runFallback(
            modelPath: modelPath,
            prompt: "Explain quantum computing in simple terms.",
            maxTokens: 96)
        XCTAssertFalse(output.isEmpty)
        assertContainsAny(output, ["quantum", "qubit", "qubits", "superposition", "entanglement"])
    }

    func testGemma4ImageRedBoxSmoke() async throws {
        try requireMLXVLM()
        let modelPath = try requireModel()
        let imagePath = try requireAsset(named: "gemma4-red-box.png")
        let output = try await runFallback(
            modelPath: modelPath,
            prompt: "What is shown in this image? Answer briefly.",
            imagePath: imagePath,
            maxTokens: 48)
        XCTAssertFalse(output.isEmpty)
        assertContainsAny(output, ["red", "box", "square", "rectangle"])
        assertContainsNone(output, ["green", "blue", "circle", "triangle"])
    }

    func testGemma4AudioSineToneSmoke() async throws {
        try requireMLXVLM()
        let modelPath = try requireModel()
        let audioPath = try requireAsset(named: "gemma4-sine-1khz-5s.wav")
        let output = try await runFallback(
            modelPath: modelPath,
            prompt: "What sound is in this audio? Answer briefly.",
            audioPath: audioPath,
            maxTokens: 48)
        XCTAssertFalse(output.isEmpty)
        assertContainsAny(output, ["tone", "sine", "beep", "single", "steady", "continuous", "hum", "buzz"])
        assertContainsNone(output, ["dog", "bark", "music", "speech", "voice"])
    }
}
