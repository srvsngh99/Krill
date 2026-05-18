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

    /// WS6 acceptance gate: native Swift+MLX audio must be **semantically
    /// equivalent to the mlx-vlm oracle** on the deterministic sine fixture
    /// — not merely non-empty. Runs both paths and holds the native output
    /// to the SAME quality rubric the bridge satisfies above. This is the
    /// numerical-validation check that must pass on the M4 target before
    /// `KRILL_NATIVE_AUDIO` is flipped default-on and the bridge is retired.
    /// Skips unless KLM_GEMMA4_MODEL_PATH + mlx-vlm + the fixture are present.
    func testWS6NativeAudioMatchesBridgeOracleOnSineTone() async throws {
        try requireMLXVLM()
        let modelPath = try requireModel()
        let audioPath = try requireAsset(named: "gemma4-sine-1khz-5s.wav")
        let prompt = "What sound is in this audio? Answer briefly."
        let expected = ["tone", "sine", "beep", "single", "steady",
                        "continuous", "hum", "buzz"]
        let forbidden = ["dog", "bark", "music", "speech", "voice"]

        // Oracle (bridge).
        let oracle = try await runFallback(
            modelPath: modelPath, prompt: prompt,
            audioPath: audioPath, maxTokens: 48)
        XCTAssertFalse(oracle.isEmpty)
        assertContainsAny(oracle, expected)
        assertContainsNone(oracle, forbidden)

        // Native Swift+MLX path.
        setenv("KRILL_NATIVE_AUDIO", "1", 1)
        defer { unsetenv("KRILL_NATIVE_AUDIO") }
        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: modelPath))
        try await engine.load()
        XCTAssertTrue(engine.canUseNativeAudio, "native audio must be active")
        let audioData = try Data(contentsOf: URL(fileURLWithPath: audioPath))
        let (stream, _) = engine.generate(
            prompt: prompt, maxTokens: 48, imageData: nil, audioData: audioData)
        var native = ""
        for await ev in stream { native += ev.text; if ev.isEnd { break } }
        native = native.trimmingCharacters(in: .whitespacesAndNewlines)

        XCTAssertFalse(native.isEmpty, "native audio produced empty output")
        // The decisive parity assertion: native meets the same rubric as
        // the oracle. (Greedy token-equality is too strict across two
        // independent runtimes; rubric equivalence is the WS6 contract.)
        assertContainsAny(native, expected)
        assertContainsNone(native, forbidden)
    }
}
