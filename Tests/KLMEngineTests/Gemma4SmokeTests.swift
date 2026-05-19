import XCTest
@testable import KLMEngine

/// Live Gemma 4 native-path smokes + the WS6 numerical-parity gate.
///
/// The mlx-vlm bridge was retired in WS6 Step 4. Its role as the
/// correctness oracle is preserved by a *recorded* baseline captured from
/// the bridge before removal (`Fixtures/ws6_oracle_baseline.json`). These
/// tests assert (a) the live native path meets each recorded rubric and
/// (b) the recorded oracle output itself still meets that rubric — a
/// baseline-integrity guard against silent rubric drift. Greedy
/// token-equality across two runtimes was always too strict; rubric
/// equivalence is the WS6 contract.
final class Gemma4SmokeTests: XCTestCase {

    // MARK: - Golden oracle baseline

    struct OracleBaseline: Decodable {
        struct Entry: Decodable {
            let name: String
            let prompt: String
            let asset: String?
            let max_tokens: Int
            let expected_any: [String]
            let forbidden: [String]
            let oracle_output: String?
        }
        let entries: [Entry]
    }

    /// Resolves `Fixtures/ws6_oracle_baseline.json` next to this source file
    /// (no SwiftPM resource bundling needed).
    private func loadBaseline(file: StaticString = #filePath) throws -> OracleBaseline {
        let here = URL(fileURLWithPath: "\(file)").deletingLastPathComponent()
        let url = here.appendingPathComponent("Fixtures/ws6_oracle_baseline.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(OracleBaseline.self, from: data)
    }

    private func entry(_ name: String) throws -> OracleBaseline.Entry {
        let base = try loadBaseline()
        guard let e = base.entries.first(where: { $0.name == name }) else {
            throw XCTSkip("baseline entry \(name) missing")
        }
        return e
    }

    // MARK: - Helpers

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
        _ output: String, _ candidates: [String],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let lower = output.lowercased()
        let hit = candidates.contains { lower.contains($0.lowercased()) }
        XCTAssertTrue(hit, "Expected output to contain one of \(candidates); got: \(output)", file: file, line: line)
    }

    private func assertContainsNone(
        _ output: String, _ forbidden: [String],
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let lower = output.lowercased()
        for term in forbidden {
            XCTAssertFalse(lower.contains(term.lowercased()),
                "Output should not contain forbidden term \"\(term)\"; got: \(output)",
                file: file, line: line)
        }
    }

    /// Runs the native Swift+MLX engine and returns trimmed output.
    private func runNative(
        modelPath: String, prompt: String,
        imageData: Data? = nil, audioData: Data? = nil, maxTokens: Int
    ) async throws -> String {
        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: modelPath))
        try await engine.load()
        if audioData != nil {
            XCTAssertTrue(engine.canUseNativeAudio, "native audio must be active (default-on, WS6)")
        }
        let (stream, _) = engine.generate(
            prompt: prompt, maxTokens: maxTokens,
            imageData: imageData, audioData: audioData)
        var out = ""
        for await ev in stream { out += ev.text; if ev.isEnd { break } }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Baseline integrity (no model required)

    /// The pinned oracle outputs must themselves satisfy their rubrics.
    /// Guards against editing a rubric into something the recorded
    /// baseline no longer meets.
    func testOracleBaselineEntriesSatisfyTheirRubrics() throws {
        let base = try loadBaseline()
        XCTAssertFalse(base.entries.isEmpty, "baseline has no entries")
        for e in base.entries {
            let out = try XCTUnwrap(e.oracle_output, "\(e.name): no recorded oracle_output")
            XCTAssertFalse(out.isEmpty, "\(e.name): empty oracle_output")
            assertContainsAny(out, e.expected_any)
            assertContainsNone(out, e.forbidden)
        }
    }

    // MARK: - Live native smokes vs the recorded baseline

    func testGemma4TextSmokeNative() async throws {
        let modelPath = try requireModel()
        let e = try entry("text-quantum")
        let out = try await runNative(
            modelPath: modelPath, prompt: e.prompt, maxTokens: e.max_tokens)
        XCTAssertFalse(out.isEmpty)
        assertContainsAny(out, e.expected_any)
        assertContainsNone(out, e.forbidden)
    }

    func testGemma4ImageRedBoxNative() async throws {
        let modelPath = try requireModel()
        let e = try entry("image-red-box")
        let assetName = (e.asset.map { ($0 as NSString).lastPathComponent }) ?? "gemma4-red-box.png"
        let imageData = try Data(contentsOf: URL(fileURLWithPath: try requireAsset(named: assetName)))
        let out = try await runNative(
            modelPath: modelPath, prompt: e.prompt,
            imageData: imageData, maxTokens: e.max_tokens)
        XCTAssertFalse(out.isEmpty)
        assertContainsAny(out, e.expected_any)
        assertContainsNone(out, e.forbidden)
    }

    /// Non-empty smoke only. A pure 1 kHz sine is out-of-distribution for a
    /// speech-understanding model: Gemma 4 E2B hallucinates
    /// non-deterministically on it, so a semantic rubric here is unstable
    /// by construction (see docs/NATIVE_GEMMA4_AUDIO_PLAN.md WS6 + Risks).
    /// The deterministic semantic gate is the speech test below.
    func testGemma4AudioSineToneSmoke() async throws {
        let modelPath = try requireModel()
        let audioPath = try requireAsset(named: "gemma4-sine-1khz-5s.wav")
        let audioData = try Data(contentsOf: URL(fileURLWithPath: audioPath))
        let out = try await runNative(
            modelPath: modelPath,
            prompt: "What sound is in this audio? Answer briefly.",
            audioData: audioData, maxTokens: 48)
        XCTAssertFalse(out.isEmpty, "native audio path produced empty output")
    }

    /// WS6 numerical-parity gate. The native Swift+MLX audio path must meet
    /// the same rubric the recorded mlx-vlm oracle satisfied on the
    /// deterministic speech pangram. Speech (not a tone) is used because a
    /// speech model hallucinates non-deterministically on non-speech audio.
    /// Skips unless KLM_GEMMA4_MODEL_PATH + the speech fixture are present.
    func testWS6NativeAudioMatchesOracleBaselineOnSpeech() async throws {
        let modelPath = try requireModel()
        let e = try entry("audio-speech-pangram")

        // Baseline integrity: the recorded oracle still meets the rubric.
        let oracle = try XCTUnwrap(e.oracle_output, "no recorded oracle baseline")
        assertContainsAny(oracle, e.expected_any)
        assertContainsNone(oracle, e.forbidden)

        // Live native path must meet the same rubric.
        let assetName = (e.asset.map { ($0 as NSString).lastPathComponent }) ?? "gemma4-speech-pangram.wav"
        let audioData = try Data(contentsOf: URL(fileURLWithPath: try requireAsset(named: assetName)))
        let native = try await runNative(
            modelPath: modelPath, prompt: e.prompt,
            audioData: audioData, maxTokens: e.max_tokens)
        XCTAssertFalse(native.isEmpty, "native audio produced empty output")
        assertContainsAny(native, e.expected_any)
        assertContainsNone(native, e.forbidden)
    }
}
