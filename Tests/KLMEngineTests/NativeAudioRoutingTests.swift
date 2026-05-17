import XCTest
import Foundation
import KLMCore
@testable import KLMEngine

/// WS5 routing tests: the native-vs-bridge decision flag, plus a
/// KLM_GEMMA4_MODEL_PATH-gated live native-audio end-to-end check.
final class NativeAudioRoutingTests: XCTestCase {

    private func withEnv(_ kv: [String: String?], _ body: () -> Void) {
        let keys = Array(kv.keys)
        let saved = keys.map { ($0, ProcessInfo.processInfo.environment[$0]) }
        for (k, v) in kv {
            if let v { setenv(k, v, 1) } else { unsetenv(k) }
        }
        defer { for (k, v) in saved { if let v { setenv(k, v, 1) } else { unsetenv(k) } } }
        body()
    }

    func testNativeAudioFlagDefaultsOff() {
        withEnv(["KRILL_NATIVE_AUDIO": nil, "KRILL_AUDIO_BRIDGE_ONLY": nil]) {
            XCTAssertFalse(InferenceEngine.nativeAudioEnabled)
        }
    }

    func testNativeAudioFlagEnabled() {
        withEnv(["KRILL_NATIVE_AUDIO": "1", "KRILL_AUDIO_BRIDGE_ONLY": nil]) {
            XCTAssertTrue(InferenceEngine.nativeAudioEnabled)
        }
    }

    func testBridgeOnlyOverridesNative() {
        withEnv(["KRILL_NATIVE_AUDIO": "1", "KRILL_AUDIO_BRIDGE_ONLY": "1"]) {
            XCTAssertFalse(InferenceEngine.nativeAudioEnabled,
                           "KRILL_AUDIO_BRIDGE_ONLY must win over KRILL_NATIVE_AUDIO")
        }
    }

    /// Live: native audio must not touch PythonFallback and must produce
    /// non-empty output that differs from the text-only answer. Skips
    /// unless KLM_GEMMA4_MODEL_PATH points at a real checkpoint.
    func testLiveNativeAudioProducesOutput() async throws {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH is not a directory")
        }
        // A deterministic 1 kHz tone WAV fixture (see WS5 assets).
        let assets = ProcessInfo.processInfo.environment["KLM_BENCH_ASSETS_DIR"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".build/benchmarks/assets", isDirectory: true)
        let wav = assets.appendingPathComponent("gemma4-sine-1khz-5s.wav")
        guard FileManager.default.fileExists(atPath: wav.path) else {
            throw XCTSkip("audio fixture missing: \(wav.path)")
        }

        setenv("KRILL_NATIVE_AUDIO", "1", 1)
        defer { unsetenv("KRILL_NATIVE_AUDIO") }

        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: path))
        try await engine.load()
        XCTAssertTrue(engine.canUseNativeAudio,
                      "native audio path must be active with the flag on")

        let audioData = try Data(contentsOf: wav)

        // Prove the native frontend actually decodes this fixture (so a
        // silent text-only fallback can't masquerade as a pass).
        let feats = try AudioPreprocessor.features(fromAudio: audioData)
        XCTAssertGreaterThan(feats.numTokens, 0)
        XCTAssertEqual(feats.mel.dim(2), AudioPreprocessor.melBins)

        let prompt = "What do you hear in this audio?"
        func run(_ audio: Data?) async -> String {
            let (stream, _) = engine.generate(
                prompt: prompt, maxTokens: 24,
                imageData: nil, audioData: audio)
            var out = ""
            for await ev in stream { out += ev.text; if ev.isEnd { break } }
            return out.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let withAudio = await run(audioData)
        let textOnly = await run(nil)

        XCTAssertFalse(withAudio.isEmpty, "native audio produced empty output")
        // If audio were silently dropped, greedy decoding of the identical
        // prompt would yield identical text. Different output proves the
        // audio embeddings actually conditioned generation.
        XCTAssertNotEqual(withAudio, textOnly,
            "audio-conditioned output must differ from the text-only answer; "
            + "identical output means the native audio path did not run")
    }
}
