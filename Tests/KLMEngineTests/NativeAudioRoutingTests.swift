import XCTest
import Foundation
import KLMCore
@testable import KLMEngine

/// Live native-audio end-to-end checks. The native-vs-bridge routing flag
/// was removed in WS6 Step 4 (the mlx-vlm bridge was retired); audio now
/// always runs on the native Swift+MLX USM path. These KLM_GEMMA4_MODEL_PATH
/// -gated tests prove the native path actually conditions on audio and
/// surfaces decode failures loudly.
final class NativeAudioRoutingTests: XCTestCase {

    /// Live: native audio must produce non-empty output that differs from
    /// the text-only answer (proving the audio embeddings conditioned
    /// generation). Skips unless KLM_GEMMA4_MODEL_PATH points at a real
    /// checkpoint.
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
        let assets = ProcessInfo.processInfo.environment["KLM_BENCH_ASSETS_DIR"]
            .map { URL(fileURLWithPath: $0, isDirectory: true) }
            ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent(".build/benchmarks/assets", isDirectory: true)
        let wav = assets.appendingPathComponent("gemma4-sine-1khz-5s.wav")
        guard FileManager.default.fileExists(atPath: wav.path) else {
            throw XCTSkip("audio fixture missing: \(wav.path)")
        }

        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: path))
        try await engine.load()
        XCTAssertTrue(engine.canUseNativeAudio,
                      "audio-capable Gemma 4 must report canUseNativeAudio")

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

    /// PR #21 rereview P1b: with the native path selected, undecodable
    /// audio must surface a loud decode error in the engine-visible
    /// response — never a silent/empty "successful" text-only answer.
    /// Consumed with the CLI/non-streaming idiom (break on `isEnd`).
    func testLiveNativeAudioDecodeErrorIsSurfaced() async throws {
        guard let path = ProcessInfo.processInfo.environment["KLM_GEMMA4_MODEL_PATH"],
              !path.isEmpty else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH not set")
        }
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDir),
              isDir.boolValue else {
            throw XCTSkip("KLM_GEMMA4_MODEL_PATH is not a directory")
        }

        let engine = InferenceEngine(modelDirectory: URL(fileURLWithPath: path))
        try await engine.load()
        XCTAssertTrue(engine.canUseNativeAudio)

        // RIFF/WAVE header but a corrupt body: passes isWAV, fails decode.
        var corrupt = Data("RIFF".utf8)
        corrupt.append(Data([0xff, 0xff, 0xff, 0xff]))
        corrupt.append(Data("WAVE".utf8))
        corrupt.append(Data(repeating: 0x7f, count: 8))

        let (stream, _) = engine.generate(
            prompt: "What do you hear?", maxTokens: 24,
            imageData: nil, audioData: corrupt)
        // CLI/non-streaming idiom: break on isEnd BEFORE appending.
        var out = ""
        for await ev in stream { if ev.isEnd { break }; out += ev.text }
        XCTAssertTrue(out.contains("native audio decode failed"),
            "decode failure must be surfaced, not a silent text answer; got: \(out)")
    }
}
