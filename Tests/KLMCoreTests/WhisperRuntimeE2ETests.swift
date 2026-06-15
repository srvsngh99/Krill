import XCTest
import MLX
@testable import KLMCore

/// Stage 4 gate: the full native runtime (mel -> encoder -> decoder -> greedy
/// decode -> detokenize) must transcribe real speech. The fixture
/// `whisper_say_fox.wav` is 2.5 s of macOS `say` speech ("the quick brown fox
/// jumps over the lazy dog"); the transformers reference yields exactly that.
///
/// Live test: skips when the converted model dir is absent (CI). See
/// `WhisperEncoderParityTests` for the `KLM_WHISPER_DIR` override.
final class WhisperRuntimeE2ETests: XCTestCase {

    private func modelDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        let dir: URL = env["KLM_WHISPER_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".krillm/whisper-small.en")
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path) ? dir : nil
    }

    /// Raw HuggingFace `transformers` snapshot (model.safetensors + vocab.json
    /// + config.json), as a first-use download would fetch with no Python.
    private func hfSnapshotDir() -> URL? {
        let base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(
                ".cache/huggingface/hub/models--openai--whisper-small.en/snapshots")
        guard let snaps = try? FileManager.default.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil) else { return nil }
        for s in snaps where FileManager.default.fileExists(
            atPath: s.appendingPathComponent("model.safetensors").path) {
            return s
        }
        return nil
    }

    private func waveform() throws -> [Float] {
        let wavURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/whisper_say_fox.wav")
        return try AudioPreprocessor.monoWaveform(fromAudio: try Data(contentsOf: wavURL))
    }

    private func normalized(_ s: String) -> String {
        s.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: " .,!?"))
    }

    func testTranscribesRealSpeech() throws {
        guard let dir = modelDir() else {
            throw XCTSkip("whisper model dir not found (set KLM_WHISPER_DIR)")
        }
        let runtime = try WhisperRuntime(modelDir: dir)
        XCTAssertEqual(normalized(runtime.transcribe(waveform: try waveform())),
                       "the quick brown fox jumps over the lazy dog")
    }

    /// The pure-Swift HF load path: same transcript from a raw HF snapshot via
    /// the in-memory key remap (no convert_whisper.py step).
    func testTranscribesFromRawHFSnapshot() throws {
        guard let dir = hfSnapshotDir() else {
            throw XCTSkip("openai/whisper-small.en HF snapshot not cached")
        }
        let runtime = try WhisperRuntime(modelDir: dir)
        XCTAssertEqual(normalized(runtime.transcribe(waveform: try waveform())),
                       "the quick brown fox jumps over the lazy dog")
    }
}
