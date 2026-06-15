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

    func testTranscribesRealSpeech() throws {
        guard let dir = modelDir() else {
            throw XCTSkip("whisper model dir not found (set KLM_WHISPER_DIR)")
        }
        let wavURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/whisper_say_fox.wav")
        let wav = try Data(contentsOf: wavURL)
        let waveform = try AudioPreprocessor.monoWaveform(fromAudio: wav)

        let runtime = try WhisperRuntime(modelDir: dir)
        let text = runtime.transcribe(waveform: waveform)
            .lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: " .,!?"))

        XCTAssertEqual(text, "the quick brown fox jumps over the lazy dog",
                       "got: \(text)")
    }
}
