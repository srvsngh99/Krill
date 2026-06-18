import XCTest
import MLX
@testable import KrillCore

/// Follow-up gate: the multilingual path (auto language detection + the
/// language-token prompt). Uses a multilingual checkpoint (vocab 51865) on two
/// `say` fixtures - English ("the quick brown fox...") and Spanish ("hola me
/// llamo Claude"). The base multilingual model is weak on synthetic speech, so
/// the gate is PARITY with the transformers reference on the same clip+model,
/// not "ideal" text - it proves language detect + multilingual decode are
/// correct.
///
/// Live test: skips when the multilingual model dir is absent (CI). Override
/// with `KRILL_WHISPER_ML_DIR`, else the default `~/.krill/models/whisper-base`.
final class WhisperMultilingualE2ETests: XCTestCase {

    private func modelDir() -> URL? {
        let env = ProcessInfo.processInfo.environment
        let dir: URL = env["KRILL_WHISPER_ML_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".krill/models/whisper-base")
        return FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("model.safetensors").path) ? dir : nil
    }

    private func waveform(_ name: String) throws -> [Float] {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/\(name)")
        return try AudioPreprocessor.monoWaveform(fromAudio: try Data(contentsOf: url))
    }

    private func normalized(_ s: String) -> String {
        s.lowercased().trimmingCharacters(in: CharacterSet(charactersIn: " .,!?"))
    }

    func testMultilingualAutoDetectsAndTranscribes() throws {
        guard let dir = modelDir() else {
            throw XCTSkip("multilingual whisper model dir not found (set KRILL_WHISPER_ML_DIR)")
        }
        let runtime = try WhisperRuntime(modelDir: dir)
        XCTAssertTrue(runtime.config.nVocab >= 51865, "expected a multilingual checkpoint")

        // English clip: language auto-detects to en, transcript matches.
        XCTAssertEqual(normalized(runtime.transcribe(waveform: try waveform("whisper_say_fox.wav"))),
                       "the quick brown fox jumps over the lazy dog")

        // Spanish clip: parity with the transformers reference (base is weak on
        // synthetic speech; what matters is detect + decode match the oracle).
        XCTAssertEqual(normalized(runtime.transcribe(waveform: try waveform("whisper_say_es.wav"))),
                       "hala mi lamao claude")
    }
}
