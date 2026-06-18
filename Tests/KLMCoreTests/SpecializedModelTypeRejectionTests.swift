import XCTest
@testable import KLMCore

/// WS7 foundation: the causal-LM `loadModel(from:)` dispatcher must
/// reject specialized non-chat model types (ASR / TTS / diffusion /
/// video-language / OCR) with an explicit, specific error rather than
/// mis-loading them through the Llama fallback. These tests pin the
/// `detectSpecializedModelType` heuristic and the loader rejection.
final class SpecializedModelTypeRejectionTests: XCTestCase {

    // MARK: - detectSpecializedModelType

    func testDetectsSpeechRecognition() {
        XCTAssertEqual(
            detectSpecializedModelType(
                arch: "whisperforconditionalgeneration", modelType: "whisper"),
            .speechRecognition)
        XCTAssertEqual(
            detectSpecializedModelType(arch: "wav2vec2forctc", modelType: "wav2vec2"),
            .speechRecognition)
    }

    func testDetectsTextToSpeech() {
        XCTAssertEqual(
            detectSpecializedModelType(
                arch: "parlerttsforconditionalgeneration", modelType: "parler_tts"),
            .textToSpeech)
        XCTAssertEqual(
            detectSpecializedModelType(arch: "barkmodel", modelType: "bark"),
            .textToSpeech)
    }

    func testDetectsImageGeneration() {
        XCTAssertEqual(
            detectSpecializedModelType(
                arch: "unet2dconditionmodel", modelType: "stable-diffusion"),
            .imageGeneration)
        XCTAssertEqual(
            detectSpecializedModelType(arch: "fluxpipeline", modelType: "flux"),
            .imageGeneration)
    }

    func testDetectsVideoLanguage() {
        XCTAssertEqual(
            detectSpecializedModelType(
                arch: "llavanextvideoforconditionalgeneration",
                modelType: "llava_next_video"),
            .videoLanguage)
    }

    func testDetectsDocumentOCR() {
        XCTAssertEqual(
            detectSpecializedModelType(arch: "donutswin", modelType: "vision-encoder-decoder"),
            .documentOCR)
        XCTAssertEqual(
            detectSpecializedModelType(arch: "trocrforcausallm", modelType: "trocr"),
            .documentOCR)
    }

    func testCausalLMArchitecturesAreNotSpecialized() {
        // Normal supported families must NOT be flagged - the loader
        // would never reach the detector for these, but the heuristic
        // itself must stay clean.
        for (arch, modelType) in [
            ("llamaforcausallm", "llama"),
            ("qwen2forcausallm", "qwen2"),
            ("mistralforcausallm", "mistral"),
            ("gemma2forcausallm", "gemma2"),
            ("phi3forcausallm", "phi3"),
            ("bertmodel", "bert"),
        ] {
            XCTAssertNil(
                detectSpecializedModelType(arch: arch, modelType: modelType),
                "\(arch) is a supported family and must not be flagged specialized")
        }
    }

    func testDisplayNameIsNonEmptyForEveryType() {
        for type in SpecializedModelType.allCases {
            XCTAssertFalse(type.displayName.isEmpty)
        }
    }

    // MARK: - loadModel rejection

    private func writeConfig(_ json: [String: Any], slug: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-ws7-\(slug)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: json)
        try data.write(to: dir.appendingPathComponent("config.json"))
        return dir
    }

    func testWhisperIsRejectedFromCausalLMDispatcher() throws {
        let dir = try writeConfig([
            "architectures": ["WhisperForConditionalGeneration"],
            "model_type": "whisper",
        ], slug: "whisper")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .specializedModelUnsupported(let msg) = modelError else {
                XCTFail("Expected specializedModelUnsupported, got \(error)")
                return
            }
            XCTAssertTrue(msg.lowercased().contains("speech-recognition"))
            XCTAssertTrue(msg.contains("WS7"))
        }
    }

    func testDiffusionModelIsRejected() throws {
        let dir = try writeConfig([
            "architectures": ["FluxPipeline"],
            "model_type": "flux",
        ], slug: "flux")
        defer { try? FileManager.default.removeItem(at: dir) }

        XCTAssertThrowsError(try loadModel(from: dir)) { error in
            guard let modelError = error as? ModelLoadError,
                  case .specializedModelUnsupported(let msg) = modelError else {
                XCTFail("Expected specializedModelUnsupported, got \(error)")
                return
            }
            XCTAssertTrue(msg.lowercased().contains("image-generation"))
        }
    }
}
