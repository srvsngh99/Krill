import Foundation

/// A non-chat model type Krill does not run as a causal LM.
///
/// These are the WS7 "specialized model types": speech recognition,
/// speech synthesis, image generation, video-language, and document
/// OCR. Krill has no native runtime for them in this build (only
/// rerankers have shipped). Loading one as a dense causal LM would
/// silently produce a garbage forward pass.
///
/// `detectSpecializedModelType` recognizes them from a checkpoint's
/// `config.json` so `loadModel` can reject them with an explicit,
/// specific error - the roadmap's "Unsupported tier: explicit error
/// before execution" - instead of mis-loading them through the Llama
/// fallback.
public enum SpecializedModelType: String, Sendable, CaseIterable {
    /// Whisper / wav2vec2-style automatic speech recognition.
    case speechRecognition
    /// Speech synthesis / audio generation (Bark, VITS, Parler-TTS).
    case textToSpeech
    /// Diffusion image generation (Stable Diffusion, FLUX, PixArt).
    case imageGeneration
    /// Video-language models (Video-LLaVA, VideoLLaMA).
    case videoLanguage
    /// Document understanding / OCR models (Donut, Nougat, TrOCR).
    case documentOCR

    /// A short human-readable name for error messages.
    public var displayName: String {
        switch self {
        case .speechRecognition: return "speech-recognition (ASR)"
        case .textToSpeech:      return "speech-synthesis / audio-generation (TTS)"
        case .imageGeneration:   return "image-generation (diffusion)"
        case .videoLanguage:     return "video-language"
        case .documentOCR:       return "document-OCR"
        }
    }
}

/// Recognize a WS7 specialized model type from a checkpoint's
/// `config.json` architecture / model_type strings.
///
/// Both arguments must already be lowercased. Returns `nil` for a
/// causal LM or any architecture this function does not recognize as
/// specialized - the caller then proceeds with normal family
/// detection.
///
/// This is intentionally a last-resort heuristic: `loadModel` only
/// consults it after every supported-family arm has failed, so a
/// loose substring match here can only ever reclassify a checkpoint
/// that would otherwise have been mis-loaded as Llama.
public func detectSpecializedModelType(
    arch: String, modelType: String
) -> SpecializedModelType? {
    let haystack = arch + " " + modelType

    func anyOf(_ needles: [String]) -> Bool {
        needles.contains { haystack.contains($0) }
    }

    // Speech recognition: Whisper and the wav2vec2 / HuBERT family.
    if anyOf(["whisper", "wav2vec2", "wav2vec", "hubert",
              "speech-encoder-decoder", "speech_to_text", "speechtotext"]) {
        return .speechRecognition
    }
    // Speech synthesis / audio generation.
    if anyOf(["texttospeech", "text_to_speech", "parler", "speecht5",
              "bark", "vits", "musicgen", "encodec", "fastspeech",
              "tacotron", "xtts"]) {
        return .textToSpeech
    }
    // Diffusion image generation.
    if anyOf(["diffusion", "unet2dcondition", "flux", "sdxl",
              "pixart", "kandinsky", "latentconsistency"]) {
        return .imageGeneration
    }
    // Video-language models.
    if anyOf(["videollava", "video_llava", "llavanextvideo",
              "llava_next_video", "videollama", "video_llama",
              "videomae", "internvideo", "video-language"]) {
        return .videoLanguage
    }
    // Document understanding / OCR.
    if anyOf(["donut", "nougat", "trocr", "layoutlm", "udop",
              "got_ocr", "gotocr"]) {
        return .documentOCR
    }
    return nil
}
