import Foundation
#if canImport(Speech)
@preconcurrency import Speech
#endif

/// On-device speech-to-text via Apple's Speech framework. Recognition runs fully
/// locally (`requiresOnDeviceRecognition`) with no model download and no cloud -
/// unlike a remote transcription service. Used for voice dictation in the chat
/// TUI; callers fall back to the multimodal model when this is unavailable.
public final class SpeechRecognizer: @unchecked Sendable {
    public init() {}

    /// True when on-device recognition is usable for the current locale on this
    /// OS (the Speech framework exists, a recognizer is available, and it
    /// supports the local-only path).
    public static var isAvailable: Bool {
#if canImport(Speech)
        guard let recognizer = SFSpeechRecognizer() else { return false }
        return recognizer.isAvailable && recognizer.supportsOnDeviceRecognition
#else
        return false
#endif
    }

    /// Prompt for (or confirm) speech-recognition authorization. Returns `true`
    /// once authorized. The first call shows the system permission dialog (which,
    /// like the microphone prompt, attributes to a code-signed bundle).
    public static func requestAuthorization() async -> Bool {
#if canImport(Speech)
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized: return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                SFSpeechRecognizer.requestAuthorization { status in
                    cont.resume(returning: status == .authorized)
                }
            }
        default: return false
        }
#else
        return false
#endif
    }

    /// Transcribe a 16-bit PCM WAV clip on-device. Returns the recognized text,
    /// or `nil` if recognition is unavailable, denied, or produced nothing.
    public func transcribe(wav: Data) async -> String? {
#if canImport(Speech)
        guard let recognizer = SFSpeechRecognizer(), recognizer.isAvailable else { return nil }
        // SFSpeech consumes a file URL; stage the clip in a temp wav.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("krill-voice-\(UUID().uuidString).wav")
        guard (try? wav.write(to: url)) != nil else { return nil }
        defer { try? FileManager.default.removeItem(at: url) }

        let request = SFSpeechURLRecognitionRequest(url: url)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false

        // The result handler may fire more than once; resume the continuation
        // exactly once (a double resume traps).
        let once = ResumeOnce()
        return await withCheckedContinuation { (cont: CheckedContinuation<String?, Never>) in
            recognizer.recognitionTask(with: request) { result, error in
                if let result, result.isFinal {
                    once.run { cont.resume(returning: result.bestTranscription.formattedString) }
                } else if error != nil {
                    once.run { cont.resume(returning: nil) }
                }
            }
        }
#else
        return nil
#endif
    }
}

/// One-shot guard so a continuation is resumed at most once across repeated
/// callback invocations.
private final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var fired = false
    func run(_ body: () -> Void) {
        lock.lock()
        let first = !fired
        fired = true
        lock.unlock()
        if first { body() }
    }
}
