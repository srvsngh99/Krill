import Foundation
#if canImport(Speech)
@preconcurrency import Speech
#endif

/// On-device speech-to-text via Apple's Speech framework. Recognition runs fully
/// locally (`requiresOnDeviceRecognition`) with no model download and no cloud -
/// unlike a remote transcription service. Used for voice dictation in the chat
/// TUI; callers fall back to the multimodal model when this is unavailable.
public final class SpeechRecognizer: @unchecked Sendable {
    /// `auto` (or an empty value) follows the current macOS locale.
    public let language: String

    public init(language: String = "auto") { self.language = language }

    /// True when on-device recognition is usable for the current locale on this
    /// OS (the Speech framework exists, a recognizer is available, and it
    /// supports the local-only path).
    public static var isAvailable: Bool { isAvailable(language: "auto") }

    /// True when local recognition is usable for `language`. Empty and `auto`
    /// deliberately use the current locale, keeping Apple recognition local.
    public static func isAvailable(language: String = "auto") -> Bool {
#if canImport(Speech)
        let settings = AppleSpeechSettings(language: language)
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: settings.localeIdentifier)) else { return false }
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
        let settings = AppleSpeechSettings(language: language)
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: settings.localeIdentifier)), recognizer.isAvailable else { return nil }
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

/// A transcript update from ``LiveAppleSpeechRecognizer``. Partial updates are
/// intended for a live caption; a final update is the stable user instruction.
public struct LiveSpeechTranscriptionUpdate: Sendable, Equatable {
    public let text: String
    public let isFinal: Bool

    public init(text: String, isFinal: Bool) {
        self.text = text
        self.isFinal = isFinal
    }
}

/// On-device streaming speech recognition for a single endpointed utterance.
/// Start a fresh instance (or call `start` again) per turn. A monotonically
/// increasing generation token ensures delayed callbacks from a cancelled turn
/// cannot overwrite a newer live transcript.
public final class LiveAppleSpeechRecognizer: @unchecked Sendable {
    public typealias UpdateHandler = @Sendable (LiveSpeechTranscriptionUpdate) -> Void

    private let settings: AppleSpeechSettings
    private let onUpdate: UpdateHandler
    /// Request mutation and `append` are serialized. Speech accepts buffers on
    /// arbitrary capture queues, while `finish`/barge-in commonly originate on
    /// the UI queue; a lock around only the pointer would still race endAudio.
    private let stateQueue = DispatchQueue(label: "krill.voice.live-stt")
    private var generation: UInt64 = 0

#if canImport(Speech) && canImport(AVFoundation)
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
#endif

    public init(settings: AppleSpeechSettings = .init(), onUpdate: @escaping UpdateHandler) {
        self.settings = settings
        self.onUpdate = onUpdate
    }

    /// Create an on-device request and begin receiving partial results. Callers
    /// should request `SpeechRecognizer` authorization before starting.
    public func start() throws {
#if canImport(Speech) && canImport(AVFoundation)
        guard SpeechRecognizer.isAvailable(language: settings.language),
              let recognizer = SFSpeechRecognizer(locale: Locale(identifier: settings.localeIdentifier)) else {
            throw VoiceTranscriptionError.unavailable
        }
        guard SFSpeechRecognizer.authorizationStatus() == .authorized else {
            throw VoiceTranscriptionError.authorizationDenied
        }

        cancel()
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true

        let currentGeneration = stateQueue.sync { () -> UInt64 in
            generation &+= 1
            self.request = request
            return generation
        }

        let task = recognizer.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }
            self.handle(result: result, error: error, generation: currentGeneration)
        }
        // A new start could have raced with recognitionTask creation. Do not
        // let this now-stale task replace the active request.
        stateQueue.sync {
            if generation == currentGeneration {
                self.task = task
            } else {
                task.cancel()
            }
        }
#else
        throw VoiceTranscriptionError.unavailable
#endif
    }

    /// Append a copied mono PCM frame. Frames before `start`, after `finish`, or
    /// from a stale generation are ignored rather than crashing a UI callback.
    public func append(_ frame: VoiceAudioFrame) {
#if canImport(Speech) && canImport(AVFoundation)
        guard frame.sampleRate > 0, !frame.samples.isEmpty else { return }
        guard let format = AVAudioFormat(standardFormatWithSampleRate: frame.sampleRate, channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frame.samples.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(frame.samples.count)
        guard let destination = buffer.floatChannelData?[0] else { return }
        frame.samples.withUnsafeBufferPointer { source in
            destination.update(from: source.baseAddress!, count: frame.samples.count)
        }
        stateQueue.sync { request?.append(buffer) }
#endif
    }

    /// Signal end-of-audio and allow Apple's recognizer to publish its final
    /// transcript. `cancel` may still be used immediately for barge-in.
    public func finish() {
#if canImport(Speech) && canImport(AVFoundation)
        stateQueue.sync { request?.endAudio() }
#endif
    }

    /// Discard the current turn. Generation is advanced before cancellation, so
    /// already queued recognition callbacks are harmlessly ignored.
    public func cancel() {
#if canImport(Speech) && canImport(AVFoundation)
        stateQueue.sync {
            generation &+= 1
            request?.endAudio()
            task?.cancel()
            request = nil
            task = nil
        }
#endif
    }

#if canImport(Speech) && canImport(AVFoundation)
    private func handle(result: SFSpeechRecognitionResult?, error: Error?, generation callbackGeneration: UInt64) {
        stateQueue.async { [weak self] in
            guard let self, self.generation == callbackGeneration else { return }
            if result?.isFinal == true || error != nil {
                self.request = nil
                self.task = nil
            }
            guard let result else { return }
            let text = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return }
            self.onUpdate(.init(text: text, isFinal: result.isFinal))
        }
    }
#endif
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
