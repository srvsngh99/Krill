import Foundation

/// A local speech-to-text engine. Implementations deliberately receive a WAV
/// clip rather than an audio-device object so they can be used by both the TUI
/// and a future continuous voice coordinator without importing UI frameworks.
public protocol VoiceTranscriber: Sendable {
    func transcribe(_ wav: Data) async throws -> String
}

/// A local text-to-speech output. Phase 3 can replace Apple's output with a
/// streaming neural voice without changing the agent or UI contracts.
public protocol VoiceOutput: Sendable {
    func speak(_ text: String) async throws
    func stop()
}

/// The result of one VAD analysis window. This is intentionally small: Phase 2
/// only establishes the seam; Phase 3 supplies continuous capture and EOU.
public enum VoiceActivityEvent: Sendable, Equatable {
    case speech
    case silence
}

/// A forward-looking VAD seam. Detectors must not retain the buffer passed to
/// `accept`; capture owns that memory only for the duration of this call.
public protocol VoiceActivityDetector: Sendable {
    func accept(_ samples: UnsafeBufferPointer<Float>) -> VoiceActivityEvent
}

/// A mono block of PCM audio produced by the continuous microphone path. Audio
/// is copied out of AVFoundation's callback buffer before this value is made, so
/// it is safe to retain or move to another queue.
public struct VoiceAudioFrame: Sendable, Equatable {
    public let samples: [Float]
    public let sampleRate: Double

    public init(samples: [Float], sampleRate: Double) {
        self.samples = samples
        self.sampleRate = sampleRate
    }
}

/// A stable measurement of signal energy for UI meters and voice endpointing.
/// Decibels are dBFS, with silence clamped to -160 dBFS to avoid infinity.
public struct VoiceSignalLevel: Sendable, Equatable {
    public let rms: Float
    public let decibels: Float

    public init(rms: Float, decibels: Float) {
        self.rms = rms
        self.decibels = decibels
    }

    public static func measure(_ samples: [Float]) -> VoiceSignalLevel {
        guard !samples.isEmpty else { return .init(rms: 0, decibels: -160) }
        var sum: Float = 0
        for sample in samples { sum += sample * sample }
        let rms = (sum / Float(samples.count)).squareRoot()
        let decibels = max(-160, 20 * log10(max(rms, 0.000_000_000_1)))
        return .init(rms: rms, decibels: decibels)
    }
}

/// Tunables for deterministic local energy endpointing. The default 620 ms
/// trailing-silence period is deliberately conversational: long enough not to
/// cut off a thinker, short enough to keep a voice coding loop responsive.
public struct VoiceEndpointingConfiguration: Sendable, Equatable {
    public var speechThresholdDecibels: Float
    public var silenceThresholdDecibels: Float
    public var speechConfirmationDuration: TimeInterval
    public var preRollDuration: TimeInterval
    public var minimumSpeechDuration: TimeInterval
    public var trailingSilenceDuration: TimeInterval
    public var maximumUtteranceDuration: TimeInterval

    public init(
        speechThresholdDecibels: Float = -38,
        silenceThresholdDecibels: Float = -45,
        speechConfirmationDuration: TimeInterval = 0.08,
        preRollDuration: TimeInterval = 0.20,
        minimumSpeechDuration: TimeInterval = 0.12,
        trailingSilenceDuration: TimeInterval = 0.62,
        maximumUtteranceDuration: TimeInterval = 20
    ) {
        // A lower silence threshold creates hysteresis. Normalize malformed
        // configuration instead of making a microphone permanently "speaking".
        self.speechThresholdDecibels = speechThresholdDecibels
        self.silenceThresholdDecibels = min(silenceThresholdDecibels, speechThresholdDecibels)
        self.speechConfirmationDuration = max(0, speechConfirmationDuration)
        self.preRollDuration = max(0, preRollDuration)
        self.minimumSpeechDuration = max(0, minimumSpeechDuration)
        self.trailingSilenceDuration = max(0, trailingSilenceDuration)
        self.maximumUtteranceDuration = max(0.01, maximumUtteranceDuration)
    }
}

/// Meaningful state transitions emitted by ``VoiceEndpointDetector``. The UI
/// can use the first two for live status/metering, while `utteranceReady` owns a
/// complete mono clip suitable for live STT or a final transcription pass.
public enum VoiceEndpointEvent: Sendable, Equatable {
    /// `audio` contains pre-roll plus every confirmation frame, so a live STT
    /// request begins with the first phoneme rather than the confirmation edge.
    case speechStarted(level: VoiceSignalLevel, audio: VoiceAudioFrame)
    /// `audio` is the newly received frame, suitable for direct live-STT append.
    case speechContinued(level: VoiceSignalLevel, audio: VoiceAudioFrame)
    case utteranceReady(samples: [Float], sampleRate: Double)
}

/// Deterministic, pure-native energy endpointing for a serial stream of mono
/// frames. It uses start/continue hysteresis, requires consecutive loud frames
/// before beginning an utterance, retains pre-roll, and ends after sustained
/// silence or the configured maximum length. This is intentionally a small
/// baseline VAD rather than a learned model; callers can replace it later
/// without changing capture or transcript APIs.
public final class VoiceEndpointDetector: @unchecked Sendable {
    /// Reference storage avoids copying the full utterance for every 1,024-frame
    /// callback. Audio is copied only when it crosses the public event boundary.
    private final class BufferedUtterance {
        var samples: [Float]
        var speechSamples: Int
        var trailingSilenceSamples: Int

        init(samples: [Float], speechSamples: Int, trailingSilenceSamples: Int = 0) {
            self.samples = samples
            self.speechSamples = speechSamples
            self.trailingSilenceSamples = trailingSilenceSamples
        }
    }

    private enum State {
        case idle
        case confirming(preRoll: [Float], utterance: BufferedUtterance)
        case active(BufferedUtterance)
    }

    public let configuration: VoiceEndpointingConfiguration
    private var state: State = .idle
    private var recentSamples: [Float] = []
    private var activeSampleRate: Double?

    public init(configuration: VoiceEndpointingConfiguration = .init()) {
        self.configuration = configuration
    }

    /// Feed one frame in capture order. The detector is intentionally mutable
    /// and should be used from the capture session's serial processing queue.
    public func process(_ frame: VoiceAudioFrame) -> [VoiceEndpointEvent] {
        guard frame.sampleRate > 0, !frame.samples.isEmpty else { return [] }
        if let rate = activeSampleRate, abs(rate - frame.sampleRate) > 0.5 {
            reset()
        }
        activeSampleRate = frame.sampleRate

        let level = VoiceSignalLevel.measure(frame.samples)
        let startThreshold = configuration.speechThresholdDecibels
        let continueThreshold = configuration.silenceThresholdDecibels
        let frameCount = frame.samples.count

        switch state {
        case .idle:
            let preRoll = recentSamples
            appendRecent(frame.samples, sampleRate: frame.sampleRate)
            guard level.decibels >= startThreshold else { return [] }
            let candidate = frame.samples
            if Double(frameCount) / frame.sampleRate >= configuration.speechConfirmationDuration {
                let utterance = BufferedUtterance(samples: preRoll + candidate, speechSamples: frameCount)
                state = .active(utterance)
                return [.speechStarted(level: level, audio: .init(samples: Array(utterance.samples), sampleRate: frame.sampleRate))]
            }
            state = .confirming(preRoll: preRoll, utterance: .init(samples: candidate, speechSamples: frameCount))
            return []

        case let .confirming(preRoll, utterance):
            appendRecent(frame.samples, sampleRate: frame.sampleRate)
            guard level.decibels >= startThreshold else {
                // Keep the rejected candidate in history so a real utterance
                // immediately after it still receives natural pre-roll.
                state = .idle
                return []
            }
            utterance.samples += frame.samples
            utterance.speechSamples += frameCount
            if Double(utterance.speechSamples) / frame.sampleRate >= configuration.speechConfirmationDuration {
                // Add pre-roll once, at confirmation time. The active buffer is
                // subsequently appended in place rather than rebuilt per frame.
                utterance.samples = preRoll + utterance.samples
                state = .active(utterance)
                return [.speechStarted(level: level, audio: .init(samples: Array(utterance.samples), sampleRate: frame.sampleRate))]
            }
            return []

        case let .active(utterance):
            utterance.samples += frame.samples
            let speaking = level.decibels >= continueThreshold
            if speaking {
                utterance.speechSamples += frameCount
                utterance.trailingSilenceSamples = 0
            } else {
                utterance.trailingSilenceSamples += frameCount
            }
            let maxSamples = Int((configuration.maximumUtteranceDuration * frame.sampleRate).rounded(.up))
            let endpointSamples = Int((configuration.trailingSilenceDuration * frame.sampleRate).rounded(.up))
            if utterance.samples.count >= maxSamples || utterance.trailingSilenceSamples >= endpointSamples {
                state = .idle
                recentSamples.removeAll(keepingCapacity: true)
                let minSpeechSamples = Int((configuration.minimumSpeechDuration * frame.sampleRate).rounded(.up))
                guard utterance.speechSamples >= minSpeechSamples else { return [] }
                // Max duration must be a real cap, not merely a notification.
                return [.utteranceReady(samples: Array(utterance.samples.prefix(maxSamples)), sampleRate: frame.sampleRate)]
            }
            return [.speechContinued(level: level, audio: frame)]
        }
    }

    /// Discard an in-progress utterance and any pre-roll. Reuse is immediate.
    public func reset() {
        state = .idle
        recentSamples.removeAll(keepingCapacity: true)
        activeSampleRate = nil
    }

    private func appendRecent(_ samples: [Float], sampleRate: Double) {
        recentSamples += samples
        let limit = Int((configuration.preRollDuration * sampleRate).rounded(.up))
        guard recentSamples.count > limit else { return }
        recentSamples.removeFirst(recentSamples.count - limit)
    }
}

/// Shared, platform-independent Apple speech selection. Empty and `auto`
/// language values intentionally mean the current system locale. A rate of zero
/// is the stable config sentinel for AVSpeech's system default.
public struct AppleSpeechSettings: Sendable, Equatable {
    public static let systemRate: Float = 0

    public var language: String
    public var voiceIdentifier: String
    public var rate: Float

    public init(language: String = "auto", voiceIdentifier: String = "", rate: Float = AppleSpeechSettings.systemRate) {
        self.language = language
        self.voiceIdentifier = voiceIdentifier
        self.rate = rate
    }

    public var localeIdentifier: String {
        let value = language.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty || value.lowercased() == "auto" ? Locale.current.identifier : value
    }

    public var requestedVoiceIdentifier: String? {
        let value = voiceIdentifier.trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    /// AVSpeech accepts a limited positive range. Invalid configuration safely
    /// returns nil so callers keep Apple's system default rather than silence.
    public var utteranceRate: Float? {
        guard rate > 0, rate <= 1 else { return nil }
        return rate
    }
}

public enum VoiceTranscriptionError: Error, Sendable, LocalizedError {
    case unavailable
    case authorizationDenied
    case noTranscription

    public var errorDescription: String? {
        switch self {
        case .unavailable: return "On-device speech recognition is unavailable for the selected language."
        case .authorizationDenied: return "Speech recognition permission was denied. Grant it in System Settings › Privacy & Security › Speech Recognition."
        case .noTranscription: return "No speech was recognized."
        }
    }
}

/// Adapter that exposes Apple's one-shot recognizer through the generic voice
/// seam while preserving `SpeechRecognizer`'s optional-returning legacy API.
public final class AppleVoiceTranscriber: VoiceTranscriber, @unchecked Sendable {
    private let recognizer: SpeechRecognizer

    public init(settings: AppleSpeechSettings = .init()) {
        self.recognizer = SpeechRecognizer(language: settings.language)
    }

    public func transcribe(_ wav: Data) async throws -> String {
        guard SpeechRecognizer.isAvailable(language: recognizer.language) else {
            throw VoiceTranscriptionError.unavailable
        }
        guard let text = await recognizer.transcribe(wav: wav),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw VoiceTranscriptionError.noTranscription
        }
        return text
    }
}

/// Adapter for the existing Apple synthesizer. `speak` starts local playback;
/// it does not wait for the utterance to finish, matching AVSpeech's semantics.
public final class AppleVoiceOutput: VoiceOutput, @unchecked Sendable {
    private let synthesizer: SpeechSynthesizer

    public init(settings: AppleSpeechSettings = .init()) {
        self.synthesizer = SpeechSynthesizer(
            language: settings.language,
            voiceIdentifier: settings.voiceIdentifier,
            rate: settings.rate)
    }

    public func speak(_ text: String) async throws {
        synthesizer.speak(text)
    }

    public func stop() { synthesizer.stop() }
}
