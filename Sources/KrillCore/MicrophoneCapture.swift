import Foundation
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// Errors raised while capturing live microphone audio.
public enum MicrophoneCaptureError: Error, CustomStringConvertible {
    /// Microphone access was denied (or restricted) by the OS / user.
    case permissionDenied
    /// Capture could not start or produced no audio.
    case unavailable(String)

    public var description: String {
        switch self {
        case .permissionDenied:
            return "Microphone access was denied. Grant it in System Settings › Privacy & Security › Microphone."
        case .unavailable(let why):
            return "Microphone capture unavailable: \(why)"
        }
    }
}

/// Records mono audio from the default input device and returns it as WAV bytes
/// suitable for `AudioPreprocessor` / `InferenceEngine.generate(audioData:)`.
///
/// Captures at the device's native sample rate (typically 44.1/48 kHz) and
/// averages channels to mono; the existing audio pipeline downsamples to the
/// model's 16 kHz. Designed for a press-to-start / press-Enter-to-stop REPL
/// flow: `start()` returns immediately and audio accumulates on the engine's
/// real-time thread until `stop()`.
///
/// macOS gates microphone access behind TCC keyed on the running app's bundle:
/// a bare CLI binary run from a terminal inherits the terminal's permission,
/// while the packaged `krill.app` (with `NSMicrophoneUsageDescription`) prompts
/// under its own identity. `requestAccess()` triggers that prompt.
public final class MicrophoneRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [Float] = []
    private var capturedRate: Int = 16_000
    private var started = false

#if canImport(AVFoundation)
    private let engine = AVAudioEngine()
#endif

    public init() {}

    /// Number of seconds captured so far (thread-safe). Useful for a live
    /// duration readout.
    public var capturedSeconds: Double {
        lock.lock(); defer { lock.unlock() }
        return capturedRate > 0 ? Double(samples.count) / Double(capturedRate) : 0
    }

    /// Prompt for (or confirm) microphone access. Returns `true` once authorized.
    public static func requestAccess() async -> Bool {
#if canImport(AVFoundation)
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return true
        case .notDetermined:
            return await withCheckedContinuation { cont in
                AVCaptureDevice.requestAccess(for: .audio) { granted in
                    cont.resume(returning: granted)
                }
            }
        default:
            return false
        }
#else
        return false
#endif
    }

    /// Begin capturing. Returns immediately; audio accumulates until `stop()`.
    public func start() throws {
#if canImport(AVFoundation)
        guard !started else { return }
        let input = engine.inputNode
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw MicrophoneCaptureError.unavailable("no usable input device format")
        }
        lock.lock(); capturedRate = Int(format.sampleRate.rounded()); samples.removeAll(); lock.unlock()

        let channelCount = Int(format.channelCount)
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
            guard let self, let chans = buffer.floatChannelData else { return }
            let frames = Int(buffer.frameLength)
            guard frames > 0 else { return }
            var mono = [Float](repeating: 0, count: frames)
            if channelCount == 1 {
                mono.withUnsafeMutableBufferPointer { dst in
                    dst.baseAddress!.update(from: chans[0], count: frames)
                }
            } else {
                for f in 0 ..< frames {
                    var sum: Float = 0
                    for c in 0 ..< channelCount { sum += chans[c][f] }
                    mono[f] = sum / Float(channelCount)
                }
            }
            self.lock.lock(); self.samples.append(contentsOf: mono); self.lock.unlock()
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw MicrophoneCaptureError.unavailable(error.localizedDescription)
        }
        started = true
#else
        throw MicrophoneCaptureError.unavailable("AVFoundation not available on this platform")
#endif
    }

    /// Stop capturing and return the recorded audio as 16-bit PCM WAV bytes.
    /// Throws if nothing was captured.
    @discardableResult
    public func stop() throws -> Data {
#if canImport(AVFoundation)
        guard started else { throw MicrophoneCaptureError.unavailable("recorder was not started") }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        started = false
#endif
        lock.lock(); let captured = samples; let rate = capturedRate; lock.unlock()
        guard !captured.isEmpty else {
            throw MicrophoneCaptureError.unavailable("no audio captured")
        }
        return MediaAttachment.encodeWAV(samples: captured, sampleRate: rate)
    }
}
