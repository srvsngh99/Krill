import Foundation
#if canImport(AVFoundation)
@preconcurrency import AVFoundation
#endif

/// Continuously captures mono microphone frames for endpointing and live
/// transcription. It owns a separate `AVAudioEngine`; do not share it with the
/// legacy ``MicrophoneRecorder`` because AVAudioInputNode supports only one tap
/// per bus.
///
/// The audio tap does the unavoidable copy/mix into a Swift array, then hands
/// the frame to a serial, non-UI queue. A lock-free ring buffer would further
/// reduce real-time allocation pressure, but is deliberately deferred from this
/// prototype to keep correctness and ownership obvious.
public final class ContinuousMicrophoneCapture: @unchecked Sendable {
    public typealias AudioFrameHandler = @Sendable (VoiceAudioFrame) -> Void

    private let lock = NSLock()
    private let lifecycleQueue = DispatchQueue(label: "krill.voice.capture.lifecycle")
    private let processingQueue = DispatchQueue(label: "krill.voice.capture.processing", qos: .userInitiated)
    private var handler: AudioFrameHandler
    private var running = false
    private var voiceProcessingEnabled = false

#if canImport(AVFoundation)
    private let engine = AVAudioEngine()
#endif

    public init(onAudioFrame: @escaping AudioFrameHandler) {
        self.handler = onAudioFrame
    }

    /// Replace the receiver used on the serial processing queue. This is useful
    /// when a UI view is recreated without restarting the microphone session.
    public func setAudioFrameHandler(_ handler: @escaping AudioFrameHandler) {
        lock.lock()
        self.handler = handler
        lock.unlock()
    }

    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return running
    }

    /// Whether best-effort system echo/noise processing was enabled on the
    /// input node for the current run. It may be false on unsupported hardware.
    public var isVoiceProcessingEnabled: Bool {
        lock.lock(); defer { lock.unlock() }
        return voiceProcessingEnabled
    }

    /// Start capture. Calling this more than once is safe and does not install a
    /// second tap.
    public func start() throws {
#if canImport(AVFoundation)
        try lifecycleQueue.sync {
            try startSerialized()
        }
#else
        throw MicrophoneCaptureError.unavailable("AVFoundation not available on this platform")
#endif
    }

#if canImport(AVFoundation)
    /// All AVAudioEngine lifecycle mutation is serialized here. Without this,
    /// two Sendable callers can both observe `running == false` and install two
    /// taps, or a concurrent stop can return before a start finishes.
    private func startSerialized() throws {
        lock.lock()
        if running {
            lock.unlock()
            return
        }
        lock.unlock()

        let input = engine.inputNode
        // This can fail for devices/virtual routes without voice-processing I/O;
        // audio capture still works, and the public state tells the UI the truth.
        let processingActive: Bool
        do {
            try input.setVoiceProcessingEnabled(true)
            processingActive = input.isVoiceProcessingEnabled
        } catch {
            processingActive = false
        }

        // Voice-processing I/O can select a different hardware route/format, so
        // inspect the input only after attempting to enable it.
        let format = input.inputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            throw MicrophoneCaptureError.unavailable("no usable input device format")
        }

        let channelCount = Int(format.channelCount)
        input.installTap(onBus: 0, bufferSize: 1_024, format: format) { [weak self] buffer, _ in
            guard let self, let channels = buffer.floatChannelData else { return }
            let frames = Int(buffer.frameLength)
            guard frames > 0 else { return }

            // Keep the audio callback's work bounded: copy/mix only. Endpointing,
            // STT and every user callback run off the real-time audio thread.
            var mono = [Float](repeating: 0, count: frames)
            if channelCount == 1 {
                mono.withUnsafeMutableBufferPointer { destination in
                    destination.baseAddress!.update(from: channels[0], count: frames)
                }
            } else {
                for index in 0 ..< frames {
                    var sum: Float = 0
                    for channel in 0 ..< channelCount { sum += channels[channel][index] }
                    mono[index] = sum / Float(channelCount)
                }
            }
            let frame = VoiceAudioFrame(samples: mono, sampleRate: format.sampleRate)
            self.processingQueue.async { [weak self] in
                guard let self else { return }
                self.lock.lock()
                let isRunning = self.running
                let callback = self.handler
                self.lock.unlock()
                guard isRunning else { return }
                callback(frame)
            }
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            throw MicrophoneCaptureError.unavailable(error.localizedDescription)
        }
        lock.lock()
        running = true
        voiceProcessingEnabled = processingActive
        lock.unlock()
    }
#endif

    /// Stop capture and discard queued-but-not-yet-delivered frames. Repeated
    /// calls are safe.
    public func stop() {
#if canImport(AVFoundation)
        lifecycleQueue.sync {
            stopSerialized()
        }
#endif
    }

#if canImport(AVFoundation)
    private func stopSerialized() {
        lock.lock()
        guard running else {
            lock.unlock()
            return
        }
        running = false
        voiceProcessingEnabled = false
        lock.unlock()

        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
#endif
}
