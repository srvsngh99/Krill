import Foundation
import MLX

// MARK: - Native Gemma 4 Audio Preprocessor (USM feature extractor)

/// Swift port of HuggingFace / `mlx-vlm` `Gemma4AudioFeatureExtractor`.
///
/// The local Gemma 4 checkpoint ships **no** `feature_extractor` config, so
/// every constant below is the fixed USM default (see
/// `docs/GEMMA4_INTERNALS.md` "Audio Encoder"). This produces the exact
/// log-mel input the native `AudioEncoder` (and the `mlx-vlm` oracle)
/// consumes, so native and bridge outputs are comparable.
///
/// Pipeline: WAV -> mono -> resample 16 kHz -> truncate 30 s -> pad to
/// multiple of 128 -> semicausal left-pad -> framed periodic-Hann STFT
/// (rfft 512) -> |.| -> HTK mel (257x128) -> log(.+1e-3) -> zero padded
/// frames.
public enum AudioPreprocessor {
    // USM fixed constants (Gemma4AudioFeatureExtractor defaults).
    public static let sampleRate = 16_000
    public static let melBins = 128
    static let frameLength = 320          // round(16000 * 20ms / 1000)
    static let hopLength = 160            // round(16000 * 10ms / 1000)
    static let fftLength = 512            // 2^ceil(log2(320))
    static let numFreqBins = 257          // fftLength/2 + 1
    static let melFloor: Float = 1e-3
    static let minFrequency: Float = 0.0
    static let maxFrequency: Float = 8000.0
    static let maxSamples = 480_000       // 30 s @ 16 kHz
    static let padToMultipleOf = 128
    /// Soft-token cadence: one audio token per 40 ms, capped at 750.
    static let audioMsPerToken = 40
    static let audioSeqLength = 750

    /// Result of preprocessing one waveform.
    public struct Features {
        /// Log-mel spectrogram, shape `[1, T, 128]`, float32.
        public let mel: MLXArray
        /// Per-frame validity, shape `[1, T]` bool. `true` = real audio,
        /// `false` = padding. (The tower wants the inverted "invalid" mask;
        /// the `AudioEncoder` wrapper performs that inversion.)
        public let validMask: MLXArray
        /// Number of `<|audio|>` soft tokens to place in the prompt:
        /// `ceil(duration_ms / 40)` capped at 750. The 2x stride-2 conv
        /// subsampling makes the tower emit this many frames.
        public let numTokens: Int
    }

    /// True iff `data` begins with a RIFF/WAVE header. The native frontend
    /// is WAV-PCM-only; callers use this to keep non-WAV codecs
    /// (mp3/flac/ogg/m4a) on the `mlx-vlm` bridge instead of failing here.
    public static func isWAV(_ data: Data) -> Bool {
        guard data.count >= 12 else { return false }
        let b = [UInt8](data.prefix(12))
        return b[0] == 0x52 && b[1] == 0x49 && b[2] == 0x46 && b[3] == 0x46
            && b[8] == 0x57 && b[9] == 0x41 && b[10] == 0x56 && b[11] == 0x45
    }

    /// Decode WAV bytes and produce native audio features.
    public static func features(fromWAV wavData: Data) throws -> Features {
        let (raw, fileSR) = try loadWAV(from: wavData)
        let mono = resampleAudio(raw, from: fileSR, to: sampleRate)
        return try features(waveform: mono)
    }

    /// Produce features from a mono 16 kHz Float waveform.
    public static func features(waveform input: [Float]) throws -> Features {
        guard !input.isEmpty else {
            throw MultimodalPreprocessingError.audioPreprocessingUnavailable
        }

        // Truncate to 30 s, like the extractor's `max_length`.
        var waveform = input
        if waveform.count > maxSamples { waveform = Array(waveform.prefix(maxSamples)) }
        let originalSamples = waveform.count

        // `padding="longest"` for a single clip == pad up to a multiple of
        // 128 samples; mask 1 for real samples, 0 for the tail pad.
        var target = waveform.count
        if target % padToMultipleOf != 0 {
            target = (target / padToMultipleOf + 1) * padToMultipleOf
        }
        var attnMask = [Int](repeating: 1, count: target)
        if waveform.count < target {
            for i in waveform.count ..< target { attnMask[i] = 0 }
            waveform.append(contentsOf: [Float](repeating: 0, count: target - waveform.count))
        }

        // Semicausal left-pad: prepend frame_length/2 zeros so the first
        // frame is centered at t=0 (HuggingFace parity). Mask pad = 0.
        let padLeft = frameLength / 2          // 160
        var padded = [Float](repeating: 0, count: padLeft)
        padded.append(contentsOf: waveform)
        var maskPadded = [Int](repeating: 0, count: padLeft)
        maskPadded.append(contentsOf: attnMask)

        // Unfold size = frame_length + 1; preemphasis is 0 so we drop the
        // trailing sample and keep `frameLength` per frame.
        let frameSizeForUnfold = frameLength + 1   // 321
        let L = padded.count
        let numFrames = L >= frameSizeForUnfold
            ? (L - frameSizeForUnfold) / hopLength + 1
            : 0
        guard numFrames > 0 else {
            throw MultimodalPreprocessingError.audioPreprocessingUnavailable
        }

        // Periodic Hann window over `frameLength` samples.
        var window = [Float](repeating: 0, count: frameLength)
        for n in 0 ..< frameLength {
            window[n] = 0.5 - 0.5 * cos(2.0 * .pi * Float(n) / Float(frameLength))
        }

        // Flatten windowed frames into one [numFrames * frameLength] buffer.
        var framesFlat = [Float](repeating: 0, count: numFrames * frameLength)
        for f in 0 ..< numFrames {
            let start = f * hopLength
            let base = f * frameLength
            for n in 0 ..< frameLength {
                framesFlat[base + n] = padded[start + n] * window[n]
            }
        }

        // rfft(n=512) of each 320-sample windowed frame == frame · DFT
        // basis truncated to the first `frameLength` rows (numpy zero-pads
        // to `fftLength`). magnitude = sqrt(re^2 + im^2).
        let (cosB, sinB) = Self.dftBasis()      // [frameLength, numFreqBins] each
        let frames = MLXArray(framesFlat, [numFrames, frameLength])
        let re = frames.matmul(cosB)            // [numFrames, 257]
        let im = frames.matmul(sinB)
        let magnitude = MLX.sqrt(re * re + im * im)

        // HTK mel projection then log(. + mel_floor). No per-bin norm.
        let melFB = Self.melFilterBank()        // [numFreqBins, melBins]
        let melSpec = magnitude.matmul(melFB)   // [numFrames, 128]
        var logMel = MLX.log(melSpec + MLXArray(melFloor))

        // Per-frame validity: a frame is valid iff its end sample is real.
        // frame_end_index = i*hop + frame_size_for_unfold - 1.
        var valid = [Bool](repeating: false, count: numFrames)
        for i in 0 ..< numFrames {
            let endIdx = i * hopLength + frameSizeForUnfold - 1
            valid[i] = endIdx < maskPadded.count ? (maskPadded[endIdx] != 0) : false
        }
        let validMLX = MLXArray(valid.map { $0 ? Int32(1) : 0 }, [numFrames])
            .asType(.bool)

        // Zero out padded spectrogram rows (HuggingFace parity).
        let validF = validMLX.asType(.float32).reshaped([numFrames, 1])
        logMel = logMel * validF

        // Soft-token count from the *original* (pre-pad) duration.
        let durationMs = Double(originalSamples) / Double(sampleRate) * 1000.0
        let numTokens = min(
            Int(ceil(durationMs / Double(audioMsPerToken))), audioSeqLength)

        return Features(
            mel: logMel.reshaped([1, numFrames, melBins]).asType(.float32),
            validMask: validMLX.reshaped([1, numFrames]),
            numTokens: max(1, numTokens))
    }

    // MARK: - Cached basis tensors

    /// Real/imag DFT basis truncated to `frameLength` input samples.
    /// `cos[n,k] = cos(-2*pi*k*n/fftLength)`, `sin[n,k] = sin(...)`.
    /// Recomputed per call (small: 320x257) to stay value-only / Sendable.
    static func dftBasis() -> (MLXArray, MLXArray) {
        var cosv = [Float](repeating: 0, count: frameLength * numFreqBins)
        var sinv = [Float](repeating: 0, count: frameLength * numFreqBins)
        for n in 0 ..< frameLength {
            for k in 0 ..< numFreqBins {
                let a = -2.0 * Double.pi * Double(k) * Double(n) / Double(fftLength)
                cosv[n * numFreqBins + k] = Float(cos(a))
                sinv[n * numFreqBins + k] = Float(sin(a))
            }
        }
        return (MLXArray(cosv, [frameLength, numFreqBins]),
                MLXArray(sinv, [frameLength, numFreqBins]))
    }

    /// HTK mel filter bank `[numFreqBins, melBins]`, `norm=None`. Exact port
    /// of `transformers.audio_utils.mel_filter_bank` (htk) / the `mlx-vlm`
    /// `_mel_filter_bank` fallback.
    static func melFilterBank() -> MLXArray {
        func hzToMel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
        func melToHz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }

        let melMin = hzToMel(Double(minFrequency))
        let melMax = hzToMel(Double(maxFrequency))
        let nPts = melBins + 2
        var melPoints = [Double](repeating: 0, count: nPts)
        for i in 0 ..< nPts {
            melPoints[i] = melMin + Double(i) * (melMax - melMin) / Double(nPts - 1)
        }
        let freqPoints = melPoints.map { melToHz($0) }

        // all_freqs[k] = k * sampling_rate / (2 * (numFreqBins - 1))
        let step = Double(sampleRate) / Double(2 * (numFreqBins - 1))
        var fb = [Float](repeating: 0, count: numFreqBins * melBins)
        for i in 0 ..< melBins {
            let lower = freqPoints[i]
            let center = freqPoints[i + 1]
            let upper = freqPoints[i + 2]
            let dLow = max(center - lower, 1e-10)
            let dHigh = max(upper - center, 1e-10)
            for k in 0 ..< numFreqBins {
                let freq = Double(k) * step
                let rising = (freq - lower) / dLow
                let falling = (upper - freq) / dHigh
                let v = max(0.0, min(rising, falling))
                fb[k * melBins + i] = Float(v)
            }
        }
        return MLXArray(fb, [numFreqBins, melBins])
    }
}
