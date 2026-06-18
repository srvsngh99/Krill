import Foundation
import MLX

// MARK: - Native Whisper log-mel front-end

/// Swift + MLX port of OpenAI Whisper's audio front-end, byte-faithful to
/// `transformers.WhisperFeatureExtractor` / `torch.stft`. Whisper always
/// operates on fixed 30 s windows, so any clip is right zero-padded (or
/// truncated) to `nSamples = 480_000` before the spectrogram, yielding the
/// canonical `[nMels, 3000]` feature map the encoder consumes.
///
/// Pipeline (matches `_torch_extract_fbank_features`):
/// pad/truncate to 30 s -> center reflect-pad `nFft/2` -> framed periodic-Hann
/// STFT (rfft 400, hop 160) -> `|.|^2` magnitudes (drop the last frame) ->
/// Slaney mel projection -> `log10(clamp(., 1e-3? no -> 1e-10))` -> global
/// `max(., max - 8)` -> `(. + 4) / 4`.
///
/// Pure DSP, self-contained, gated against a `WhisperFeatureExtractor` golden
/// reference (`WhisperMelTests`).
public enum WhisperMel {
    public static let sampleRate = 16_000
    public static let nFft = 400
    public static let hopLength = 160
    public static let chunkSeconds = 30
    /// 30 s @ 16 kHz.
    public static let nSamples = sampleRate * chunkSeconds      // 480_000
    /// Canonical frame count for a 30 s window (`nSamples / hop`).
    public static let nFrames = nSamples / hopLength            // 3000
    public static let numFreqBins = nFft / 2 + 1               // 201
    public static let maxFrequency: Float = 8000.0             // sampleRate / 2

    /// Compute the Whisper log-mel features for a mono 16 kHz waveform.
    ///
    /// - Parameters:
    ///   - waveform: mono PCM at 16 kHz.
    ///   - nMels: 80 for tiny/base/small (the published `.en` SKUs), 128 for
    ///     large-v3.
    /// - Returns: `[nFrames, nMels]` float32 (MLX conv1d-friendly: time-major,
    ///   channels last). Transpose for the `[nMels, nFrames]` reference layout.
    public static func logMel(waveform input: [Float], nMels: Int = 80) -> MLXArray {
        // 1. Pad/truncate to a fixed 30 s window (right zero-pad).
        var wave = input
        if wave.count > nSamples {
            wave = Array(wave.prefix(nSamples))
        } else if wave.count < nSamples {
            wave.append(contentsOf: [Float](repeating: 0, count: nSamples - wave.count))
        }

        // 2. Center reflect-pad by nFft/2 (torch.stft center=True default).
        let pad = nFft / 2                                     // 200
        let padded = reflectPad(wave, pad)                    // 480_400

        // 3. Periodic Hann window (torch.hann_window default periodic=True).
        var window = [Float](repeating: 0, count: nFft)
        for n in 0 ..< nFft {
            window[n] = 0.5 - 0.5 * cos(2.0 * .pi * Float(n) / Float(nFft))
        }

        // 4. Frame the padded signal. torch.stft yields 1 + (L - nFft)/hop
        //    frames; Whisper drops the trailing frame to land on `nFrames`.
        let totalFrames = (padded.count - nFft) / hopLength + 1
        let keep = min(nFrames, totalFrames - 1)
        var framesFlat = [Float](repeating: 0, count: keep * nFft)
        for f in 0 ..< keep {
            let start = f * hopLength
            let base = f * nFft
            for n in 0 ..< nFft {
                framesFlat[base + n] = padded[start + n] * window[n]
            }
        }

        // 5. rfft(400) via truncated DFT basis: magnitude^2 = re^2 + im^2.
        let (cosB, sinB) = dftBasis()                         // [nFft, numFreqBins]
        let frames = MLXArray(framesFlat, [keep, nFft])
        let re = frames.matmul(cosB)                          // [keep, 201]
        let im = frames.matmul(sinB)
        let power = re * re + im * im                         // [keep, 201]

        // 6. Slaney mel projection: power[keep,201] @ melFB[201, nMels].
        let melFB = melFilterBank(nMels: nMels)               // [201, nMels]
        var logSpec = MLX.log10(MLX.maximum(power.matmul(melFB), MLXArray(Float(1e-10))))

        // 7. Global floor then affine normalization.
        let maxVal = MLX.max(logSpec)
        logSpec = MLX.maximum(logSpec, maxVal - MLXArray(Float(8.0)))
        logSpec = (logSpec + MLXArray(Float(4.0))) / MLXArray(Float(4.0))

        return logSpec.asType(.float32)                       // [keep, nMels]
    }

    // MARK: - Reflect padding

    /// numpy/torch `mode="reflect"`: mirror without repeating the edge sample.
    static func reflectPad(_ x: [Float], _ pad: Int) -> [Float] {
        guard pad > 0, x.count > 1 else {
            return [Float](repeating: 0, count: pad) + x + [Float](repeating: 0, count: pad)
        }
        var out = [Float]()
        out.reserveCapacity(x.count + 2 * pad)
        for i in 0 ..< pad { out.append(x[pad - i]) }          // x[pad]..x[1]
        out.append(contentsOf: x)
        let n = x.count
        for i in 0 ..< pad { out.append(x[n - 2 - i]) }        // x[n-2]..x[n-1-pad]
        return out
    }

    // MARK: - Cached basis tensors

    /// Real/imag rfft basis truncated to `nFft` input samples.
    /// `cos[n,k] = cos(-2*pi*k*n/nFft)`, `sin[n,k] = sin(...)`.
    static func dftBasis() -> (MLXArray, MLXArray) {
        var cosv = [Float](repeating: 0, count: nFft * numFreqBins)
        var sinv = [Float](repeating: 0, count: nFft * numFreqBins)
        for n in 0 ..< nFft {
            for k in 0 ..< numFreqBins {
                let a = -2.0 * Double.pi * Double(k) * Double(n) / Double(nFft)
                cosv[n * numFreqBins + k] = Float(cos(a))
                sinv[n * numFreqBins + k] = Float(sin(a))
            }
        }
        return (MLXArray(cosv, [nFft, numFreqBins]),
                MLXArray(sinv, [nFft, numFreqBins]))
    }

    // MARK: - Slaney mel filter bank

    /// Slaney-scale, Slaney-normalized triangular mel filter bank,
    /// `[numFreqBins, nMels]`. Exact port of
    /// `transformers.audio_utils.mel_filter_bank(norm="slaney",
    /// mel_scale="slaney")`.
    static func melFilterBank(nMels: Int) -> MLXArray {
        let melMin = hzToMelSlaney(0.0)
        let melMax = hzToMelSlaney(Double(maxFrequency))
        let nPts = nMels + 2
        var filterFreqs = [Double](repeating: 0, count: nPts)
        for i in 0 ..< nPts {
            let mel = melMin + Double(i) * (melMax - melMin) / Double(nPts - 1)
            filterFreqs[i] = melToHzSlaney(mel)
        }

        // fft bin center frequencies: linspace(0, sr/2, numFreqBins).
        var fftFreqs = [Double](repeating: 0, count: numFreqBins)
        let nyquist = Double(sampleRate) / 2.0
        for k in 0 ..< numFreqBins {
            fftFreqs[k] = Double(k) * nyquist / Double(numFreqBins - 1)
        }

        // Triangular filters + Slaney area normalization.
        var fb = [Float](repeating: 0, count: numFreqBins * nMels)
        for m in 0 ..< nMels {
            let lower = filterFreqs[m]
            let center = filterFreqs[m + 1]
            let upper = filterFreqs[m + 2]
            let dLow = center - lower
            let dHigh = upper - center
            let enorm = 2.0 / (upper - lower)                 // Slaney norm
            for k in 0 ..< numFreqBins {
                let freq = fftFreqs[k]
                let down = (freq - lower) / dLow
                let up = (upper - freq) / dHigh
                let v = max(0.0, min(down, up)) * enorm
                fb[k * nMels + m] = Float(v)
            }
        }
        return MLXArray(fb, [numFreqBins, nMels])
    }

    /// Slaney hertz->mel: linear below 1 kHz, log above.
    static func hzToMelSlaney(_ freq: Double) -> Double {
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp                        // 15.0
        let logstep = log(6.4) / 27.0
        if freq >= minLogHz {
            return minLogMel + log(freq / minLogHz) / logstep
        }
        return freq / fSp
    }

    /// Slaney mel->hertz (inverse of `hzToMelSlaney`).
    static func melToHzSlaney(_ mel: Double) -> Double {
        let fSp = 200.0 / 3.0
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp                        // 15.0
        let logstep = log(6.4) / 27.0
        if mel >= minLogMel {
            return minLogHz * exp(logstep * (mel - minLogMel))
        }
        return fSp * mel
    }
}
