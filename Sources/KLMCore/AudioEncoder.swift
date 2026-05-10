import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Audio Encoder (Conformer for Gemma 4)

/// Conformer-based audio encoder for Gemma 4 multimodal models.
///
/// Pipeline: Waveform (16kHz) -> Mel Spectrogram -> Conformer -> Project
/// Produces audio tokens at 25 tokens/second (max 30s = 750 tokens).
public class AudioEncoder: Module {
    @ModuleInfo(key: "conformer") var conformer: ConformerStack
    @ModuleInfo(key: "projector") var projector: Linear

    let sampleRate: Int
    let melBins: Int
    let frameMs: Int
    let maxSeconds: Int

    public init(
        audioHiddenSize: Int = 512,
        conformerLayers: Int = 12,
        conformerHeads: Int = 8,
        melBins: Int = 128,
        projectedSize: Int = 2048,
        sampleRate: Int = 16000,
        frameMs: Int = 40,
        maxSeconds: Int = 30
    ) {
        self.sampleRate = sampleRate
        self.melBins = melBins
        self.frameMs = frameMs
        self.maxSeconds = maxSeconds

        _conformer = ModuleInfo(
            wrappedValue: ConformerStack(
                inputSize: melBins, hiddenSize: audioHiddenSize,
                numLayers: conformerLayers, numHeads: conformerHeads),
            key: "conformer")
        _projector = ModuleInfo(
            wrappedValue: Linear(audioHiddenSize, projectedSize, bias: true),
            key: "projector")
    }

    /// Encode audio waveform into tokens for the LLM.
    ///
    /// - Parameter melSpectrogram: Pre-computed mel spectrogram [1, numFrames, melBins]
    /// - Returns: Audio tokens [1, numTokens, projectedSize]
    public func callAsFunction(_ melSpectrogram: MLXArray) -> MLXArray {
        let encoded = conformer(melSpectrogram)
        return projector(encoded)
    }

    /// Maximum number of audio tokens (30s at 25 tok/s).
    public var maxTokens: Int { maxSeconds * (1000 / frameMs) }
}

// MARK: - Conformer Stack

/// Stack of Conformer blocks for audio processing.
class ConformerStack: Module {
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "layers") var layers: [ConformerBlock]

    init(inputSize: Int, hiddenSize: Int, numLayers: Int, numHeads: Int) {
        _inputProj = ModuleInfo(
            wrappedValue: Linear(inputSize, hiddenSize, bias: true),
            key: "input_proj")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< numLayers).map { _ in
                ConformerBlock(hiddenSize: hiddenSize, numHeads: numHeads)
            },
            key: "layers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = inputProj(x)
        for layer in layers {
            hidden = layer(hidden)
        }
        return hidden
    }
}

// MARK: - Conformer Block

/// A single Conformer block: FFN -> Self-Attention -> Conv1D -> FFN -> LayerNorm
class ConformerBlock: Module {
    @ModuleInfo(key: "ffn1") var ffn1: ConformerFFN
    @ModuleInfo(key: "self_attn") var selfAttn: ConformerAttention
    @ModuleInfo(key: "conv") var conv: ConformerConv
    @ModuleInfo(key: "ffn2") var ffn2: ConformerFFN
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(hiddenSize: Int, numHeads: Int) {
        _ffn1 = ModuleInfo(wrappedValue: ConformerFFN(hiddenSize: hiddenSize), key: "ffn1")
        _selfAttn = ModuleInfo(
            wrappedValue: ConformerAttention(hiddenSize: hiddenSize, numHeads: numHeads),
            key: "self_attn")
        _conv = ModuleInfo(wrappedValue: ConformerConv(hiddenSize: hiddenSize), key: "conv")
        _ffn2 = ModuleInfo(wrappedValue: ConformerFFN(hiddenSize: hiddenSize), key: "ffn2")
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Macaron-style: half-step FFN + attention + conv + half-step FFN + norm
        var h = x + 0.5 * ffn1(x)
        h = h + selfAttn(h)
        h = h + conv(h)
        h = h + 0.5 * ffn2(h)
        return norm(h)
    }
}

// MARK: - Conformer Sub-modules

class ConformerFFN: Module {
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(hiddenSize: Int) {
        _linear1 = ModuleInfo(
            wrappedValue: Linear(hiddenSize, hiddenSize * 4, bias: true), key: "linear1")
        _linear2 = ModuleInfo(
            wrappedValue: Linear(hiddenSize * 4, hiddenSize, bias: true), key: "linear2")
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(silu(linear1(norm(x))))
    }
}

class ConformerAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Float(hiddenSize / numHeads).squareRoot()

        _qProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "o_proj")
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = norm(x)
        let B = normed.dim(0)
        let L = normed.dim(1)

        let q = qProj(normed).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(normed).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(normed).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class ConformerConv: Module {
    @ModuleInfo(key: "pointwise1") var pointwise1: Linear
    @ModuleInfo(key: "pointwise2") var pointwise2: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(hiddenSize: Int) {
        // Pointwise convolutions (1x1 conv = linear on last dim)
        _pointwise1 = ModuleInfo(
            wrappedValue: Linear(hiddenSize, hiddenSize * 2, bias: true), key: "pointwise1")
        _pointwise2 = ModuleInfo(
            wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "pointwise2")
        _norm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = norm(x)
        // GLU gating: split pointwise1 output into two halves
        let pw1 = pointwise1(normed)
        let hiddenSize = x.dim(2)
        let gate = pw1[0..., 0..., ..<hiddenSize]
        let value = pw1[0..., 0..., hiddenSize...]
        let gated = silu(gate) * value
        return pointwise2(gated)
    }
}

// MARK: - Audio Preprocessing

/// Load a WAV file and return PCM samples as a Float array at the file's sample rate.
///
/// Supports 16-bit and 32-bit float PCM in standard RIFF/WAVE format.
/// Multi-channel audio is mixed down to mono by averaging channels.
public func loadWAV(from data: Data) throws -> (samples: [Float], sampleRate: Int) {
    guard data.count > 44 else {
        throw MultimodalPreprocessingError.audioPreprocessingUnavailable
    }

    // Parse RIFF/WAVE header
    let bytes = [UInt8](data)
    guard bytes[0] == 0x52, bytes[1] == 0x49, bytes[2] == 0x46, bytes[3] == 0x46, // "RIFF"
          bytes[8] == 0x57, bytes[9] == 0x41, bytes[10] == 0x56, bytes[11] == 0x45  // "WAVE"
    else {
        throw MultimodalPreprocessingError.audioPreprocessingUnavailable
    }

    // Find fmt chunk
    var offset = 12
    var audioFormat: UInt16 = 0
    var numChannels: UInt16 = 0
    var fileSampleRate: UInt32 = 0
    var bitsPerSample: UInt16 = 0

    while offset + 8 <= bytes.count {
        let chunkId = String(bytes: bytes[offset..<offset+4], encoding: .ascii) ?? ""
        let chunkSize = Int(UInt32(bytes[offset+4]) | (UInt32(bytes[offset+5]) << 8)
                            | (UInt32(bytes[offset+6]) << 16) | (UInt32(bytes[offset+7]) << 24))
        offset += 8

        if chunkId == "fmt " {
            guard chunkSize >= 16, offset + 16 <= bytes.count else { break }
            audioFormat = UInt16(bytes[offset]) | (UInt16(bytes[offset+1]) << 8)
            numChannels = UInt16(bytes[offset+2]) | (UInt16(bytes[offset+3]) << 8)
            fileSampleRate = UInt32(bytes[offset+4]) | (UInt32(bytes[offset+5]) << 8)
                           | (UInt32(bytes[offset+6]) << 16) | (UInt32(bytes[offset+7]) << 24)
            bitsPerSample = UInt16(bytes[offset+14]) | (UInt16(bytes[offset+15]) << 8)
            offset += chunkSize
            continue
        }

        if chunkId == "data" {
            guard audioFormat == 1 || audioFormat == 3 else {
                throw MultimodalPreprocessingError.audioPreprocessingUnavailable
            }
            let channels = Int(numChannels)
            let dataEnd = min(offset + chunkSize, bytes.count)

            var samples: [Float]
            if audioFormat == 1 && bitsPerSample == 16 {
                // PCM 16-bit signed
                let sampleCount = (dataEnd - offset) / 2
                var raw = [Float](repeating: 0, count: sampleCount)
                for i in 0 ..< sampleCount {
                    let idx = offset + i * 2
                    guard idx + 1 < bytes.count else { break }
                    let val = Int16(bitPattern: UInt16(bytes[idx]) | (UInt16(bytes[idx+1]) << 8))
                    raw[i] = Float(val) / 32768.0
                }
                samples = raw
            } else if audioFormat == 3 && bitsPerSample == 32 {
                // IEEE Float 32-bit
                let sampleCount = (dataEnd - offset) / 4
                var raw = [Float](repeating: 0, count: sampleCount)
                data.withUnsafeBytes { buf in
                    let floatPtr = buf.baseAddress!.advanced(by: offset)
                        .assumingMemoryBound(to: Float.self)
                    for i in 0 ..< sampleCount {
                        raw[i] = floatPtr[i]
                    }
                }
                samples = raw
            } else {
                throw MultimodalPreprocessingError.audioPreprocessingUnavailable
            }

            // Mix to mono if multi-channel
            if channels > 1 {
                let monoCount = samples.count / channels
                var mono = [Float](repeating: 0, count: monoCount)
                for i in 0 ..< monoCount {
                    var sum: Float = 0
                    for ch in 0 ..< channels {
                        sum += samples[i * channels + ch]
                    }
                    mono[i] = sum / Float(channels)
                }
                samples = mono
            }

            return (samples, Int(fileSampleRate))
        }

        offset += chunkSize
    }

    throw MultimodalPreprocessingError.audioPreprocessingUnavailable
}

/// Resample audio from one sample rate to another using linear interpolation.
func resampleAudio(_ samples: [Float], from srcRate: Int, to dstRate: Int) -> [Float] {
    guard srcRate != dstRate, !samples.isEmpty else { return samples }
    let ratio = Float(srcRate) / Float(dstRate)
    let outCount = Int(Float(samples.count) / ratio)
    var result = [Float](repeating: 0, count: outCount)
    for i in 0 ..< outCount {
        let srcPos = Float(i) * ratio
        let idx = Int(srcPos)
        let frac = srcPos - Float(idx)
        if idx + 1 < samples.count {
            result[i] = samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
        } else if idx < samples.count {
            result[i] = samples[idx]
        }
    }
    return result
}

/// Build a mel filterbank matrix.
///
/// - Parameters:
///   - nMels: Number of mel bands
///   - nFft: FFT size
///   - sampleRate: Audio sample rate
/// - Returns: Filterbank matrix [nFft/2 + 1, nMels]
func melFilterbank(nMels: Int, nFft: Int, sampleRate: Int) -> [[Float]] {
    func hzToMel(_ hz: Float) -> Float { 2595.0 * log10(1.0 + hz / 700.0) }
    func melToHz(_ mel: Float) -> Float { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

    let fMax = Float(sampleRate) / 2.0
    let melMin = hzToMel(0)
    let melMax = hzToMel(fMax)
    let nFreqs = nFft / 2 + 1

    // Equally spaced mel points
    var melPoints = [Float](repeating: 0, count: nMels + 2)
    for i in 0 ..< nMels + 2 {
        melPoints[i] = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
    }

    // Convert to frequency bin indices
    var binPoints = [Float](repeating: 0, count: nMels + 2)
    for i in 0 ..< nMels + 2 {
        binPoints[i] = melToHz(melPoints[i]) * Float(nFft) / Float(sampleRate)
    }

    // Create triangular filters
    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nMels), count: nFreqs)
    for m in 0 ..< nMels {
        let left = binPoints[m]
        let center = binPoints[m + 1]
        let right = binPoints[m + 2]

        for k in 0 ..< nFreqs {
            let freq = Float(k)
            if freq >= left && freq <= center && center > left {
                filterbank[k][m] = (freq - left) / (center - left)
            } else if freq > center && freq <= right && right > center {
                filterbank[k][m] = (right - freq) / (right - center)
            }
        }
    }
    return filterbank
}

/// Compute log-mel spectrogram from raw audio waveform.
///
/// Uses a simple DFT implementation (no FFT dependency needed for short frames).
///
/// - Parameters:
///   - waveform: Audio samples at target sample rate as [Float]
///   - sampleRate: Sample rate of the waveform (default 16000)
///   - melBins: Number of mel frequency bins (default 128)
///   - frameMs: Frame duration in milliseconds (default 40)
///   - hopMs: Hop duration in milliseconds (default frameMs / 2)
/// - Returns: Log-mel spectrogram as MLXArray [1, numFrames, melBins]
public func computeMelSpectrogram(
    waveform: [Float]? = nil,
    sampleRate: Int = 16000,
    melBins: Int = 128,
    frameMs: Int = 40,
    hopMs: Int? = nil
) throws -> MLXArray {
    guard let waveform, !waveform.isEmpty else {
        throw MultimodalPreprocessingError.audioPreprocessingUnavailable
    }

    let nFft = sampleRate * frameMs / 1000  // 640 for 16kHz @ 40ms
    let hopLength = sampleRate * (hopMs ?? (frameMs / 2)) / 1000  // 320 for 20ms hop
    let numFrames = max(1, (waveform.count - nFft) / hopLength + 1)
    let nFreqs = nFft / 2 + 1

    // Hann window
    var window = [Float](repeating: 0, count: nFft)
    for i in 0 ..< nFft {
        window[i] = 0.5 * (1.0 - cos(2.0 * .pi * Float(i) / Float(nFft)))
    }

    // Mel filterbank
    let filters = melFilterbank(nMels: melBins, nFft: nFft, sampleRate: sampleRate)

    // Compute spectrogram frame by frame
    var melSpec = [Float](repeating: 0, count: numFrames * melBins)

    for frame in 0 ..< numFrames {
        let start = frame * hopLength

        // Windowed frame
        var windowed = [Float](repeating: 0, count: nFft)
        for i in 0 ..< nFft {
            let idx = start + i
            if idx < waveform.count {
                windowed[i] = waveform[idx] * window[i]
            }
        }

        // DFT magnitude squared (real-valued input, only need positive freqs)
        var powerSpectrum = [Float](repeating: 0, count: nFreqs)
        for k in 0 ..< nFreqs {
            var real: Float = 0
            var imag: Float = 0
            let freqK = -2.0 * .pi * Float(k) / Float(nFft)
            for n in 0 ..< nFft {
                let angle = freqK * Float(n)
                real += windowed[n] * cos(angle)
                imag += windowed[n] * sin(angle)
            }
            powerSpectrum[k] = real * real + imag * imag
        }

        // Apply mel filterbank and take log
        for m in 0 ..< melBins {
            var energy: Float = 0
            for k in 0 ..< nFreqs {
                energy += powerSpectrum[k] * filters[k][m]
            }
            melSpec[frame * melBins + m] = log(max(energy, 1e-10))
        }
    }

    return MLXArray(melSpec, [1, numFrames, melBins]).asType(.bfloat16)
}

/// Compute log-mel spectrogram from WAV file data.
///
/// Convenience that loads the WAV, resamples to 16kHz, and computes the spectrogram.
public func computeMelSpectrogramFromWAV(
    _ wavData: Data,
    targetSampleRate: Int = 16000,
    melBins: Int = 128,
    frameMs: Int = 40
) throws -> MLXArray {
    let (samples, fileSR) = try loadWAV(from: wavData)
    let resampled = resampleAudio(samples, from: fileSR, to: targetSampleRate)
    return try computeMelSpectrogram(
        waveform: resampled,
        sampleRate: targetSampleRate,
        melBins: melBins,
        frameMs: frameMs
    )
}
