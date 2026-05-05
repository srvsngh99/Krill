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

/// Compute log-mel spectrogram from raw audio waveform.
///
/// - Parameters:
///   - waveform: Audio samples at 16kHz as [numSamples]
///   - melBins: Number of mel frequency bins (default 128)
///   - frameMs: Frame duration in milliseconds (default 40)
/// - Returns: Log-mel spectrogram [1, numFrames, melBins]
public func computeMelSpectrogram(
    waveform: MLXArray,
    sampleRate: Int = 16000,
    melBins: Int = 128,
    frameMs: Int = 40
) -> MLXArray? {
    // TODO: Not yet implemented. Full implementation requires FFT + mel filterbank
    // via vDSP/Accelerate for the STFT.
    // Real implementation: STFT -> power spectrum -> mel filterbank -> log
    print("[KrillLM] Warning: Audio preprocessing not yet implemented. Audio input will be ignored.")
    return nil
}
