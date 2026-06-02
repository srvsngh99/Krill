import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXFast
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

// MARK: - VisionEncoderCache

/// LRU cache for SigLIP2 vision encoder outputs, keyed by SHA-256 of input image bytes.
///
/// The cached value is the post-encoder, post-projection embedding tensor that the
/// language model consumes. Cache lifetime is bound to the owning model instance —
/// reloading the model (different weights) gets a fresh cache.
public final class VisionEncoderCache: @unchecked Sendable {
    public static let defaultCapacity: Int = 4

    private let lock = NSLock()
    private let capacity: Int
    private var entries: [String: MLXArray] = [:]
    private var order: [String] = []

    private var _hits: Int = 0
    private var _misses: Int = 0

    public init(capacity: Int = VisionEncoderCache.defaultCapacity) {
        self.capacity = max(1, capacity)
    }

    public static func key(forImageBytes data: Data) -> String {
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    func lookup(_ key: String) -> MLXArray? {
        lock.lock()
        defer { lock.unlock() }
        guard let value = entries[key] else {
            _misses += 1
            return nil
        }
        _hits += 1
        if let idx = order.firstIndex(of: key) {
            order.remove(at: idx)
        }
        order.append(key)
        return value
    }

    func store(_ key: String, value: MLXArray) {
        lock.lock()
        defer { lock.unlock() }
        if entries[key] != nil {
            if let idx = order.firstIndex(of: key) {
                order.remove(at: idx)
            }
            order.append(key)
            entries[key] = value
            return
        }
        if order.count >= capacity, let evict = order.first {
            order.removeFirst()
            entries.removeValue(forKey: evict)
        }
        entries[key] = value
        order.append(key)
    }

    var hits: Int {
        lock.lock(); defer { lock.unlock() }
        return _hits
    }

    var misses: Int {
        lock.lock(); defer { lock.unlock() }
        return _misses
    }

    var count: Int {
        lock.lock(); defer { lock.unlock() }
        return entries.count
    }

    func resetCounters() {
        lock.lock(); defer { lock.unlock() }
        _hits = 0
        _misses = 0
    }
}

// MARK: - Preprocessing Errors

public enum MultimodalPreprocessingError: Error, CustomStringConvertible {
    case emptyImageData
    case imagePreprocessingUnavailable
    case audioPreprocessingUnavailable

    public var description: String {
        switch self {
        case .emptyImageData:
            return "Image preprocessing failed: image data is empty"
        case .imagePreprocessingUnavailable:
            return "Native image preprocessing is not available on this platform."
        case .audioPreprocessingUnavailable:
            return "Native audio preprocessing is not implemented. Use Gemma 4 through the mlx-vlm Python bridge."
        }
    }
}

// MARK: - ClippableLinear

/// Linear layer with optional input/output clamping.
///
/// Matches Gemma4ClippableLinear: wraps nn.Linear, clamps input/output.
/// Clip bounds are stored as buffers (scalar tensors). Initialized to +/-inf
/// so clamping is a no-op until real values are loaded from the checkpoint.
///
/// Weight key structure: `xxx.linear.weight`, `xxx.input_min`, `xxx.input_max`,
/// `xxx.output_min`, `xxx.output_max`
class ClippableLinear: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "input_min") var inputMin: MLXArray
    @ModuleInfo(key: "input_max") var inputMax: MLXArray
    @ModuleInfo(key: "output_min") var outputMin: MLXArray
    @ModuleInfo(key: "output_max") var outputMax: MLXArray

    init(_ inputDims: Int, _ outputDims: Int, bias: Bool = false) {
        _linear = ModuleInfo(wrappedValue: Linear(inputDims, outputDims, bias: bias), key: "linear")
        _inputMin = ModuleInfo(wrappedValue: MLXArray(Float(-1e38)), key: "input_min")
        _inputMax = ModuleInfo(wrappedValue: MLXArray(Float(1e38)), key: "input_max")
        _outputMin = ModuleInfo(wrappedValue: MLXArray(Float(-1e38)), key: "output_min")
        _outputMax = ModuleInfo(wrappedValue: MLXArray(Float(1e38)), key: "output_max")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = MLX.clip(x, min: inputMin, max: inputMax)
        h = linear(h)
        return MLX.clip(h, min: outputMin, max: outputMax)
    }
}

// MARK: - RMSNorm variants for Vision

/// RMSNorm with learned scale — used for q_norm, k_norm in vision attention.
class VisionRMSNorm: Module {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight = ModuleInfo(wrappedValue: MLXArray.ones([dimensions]))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = MLX.mean(xf * xf, axis: -1, keepDims: true)
        let normed = xf * MLX.rsqrt(variance + MLXArray(eps))
        return (normed * weight.asType(.float32)).asType(x.dtype)
    }
}

/// Parameter-free RMSNorm — used for v_norm in vision attention.
class VisionRMSNormNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let variance = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return (xf * MLX.rsqrt(variance + MLXArray(eps))).asType(x.dtype)
    }
}

// MARK: - 2D Multidimensional RoPE

/// Rotate half: [-x2, x1], matching PyTorch's rotate_half.
/// Input is 4D: [B, L, numHeads, headDim]. Splits on the LAST axis.
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[0..., 0..., 0..., ..<half]
    let x2 = x[0..., 0..., 0..., half...]
    return concatenated([-x2, x1], axis: -1)
}

/// Apply 2D multidimensional RoPE to vision attention queries/keys.
///
/// Splits the head dimension into per-spatial-dimension parts and applies
/// rotate_half independently per dimension. For head_dim=64 and ndim=2,
/// each dimension gets 32 channels (16 rotary pairs).
///
/// - Parameters:
///   - inputs: [B, L, numHeads, headDim]
///   - positions: [B, L, 2] — (x, y) patch grid coordinates
///   - baseFrequency: RoPE base frequency (default 100.0)
/// - Returns: RoPE-applied tensor, same shape as inputs
func applyMultidimensionalRoPE(
    _ inputs: MLXArray, positions: MLXArray, baseFrequency: Float = 100.0
) -> MLXArray {
    let headDim = inputs.dim(-1)
    let ndim = positions.dim(-1)
    let channelsPerDim = 2 * (headDim / (2 * ndim))
    let halfPerDim = channelsPerDim / 2

    var resultParts: [MLXArray] = []
    for d in 0 ..< ndim {
        let xPart = inputs[0..., 0..., 0..., (d * channelsPerDim) ..< ((d + 1) * channelsPerDim)]

        let freqExp = (2.0 / Float(channelsPerDim)) * MLXArray(Array(0 ..< halfPerDim).map { Float($0) })
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExp)
        let sinusoidInp = positions[0..., 0..., d ..< (d + 1)].asType(.float32) / timescale

        let cosD = MLX.cos(sinusoidInp)
        let sinD = MLX.sin(sinusoidInp)
        let cosFull = concatenated([cosD, cosD], axis: -1).asType(inputs.dtype)
        let sinFull = concatenated([sinD, sinD], axis: -1).asType(inputs.dtype)
        // Expand [B, L, channelsPerDim] -> [B, L, 1, channelsPerDim]
        let cosExp = expandedDimensions(cosFull, axis: 2)
        let sinExp = expandedDimensions(sinFull, axis: 2)

        let yPart = xPart * cosExp + rotateHalf(xPart) * sinExp
        resultParts.append(yPart)
    }
    return concatenated(resultParts, axis: -1)
}

// MARK: - Vision Attention

/// SigLIP2 vision attention with ClippableLinear, q/k/v norms, and 2D RoPE.
class VisionAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeBaseFrequency: Float

    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo(key: "o_proj") var oProj: ClippableLinear
    @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm

    let vNorm = VisionRMSNormNoScale()

    init(hiddenSize: Int, numHeads: Int, numKVHeads: Int, headDim: Int,
         ropeTheta: Float = 100.0) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.ropeBaseFrequency = ropeTheta

        _qProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, numHeads * headDim), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, numKVHeads * headDim), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, numKVHeads * headDim), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: ClippableLinear(numHeads * headDim, hiddenSize), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: VisionRMSNorm(dimensions: headDim), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: VisionRMSNorm(dimensions: headDim), key: "k_norm")
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim)

        q = qNorm(q)
        k = kNorm(k)
        v = vNorm(v)

        q = applyMultidimensionalRoPE(q, positions: positions, baseFrequency: ropeBaseFrequency)
        k = applyMultidimensionalRoPE(k, positions: positions, baseFrequency: ropeBaseFrequency)

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask)

        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Vision MLP (GeGLU)

class VisionMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: ClippableLinear
    @ModuleInfo(key: "up_proj") var upProj: ClippableLinear
    @ModuleInfo(key: "down_proj") var downProj: ClippableLinear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, intermediateSize), key: "gate_proj")
        _upProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, intermediateSize), key: "up_proj")
        _downProj = ModuleInfo(wrappedValue: ClippableLinear(intermediateSize, hiddenSize), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

// MARK: - Vision Transformer Block (4 norms)

class VisionTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: RMSNorm

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int, numKVHeads: Int,
         headDim: Int, ropeTheta: Float, eps: Float) {
        _selfAttn = ModuleInfo(wrappedValue: VisionAttention(
            hiddenSize: hiddenSize, numHeads: numHeads, numKVHeads: numKVHeads,
            headDim: headDim, ropeTheta: ropeTheta), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: VisionMLP(
            hiddenSize: hiddenSize, intermediateSize: intermediateSize), key: "mlp")
        _inputLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: hiddenSize, eps: eps), key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: hiddenSize, eps: eps), key: "post_attention_layernorm")
        _preFeedforwardLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: hiddenSize, eps: eps), key: "pre_feedforward_layernorm")
        _postFeedforwardLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: hiddenSize, eps: eps), key: "post_feedforward_layernorm")
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        let normed = inputLayernorm(x)
        let attnOut = selfAttn(normed, positions: positions, mask: mask)
        let attnNormed = postAttentionLayernorm(attnOut)
        let h = x + attnNormed

        let ffnIn = preFeedforwardLayernorm(h)
        let ffnOut = mlp(ffnIn)
        let ffnNormed = postFeedforwardLayernorm(ffnOut)
        return h + ffnNormed
    }
}

// MARK: - Vision Transformer (encoder)

/// Holds the transformer layers — maps to `vision_tower.encoder` in weights.
class VisionTransformerModel: Module {
    @ModuleInfo(key: "layers") var layers: [VisionTransformerBlock]

    init(hiddenSize: Int, intermediateSize: Int, numLayers: Int, numHeads: Int,
         numKVHeads: Int, headDim: Int, ropeTheta: Float, eps: Float) {
        _layers = ModuleInfo(wrappedValue: (0 ..< numLayers).map { _ in
            VisionTransformerBlock(
                hiddenSize: hiddenSize, intermediateSize: intermediateSize,
                numHeads: numHeads, numKVHeads: numKVHeads,
                headDim: headDim, ropeTheta: ropeTheta, eps: eps)
        }, key: "layers")
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h, positions: positions, mask: mask)
        }
        return h
    }
}

// MARK: - Patch Embedder

/// Patchify image + Linear projection + factored 2D position embeddings.
///
/// Weight key structure:
///   `patch_embedder.input_proj.weight` — Linear(3*patchSize^2, hiddenSize)
///   `patch_embedder.position_embedding_table` — [2, posEmbedSize, hiddenSize]
class VisionPatchEmbedder: Module {
    let hiddenSize: Int
    let patchSize: Int
    let positionEmbeddingSize: Int

    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

    init(hiddenSize: Int, patchSize: Int, positionEmbeddingSize: Int) {
        self.hiddenSize = hiddenSize
        self.patchSize = patchSize
        self.positionEmbeddingSize = positionEmbeddingSize
        _inputProj = ModuleInfo(
            wrappedValue: Linear(3 * patchSize * patchSize, hiddenSize, bias: false),
            key: "input_proj")
        _positionEmbeddingTable = ModuleInfo(
            wrappedValue: MLXArray.ones([2, positionEmbeddingSize, hiddenSize]),
            key: "position_embedding_table")
    }

    /// Patchify: [B, C, H, W] -> [B, numPatches, patchSize^2 * C]
    func patchify(_ pixelValues: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let C = pixelValues.dim(1)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)
        let p = patchSize
        let pH = H / p
        let pW = W / p

        // [B, C, pH, p, pW, p] -> [B, pH, pW, p, p, C] -> [B, pH*pW, p*p*C]
        let reshaped = pixelValues.reshaped(B, C, pH, p, pW, p)
        let transposed = reshaped.transposed(0, 2, 4, 3, 5, 1)
        let patches = transposed.reshaped(B, pH * pW, C * p * p)
        // Normalize: 2 * (x - 0.5) maps [0, 1] to [-1, 1]
        return inputProj((2 * (patches - 0.5)).asType(inputProj.weight.dtype))
    }

    /// Compute position embeddings from patch grid coordinates.
    func positionEmbeddings(patchPositions: MLXArray, paddingPositions: MLXArray) -> MLXArray {
        // One-hot encode positions: [B, numPatches, 2] -> [B, numPatches, 2, posEmbedSize]
        let indices = expandedDimensions(patchPositions, axis: -1)
        let oneHot = (indices .== MLXArray(Array(0 ..< positionEmbeddingSize).map { Int32($0) }))
            .asType(.float32)
        // [B, numPatches, 2, posEmbedSize] -> [B, 2, numPatches, posEmbedSize]
        let oh = oneHot.transposed(0, 2, 1, 3).asType(positionEmbeddingTable.dtype)
        // [B, 2, numPatches, posEmbedSize] @ [2, posEmbedSize, hiddenSize] -> [B, 2, numPatches, hiddenSize]
        let posEmbed = MLX.matmul(oh, positionEmbeddingTable)
        // Sum over 2 spatial dims -> [B, numPatches, hiddenSize]
        let summed = posEmbed.sum(axis: 1)
        // Zero out padding positions
        let padMask = expandedDimensions(paddingPositions, axis: -1)
        return MLX.where(padMask, MLXArray(Float(0)), summed)
    }

    func callAsFunction(_ pixelValues: MLXArray, patchPositions: MLXArray,
                        paddingPositions: MLXArray) -> MLXArray {
        let hidden = patchify(pixelValues)
        let posEmbed = positionEmbeddings(patchPositions: patchPositions, paddingPositions: paddingPositions)
        return hidden + posEmbed
    }
}

// MARK: - Vision Pooler

/// Position-aware average pooling to fixed output length (default 280 tokens).
class VisionPooler: Module {
    let hiddenSize: Int
    let defaultOutputLength: Int
    let rootHiddenSize: Float

    init(hiddenSize: Int, defaultOutputLength: Int) {
        self.hiddenSize = hiddenSize
        self.defaultOutputLength = defaultOutputLength
        self.rootHiddenSize = Float(hiddenSize).squareRoot()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, patchPositions: MLXArray,
        paddingPositions: MLXArray
    ) -> (MLXArray, MLXArray) {
        // Zero out padding tokens
        let padMask = expandedDimensions(paddingPositions, axis: -1)
        var h = MLX.where(padMask, MLXArray(Float(0)), hiddenStates)

        let inputSeqLen = h.dim(1)
        let length = defaultOutputLength

        if inputSeqLen == length {
            return (h * MLXArray(rootHiddenSize), paddingPositions)
        }

        // Position-aware average pooling
        let k = Int(Float(inputSeqLen / length).squareRoot())
        let kSquared = Float(k * k)

        let clamped = MLX.maximum(patchPositions, MLXArray(Int32(0)))
        let maxX = MLX.max(clamped[0..., 0..., ..<1], axis: 1, keepDims: true) + 1
        let kernelIdxs = MLX.floor(clamped.asType(.float32) / MLXArray(Float(k))).asType(.int32)
        let binIdx = kernelIdxs[0..., 0..., ..<1] + (maxX / MLXArray(Int32(k))) * kernelIdxs[0..., 0..., 1...]
        let binIdxFlat = binIdx.squeezed(axis: -1)

        // One-hot encode bins -> weights
        let binOneHot = (expandedDimensions(binIdxFlat, axis: -1)
            .== MLXArray(Array(0 ..< length).map { Int32($0) })).asType(.float32)
        let weights = binOneHot / MLXArray(kSquared)

        // Weighted sum: [B, L, length]^T @ [B, L, D] -> [B, length, D]
        let pooled = MLX.matmul(weights.transposed(0, 2, 1), h.asType(.float32)).asType(h.dtype)

        // Mask: which output bins have at least one valid patch
        let mask = 1 - MLX.all(weights .== MLXArray(Float(0)), axis: 1).asType(.int32)

        return (pooled * MLXArray(rootHiddenSize), mask)
    }
}

// MARK: - VisionModel (top-level, maps to vision_tower.*)

/// Complete SigLIP2 vision encoder for Gemma 4.
///
/// Weight key structure:
///   `vision_tower.patch_embedder.*`
///   `vision_tower.encoder.layers.*`
///
/// Input: pixel values [B, C, H, W] in [0, 1] (channel-first)
/// Output: soft tokens [1, numTokens, hiddenSize] — typically [1, 280, 768]
public class VisionEncoder: Module {
    let patchSize: Int
    let poolingKernelSize: Int
    let defaultOutputLength: Int
    let maxPatches: Int

    @ModuleInfo(key: "patch_embedder") var patchEmbedder: VisionPatchEmbedder
    @ModuleInfo(key: "encoder") var encoder: VisionTransformerModel

    let pooler: VisionPooler

    public init(
        hiddenSize: Int = 768,
        intermediateSize: Int = 3072,
        numLayers: Int = 16,
        numHeads: Int = 12,
        numKVHeads: Int = 12,
        headDim: Int = 64,
        patchSize: Int = 16,
        poolingKernelSize: Int = 3,
        defaultOutputLength: Int = 280,
        positionEmbeddingSize: Int = 10240,
        ropeTheta: Float = 100.0,
        eps: Float = 1e-6
    ) {
        self.patchSize = patchSize
        self.poolingKernelSize = poolingKernelSize
        self.defaultOutputLength = defaultOutputLength
        self.maxPatches = defaultOutputLength * poolingKernelSize * poolingKernelSize

        _patchEmbedder = ModuleInfo(
            wrappedValue: VisionPatchEmbedder(
                hiddenSize: hiddenSize, patchSize: patchSize,
                positionEmbeddingSize: positionEmbeddingSize),
            key: "patch_embedder")
        _encoder = ModuleInfo(
            wrappedValue: VisionTransformerModel(
                hiddenSize: hiddenSize, intermediateSize: intermediateSize,
                numLayers: numLayers, numHeads: numHeads,
                numKVHeads: numKVHeads, headDim: headDim,
                ropeTheta: ropeTheta, eps: eps),
            key: "encoder")
        self.pooler = VisionPooler(
            hiddenSize: hiddenSize, defaultOutputLength: defaultOutputLength)
    }

    /// Compute patch positions and padding mask for a single image.
    func patchPositions(height: Int, width: Int) -> (positions: MLXArray, padding: MLXArray, numReal: Int) {
        let pH = height / patchSize
        let pW = width / patchSize
        let numPatches = pH * pW
        let numReal = min(numPatches, maxPatches)

        // Grid coordinates: (x, y)
        var positions = [Int32](repeating: 0, count: maxPatches * 2)
        for y in 0 ..< pH {
            for x in 0 ..< pW {
                let idx = y * pW + x
                if idx >= maxPatches { break }
                positions[idx * 2] = Int32(x)
                positions[idx * 2 + 1] = Int32(y)
            }
        }
        // Padding positions get -1
        for i in numReal ..< maxPatches {
            positions[i * 2] = -1
            positions[i * 2 + 1] = -1
        }

        var paddingMask = [Bool](repeating: false, count: maxPatches)
        for i in numReal ..< maxPatches {
            paddingMask[i] = true
        }

        let posArray = MLXArray(positions, [1, maxPatches, 2])
        let padArray = MLXArray(paddingMask, [1, maxPatches])
        return (posArray, padArray, numReal)
    }

    /// Encode pixel values into soft tokens.
    ///
    /// - Parameter pixelValues: [B, C, H, W] in [0, 1] range (channel-first)
    /// - Returns: Soft tokens [1, numTokens, hiddenSize] — typically [1, 280, 768]
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let B = pixelValues.dim(0)
        let H = pixelValues.dim(2)
        let W = pixelValues.dim(3)

        let (patchPos, paddingPos, numReal) = patchPositions(height: H, width: W)

        // Embed patches (only real, not padding)
        var inputsEmbeds = patchEmbedder(
            pixelValues,
            patchPositions: patchPos[0..., ..<numReal, 0...],
            paddingPositions: paddingPos[0..., ..<numReal])

        // Pad to maxPatches if needed
        let numPadding = maxPatches - numReal
        if numPadding > 0 {
            let padEmbeds = MLXArray.zeros([B, numPadding, inputsEmbeds.dim(2)]).asType(inputsEmbeds.dtype)
            inputsEmbeds = concatenated([inputsEmbeds, padEmbeds], axis: 1)
        }

        // Bidirectional attention mask: [B, 1, L, L]
        let validMask = paddingPos .== MLXArray(false)
        let attnMask2D = expandedDimensions(validMask, axis: 1) * expandedDimensions(validMask, axis: 2)
        let maskFill = MLXArray(Float(-1e4)).asType(inputsEmbeds.dtype)
        let zeroFill = MLXArray(Float(0)).asType(inputsEmbeds.dtype)
        let attnMask = expandedDimensions(
            MLX.where(attnMask2D, zeroFill, maskFill), axis: 1)

        // Transformer encoder
        let hiddenStates = encoder(inputsEmbeds, positions: patchPos, mask: attnMask)

        // Pool to defaultOutputLength tokens
        let (pooled, poolMask) = pooler(hiddenStates, patchPositions: patchPos, paddingPositions: paddingPos)

        // Extract valid (non-padding) tokens
        let validCount = poolMask.asType(.int32).sum()
        MLX.eval(validCount)
        let nValid = Int(truncating: validCount.item(Int32.self) as NSNumber) / B
        let result = pooled[0..., ..<nValid, 0...]

        return result
    }
}

// MARK: - Image Preprocessing

/// Preprocess an image file for the Gemma 4 vision encoder (SigLIP2).
///
/// Pipeline:
///   1. Decode PNG/JPEG via CoreGraphics
///   2. Resize longest side to targetSize, maintain aspect ratio
///   3. Pad to make both dimensions divisible by (patchSize * poolingKernel) = 48
///   4. Output as [1, 3, H, W] in [0, 1] range (channel-first, float32)
///
/// - Parameters:
///   - imageData: Raw image data (PNG/JPEG)
///   - targetSize: Target longest-side size (default 672, must be divisible by 48)
///   - patchSize: Patch size for the vision encoder (default 16)
///   - poolingKernel: Spatial pooling kernel size (default 3)
/// - Returns: Image tensor [1, 3, H, W] in [0, 1], bfloat16
public func preprocessImage(
    _ imageData: Data,
    targetSize: Int = 672,
    patchSize: Int = 16,
    poolingKernel: Int = 3
) throws -> MLXArray {
    guard !imageData.isEmpty else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    #if canImport(CoreGraphics) && canImport(ImageIO)
    let blockSize = patchSize * poolingKernel  // 48

    guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    let origW = cgImage.width
    let origH = cgImage.height

    // Resize: scale to targetSize, ensure both dims are at least targetSize
    // for small images, then round up to nearest blockSize multiple.
    let minDimTarget = max(targetSize, blockSize * 16)  // minimum 768 for small images
    let scale: Float
    if max(origW, origH) <= minDimTarget {
        scale = Float(minDimTarget) / Float(max(origW, origH))
    } else {
        scale = Float(targetSize) / Float(max(origW, origH))
    }
    var newW = Int(Float(origW) * scale)
    var newH = Int(Float(origH) * scale)

    newW = ((newW + blockSize - 1) / blockSize) * blockSize
    newH = ((newH + blockSize - 1) / blockSize) * blockSize

    let bitsPerComponent = 8
    let bytesPerRow = newW * 4
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil, width: newW, height: newH,
        bitsPerComponent: bitsPerComponent, bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: newW, height: newH))

    let drawW = Int(Float(origW) * scale)
    let drawH = Int(Float(origH) * scale)
    context.draw(cgImage, in: CGRect(x: 0, y: newH - drawH, width: drawW, height: drawH))

    guard let data = context.data else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    // Extract RGB from RGBA into channel-first [1, 3, H, W] in [0, 1]
    // CGContext stores pixels bottom-to-top (row 0 = bottom of image).
    // Vision models expect top-to-bottom, so we flip rows during readout.
    let pixelCount = newH * newW
    var floats = [Float](repeating: 0, count: 3 * pixelCount)
    let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)

    // Channel-first: R plane, then G plane, then B plane (with row flip)
    for row in 0 ..< newH {
        let flippedRow = newH - 1 - row  // Flip: CG bottom row → array top row
        for col in 0 ..< newW {
            let srcIdx = (flippedRow * newW + col) * 4  // CG pixel (bottom-to-top)
            let dstIdx = row * newW + col                // Array pixel (top-to-bottom)
            floats[dstIdx] = Float(ptr[srcIdx]) / 255.0                     // R
            floats[pixelCount + dstIdx] = Float(ptr[srcIdx + 1]) / 255.0    // G
            floats[2 * pixelCount + dstIdx] = Float(ptr[srcIdx + 2]) / 255.0 // B
        }
    }

    // Vision encoder expects float32 input (patchify casts to weight dtype internally)
    return MLXArray(floats, [1, 3, newH, newW])

    #else
    throw MultimodalPreprocessingError.imagePreprocessingUnavailable
    #endif
}

// MARK: - LLaVA / CLIP image preprocessing

/// Image preprocessing for the native LLaVA-1.5 runtime, matching HF's
/// `CLIPImageProcessor` (the processor llava-1.5 ships): resize the shortest
/// edge to `imageSize` preserving aspect ratio, center-crop to a square
/// `imageSize x imageSize`, rescale to `[0, 1]`, then normalize with the CLIP
/// channel mean/std. Returns a channels-first `[1, 3, imageSize, imageSize]`
/// float32 tensor -- the layout `LlavaForCausalLM.imageFeatures` transposes to
/// channels-last before the CLIP patch-embed Conv2d.
///
/// This is the LLaVA analogue of `preprocessImage` (the Gemma 4 longest-side /
/// block-aligned preprocessor); the two differ in resize geometry and in that
/// CLIP normalizes, so LLaVA gets its own path rather than overloading the
/// Gemma one.
public enum LlavaImagePreprocessor {
    /// CLIP ViT-L/14-336 normalization constants (OpenAI CLIP; HF
    /// `image_mean` / `image_std` for the llava-1.5 processor).
    public static let imageMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    public static let imageStd: [Float] = [0.26862954, 0.26130258, 0.27577711]

    /// Preprocess raw image bytes into `[1, 3, imageSize, imageSize]` float32.
    /// `imageSize` is the CLIP tower's `image_size` (336 for llava-1.5).
    public static func preprocess(_ imageData: Data, imageSize: Int = 336) throws -> MLXArray {
        guard !imageData.isEmpty else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        #if canImport(CoreGraphics) && canImport(ImageIO)
        guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
              let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        let origW = cgImage.width
        let origH = cgImage.height
        guard origW > 0, origH > 0 else {
            throw MultimodalPreprocessingError.emptyImageData
        }

        // Resize the SHORTEST edge to `imageSize`, preserving aspect ratio
        // (HF `CLIPImageProcessor` size={"shortest_edge": imageSize}).
        let scale = Float(imageSize) / Float(min(origW, origH))
        let resizedW = max(imageSize, Int((Float(origW) * scale).rounded()))
        let resizedH = max(imageSize, Int((Float(origH) * scale).rounded()))

        let bytesPerRow = resizedW * 4
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: nil, width: resizedW, height: resizedH,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        context.interpolationQuality = .high
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: resizedW, height: resizedH))

        guard let data = context.data else {
            throw MultimodalPreprocessingError.emptyImageData
        }
        let ptr = data.bindMemory(to: UInt8.self, capacity: resizedW * resizedH * 4)

        // Center-crop a square `imageSize x imageSize` window out of the
        // resized image (HF `do_center_crop`, crop_size=imageSize). Read RGB
        // directly into channel-first planes, flipping rows (CGContext stores
        // bottom-to-top) and normalizing in the same pass.
        let cropX = (resizedW - imageSize) / 2
        let cropY = (resizedH - imageSize) / 2
        let plane = imageSize * imageSize
        var floats = [Float](repeating: 0, count: 3 * plane)
        for row in 0 ..< imageSize {
            // Destination row `row` is top-to-bottom; the source crop row maps
            // to a bottom-to-top CGContext row.
            let srcRow = resizedH - 1 - (cropY + row)
            for col in 0 ..< imageSize {
                let srcIdx = (srcRow * resizedW + (cropX + col)) * 4
                let dstIdx = row * imageSize + col
                let r = Float(ptr[srcIdx]) / 255.0
                let g = Float(ptr[srcIdx + 1]) / 255.0
                let b = Float(ptr[srcIdx + 2]) / 255.0
                floats[dstIdx] = (r - imageMean[0]) / imageStd[0]
                floats[plane + dstIdx] = (g - imageMean[1]) / imageStd[1]
                floats[2 * plane + dstIdx] = (b - imageMean[2]) / imageStd[2]
            }
        }
        return MLXArray(floats, [1, 3, imageSize, imageSize])

        #else
        throw MultimodalPreprocessingError.imagePreprocessingUnavailable
        #endif
    }
}

/// GELU approximate activation matching PyTorch's gelu_pytorch_tanh.
private func geluApproximate(_ x: MLXArray) -> MLXArray {
    // gelu_approx(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let sqrt2pi = MLXArray(Float(0.7978845608028654))
    let coeff = MLXArray(Float(0.044715))
    let inner = sqrt2pi * (x + coeff * x * x * x)
    return MLXArray(Float(0.5)) * x * (MLXArray(Float(1.0)) + MLX.tanh(inner))
}
