import Foundation
import MLX
import MLXNN
import MLXFast
#if canImport(CoreGraphics) && canImport(ImageIO)
import CoreGraphics
import ImageIO
#endif

public enum MultimodalPreprocessingError: Error, CustomStringConvertible {
    case emptyImageData
    case imagePreprocessingUnavailable
    case audioPreprocessingUnavailable

    public var description: String {
        switch self {
        case .emptyImageData:
            return "Image preprocessing failed: image data is empty"
        case .imagePreprocessingUnavailable:
            return "Native image preprocessing is not implemented. Use Gemma 4 through the mlx-vlm Python bridge."
        case .audioPreprocessingUnavailable:
            return "Native audio preprocessing is not implemented. Use Gemma 4 through the mlx-vlm Python bridge."
        }
    }
}

// MARK: - Vision Encoder (SigLIP2-style for Gemma 4)

/// SigLIP2-based vision encoder for Gemma 4 multimodal models.
///
/// Pipeline: Image -> Patch Embed -> Position Embed -> Transformer -> Pool -> Project
/// Produces soft tokens that are prepended to the text token stream.
public class VisionEncoder: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: PatchEmbedding
    @ModuleInfo(key: "encoder") var encoder: VisionTransformer
    @ModuleInfo(key: "multi_modal_projector") var projector: Linear

    let patchSize: Int
    let poolingKernel: Int
    let hiddenSize: Int

    public init(
        imageSize: Int = 896,
        patchSize: Int = 16,
        poolingKernel: Int = 3,
        visionHiddenSize: Int = 1152,
        visionLayers: Int = 16,
        visionHeads: Int = 12,
        visionIntermediateSize: Int = 4304,
        projectedSize: Int = 2048
    ) {
        self.patchSize = patchSize
        self.poolingKernel = poolingKernel
        self.hiddenSize = visionHiddenSize

        _patchEmbedding = ModuleInfo(
            wrappedValue: PatchEmbedding(
                patchSize: patchSize, inChannels: 3, hiddenSize: visionHiddenSize),
            key: "patch_embedding")
        _encoder = ModuleInfo(
            wrappedValue: VisionTransformer(
                hiddenSize: visionHiddenSize, numLayers: visionLayers,
                numHeads: visionHeads, intermediateSize: visionIntermediateSize),
            key: "encoder")
        _projector = ModuleInfo(
            wrappedValue: Linear(visionHiddenSize, projectedSize, bias: true),
            key: "multi_modal_projector")
    }

    /// Encode an image into soft tokens for the LLM.
    ///
    /// - Parameter image: Normalized image tensor [1, H, W, 3] in [-1, 1]
    /// - Returns: Soft tokens [1, numTokens, projectedSize]
    public func callAsFunction(_ image: MLXArray) -> MLXArray {
        // Patch embedding
        var x = patchEmbedding(image)

        // Transformer encoder
        x = encoder(x)

        // Spatial pooling (average pooling to reduce token count)
        x = spatialPool(x)

        // Project to LLM hidden size
        return projector(x)
    }

    /// Average pool patches spatially to reduce token count.
    private func spatialPool(_ x: MLXArray) -> MLXArray {
        // x shape: [B, numPatches, hiddenSize]
        // For simplicity, use a stride-based approach
        // Full impl would reshape to 2D grid and pool
        let B = x.dim(0)
        let N = x.dim(1)
        let D = x.dim(2)

        // Simple stride pooling: take every poolingKernel-th patch
        let pooledN = N / (poolingKernel * poolingKernel)
        if pooledN <= 0 { return x }

        // Average non-overlapping blocks
        let blockSize = poolingKernel * poolingKernel
        let trimmedN = (N / blockSize) * blockSize
        let trimmed = x[0..., ..<trimmedN, 0...]
        let reshaped = trimmed.reshaped(B, trimmedN / blockSize, blockSize, D)
        return MLX.mean(reshaped, axis: 2)
    }
}

// MARK: - Patch Embedding

/// Converts image pixels into patch embeddings via convolution.
class PatchEmbedding: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(patchSize: Int, inChannels: Int, hiddenSize: Int) {
        _projection = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inChannels, outputChannels: hiddenSize,
                kernelSize: IntOrPair(patchSize),
                stride: IntOrPair(patchSize)),
            key: "projection")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, H, W, C] -> conv -> [B, H/P, W/P, D] -> flatten -> [B, N, D]
        let patches = projection(x)
        let B = patches.dim(0)
        let D = patches.dim(3)
        return patches.reshaped(B, -1, D)
    }
}

// MARK: - Vision Transformer (encoder blocks)

class VisionTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [VisionBlock]

    init(hiddenSize: Int, numLayers: Int, numHeads: Int, intermediateSize: Int) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< numLayers).map { _ in
                VisionBlock(hiddenSize: hiddenSize, numHeads: numHeads,
                           intermediateSize: intermediateSize)
            },
            key: "layers")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for layer in layers {
            hidden = layer(hidden)
        }
        return hidden
    }
}

class VisionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm

    init(hiddenSize: Int, numHeads: Int, intermediateSize: Int) {
        _selfAttn = ModuleInfo(
            wrappedValue: VisionAttention(hiddenSize: hiddenSize, numHeads: numHeads),
            key: "self_attn")
        _mlp = ModuleInfo(
            wrappedValue: VisionMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize),
            key: "mlp")
        _norm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm1")
        _norm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(norm1(x))
        return h + mlp(norm2(h))
    }
}

class VisionAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Float(headDim).squareRoot()

        _qProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize, bias: true), key: "o_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class VisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _fc1 = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize, bias: true), key: "fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

// MARK: - Image Preprocessing

/// Preprocess an image file for the Gemma 4 vision encoder (SigLIP2).
///
/// Pipeline:
///   1. Decode PNG/JPEG via CoreGraphics
///   2. Resize longest side to targetSize, maintain aspect ratio
///   3. Pad to make both dimensions divisible by (patchSize * poolingKernel) = 48
///   4. Normalize: pixel / 255 → [0,1] → (x - 0.5) / 0.5 → [-1, 1]
///
/// - Parameters:
///   - imageData: Raw image data (PNG/JPEG)
///   - targetSize: Target longest-side size (default 672, must be divisible by 48)
///   - patchSize: Patch size for the vision encoder (default 16)
///   - poolingKernel: Spatial pooling kernel size (default 3)
/// - Returns: Normalized image tensor [1, H, W, 3] in bfloat16
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

    // Decode image
    guard let source = CGImageSourceCreateWithData(imageData as CFData, nil),
          let cgImage = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    let origW = cgImage.width
    let origH = cgImage.height

    // Resize: fit longest side to targetSize, maintain aspect ratio
    let scale = Float(targetSize) / Float(max(origW, origH))
    var newW = Int(Float(origW) * scale)
    var newH = Int(Float(origH) * scale)

    // Pad up to nearest multiple of blockSize
    newW = ((newW + blockSize - 1) / blockSize) * blockSize
    newH = ((newH + blockSize - 1) / blockSize) * blockSize

    // Render into RGBA bitmap
    let bitsPerComponent = 8
    let bytesPerRow = newW * 4
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(
        data: nil,
        width: newW,
        height: newH,
        bitsPerComponent: bitsPerComponent,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    // White background for padding area
    context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    context.fill(CGRect(x: 0, y: 0, width: newW, height: newH))

    // Draw image centered (padding on right/bottom)
    let drawW = Int(Float(origW) * scale)
    let drawH = Int(Float(origH) * scale)
    context.draw(cgImage, in: CGRect(x: 0, y: newH - drawH, width: drawW, height: drawH))

    guard let data = context.data else {
        throw MultimodalPreprocessingError.emptyImageData
    }

    // Extract RGB from RGBA, normalize to [-1, 1]
    let pixelCount = newH * newW
    var floats = [Float](repeating: 0, count: pixelCount * 3)
    let ptr = data.bindMemory(to: UInt8.self, capacity: pixelCount * 4)

    for i in 0 ..< pixelCount {
        let base = i * 4
        // SigLIP2 normalization: (pixel/255 - 0.5) / 0.5 = pixel/127.5 - 1.0
        floats[i * 3 + 0] = Float(ptr[base + 0]) / 127.5 - 1.0  // R
        floats[i * 3 + 1] = Float(ptr[base + 1]) / 127.5 - 1.0  // G
        floats[i * 3 + 2] = Float(ptr[base + 2]) / 127.5 - 1.0  // B
    }

    // Shape: [1, H, W, 3] — MLX vision convention (NHWC)
    let array = MLXArray(floats, [1, newH, newW, 3])
    return array.asType(.bfloat16)

    #else
    throw MultimodalPreprocessingError.imagePreprocessingUnavailable
    #endif
}
