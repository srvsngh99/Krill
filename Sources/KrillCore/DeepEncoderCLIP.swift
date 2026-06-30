import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - DeepEncoder CLIP-L tower (Unlimited-OCR / DeepSeek-OCR `vision_model`)

/// The CLIP-L half of the DeepEncoder: a clean pre-norm ViT (24 layers, width
/// 1024, 16 heads, quick-GELU MLP, fused QKV). It consumes `patch_embeds` from
/// the SAM tower (NOT raw pixels: its own `patch_embedding` Conv2d is bypassed
/// in the DeepEncoder path), prepends a class token, adds (non-interpolated, at
/// the 16x16 grid) position embeddings, and runs the transformer.
///
/// Mirrors `deepencoder.VitModel` / `NoTPTransformer`. `get_abs_pos` is a no-op
/// here because the SAM output grid (16x16) equals the CLIP position grid
/// (224/14 = 16), so no bicubic resample is needed for the standard 1024 path.
public struct CLIPVisionConfig {
    public var hiddenSize = 1024
    public var numLayers = 24
    public var numHeads = 16
    public var ffnHiddenSize = 4096
    public var imageSize = 224
    public var patchSize = 14
    public var layerNormEps: Float = 1e-5
    public init() {}
    public var headDim: Int { hiddenSize / numHeads }
    public var numPositions: Int { (imageSize / patchSize) * (imageSize / patchSize) + 1 }
}

private func quickGELU(_ x: MLXArray) -> MLXArray { x * sigmoid(1.702 * x) }

final class CLIPNoTPAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ cfg: CLIPVisionConfig) {
        self.numHeads = cfg.numHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / Float(cfg.headDim).squareRoot()
        _qkvProj = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize * 3, bias: true), key: "qkv_proj")
        _outProj = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.hiddenSize, bias: true), key: "out_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        // [B, S, 3*H*D] -> [B, S, 3, H, D] -> per-tensor [B, H, S, D]
        let qkv = qkvProj(x).reshaped(B, S, 3, numHeads, headDim)
        let parts = split(qkv, parts: 3, axis: 2)
        let q = parts[0].squeezed(axis: 2).transposed(0, 2, 1, 3)
        let k = parts[1].squeezed(axis: 2).transposed(0, 2, 1, 3)
        let v = parts[2].squeezed(axis: 2).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(B, S, -1))
    }
}

final class CLIPNoTPBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CLIPNoTPAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: CLIPNoTPMLP

    init(_ cfg: CLIPVisionConfig) {
        _selfAttn = ModuleInfo(wrappedValue: CLIPNoTPAttention(cfg), key: "self_attn")
        _layerNorm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "layer_norm1")
        _layerNorm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "layer_norm2")
        _mlp = ModuleInfo(wrappedValue: CLIPNoTPMLP(cfg), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

final class CLIPNoTPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ cfg: CLIPVisionConfig) {
        _fc1 = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.ffnHiddenSize, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(cfg.ffnHiddenSize, cfg.hiddenSize, bias: true), key: "fc2")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(quickGELU(fc1(x))) }
}

final class CLIPVisionEmbeddings: Module {
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    // NOTE: the checkpoint's `patch_embedding` Conv2d is intentionally absent:
    // the DeepEncoder feeds CLIP `patch_embeds` from SAM, so that Conv2d never
    // runs. Its weight is dropped at load (see the loader / parity harness).

    init(_ cfg: CLIPVisionConfig) {
        _classEmbedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([cfg.hiddenSize]), key: "class_embedding")
        _positionEmbedding = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.numPositions, dimensions: cfg.hiddenSize),
            key: "position_embedding")
    }

    /// `patchEmbeds`: SAM output `[B, C, gh, gw]`. Returns `[B, gh*gw + 1, C]`.
    func callAsFunction(patchEmbeds: MLXArray) -> MLXArray {
        let B = patchEmbeds.dim(0), C = patchEmbeds.dim(1)
        // flatten(2).transpose(1,2): [B,C,gh,gw] -> [B,C,N] -> [B,N,C]
        let flat = patchEmbeds.reshaped(B, C, -1).transposed(0, 2, 1)
        let cls = broadcast(classEmbedding.reshaped(1, 1, C), to: [B, 1, C])
        let embeds = concatenated([cls, flat], axis: 1)                 // [B, N+1, C]
        // get_abs_pos is identity at the matching 16x16 grid: add the full table.
        return embeds + positionEmbedding.weight.reshaped(1, -1, C)
    }
}

public final class DeepEncoderCLIP: Module {
    @ModuleInfo(key: "embeddings") var embeddings: CLIPVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayrnorm: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: CLIPTransformer

    public init(_ cfg: CLIPVisionConfig = CLIPVisionConfig()) {
        _embeddings = ModuleInfo(wrappedValue: CLIPVisionEmbeddings(cfg), key: "embeddings")
        _preLayrnorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: cfg.hiddenSize, eps: cfg.layerNormEps),
            key: "pre_layrnorm")
        _transformer = ModuleInfo(wrappedValue: CLIPTransformer(cfg), key: "transformer")
    }

    /// `patchEmbeds`: SAM output `[B, 1024, 16, 16]`. Returns `[B, 257, 1024]`.
    public func callAsFunction(patchEmbeds: MLXArray) -> MLXArray {
        transformer(preLayrnorm(embeddings(patchEmbeds: patchEmbeds)))
    }
}

final class CLIPTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [CLIPNoTPBlock]
    init(_ cfg: CLIPVisionConfig) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< cfg.numLayers).map { _ in CLIPNoTPBlock(cfg) }, key: "layers")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }
}
