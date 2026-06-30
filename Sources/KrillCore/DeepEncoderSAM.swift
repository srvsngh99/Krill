import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - DeepEncoder SAM-ViT-B tower (Unlimited-OCR / DeepSeek-OCR `sam_model`)

/// The SAM half of the DeepEncoder (ViTDet-style): a 12-layer ViT-B (768-d, 12
/// heads) with windowed attention (window 14) and global attention at layers
/// [2,5,8,11], decomposed relative-position bias, then a neck + two stride-2
/// downsample convs that turn the 64x64 patch grid into the [B,1024,16,16]
/// `patch_embeds` the CLIP tower consumes. Mirrors `deepencoder.ImageEncoderViT`.
///
/// All convolutions run channels-last (MLX convention); the loader transposes
/// the PyTorch [out,in,kH,kW] conv weights to [out,kH,kW,in]. `get_abs_pos_sam`
/// is identity at the matching 64x64 grid.
public struct SAMConfig {
    public var embedDim = 768
    public var depth = 12
    public var numHeads = 12
    public var mlpRatio = 4
    public var outChans = 256
    public var imgSize = 1024
    public var patchSize = 16
    public var windowSize = 14
    public var globalAttnIndexes: Set<Int> = [2, 5, 8, 11]
    public var eps: Float = 1e-6
    public init() {}
    public var headDim: Int { embedDim / numHeads }
    public var grid: Int { imgSize / patchSize }   // 64
}

// MARK: - relative position helpers

/// `get_rel_pos`: no interpolation here (the checkpoint's rel_pos table length
/// already equals 2*size-1 for both window=14 and global=64). Returns [q,k,hd].
private func getRelPos(_ qSize: Int, _ kSize: Int, _ relPos: MLXArray) -> MLXArray {
    let q = MLXArray(0 ..< qSize).reshaped(qSize, 1)
    let k = MLXArray(0 ..< kSize).reshaped(1, kSize)
    let coords = (q - k) + Int32(kSize - 1)                 // [q,k], values in [0, 2*size-2]
    let hd = relPos.dim(1)
    return take(relPos, coords.reshaped(-1), axis: 0).reshaped(qSize, kSize, hd)
}

/// Decomposed relative-position bias (mvitv2). `q`: [Bh, h, w, hd] (batch*heads
/// folded). Returns the additive attention bias [Bh, h*w, h*w].
private func decomposedRelPosBias(_ q: MLXArray, _ relPosH: MLXArray, _ relPosW: MLXArray,
                                  _ h: Int, _ w: Int) -> MLXArray {
    let Bh = q.dim(0), hd = q.dim(3)
    let Rh = getRelPos(h, h, relPosH)                       // [h, h, hd]
    let Rw = getRelPos(w, w, relPosW)                       // [w, w, hd]
    // einsum bhwc,hkc->bhwk : matmul over hd, aligned on the h axis.
    let relH = matmul(q, Rh.transposed(0, 2, 1).reshaped(1, h, hd, h))   // [Bh, h, w, h]
    // einsum bhwc,wkc->bhwk : align on w (transpose w to a batch axis, matmul, back).
    let qW = q.transposed(0, 2, 1, 3)                                    // [Bh, w, h, hd]
    let relWt = matmul(qW, Rw.transposed(0, 2, 1).reshaped(1, w, hd, w)) // [Bh, w, h, w]
    let relW = relWt.transposed(0, 2, 1, 3)                              // [Bh, h, w, w]
    // bias[Bh,h,w,kh,kw] = relH[...,kh] + relW[...,kw] -> [Bh, h*w, h*w]
    let bias = relH.reshaped(Bh, h, w, h, 1) + relW.reshaped(Bh, h, w, 1, w)
    return bias.reshaped(Bh, h * w, h * w)
}

// MARK: - window partition / unpartition (channels-last [B,H,W,C])

private func windowPartition(_ x: MLXArray, _ window: Int) -> (MLXArray, Int, Int) {
    let B = x.dim(0), H = x.dim(1), W = x.dim(2), C = x.dim(3)
    let padH = (window - H % window) % window
    let padW = (window - W % window) % window
    var xp = x
    if padH > 0 || padW > 0 {
        xp = padded(x, widths: [.init((0, 0)), .init((0, padH)), .init((0, padW)), .init((0, 0))])
    }
    let Hp = H + padH, Wp = W + padW
    let r = xp.reshaped(B, Hp / window, window, Wp / window, window, C)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(-1, window, window, C)
    return (r, Hp, Wp)
}

private func windowUnpartition(_ windows: MLXArray, _ window: Int,
                               _ Hp: Int, _ Wp: Int, _ H: Int, _ W: Int) -> MLXArray {
    let C = windows.dim(3)
    let B = windows.dim(0) / (Hp * Wp / window / window)
    var x = windows.reshaped(B, Hp / window, Wp / window, window, window, C)
        .transposed(0, 1, 3, 2, 4, 5)
        .reshaped(B, Hp, Wp, C)
    if Hp > H || Wp > W { x = x[0..., 0 ..< H, 0 ..< W, 0...] }
    return x
}

// MARK: - attention

final class SAMAttention: Module {
    let numHeads: Int, headDim: Int, scale: Float
    let useRelPos: Bool
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

    init(_ cfg: SAMConfig, inputSize: Int) {
        self.numHeads = cfg.numHeads
        self.headDim = cfg.headDim
        self.scale = 1.0 / Float(cfg.headDim).squareRoot()
        self.useRelPos = true
        _qkv = ModuleInfo(wrappedValue: Linear(cfg.embedDim, cfg.embedDim * 3, bias: true), key: "qkv")
        _proj = ModuleInfo(wrappedValue: Linear(cfg.embedDim, cfg.embedDim, bias: true), key: "proj")
        _relPosH = ParameterInfo(
            wrappedValue: MLXArray.zeros([2 * inputSize - 1, cfg.headDim]), key: "rel_pos_h")
        _relPosW = ParameterInfo(
            wrappedValue: MLXArray.zeros([2 * inputSize - 1, cfg.headDim]), key: "rel_pos_w")
    }

    /// `x`: [B, H, W, C]. Returns [B, H, W, C].
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), H = x.dim(1), W = x.dim(2)
        let N = H * W
        // qkv -> [B, N, 3, nH, hd] -> [3, B, nH, N, hd]
        let qkvAll = qkv(x).reshaped(B, N, 3, numHeads, headDim).transposed(2, 0, 3, 1, 4)
        let q = qkvAll[0], k = qkvAll[1], v = qkvAll[2]      // each [B, nH, N, hd]

        // decomposed rel-pos bias on q folded to [B*nH, H, W, hd]
        let qf = q.reshaped(B * numHeads, H, W, headDim)
        let bias = decomposedRelPosBias(qf, relPosH, relPosW, H, W)   // [B*nH, N, N]
        let mask = bias.reshaped(B, numHeads, N, N)

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)  // [B,nH,N,hd]
        let merged = out.reshaped(B, numHeads, H, W, headDim)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(B, H, W, numHeads * headDim)
        return proj(merged)
    }
}

final class SAMMLPBlock: Module {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear
    init(_ cfg: SAMConfig) {
        _lin1 = ModuleInfo(wrappedValue: Linear(cfg.embedDim, cfg.embedDim * cfg.mlpRatio, bias: true), key: "lin1")
        _lin2 = ModuleInfo(wrappedValue: Linear(cfg.embedDim * cfg.mlpRatio, cfg.embedDim, bias: true), key: "lin2")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { lin2(gelu(lin1(x))) }   // exact GELU
}

final class SAMBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SAMAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SAMMLPBlock
    let windowSize: Int

    init(_ cfg: SAMConfig, global: Bool) {
        self.windowSize = global ? 0 : cfg.windowSize
        let inputSize = global ? cfg.grid : cfg.windowSize
        _norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.embedDim, eps: cfg.eps), key: "norm1")
        _attn = ModuleInfo(wrappedValue: SAMAttention(cfg, inputSize: inputSize), key: "attn")
        _norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: cfg.embedDim, eps: cfg.eps), key: "norm2")
        _mlp = ModuleInfo(wrappedValue: SAMMLPBlock(cfg), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var h = norm1(x)
        let H = x.dim(1), W = x.dim(2)
        var Hp = H, Wp = W
        if windowSize > 0 { (h, Hp, Wp) = windowPartition(h, windowSize) }
        h = attn(h)
        if windowSize > 0 { h = windowUnpartition(h, windowSize, Hp, Wp, H, W) }
        let y = shortcut + h
        return y + mlp(norm2(y))
    }
}

// MARK: - LayerNorm2d (normalize across channels; channels-last here)

final class SAMLayerNorm2d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float
    init(_ channels: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight = ParameterInfo(wrappedValue: MLXArray.ones([channels]), key: "weight")
        _bias = ParameterInfo(wrappedValue: MLXArray.zeros([channels]), key: "bias")
    }
    /// `x`: [B,H,W,C]; normalize over C (last axis).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let u = mean(x, axis: -1, keepDims: true)
        let s = mean(square(x - u), axis: -1, keepDims: true)
        return weight * ((x - u) / sqrt(s + eps)) + bias
    }
}

// MARK: - patch embed

final class SAMPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    init(_ cfg: SAMConfig) {
        _proj = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: 3, outputChannels: cfg.embedDim,
                kernelSize: IntOrPair(cfg.patchSize), stride: IntOrPair(cfg.patchSize), bias: true),
            key: "proj")
    }
    /// `x`: channels-last image [B, H, W, 3]. Returns [B, H/16, W/16, C].
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

// MARK: - encoder

public final class DeepEncoderSAM: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: SAMPatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    @ModuleInfo(key: "blocks") var blocks: [SAMBlock]
    // neck is an nn.Sequential (Conv, LN2d, Conv, LN2d) -> numeric child keys
    // 0..3, which MLX unflattens as an ARRAY, so it must be a [Module].
    @ModuleInfo(key: "neck") var neck: [Module]
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    public init(_ cfg: SAMConfig = SAMConfig()) {
        _patchEmbed = ModuleInfo(wrappedValue: SAMPatchEmbed(cfg), key: "patch_embed")
        _posEmbed = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, cfg.grid, cfg.grid, cfg.embedDim]), key: "pos_embed")
        _blocks = ModuleInfo(
            wrappedValue: (0 ..< cfg.depth).map { SAMBlock(cfg, global: cfg.globalAttnIndexes.contains($0)) },
            key: "blocks")
        _neck = ModuleInfo(wrappedValue: [
            Conv2d(inputChannels: cfg.embedDim, outputChannels: cfg.outChans,
                   kernelSize: IntOrPair(1), bias: false),               // 0
            SAMLayerNorm2d(cfg.outChans),                                // 1
            Conv2d(inputChannels: cfg.outChans, outputChannels: cfg.outChans,
                   kernelSize: IntOrPair(3), padding: IntOrPair(1), bias: false),  // 2
            SAMLayerNorm2d(cfg.outChans),                                // 3
        ], key: "neck")
        _net2 = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: 256, outputChannels: 512, kernelSize: IntOrPair(3),
                                 stride: IntOrPair(2), padding: IntOrPair(1), bias: false), key: "net_2")
        _net3 = ModuleInfo(
            wrappedValue: Conv2d(inputChannels: 512, outputChannels: 1024, kernelSize: IntOrPair(3),
                                 stride: IntOrPair(2), padding: IntOrPair(1), bias: false), key: "net_3")
    }

    /// `image`: channels-last [B, 1024, 1024, 3]. Returns patch_embeds
    /// [B, 1024, 16, 16] (channels-first, as the CLIP tower expects).
    public func callAsFunction(image: MLXArray) -> MLXArray {
        var x = patchEmbed(image) + posEmbed                  // [B,64,64,768]
        for blk in blocks { x = blk(x) }
        for layer in neck {                                   // Conv, LN2d, Conv, LN2d
            if let c = layer as? Conv2d { x = c(x) }
            else if let ln = layer as? SAMLayerNorm2d { x = ln(x) }
        }                                                     // [B,64,64,256]
        x = net2(x)                                           // [B,32,32,512]
        x = net3(x)                                           // [B,16,16,1024]
        return x.transposed(0, 3, 1, 2)                       // -> [B,1024,16,16]
    }
}

