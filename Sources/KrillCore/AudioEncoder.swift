import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Native Gemma 4 Audio Encoder (USM Conformer)

/// Swift + MLX port of the Gemma 4 USM Conformer audio tower
/// (`mlx-vlm` `models/gemma4/audio.py`, the correctness oracle). Module
/// keys mirror the checkpoint `audio_tower.*` hierarchy exactly so weights
/// load with no remapping. See `docs/GEMMA4_INTERNALS.md` "Audio Encoder".

/// USM audio config (checkpoint `audio_config`; all values have fixed
/// defaults so a missing field never breaks load).
public struct AudioConfig: Sendable {
    public var hiddenSize = 1024
    public var numHiddenLayers = 12
    public var numAttentionHeads = 8
    public var subsamplingConvChannels = [128, 32]
    public var convKernelSize = 5
    public var residualWeight: Float = 0.5
    public var attentionChunkSize = 12
    public var attentionContextLeft = 13
    public var attentionContextRight = 0
    public var attentionLogitCap: Float = 50.0
    public var attentionInvalidLogitsValue: Float = -1e9
    public var rmsNormEps: Float = 1e-6
    public var gradientClipping: Float = 1e10
    public var outputProjDims = 1536

    public init() {}

    /// Build from the raw `config.json` `audio_config` dict; absent keys
    /// keep the USM default.
    public init(from dict: [String: Any]?) {
        guard let d = dict else { return }
        if let v = d["hidden_size"] as? Int { hiddenSize = v }
        if let v = d["num_hidden_layers"] as? Int { numHiddenLayers = v }
        if let v = d["num_attention_heads"] as? Int { numAttentionHeads = v }
        if let v = d["subsampling_conv_channels"] as? [Int] { subsamplingConvChannels = v }
        if let v = d["conv_kernel_size"] as? Int { convKernelSize = v }
        if let v = d["residual_weight"] as? Double { residualWeight = Float(v) }
        if let v = d["attention_chunk_size"] as? Int { attentionChunkSize = v }
        if let v = d["attention_context_left"] as? Int { attentionContextLeft = v }
        if let v = d["attention_context_right"] as? Int { attentionContextRight = v }
        if let v = d["attention_logit_cap"] as? Double { attentionLogitCap = Float(v) }
        if let v = d["attention_invalid_logits_value"] as? Double {
            attentionInvalidLogitsValue = Float(v)
        }
        if let v = d["rms_norm_eps"] as? Double { rmsNormEps = Float(v) }
        if let v = d["gradient_clipping"] as? Double { gradientClipping = Float(v) }
        if let v = d["output_proj_dims"] as? Int { outputProjDims = v }
    }

    var headDim: Int { hiddenSize / numAttentionHeads }
}

/// Weighted RMSNorm (`mx.fast.rms_norm`-equivalent; weight applied, no bias).
final class AudioRMSNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight = ModuleInfo(wrappedValue: MLXArray.ones([dim]), key: "weight")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
        return (xf * MLX.rsqrt(v + MLXArray(eps)) * weight.asType(.float32))
            .asType(x.dtype)
    }
}

/// LayerNorm over the channel (last) dim, weight only (no bias), matching
/// the SSCP `norm.weight`-only checkpoint keys.
final class AudioChannelLayerNorm: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight = ModuleInfo(wrappedValue: MLXArray.ones([dim]), key: "weight")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let mean = MLX.mean(xf, axis: -1, keepDims: true)
        let v = MLX.variance(xf, axis: -1, keepDims: true)
        let n = (xf - mean) * MLX.rsqrt(v + MLXArray(eps))
        return (n * weight.asType(.float32)).asType(x.dtype)
    }
}

/// Depthwise Conv1d (groups == channels); weight `[C, kW, 1]` (MLX layout).
final class DepthwiseConv1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let groups: Int
    init(channels: Int, kernel: Int) {
        groups = channels
        _weight = ModuleInfo(
            wrappedValue: MLXArray.zeros([channels, kernel, 1]), key: "weight")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv1d(x, weight, stride: 1, padding: 0, dilation: 1, groups: groups)
    }
}

private func clipGrad(_ x: MLXArray, _ c: Float) -> MLXArray {
    MLX.clip(x, min: MLXArray(-c), max: MLXArray(c))
}

/// SSCP Conv2d block: zero invalid -> pad (1,1,1,1) -> Conv2d s2 ->
/// LayerNorm(C) -> ReLU, with the time mask downsampled by stride 2.
final class SSCPConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "norm") var norm: AudioChannelLayerNorm
    let timeStride = 2

    init(_ config: AudioConfig, idx: Int) {
        let inC = idx == 0 ? 1 : config.subsamplingConvChannels[idx - 1]
        let outC = config.subsamplingConvChannels[idx]
        _conv = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: inC, outputChannels: outC,
                kernelSize: IntOrPair((3, 3)), stride: IntOrPair((2, 2)),
                padding: IntOrPair((0, 0)), bias: false),
            key: "conv")
        _norm = ModuleInfo(
            wrappedValue: AudioChannelLayerNorm(outC, eps: config.rmsNormEps),
            key: "norm")
    }

    /// x: [B,T,F,C]  mask: [B,T] (true = invalid/padding)
    func callAsFunction(_ x: MLXArray, _ mask: MLXArray) -> (MLXArray, MLXArray) {
        let m4 = mask.expandedDimensions(axis: -1).expandedDimensions(axis: -1)
        var h = MLX.where(m4, MLXArray(Float(0)), x)
        h = padded(h, widths: [
            IntOrPair((0, 0)), IntOrPair((1, 1)),
            IntOrPair((1, 1)), IntOrPair((0, 0)),
        ])
        h = conv(h)                              // [B,T',F',C']
        let tOut = h.dim(1)
        let T = mask.dim(1)
        let idx = Array(stride(from: 0, to: T, by: timeStride)).map { Int32($0) }
        var outMask = MLX.take(mask, MLXArray(idx), axis: 1)
        if outMask.dim(1) > tOut { outMask = outMask[0..., 0 ..< tOut] }
        h = norm(h)
        h = MLX.maximum(h, MLXArray(Float(0)))   // ReLU
        return (h, outMask)
    }
}

/// SSCP: 2 conv blocks -> flatten(F*C) -> Linear -> [B,T,hidden].
final class SubSampleConvProjection: Module {
    static let inputFeatSize = 128
    @ModuleInfo(key: "layer0") var layer0: SSCPConvBlock
    @ModuleInfo(key: "layer1") var layer1: SSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjLinear: Linear

    init(_ config: AudioConfig) {
        _layer0 = ModuleInfo(wrappedValue: SSCPConvBlock(config, idx: 0), key: "layer0")
        _layer1 = ModuleInfo(wrappedValue: SSCPConvBlock(config, idx: 1), key: "layer1")
        var freq = Self.inputFeatSize
        for _ in 0 ..< 2 { freq = (freq + 2 - 3) / 2 + 1 }
        let projInDim = freq * config.subsamplingConvChannels[config.subsamplingConvChannels.count - 1]
        _inputProjLinear = ModuleInfo(
            wrappedValue: Linear(projInDim, config.hiddenSize, bias: false),
            key: "input_proj_linear")
    }

    func callAsFunction(_ audioMel: MLXArray, _ mask: MLXArray) -> (MLXArray, MLXArray) {
        var x = audioMel.expandedDimensions(axis: -1)   // [B,T,F,1]
        var m = mask
        (x, m) = layer0(x, m)
        (x, m) = layer1(x, m)
        let B = x.dim(0), T = x.dim(1), F = x.dim(2), C = x.dim(3)
        x = x.reshaped([B, T, F * C])
        return (inputProjLinear(x), m)
    }
}

/// Macaron FFW: clip -> pre_norm -> ffw1 -> SiLU -> ffw2 -> clip ->
/// post_norm; residual + x * residual_weight.
final class ConformerFeedForward: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var ffwLayer1: ClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var ffwLayer2: ClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: AudioRMSNorm
    let grad: Float
    let residualWeight: Float

    init(_ config: AudioConfig) {
        grad = config.gradientClipping
        residualWeight = config.residualWeight
        let h = config.hiddenSize
        _preLayerNorm = ModuleInfo(
            wrappedValue: AudioRMSNorm(h, eps: config.rmsNormEps), key: "pre_layer_norm")
        _ffwLayer1 = ModuleInfo(wrappedValue: ClippableLinear(h, h * 4), key: "ffw_layer_1")
        _ffwLayer2 = ModuleInfo(wrappedValue: ClippableLinear(h * 4, h), key: "ffw_layer_2")
        _postLayerNorm = ModuleInfo(
            wrappedValue: AudioRMSNorm(h, eps: config.rmsNormEps), key: "post_layer_norm")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = clipGrad(x, grad)
        h = preLayerNorm(h)
        h = ffwLayer1(h)
        h = silu(h)
        h = ffwLayer2(h)
        h = clipGrad(h, grad)
        h = postLayerNorm(h)
        return residual + h * MLXArray(residualWeight)
    }
}

/// Chunked local attention with sinusoidal relative-position bias and
/// logit softcapping. Faithful port of `AudioAttention` + the embedded
/// `AudioRelativePositionEmbedding` logic.
final class AudioAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: ClippableLinear
    @ModuleInfo(key: "k_proj") var kProj: ClippableLinear
    @ModuleInfo(key: "v_proj") var vProj: ClippableLinear
    @ModuleInfo(key: "post") var post: ClippableLinear
    @ModuleInfo(key: "relative_k_proj") var relativeKProj: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimScale: MLXArray

    let numHeads: Int, hiddenSize: Int, headDim: Int
    let chunkSize: Int, maxFuture: Int, maxPast: Int, contextSize: Int
    let invalidValue: Float, softcap: Float
    let qScale: Float, kScale: Float

    init(_ config: AudioConfig) {
        numHeads = config.numAttentionHeads
        hiddenSize = config.hiddenSize
        headDim = config.headDim
        chunkSize = config.attentionChunkSize
        maxFuture = config.attentionContextRight
        maxPast = max(0, config.attentionContextLeft - 1)
        contextSize = chunkSize + maxPast + maxFuture
        invalidValue = config.attentionInvalidLogitsValue
        softcap = config.attentionLogitCap
        qScale = Float(pow(Double(headDim), -0.5)) / Float(log(2.0))
        kScale = Float(log(1.0 + M_E) / log(2.0))

        let proj = numHeads * headDim
        _qProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, proj), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, proj), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, proj), key: "v_proj")
        _post = ModuleInfo(wrappedValue: ClippableLinear(hiddenSize, hiddenSize), key: "post")
        _relativeKProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, proj, bias: false), key: "relative_k_proj")
        _perDimScale = ModuleInfo(
            wrappedValue: MLXArray.zeros([headDim]), key: "per_dim_scale")
    }

    private func padDim1(_ x: MLXArray, _ l: Int, _ r: Int) -> MLXArray {
        var widths = Array(repeating: IntOrPair((0, 0)), count: x.ndim)
        widths[1] = IntOrPair((l, r))
        return padded(x, widths: widths)
    }

    /// [B,T,...] -> [B,U,chunk,...] (right-pad T to a multiple of chunk).
    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), T = x.dim(1)
        let rest = Array(x.shape.dropFirst(2))
        let numBlocks = (T + chunkSize - 1) / chunkSize
        let padLen = numBlocks * chunkSize - T
        var h = x
        if padLen > 0 { h = padDim1(h, 0, padLen) }
        return h.reshaped([B, numBlocks, chunkSize] + rest)
    }

    /// [B,T,...] -> [B,U,context,...] sliding windows (past/future pad).
    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPast
        let padRight = maxFuture + chunkSize - 1
        let h = padDim1(x, padLeft, padRight)
        let Tp = h.dim(1)
        let numBlocks = (Tp - contextSize) / chunkSize + 1
        var idx = [Int32]()
        idx.reserveCapacity(numBlocks * contextSize)
        for b in 0 ..< numBlocks {
            let start = b * chunkSize
            for o in 0 ..< contextSize { idx.append(Int32(start + o)) }
        }
        let rest = Array(h.shape.dropFirst(2))
        let gathered = MLX.take(h, MLXArray(idx), axis: 1)   // [B, U*C, ...]
        return gathered.reshaped([h.dim(0), numBlocks, contextSize] + rest)
    }

    private func timingSignal(_ position: [Float], _ dtype: DType) -> MLXArray {
        let numTs = hiddenSize / 2
        let minTs = 1.0, maxTs = 10000.0
        let logInc = log(maxTs / minTs) / Double(max(numTs - 1, 1))
        var inv = [Float](repeating: 0, count: numTs)
        for i in 0 ..< numTs { inv[i] = Float(minTs * exp(Double(i) * -logInc)) }
        let P = position.count
        var sinv = [Float](repeating: 0, count: P * numTs)
        var cosv = [Float](repeating: 0, count: P * numTs)
        for p in 0 ..< P {
            for i in 0 ..< numTs {
                let t = position[p] * inv[i]
                sinv[p * numTs + i] = sin(t)
                cosv[p * numTs + i] = cos(t)
            }
        }
        let s = MLXArray(sinv, [P, numTs])
        let c = MLXArray(cosv, [P, numTs])
        return MLX.concatenated([s, c], axis: -1).asType(dtype)   // [P, hidden]
    }

    private func relativeShift(_ termBD: MLXArray, _ B: Int, _ N: Int,
                               _ U: Int, _ W: Int, _ C: Int,
                               _ maxSpanP1: Int) -> MLXArray {
        let padAmount = (C + 1) - maxSpanP1
        var h = padded(termBD, widths: [
            IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 0)),
            IntOrPair((0, 0)), IntOrPair((0, padAmount)),
        ])
        h = h.reshaped([B, N, U, W * (C + 1)])
        h = h[0..., 0..., 0..., 0 ..< (W * C)]
        return h.reshaped([B, N, U, W, C])
    }

    /// queries [B,U,W,N,H]  keys [B,U,C,N,H] -> logits [B,N,U,W,C]
    private func relPosLogits(_ queries: MLXArray, _ keys: MLXArray) -> MLXArray {
        let B = queries.dim(0), U = queries.dim(1), W = queries.dim(2)
        let N = queries.dim(3), H = queries.dim(4)
        let C = keys.dim(2)

        var pos = [Float]()
        var p = maxPast
        while p >= -maxFuture { pos.append(Float(p)); p -= 1 }
        let maxSpanP1 = pos.count

        var sinEmb = timingSignal(pos, queries.dtype)               // [S, hidden]
        sinEmb = relativeKProj(sinEmb.asType(relativeKProj.weight.dtype))
        sinEmb = sinEmb.reshaped([maxSpanP1, numHeads, headDim]).asType(queries.dtype)

        let qp = queries.transposed(0, 3, 1, 2, 4)                  // [B,N,U,W,H]
        let kp = keys.transposed(0, 3, 1, 4, 2)                     // [B,N,U,H,C]
        let termAC = qp.matmul(kp)                                  // [B,N,U,W,C]

        let sinT = sinEmb.transposed(1, 2, 0)                       // [N,H,S]
        let qr = qp.reshaped([B, N, U * W, H])
        var termBD = qr.matmul(sinT)                                // [B,N,U*W,S]
        termBD = termBD.reshaped([B, N, U, W, maxSpanP1])
        termBD = relativeShift(termBD, B, N, U, W, C, maxSpanP1)
        return termAC + termBD
    }

    /// hidden [B,T,D]  mask [B,T] (true=invalid)  cvm [chunk,context] bool
    func callAsFunction(_ hidden: MLXArray, _ mask: MLXArray,
                        _ causalValid: MLXArray) -> MLXArray {
        let B = hidden.dim(0), T = hidden.dim(1)
        let qkv = [B, T, numHeads, headDim]
        var q = qProj(hidden).asType(.float32).reshaped(qkv)
        var k = kProj(hidden).asType(.float32).reshaped(qkv)
        let v = vProj(hidden).asType(.float32).reshaped(qkv)

        let pds = softplus(perDimScale.asType(.float32))
        q = q * (MLXArray(qScale) * pds)
        k = k * MLXArray(kScale)

        let qBlocks = convertToBlock(q)                  // [B,U,W,N,H]
        let kBlocks = extractBlockContext(k)             // [B,U,C,N,H]
        let vBlocks = extractBlockContext(v)             // [B,U,C,N,H]
        let U = qBlocks.dim(1)

        let validMask = mask .== MLXArray(false)         // ~mask
        let extractedValid = extractBlockContext(validMask)   // [B,U,C]
        // [B,1,U,1,C] & [1,1,1,W,C] -> broadcasts with logits [B,N,U,W,C]
        let ev = extractedValid.expandedDimensions(axis: 1)
            .expandedDimensions(axis: 3)
        let cv = causalValid.expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let condition = ev .&& cv

        var logits = relPosLogits(qBlocks, kBlocks)      // [B,N,U,W,C]
        logits = MLX.tanh(logits / MLXArray(softcap)) * MLXArray(softcap)
        logits = MLX.where(condition, logits, MLXArray(invalidValue))
        let probs = MLX.softmax(logits, axis: -1)        // [B,N,U,W,C]

        // einsum bnuwc,bucnh->buwnh
        let context = MLX.einsum("bnuwc,bucnh->buwnh", probs, vBlocks)
        var ctx = context.reshaped([B, U * chunkSize, numHeads, headDim])
        ctx = ctx[0..., 0 ..< T]
        ctx = ctx.reshaped([B, T, numHeads * headDim]).asType(hidden.dtype)
        return post(ctx)
    }
}

/// Light conv module: pre_norm -> linear_start -> GLU -> causal depthwise
/// conv -> clip -> conv_norm -> SiLU -> linear_end + residual.
final class ConformerLightConv1d: Module {
    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: AudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: ClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: DepthwiseConv1d
    @ModuleInfo(key: "conv_norm") var convNorm: AudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: ClippableLinear
    let grad: Float
    let causalPad: Int

    init(_ config: AudioConfig) {
        grad = config.gradientClipping
        causalPad = config.convKernelSize - 1
        let h = config.hiddenSize
        _preLayerNorm = ModuleInfo(
            wrappedValue: AudioRMSNorm(h, eps: config.rmsNormEps), key: "pre_layer_norm")
        _linearStart = ModuleInfo(
            wrappedValue: ClippableLinear(h, h * 2), key: "linear_start")
        _depthwiseConv1d = ModuleInfo(
            wrappedValue: DepthwiseConv1d(channels: h, kernel: config.convKernelSize),
            key: "depthwise_conv1d")
        _convNorm = ModuleInfo(
            wrappedValue: AudioRMSNorm(h, eps: config.rmsNormEps), key: "conv_norm")
        _linearEnd = ModuleInfo(
            wrappedValue: ClippableLinear(h, h), key: "linear_end")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var h = preLayerNorm(x)
        h = linearStart(h)
        let parts = MLX.split(h, parts: 2, axis: -1)
        h = parts[0] * MLX.sigmoid(parts[1])              // GLU
        h = padded(h, widths: [
            IntOrPair((0, 0)), IntOrPair((causalPad, 0)), IntOrPair((0, 0)),
        ])
        h = depthwiseConv1d(h)
        h = clipGrad(h, grad)
        h = convNorm(h)
        h = silu(h)
        h = linearEnd(h)
        return h + residual
    }
}

/// One macaron Conformer block.
final class AudioConformerBlock: Module {
    @ModuleInfo(key: "feed_forward1") var feedForward1: ConformerFeedForward
    @ModuleInfo(key: "self_attn") var selfAttn: AudioAttention
    @ModuleInfo(key: "lconv1d") var lconv1d: ConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: ConformerFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttn: AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: AudioRMSNorm
    let grad: Float

    init(_ config: AudioConfig) {
        grad = config.gradientClipping
        _feedForward1 = ModuleInfo(wrappedValue: ConformerFeedForward(config), key: "feed_forward1")
        _selfAttn = ModuleInfo(wrappedValue: AudioAttention(config), key: "self_attn")
        _lconv1d = ModuleInfo(wrappedValue: ConformerLightConv1d(config), key: "lconv1d")
        _feedForward2 = ModuleInfo(wrappedValue: ConformerFeedForward(config), key: "feed_forward2")
        _normPreAttn = ModuleInfo(
            wrappedValue: AudioRMSNorm(config.hiddenSize, eps: config.rmsNormEps),
            key: "norm_pre_attn")
        _normPostAttn = ModuleInfo(
            wrappedValue: AudioRMSNorm(config.hiddenSize, eps: config.rmsNormEps),
            key: "norm_post_attn")
        _normOut = ModuleInfo(
            wrappedValue: AudioRMSNorm(config.hiddenSize, eps: config.rmsNormEps),
            key: "norm_out")
    }

    func callAsFunction(_ x: MLXArray, _ mask: MLXArray,
                        _ causalValid: MLXArray) -> MLXArray {
        var h = feedForward1(x)
        let residual = h
        h = clipGrad(h, grad)
        h = normPreAttn(h)
        h = selfAttn(h, mask, causalValid)
        h = clipGrad(h, grad)
        h = residual + normPostAttn(h)
        let validity = (mask .== MLXArray(false))
            .expandedDimensions(axis: -1).asType(h.dtype)
        h = h * validity
        h = lconv1d(h)
        h = feedForward2(h)
        h = clipGrad(h, grad)
        return normOut(h)
    }
}

/// Gemma 4 USM audio encoder. Loaded under checkpoint key `audio_tower`.
public final class AudioEncoder: Module {
    @ModuleInfo(key: "subsample_conv_projection") var subsampleConvProjection: SubSampleConvProjection
    @ModuleInfo(key: "layers") var layers: [AudioConformerBlock]
    @ModuleInfo(key: "output_proj") var outputProj: Linear
    let config: AudioConfig

    public init(_ config: AudioConfig = AudioConfig()) {
        self.config = config
        _subsampleConvProjection = ModuleInfo(
            wrappedValue: SubSampleConvProjection(config),
            key: "subsample_conv_projection")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in
                AudioConformerBlock(config)
            }, key: "layers")
        _outputProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.outputProjDims, bias: true),
            key: "output_proj")
    }

    /// Local causal+context validity mask, [chunk, context] bool.
    private func buildCausalValidMask() -> MLXArray {
        let chunk = config.attentionChunkSize
        let maxFuture = config.attentionContextRight
        let maxPast = max(0, config.attentionContextLeft - 1)
        let upperDiag = maxPast + maxFuture
        let ctx = chunk + maxPast + maxFuture
        let lowerCausal = MLX.tril(MLX.ones([ctx, chunk])).transposed(1, 0)
        let upperCausal = MLX.tril(MLX.ones([chunk, ctx]), k: upperDiag)
        return (lowerCausal * upperCausal) .!= MLXArray(Float(0))
    }

    /// audioMel [1,T,128]  validMask [1,T] bool (true = valid audio).
    /// Returns (encodings [1,T',1536], invalidMask [1,T'] true=padding).
    public func callAsFunction(_ audioMel: MLXArray,
                               validMask: MLXArray) -> (MLXArray, MLXArray) {
        // The tower's internal mask is "invalid" (true = padding).
        let invalid = validMask .== MLXArray(false)
        var (enc, current) = subsampleConvProjection(audioMel, invalid)
        let cvm = buildCausalValidMask()
        for block in layers { enc = block(enc, current, cvm) }
        enc = outputProj(enc)
        if current.dim(1) != enc.dim(1) {
            current = current[0..., 0 ..< enc.dim(1)]
        }
        enc = MLX.where(current.expandedDimensions(axis: -1),
                        MLXArray(Float(0)), enc)
        return (enc, current)
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
