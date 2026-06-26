import Foundation
import MLX
import MLXNN
import MLXFast
import KrillCache

// Native Swift+MLX runtime for the Qwen3.5 (`qwen3_5_text`) decoder — the text
// backbone of Ornith-1.0-9B. This is a Qwen3-Next-class HYBRID: most layers are
// GatedDeltaNet linear attention (SSM / delta-rule), a minority are full softmax
// attention (every `fullAttentionInterval`-th). Spec + weight contract in
// `.claude/skills/native-port/families/qwen3_5.md`.
//
// Vision tower is intentionally NOT implemented here (text-only port); image
// inference stays on mlx_vlm. Parity oracle: `mlx_lm.models.qwen3_5`.
//
// Status: core GatedDeltaNet (this file) lands first, parity-gated in isolation
// before the full decoder is wired up.

// MARK: - Config

public struct Qwen35Config: Codable {
    public var hiddenSize: Int
    public var intermediateSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var rmsNormEps: Float
    public var vocabSize: Int
    public var tieWordEmbeddings: Bool
    public var fullAttentionInterval: Int

    // Linear-attention (GatedDeltaNet) dims
    public var linearNumValueHeads: Int
    public var linearNumKeyHeads: Int
    public var linearKeyHeadDim: Int
    public var linearValueHeadDim: Int
    public var linearConvKernelDim: Int

    // RoPE (partial; mRoPE collapses to standard RoPE for text-only)
    public var ropeTheta: Float
    public var partialRotaryFactor: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case fullAttentionInterval = "full_attention_interval"
        case linearNumValueHeads = "linear_num_value_heads"
        case linearNumKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (try c.decode(Int.self, forKey: .hiddenSize) / c.decode(Int.self, forKey: .numAttentionHeads))
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        fullAttentionInterval = try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        linearNumValueHeads = try c.decode(Int.self, forKey: .linearNumValueHeads)
        linearNumKeyHeads = try c.decode(Int.self, forKey: .linearNumKeyHeads)
        linearKeyHeadDim = try c.decode(Int.self, forKey: .linearKeyHeadDim)
        linearValueHeadDim = try c.decode(Int.self, forKey: .linearValueHeadDim)
        linearConvKernelDim = try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000_000
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
    }

    /// True for GatedDeltaNet (linear-attention) layers under the hybrid schedule.
    public func isLinearLayer(_ idx: Int) -> Bool {
        (idx + 1) % fullAttentionInterval != 0
    }
}

// MARK: - Helpers

/// RMS-normalise over the last axis with NO learned weight (mlx `rms_norm(x, None, eps)`).
@inline(__always)
func rmsNormNoWeight(_ x: MLXArray, eps: Float) -> MLXArray {
    let xf = x.asType(.float32)
    let norm = rsqrt(xf.square().mean(axis: -1, keepDims: true) + eps)
    return (xf * norm).asType(x.dtype)
}

/// Qwen3-Next gated RMSNorm: rms_norm(h, weight) then, if gated, `silu(z)·x` in fp32.
final class Qwen35RMSNormGated: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(_ dim: Int, eps: Float) {
        self.eps = eps
        _weight = ModuleInfo(wrappedValue: MLXArray.ones([dim]), key: "weight")
    }

    func callAsFunction(_ hidden: MLXArray, gate: MLXArray) -> MLXArray {
        let x = MLXFast.rmsNorm(hidden, weight: weight, eps: eps)
        let g = silu(gate.asType(.float32))
        return (g * x.asType(.float32)).asType(hidden.dtype)
    }
}

/// Depthwise causal Conv1d, weight `[C, kW, 1]` (MLX layout) — same as AudioEncoder's.
final class Qwen35DepthwiseConv1d: Module {
    @ModuleInfo(key: "weight") var weight: MLXArray
    let groups: Int
    init(channels: Int, kernel: Int) {
        groups = channels
        _weight = ModuleInfo(wrappedValue: MLXArray.zeros([channels, kernel, 1]), key: "weight")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv1d(x, weight, stride: 1, padding: 0, dilation: 1, groups: groups)
    }
}

/// Sequential delta-rule scan (ops reference port of `gated_delta_ops`).
/// Inputs (already split into heads, q/k at `numKHeads`, repeated to `numVHeads`
/// internally): q,k `[B,S,Hk,Dk]`, v `[B,S,Hv,Dv]`, g,beta `[B,S,Hv]`.
/// Returns y `[B,S,Hv,Dv]` and final state `[B,Hv,Dv,Dk]` (float32).
func gatedDeltaScan(
    q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray,
    state initialState: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let B = q.dim(0), S = q.dim(1), Hk = q.dim(2), Dk = q.dim(3)
    let Hv = v.dim(2), Dv = v.dim(3)

    var qh = q, kh = k
    let rep = Hv / Hk
    if rep > 1 {
        // repeat along the head axis (Hk -> Hv), matching mx.repeat(..., -2)
        qh = repeated(q, count: rep, axis: 2)
        kh = repeated(k, count: rep, axis: 2)
    }

    var state = initialState ?? MLXArray.zeros([B, Hv, Dv, Dk], dtype: .float32)
    var ys: [MLXArray] = []
    ys.reserveCapacity(S)

    for t in 0 ..< S {
        let qt = qh[0..., t].asType(.float32)        // [B,Hv,Dk]
        let kt = kh[0..., t].asType(.float32)        // [B,Hv,Dk]
        let vt = v[0..., t].asType(.float32)         // [B,Hv,Dv]
        let gt = g[0..., t].asType(.float32)         // [B,Hv]
        let bt = beta[0..., t].asType(.float32)      // [B,Hv]

        // decay: state *= g[...,None,None]
        let decay = gt.expandedDimensions(axis: -1).expandedDimensions(axis: -1)  // [B,Hv,1,1]
        state = state * decay

        let ktE = kt.expandedDimensions(axis: 2)     // [B,Hv,1,Dk]
        let kvMem = (state * ktE).sum(axis: -1)      // [B,Hv,Dv]
        let delta = (vt - kvMem) * bt.expandedDimensions(axis: -1)  // [B,Hv,Dv]
        state = state + ktE * delta.expandedDimensions(axis: -1)    // [B,Hv,Dv,Dk]

        let qtE = qt.expandedDimensions(axis: 2)     // [B,Hv,1,Dk]
        let yt = (state * qtE).sum(axis: -1)         // [B,Hv,Dv]
        ys.append(yt)
    }

    let y = stacked(ys, axis: 1).asType(q.dtype)     // [B,S,Hv,Dv]
    return (y, state)
}

// MARK: - GatedDeltaNet (linear-attention layer)

final class Qwen35GatedDeltaNet: Module {
    let hiddenSize: Int
    let numVHeads: Int
    let numKHeads: Int
    let headKDim: Int
    let headVDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernel: Int
    let convDim: Int
    let eps: Float

    @ModuleInfo(key: "in_proj_qkv") var inProjQkv: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "conv1d") var conv1dLayer: Qwen35DepthwiseConv1d
    @ModuleInfo(key: "norm") var norm: Qwen35RMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ c: Qwen35Config) {
        hiddenSize = c.hiddenSize
        numVHeads = c.linearNumValueHeads
        numKHeads = c.linearNumKeyHeads
        headKDim = c.linearKeyHeadDim
        headVDim = c.linearValueHeadDim
        keyDim = headKDim * numKHeads
        valueDim = headVDim * numVHeads
        convKernel = c.linearConvKernelDim
        convDim = keyDim * 2 + valueDim
        eps = c.rmsNormEps

        _inProjQkv = ModuleInfo(wrappedValue: Linear(hiddenSize, keyDim * 2 + valueDim, bias: false), key: "in_proj_qkv")
        _inProjZ = ModuleInfo(wrappedValue: Linear(hiddenSize, valueDim, bias: false), key: "in_proj_z")
        _inProjB = ModuleInfo(wrappedValue: Linear(hiddenSize, numVHeads, bias: false), key: "in_proj_b")
        _inProjA = ModuleInfo(wrappedValue: Linear(hiddenSize, numVHeads, bias: false), key: "in_proj_a")
        _conv1dLayer = ModuleInfo(wrappedValue: Qwen35DepthwiseConv1d(channels: convDim, kernel: convKernel), key: "conv1d")
        _norm = ModuleInfo(wrappedValue: Qwen35RMSNormGated(headVDim, eps: eps), key: "norm")
        _outProj = ModuleInfo(wrappedValue: Linear(valueDim, hiddenSize, bias: false), key: "out_proj")
        _aLog = ParameterInfo(wrappedValue: MLXArray.zeros([numVHeads]), key: "A_log")
        _dtBias = ParameterInfo(wrappedValue: MLXArray.ones([numVHeads]), key: "dt_bias")
    }

    /// Full-sequence (prefill) forward, no cache. `x`: `[B,S,hiddenSize]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)

        let qkv = inProjQkv(x)                                   // [B,S,convDim]
        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let a = inProjA(x)                                       // [B,S,numVHeads]
        let b = inProjB(x)                                       // [B,S,numVHeads]

        // causal depthwise conv: prepend (k-1) zeros, conv (padding 0), silu
        let pad = MLXArray.zeros([B, convKernel - 1, convDim], dtype: x.dtype)
        let convInput = concatenated([pad, qkv], axis: 1)        // [B,S+k-1,convDim]
        let convOut = silu(conv1dLayer(convInput))               // [B,S,convDim]

        let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        var qh = parts[0].reshaped(B, S, numKHeads, headKDim)
        var kh = parts[1].reshaped(B, S, numKHeads, headKDim)
        let vh = parts[2].reshaped(B, S, numVHeads, headVDim)

        let invScale = Float(headKDim).squareRoot()  // == headKDim^0.5 ; inv = 1/that
        let inv = 1.0 / invScale
        qh = (inv * inv) * rmsNormNoWeight(qh, eps: 1e-6)
        kh = inv * rmsNormNoWeight(kh, eps: 1e-6)

        // g = exp(-exp(A_log_f32) * softplus(a + dt_bias)) ; beta = sigmoid(b)
        let g = exp(-exp(aLog.asType(.float32)) * softplus(a.asType(.float32) + dtBias.asType(.float32)))
        let beta = sigmoid(b)

        let (y, _) = gatedDeltaScan(q: qh, k: kh, v: vh, g: g, beta: beta)
        let out = norm(y, gate: z)                               // [B,S,numVHeads,headVDim]
        return outProj(out.reshaped(B, S, numVHeads * headVDim))
    }
}
