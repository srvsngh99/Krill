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

    /// Forward. `x`: `[B,S,hiddenSize]`. With `cache`, carries conv + SSM state
    /// across calls for incremental decode; without it, a fresh prefill.
    func callAsFunction(_ x: MLXArray, cache: GatedDeltaCache? = nil) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)

        let qkv = inProjQkv(x)                                   // [B,S,convDim]
        let z = inProjZ(x).reshaped(B, S, numVHeads, headVDim)
        let a = inProjA(x)                                       // [B,S,numVHeads]
        let b = inProjB(x)                                       // [B,S,numVHeads]

        // causal depthwise conv: prepend the (k-1)-col conv state, conv, silu.
        let convState = cache?.convState ?? MLXArray.zeros([B, convKernel - 1, convDim], dtype: x.dtype)
        let convInput = concatenated([convState, qkv], axis: 1)  // [B,S+k-1,convDim]
        if let cache {
            let nKeep = convKernel - 1
            let total = convInput.dim(1)
            cache.convState = convInput[0..., (total - nKeep) ..< total].asType(x.dtype)
        }
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

        let (y, newState) = gatedDeltaScan(q: qh, k: kh, v: vh, g: g, beta: beta, state: cache?.ssmState)
        if let cache {
            cache.ssmState = newState
            cache.advance(S)
        }
        let out = norm(y, gate: z)                               // [B,S,numVHeads,headVDim]
        return outProj(out.reshaped(B, S, numVHeads * headVDim))
    }
}

// The per-layer GatedDeltaNet cache (`GatedDeltaCache`: conv-state + SSM
// recurrent state) lives in KrillCache so the engine's `makeKVCaches(spec:)`
// can build it for `.ssm` layers.

// MARK: - Full softmax attention (Qwen3-Next gated attention)

final class Qwen35Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPE

    init(_ c: Qwen35Config) {
        numHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = 1.0 / Float(c.headDim).squareRoot()
        _qProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numHeads * headDim * 2, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numKVHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numKVHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, c.hiddenSize, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps), key: "k_norm")
        // Partial rotary: rotate the first floor(headDim * partialRotaryFactor) dims.
        // mRoPE collapses to standard RoPE for text-only positions.
        let rotaryDim = Int(Float(headDim) * c.partialRotaryFactor)
        self.rope = RoPE(dimensions: rotaryDim, traditional: false, base: c.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        let qp = qProj(x).reshaped(B, L, numHeads, headDim * 2)
        let qParts = split(qp, indices: [headDim], axis: -1)
        var queries = qNorm(qParts[0])                                  // [B,L,numHeads,headDim]
        let gate = qParts[1].reshaped(B, L, numHeads * headDim)
        var keys = kNorm(kProj(x).reshaped(B, L, numKVHeads, headDim))
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim)

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.sequenceLength ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        let o = out.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim)
        return oProj(o * sigmoid(gate))
    }
}

// MARK: - MLP

final class Qwen35MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ dim: Int, _ hidden: Int) {
        _gateProj = ModuleInfo(wrappedValue: Linear(dim, hidden, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(wrappedValue: Linear(dim, hidden, bias: false), key: "up_proj")
        _downProj = ModuleInfo(wrappedValue: Linear(hidden, dim, bias: false), key: "down_proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Decoder layer (hybrid: linear vs full attention)

final class Qwen35DecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "linear_attn") var linearAttn: Qwen35GatedDeltaNet?
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention?
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Qwen35MLP

    init(_ c: Qwen35Config, layerIdx: Int) {
        isLinear = c.isLinearLayer(layerIdx)
        if isLinear {
            _linearAttn = ModuleInfo(wrappedValue: Qwen35GatedDeltaNet(c), key: "linear_attn")
            _selfAttn = ModuleInfo(wrappedValue: nil, key: "self_attn")
        } else {
            _linearAttn = ModuleInfo(wrappedValue: nil, key: "linear_attn")
            _selfAttn = ModuleInfo(wrappedValue: Qwen35Attention(c), key: "self_attn")
        }
        _inputLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "post_attention_layernorm")
        _mlp = ModuleInfo(wrappedValue: Qwen35MLP(c.hiddenSize, c.intermediateSize), key: "mlp")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCacheProtocol? = nil) -> MLXArray {
        let normed = inputLayernorm(x)
        let r: MLXArray
        if isLinear {
            r = linearAttn!(normed, cache: cache as? GatedDeltaCache)
        } else {
            r = selfAttn!(normed, mask: mask, cache: cache as? KVCache)
        }
        let h = x + r
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - Model + LM head

/// Additive causal mask `[L, L]` (0 on/below diagonal, -inf above).
@inline(__always)
func qwen35CausalMask(_ L: Int, _ dtype: DType) -> MLXArray {
    let idx = MLXArray(Int32(0) ..< Int32(L))
    let upper = idx.expandedDimensions(axis: 0) .> idx.expandedDimensions(axis: 1)  // [L,L] true above diag
    return MLX.where(upper, MLXArray(-Float.greatestFiniteMagnitude), MLXArray(Float(0))).asType(dtype)
}

final class Qwen35Model: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [Qwen35DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: Qwen35Config) {
        _embedTokens = ModuleInfo(wrappedValue: Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize), key: "embed_tokens")
        layers = (0 ..< c.numHiddenLayers).map { Qwen35DecoderLayer(c, layerIdx: $0) }
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "norm")
    }

    /// Forward. `tokens`: `[B, L]`. With per-layer `caches` (one `KVCache` per
    /// full-attn layer, one `GatedDeltaCache` per linear layer) this carries
    /// state for incremental decode; without them it is a cacheless prefill.
    func callAsFunction(_ tokens: MLXArray, caches: [KVCacheProtocol]? = nil) -> MLXArray {
        var h = embedTokens(tokens)
        let L = h.dim(1)
        let mask: MLXArray? = L > 1 ? qwen35CausalMask(L, h.dtype) : nil
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: caches?[i])
        }
        return norm(h)
    }
}

public final class Qwen35ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Qwen35Model
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    public let config: Qwen35Config

    public init(_ c: Qwen35Config) {
        self.config = c
        _model = ModuleInfo(wrappedValue: Qwen35Model(c), key: "model")
        _lmHead = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.vocabSize, bias: false), key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCacheProtocol]? = nil) -> MLXArray {
        lmHead(model(tokens, caches: caches))
    }

    /// One per-layer cache: `KVCache` for full-attn layers, `GatedDeltaCache`
    /// for GatedDeltaNet layers. Pass to `callAsFunction` for incremental decode.
    public func makeCaches() -> [KVCacheProtocol] {
        (0 ..< config.numHiddenLayers).map { idx -> KVCacheProtocol in
            config.isLinearLayer(idx) ? GatedDeltaCache() : KVCache()
        }
    }
}
