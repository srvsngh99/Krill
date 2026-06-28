import Foundation
import MLX
import MLXNN
import MLXFast
import KrillKernels
import KrillCache

// MARK: - DeepSeek (V2 / V3) Config

/// Configuration for the DeepSeek MoE family (`DeepseekV2ForCausalLM` /
/// `DeepseekV3ForCausalLM`, `model_type: deepseek_v2` / `deepseek_v3`).
///
/// This is the most involved MoE port: it adds Multi-head Latent Attention
/// (MLA, a low-rank Q/KV bottleneck with split rope/nope head dims), YaRN
/// RoPE, an always-on shared expert, the `first_k_dense_replace` dense-layer
/// prefix, a `routed_scaling_factor`, and fine-grained group gating. The two
/// generations differ only in the router scoring: V2 softmaxes the gate
/// logits (optionally `group_limited_greedy`); V3 (`noaux_tc`) uses sigmoid
/// scores plus a learned `e_score_correction_bias` for selection and
/// renormalizes the chosen weights.
///
/// DeepSeek-V2-Lite is the runnable reference checkpoint (the full V3 671B is
/// RAM-blocked); both gating paths are validated numerically against mlx-lm
/// on tiny quantized fixtures.
public struct DeepSeekRopeScaling: Codable, Sendable {
    public let factor: Float
    public let betaFast: Float
    public let betaSlow: Float
    public let mscale: Float
    public let mscaleAllDim: Float
    public let originalMaxPositionEmbeddings: Int

    enum CodingKeys: String, CodingKey {
        case factor
        case betaFast = "beta_fast"
        case betaSlow = "beta_slow"
        case mscale
        case mscaleAllDim = "mscale_all_dim"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        factor = try c.decodeIfPresent(Float.self, forKey: .factor) ?? 1.0
        betaFast = try c.decodeIfPresent(Float.self, forKey: .betaFast) ?? 32
        betaSlow = try c.decodeIfPresent(Float.self, forKey: .betaSlow) ?? 1
        mscale = try c.decodeIfPresent(Float.self, forKey: .mscale) ?? 1
        mscaleAllDim = try c.decodeIfPresent(Float.self, forKey: .mscaleAllDim) ?? 0
        originalMaxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .originalMaxPositionEmbeddings) ?? 4096
    }

    /// Identity (no-op) scaling, used when a checkpoint omits `rope_scaling`
    /// entirely (e.g. the DeepSeek-OCR language backbone). `factor <= 1` makes
    /// every YaRN ramp/mscale collapse to 1, so the rope is plain RoPE.
    public init() {
        factor = 1.0
        betaFast = 32
        betaSlow = 1
        mscale = 1
        mscaleAllDim = 0
        originalMaxPositionEmbeddings = 4096
    }
}

public struct DeepSeekConfig: ModelConfig, Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let quantization: QuantizationConfig?
    public let attentionBias: Bool

    // MLA
    public let qLoraRank: Int?
    public let kvLoraRank: Int
    public let qkRopeHeadDim: Int
    public let qkNopeHeadDim: Int
    public let vHeadDim: Int

    // MoE
    public let nRoutedExperts: Int
    public let nSharedExperts: Int
    public let numExpertsPerToken: Int
    public let routedScalingFactor: Float
    public let topkMethod: String
    public let scoringFunc: String
    public let normTopKProb: Bool
    public let nGroup: Int
    public let topkGroup: Int
    public let firstKDenseReplace: Int
    public let moeLayerFreq: Int

    public let ropeScaling: DeepSeekRopeScaling
    public let modelType: String

    /// Whether the decoder uses Multi-head Latent Attention. False for the
    /// DeepSeek-OCR language backbone (`use_mla: false`, `kv_lora_rank: null`,
    /// `qk_*_head_dim: 0`), which is plain multi-head attention with a DeepSeek
    /// MoE FFN. Defaults true so existing V2/V2-Lite checkpoints are unaffected.
    public let useMLA: Bool
    /// Sliding-window size if the config declares one (the OCR backbone carries
    /// `sliding_window_size: 128`); nil when absent. See note in
    /// `DeepSeekStandardAttention` on whether it is actually applied.
    public let slidingWindowSize: Int?

    public var headDim: Int { qkNopeHeadDim + qkRopeHeadDim }

    /// Head dim for the standard (non-MLA) attention path: `v_head_dim` when
    /// set, else `hidden / heads`. (The MLA `headDim` above is the split
    /// nope+rope dim, which is 0 for a non-MLA checkpoint.)
    public var standardHeadDim: Int {
        vHeadDim > 0 ? vHeadDim : hiddenSize / numAttentionHeads
    }

    /// DeepSeek-V3 (and V3.2) ship an *absorbed* MLA representation
    /// (`embed_q` / `unembed_out` per-head linears, a latent KV cache, and a
    /// split rope/nope attention) that is structurally distinct from the V2
    /// `kv_b_proj` form this native runtime implements. The native DeepSeek
    /// runtime currently serves DeepSeek-V2 / V2-Lite; V3 absorbed-MLA loading
    /// is a tracked follow-up (docs/BACKLOG.md), and the 671B V3 is RAM-blocked
    /// on a 24 GB host regardless.
    public var usesAbsorbedMLA: Bool {
        modelType == "deepseek_v3" || modelType == "deepseek_v32"
    }

    /// Whether layer `i` is a sparse MoE layer. The first
    /// `firstKDenseReplace` layers (and any off the `moeLayerFreq` cadence)
    /// use a plain dense MLP.
    public func isMoELayer(_ i: Int) -> Bool {
        i >= firstKDenseReplace && i % moeLayerFreq == 0
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case quantization
        case attentionBias = "attention_bias"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case routedScalingFactor = "routed_scaling_factor"
        case topkMethod = "topk_method"
        case scoringFunc = "scoring_func"
        case normTopKProb = "norm_topk_prob"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeLayerFreq = "moe_layer_freq"
        case ropeScaling = "rope_scaling"
        case modelType = "model_type"
        case useMLA = "use_mla"
        case slidingWindowSize = "sliding_window_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            ?? intermediateSize
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? (try c.decode(Int.self, forKey: .numAttentionHeads))
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 2048
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false

        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128

        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 64
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 0
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 6
        routedScalingFactor = try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
        topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "greedy"
        scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "softmax"
        normTopKProb = try c.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? false
        nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
        topkGroup = try c.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
        ropeScaling = try c.decodeIfPresent(DeepSeekRopeScaling.self, forKey: .ropeScaling)
            ?? DeepSeekRopeScaling()
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v2"
        useMLA = try c.decodeIfPresent(Bool.self, forKey: .useMLA) ?? true
        slidingWindowSize = try c.decodeIfPresent(Int.self, forKey: .slidingWindowSize)
    }
}

// MARK: - YaRN RoPE

private func yarnGetMscale(_ scale: Float, _ mscale: Float) -> Float {
    if scale <= 1 { return 1.0 }
    return 0.1 * mscale * logf(scale) + 1.0
}

/// YaRN rotary embedding mirroring mlx-lm's `DeepseekV2YarnRotaryEmbedding`:
/// precompute the interpolated/extrapolated frequency table and the
/// attention `mscale`, then apply `mx.fast.rope(..., traditional: true,
/// freqs:)`. Only the `qk_rope_head_dim` slice of Q/K is rotated.
final class DeepSeekYaRNRoPE {
    let freqs: MLXArray
    let mscale: Float

    init(dim: Int, base: Float, scaling: DeepSeekRopeScaling) {
        let factor = scaling.factor
        // mscale applied to the rotated activations before rope.
        self.mscale = yarnGetMscale(factor, scaling.mscale)
            / yarnGetMscale(factor, scaling.mscaleAllDim)

        // Correction range (in dims) for the linear interpolation ramp.
        func correctionDim(_ numRotations: Float) -> Float {
            (Float(dim) * logf(Float(scaling.originalMaxPositionEmbeddings)
                / (numRotations * 2 * Float.pi))) / (2 * logf(base))
        }
        let lowRaw = floorf(correctionDim(scaling.betaFast))
        let highRaw = ceilf(correctionDim(scaling.betaSlow))
        let low = max(lowRaw, 0)
        let high = min(highRaw, Float(dim - 1))

        let half = dim / 2
        var table = [Float](repeating: 0, count: half)
        let denom = (high == low) ? (high - low + 0.001) : (high - low)
        for k in 0 ..< half {
            let exponent = Float(2 * k) / Float(dim)
            let freqExtra = powf(base, exponent)
            let freqInter = factor * powf(base, exponent)
            // ramp mask over dim//2; freq_mask = 1 - clamp((k-low)/(high-low),0,1)
            let ramp = min(max((Float(k) - low) / denom, 0), 1)
            let freqMask = 1 - ramp
            table[k] = (freqInter * freqExtra)
                / (freqInter * freqMask + freqExtra * (1 - freqMask))
        }
        self.freqs = MLXArray(table)
    }

    /// Apply YaRN rope to `x` (`[..., L, dim]`) at a scalar position offset.
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let scaled = mscale != 1.0 ? (mscale * x) : x
        return MLXFast.RoPE(
            scaled, dimensions: x.dim(-1), traditional: true,
            base: nil, scale: 1.0, offset: offset, freqs: freqs)
    }

    /// Per-row variant for the batched ragged-decode path: each row carries
    /// its own next position (mirrors `applyRoPEPerRow`).
    func perRow(_ x: MLXArray, offsets: [Int]) -> MLXArray {
        var rows: [MLXArray] = []
        rows.reserveCapacity(offsets.count)
        for (r, off) in offsets.enumerated() {
            rows.append(callAsFunction(x[r ..< (r + 1)], offset: off))
        }
        return concatenated(rows, axis: 0)
    }
}

// MARK: - MLA Attention

/// DeepSeek Multi-head Latent Attention. Q is either a direct projection or
/// a low-rank `q_a_proj -> q_a_layernorm -> q_b_proj` bottleneck
/// (`q_lora_rank`). KV is always compressed: `kv_a_proj_with_mqa` produces a
/// `kv_lora_rank` latent plus a shared rope key `k_pe`; `kv_a_layernorm` +
/// `kv_b_proj` expand the latent into per-head nope-keys and values. Only the
/// `qk_rope_head_dim` slice carries YaRN RoPE; the nope slice does not. The
/// key head dim (`qk_nope_head_dim + qk_rope_head_dim`) differs from the value
/// head dim (`v_head_dim`).
class DeepSeekAttention: Module {
    let numHeads: Int
    let qLoraRank: Int?
    let kvLoraRank: Int
    let qkRopeHeadDim: Int
    let qkNopeHeadDim: Int
    let vHeadDim: Int
    let qHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: DeepSeekYaRNRoPE

    init(_ config: DeepSeekConfig) {
        self.numHeads = config.numAttentionHeads
        self.qLoraRank = config.qLoraRank
        self.kvLoraRank = config.kvLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        var s = 1.0 / Float(qHeadDim).squareRoot()
        // mscale_all_dim adjusts the attention softmax scale (mlx-lm).
        if config.ropeScaling.mscaleAllDim != 0 {
            let m = yarnGetMscale(config.ropeScaling.factor, config.ropeScaling.mscaleAllDim)
            s = s * m * m
        }
        self.scale = s

        let dim = config.hiddenSize
        let bias = config.attentionBias
        if let qLora = config.qLoraRank {
            _qAProj = ModuleInfo(
                wrappedValue: Linear(dim, qLora, bias: bias), key: "q_a_proj")
            _qANorm = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: qLora, eps: 1e-6), key: "q_a_layernorm")
            _qBProj = ModuleInfo(
                wrappedValue: Linear(qLora, numHeads * qHeadDim, bias: false), key: "q_b_proj")
        } else {
            _qProj = ModuleInfo(
                wrappedValue: Linear(dim, numHeads * qHeadDim, bias: false), key: "q_proj")
        }

        _kvAProj = ModuleInfo(
            wrappedValue: Linear(dim, kvLoraRank + qkRopeHeadDim, bias: bias),
            key: "kv_a_proj_with_mqa")
        _kvANorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: kvLoraRank, eps: 1e-6), key: "kv_a_layernorm")
        _kvBProj = ModuleInfo(
            wrappedValue: Linear(
                kvLoraRank, numHeads * (qkNopeHeadDim + vHeadDim), bias: false),
            key: "kv_b_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * vHeadDim, dim, bias: bias), key: "o_proj")

        self.rope = DeepSeekYaRNRoPE(
            dim: qkRopeHeadDim, base: config.ropeTheta, scaling: config.ropeScaling)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
                        rowOffsets: [Int]? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q: MLXArray
        if let qProj {
            q = qProj(x)
        } else {
            q = qBProj!(qANorm!(qAProj!(x)))
        }
        let qHeads = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qParts = split(qHeads, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qParts[0]
        var qPe = qParts[1]

        let compressed = kvAProj(x)
        let kvParts = split(compressed, indices: [kvLoraRank], axis: -1)
        let compressedKv = kvParts[0]
        var kPe = kvParts[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        let kv = kvBProj(kvANorm(compressedKv))
            .reshaped(B, L, numHeads, qkNopeHeadDim + vHeadDim)
            .transposed(0, 2, 1, 3)
        let kvSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = kvSplit[0]
        let values = kvSplit[1]

        // RoPE on the rope slice only.
        if let rowOffsets {
            qPe = rope.perRow(qPe, offsets: rowOffsets)
            kPe = rope.perRow(kPe, offsets: rowOffsets)
        } else {
            let offset = cache?.sequenceLength ?? 0
            qPe = rope(qPe, offset: offset)
            kPe = rope(kPe, offset: offset)
        }
        // k_pe is shared across heads (MQA-style): broadcast to all heads.
        let kPeFull = repeated(kPe, count: numHeads, axis: 1)

        let queries = concatenated([qNope, qPe], axis: -1)         // [B, H, L, qHeadDim]
        var keys = concatenated([kNope, kPeFull], axis: -1)        // [B, H, L, qHeadDim]
        var vals = values                                          // [B, H, L, vHeadDim]

        if let cache {
            (keys, vals) = cache.update(keys: keys, values: vals)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: vals, scale: scale, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Standard (non-MLA) attention

/// Plain multi-head attention for DeepSeek configs with `use_mla: false`
/// (the DeepSeek-OCR language backbone): direct `q/k/v/o_proj`, GQA via
/// `num_key_value_heads`, and standard half-split (NeoX / `rotate_half`) RoPE
/// over the full head dim. No latent compression and no split nope/rope slices.
///
/// Note on `sliding_window_size`: the OCR backbone's config carries a 128-token
/// window, but the reference `modeling_deepseekv2` attention is full-causal
/// (the field is vestigial there). This implementation is full-causal; the
/// Phase-2 logit-parity gate is what confirms that choice (and the RoPE
/// convention) against the HF reference before we trust it.
class DeepSeekStandardAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let ropeTheta: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: DeepSeekConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.standardHeadDim
        self.scale = 1.0 / Float(headDim).squareRoot()
        self.ropeTheta = config.ropeTheta

        let dim = config.hiddenSize
        let bias = config.attentionBias
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: bias), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: bias), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: bias), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: bias), key: "o_proj")
    }

    private func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: headDim, traditional: false,
            base: ropeTheta, scale: 1.0, offset: offset)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
                        rowOffsets: [Int]? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let rowOffsets {
            // Batched ragged decode: each row carries its own next position.
            var qRows: [MLXArray] = []
            var kRows: [MLXArray] = []
            qRows.reserveCapacity(rowOffsets.count)
            kRows.reserveCapacity(rowOffsets.count)
            for (r, off) in rowOffsets.enumerated() {
                qRows.append(rope(q[r ..< (r + 1)], offset: off))
                kRows.append(rope(k[r ..< (r + 1)], offset: off))
            }
            q = concatenated(qRows, axis: 0)
            k = concatenated(kRows, axis: 0)
        } else {
            let offset = cache?.sequenceLength ?? 0
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        // GQA: expand the compact KV heads to the query head count (after the
        // cache update, which stores the compact KV).
        if numKVHeads < numHeads {
            let rep = numHeads / numKVHeads
            k = repeated(k, count: rep, axis: 1)
            v = repeated(v, count: rep, axis: 1)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - V3 absorbed-MLA per-head quantized linear

/// A stacked per-head quantized linear, mirroring mlx-lm's
/// `mla.QuantizedMultiLinear`. Holds a `[numHeads, outputDims, inputDims]`
/// weight quantized along the last (`inputDims`) axis, and contracts a
/// per-head matmul in a single `quantizedMatmul` -- the matrix the absorbed-MLA
/// `embed_q` / `unembed_out` apply per head.
///
/// `transpose` selects which axis the matmul contracts (matching mlx's
/// `quantized_matmul` semantics on one packed tensor):
///   - `true`  (default): `x @ W.T`, contracts the packed `inputDims` axis,
///     so `x: [..., H, ..., inputDims] -> [..., H, ..., outputDims]`.
///   - `false`: `x @ W`, contracts `outputDims`, so
///     `x: [..., H, ..., outputDims] -> [..., H, ..., inputDims]`.
/// The same packed tensor serves both, which is exactly how V3 uses `embed_q`
/// (transpose at decode to project the query into the latent space; no-transpose
/// at prefill to expand the latent into per-head nope-keys).
///
/// Born quantized (like the SwitchGLU experts): the loader's quantize pass only
/// touches `nn.Linear`-shaped modules, so this pre-allocates the packed
/// `weight` / `scales` / `biases` at the checkpoint's shapes and binds them
/// directly.
class DeepSeekQuantizedMultiLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray

    let groupSize: Int
    let bits: Int

    init(inputDims: Int, outputDims: Int, numHeads: Int, groupSize: Int, bits: Int) {
        self.groupSize = groupSize
        self.bits = bits
        let packedIn = inputDims * bits / 32
        let groupsIn = inputDims / groupSize
        _weight = ParameterInfo(
            wrappedValue: MLXArray.zeros([numHeads, outputDims, packedIn], dtype: .uint32),
            key: "weight")
        _scales = ParameterInfo(
            wrappedValue: MLXArray.zeros([numHeads, outputDims, groupsIn], dtype: .bfloat16),
            key: "scales")
        _biases = ParameterInfo(
            wrappedValue: MLXArray.zeros([numHeads, outputDims, groupsIn], dtype: .bfloat16),
            key: "biases")
    }

    func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        return quantizedMatmul(
            x, weight, scales: scales, biases: biases,
            transpose: transpose, groupSize: groupSize, bits: bits, mode: .affine)
    }
}

// MARK: - V3 absorbed Multi-head Latent Attention

/// DeepSeek-V3 attention. Same low-rank Q/KV projections and YaRN RoPE as V2,
/// but the `kv_b_proj` expansion is *absorbed* into two per-head linears
/// (`embed_q` / `unembed_out`) and the KV cache stores the compressed latent
/// (`kv_latent`) plus the shared rope key (`k_pe`) instead of full keys/values.
/// Mirrors `mlx_lm.models.deepseek_v3.DeepseekV3Attention`.
///
/// The rope-slice scores are computed separately (`pe_scores`) and folded into
/// the attention as an additive bias, so the nope-slice attention can run in
/// the compact latent space:
///   - **decode (L == 1):** project the query nope-slice into the latent with
///     `embed_q`, attend directly against the cached `kv_latent` (k == v ==
///     latent), then map the latent output back to `v_head_dim` with
///     `unembed_out`. Keeps the per-step KV footprint at `kv_lora_rank + rope`.
///   - **prefill (L > 1):** expand the latent into per-head nope-keys
///     (`embed_q`, no transpose) and values (`unembed_out`) and run a standard
///     SDPA.
/// Both paths are the same identity, so they produce equal logits.
class DeepSeekV3Attention: Module {
    let numHeads: Int
    let kvLoraRank: Int
    let qkRopeHeadDim: Int
    let qkNopeHeadDim: Int
    let vHeadDim: Int
    let qHeadDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: DeepSeekQuantizedMultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOut: DeepSeekQuantizedMultiLinear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: DeepSeekYaRNRoPE

    init(_ config: DeepSeekConfig) {
        self.numHeads = config.numAttentionHeads
        self.kvLoraRank = config.kvLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.vHeadDim = config.vHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim

        var s = 1.0 / Float(qHeadDim).squareRoot()
        if config.ropeScaling.mscaleAllDim != 0 {
            let m = yarnGetMscale(config.ropeScaling.factor, config.ropeScaling.mscaleAllDim)
            s = s * m * m
        }
        self.scale = s

        let dim = config.hiddenSize
        let bias = config.attentionBias
        if let qLora = config.qLoraRank {
            _qAProj = ModuleInfo(
                wrappedValue: Linear(dim, qLora, bias: bias), key: "q_a_proj")
            _qANorm = ModuleInfo(
                wrappedValue: RMSNorm(dimensions: qLora, eps: 1e-6), key: "q_a_layernorm")
            _qBProj = ModuleInfo(
                wrappedValue: Linear(qLora, numHeads * qHeadDim, bias: false), key: "q_b_proj")
        } else {
            _qProj = ModuleInfo(
                wrappedValue: Linear(dim, numHeads * qHeadDim, bias: false), key: "q_proj")
        }

        _kvAProj = ModuleInfo(
            wrappedValue: Linear(dim, kvLoraRank + qkRopeHeadDim, bias: bias),
            key: "kv_a_proj_with_mqa")
        _kvANorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: kvLoraRank, eps: 1e-6), key: "kv_a_layernorm")
        // embed_q: per-head [numHeads, kvLoraRank, qkNopeHeadDim]; unembed_out:
        // [numHeads, vHeadDim, kvLoraRank]. quantization (group_size / bits)
        // comes from the checkpoint, matching how mlx-lm quantizes these.
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4
        _embedQ = ModuleInfo(
            wrappedValue: DeepSeekQuantizedMultiLinear(
                inputDims: qkNopeHeadDim, outputDims: kvLoraRank,
                numHeads: numHeads, groupSize: groupSize, bits: bits),
            key: "embed_q")
        _unembedOut = ModuleInfo(
            wrappedValue: DeepSeekQuantizedMultiLinear(
                inputDims: kvLoraRank, outputDims: vHeadDim,
                numHeads: numHeads, groupSize: groupSize, bits: bits),
            key: "unembed_out")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * vHeadDim, dim, bias: bias), key: "o_proj")

        self.rope = DeepSeekYaRNRoPE(
            dim: qkRopeHeadDim, base: config.ropeTheta, scaling: config.ropeScaling)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
                        rowOffsets: [Int]? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let q: MLXArray
        if let qProj {
            q = qProj(x)
        } else {
            q = qBProj!(qANorm!(qAProj!(x)))
        }
        let qHeads = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let qParts = split(qHeads, indices: [qkNopeHeadDim], axis: -1)
        let qNope = qParts[0]                                  // [B, H, L, nope]
        var qPe = qParts[1]                                    // [B, H, L, rope]

        let compressed = kvAProj(x)
        let kvParts = split(compressed, indices: [kvLoraRank], axis: -1)
        let compressedKv = kvParts[0]                          // [B, L, kvLoraRank]
        var kPe = kvParts[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3) // [B,1,L,rope]
        // [B, 1, L, kvLoraRank] -- one "head" so the latent cache stays compact.
        var kvLatent = kvANorm(compressedKv).reshaped(B, 1, L, kvLoraRank)

        // RoPE on the rope slice only (the nope slice is rope-free).
        if let rowOffsets {
            qPe = rope.perRow(qPe, offsets: rowOffsets)
            kPe = rope.perRow(kPe, offsets: rowOffsets)
        } else {
            let offset = cache?.sequenceLength ?? 0
            qPe = rope(qPe, offset: offset)
            kPe = rope(kPe, offset: offset)
        }

        // Cache the compressed latent + shared rope key (not full K/V): store
        // kv_latent as the cache's "keys" and k_pe as its "values".
        if let cache {
            (kvLatent, kPe) = cache.update(keys: kvLatent, values: kPe)
        }

        // Rope-slice scores, folded into the additive attention bias. k_pe has
        // one head and broadcasts over the query heads (MQA-style). The causal
        // mask (additive float; nil at decode) is added on top.
        var peBias = matmul(qPe * scale, kPe.swappedAxes(-1, -2))   // [B, H, L, Lk]
        if let mask {
            peBias = peBias + mask
        }

        let qAttn: MLXArray
        let keys: MLXArray
        let vals: MLXArray
        if L == 1 {
            // Decode: project the query nope-slice into the latent and attend
            // directly against the cached latent (k == v == latent).
            qAttn = embedQ(qNope)                               // [B, H, 1, kvLoraRank]
            keys = kvLatent                                     // [B, 1, Lk, kvLoraRank]
            vals = kvLatent
        } else {
            // Prefill: expand the latent into per-head nope-keys and values.
            qAttn = qNope                                       // [B, H, L, nope]
            keys = embedQ(kvLatent, transpose: false)           // [B, H, Lk, nope]
            vals = unembedOut(kvLatent)                         // [B, H, Lk, vHeadDim]
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qAttn, keys: keys, values: vals, scale: scale, mask: peBias)
        if L == 1 {
            // Map the latent-space attention output back to v_head_dim.
            output = unembedOut(output)                         // [B, H, 1, vHeadDim]
        }

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Dense MLP (SwiGLU)

class DeepSeekMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, intermediateSize, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, intermediateSize, bias: false), key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(intermediateSize, hiddenSize, bias: false), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(KrillKernels.fusedSwiGLU(gate: gateProj(x), up: upProj(x)))
    }
}

// MARK: - MoE experts: shared `MoESwitchGLU` (see MoESwitchGLU.swift)
//
// The stacked `gatherQuantizedMM` expert dispatch
// (`MoEQuantizedSwitchedLinear` + `MoESwitchGLU`, with the prefill
// `(token, expert)` sort path in `MoESortPath.swift`) is shared across all
// native MoE families. This family uses `MoEActivation.swiglu`.

// MARK: - MoE gate (V2 softmax/group_limited_greedy + V3 noaux_tc sigmoid)

/// DeepSeek router. The gate is a raw `[n_routed_experts, hidden]` weight
/// (NOT an nn.Linear, so it stays unquantized through the loader's quantize
/// pass, matching mlx-lm's `MoEGate.weight`). V3 (`noaux_tc`) adds a learned
/// `e_score_correction_bias` used only for expert SELECTION; the routed
/// weights come from the unbiased sigmoid scores.
class DeepSeekMoEGate: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var eScoreBias: MLXArray?

    let topK: Int
    let numExperts: Int
    let routedScaling: Float
    let useSigmoid: Bool
    let nGroup: Int
    let topkGroup: Int
    let groupTop2Sum: Bool   // V3 noaux_tc uses top-2 sum; V2 group uses max
    let normTopK: Bool

    init(_ config: DeepSeekConfig) {
        self.topK = config.numExpertsPerToken
        self.numExperts = config.nRoutedExperts
        self.routedScaling = config.routedScalingFactor
        self.useSigmoid = config.scoringFunc == "sigmoid" || config.topkMethod == "noaux_tc"
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.groupTop2Sum = config.topkMethod == "noaux_tc"
        self.normTopK = config.normTopKProb

        _weight = ParameterInfo(
            wrappedValue: MLXArray.zeros([config.nRoutedExperts, config.hiddenSize]),
            key: "weight")
        if useSigmoid {
            _eScoreBias = ParameterInfo(
                wrappedValue: MLXArray.zeros([config.nRoutedExperts]),
                key: "e_score_correction_bias")
        }
    }

    /// Returns `(inds [N, topK] Int32, weights [N, topK] float32)`.
    func callAsFunction(_ flat: MLXArray) -> (MLXArray, MLXArray) {
        let N = flat.dim(0)
        let gates = matmul(flat, weight.transposed(1, 0))     // [N, E]

        var scores: MLXArray
        let orig: MLXArray
        if useSigmoid {
            let s = sigmoid(gates.asType(.float32))
            orig = s
            scores = s + (eScoreBias ?? MLXArray.zeros([numExperts])).asType(.float32)
        } else {
            scores = softmax(gates.asType(.float32), axis: -1)
            orig = scores
        }

        if nGroup > 1 {
            let perGroup = numExperts / nGroup
            var s = scores.reshaped(N, nGroup, perGroup)
            let groupScores: MLXArray
            if groupTop2Sum {
                let asc = sorted(s, axis: -1)                  // ascending
                let top2 = asc[0..., 0..., (perGroup - 2) ..< perGroup]
                groupScores = top2.sum(axis: -1)               // [N, nGroup]
            } else {
                groupScores = s.max(axis: -1)                  // [N, nGroup]
            }
            // Keep the top `topkGroup` groups; zero the rest.
            let negG = MLXArray(0) - groupScores
            let gSort = argSort(negG, axis: -1)
            let gRank = argSort(gSort, axis: -1)
            let keep = (gRank .< MLXArray(Int32(topkGroup))).reshaped(N, nGroup, 1)
            s = MLX.where(keep, s, MLXArray(Float(0)).asType(s.dtype))
            scores = s.reshaped(N, numExperts)
        }

        let neg = MLXArray(0) - scores
        let sortedByScore = argSort(neg, axis: -1)
        let inds = sortedByScore[0..., 0 ..< topK]             // [N, topK]
        var w = takeAlong(orig, inds, axis: 1)                 // [N, topK] (unbiased)
        if topK > 1 && normTopK {
            w = w / w.sum(axis: -1, keepDims: true)
        }
        w = w * routedScaling
        return (inds.asType(.int32), w)
    }
}

// MARK: - MoE block (routed experts + always-on shared experts)

class DeepSeekMoE: Module {
    @ModuleInfo(key: "gate") var gate: DeepSeekMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: MoESwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepSeekMLP?

    let numExperts: Int
    let topK: Int
    private let accumulator: ExpertCountAccumulator

    var cumulativeExpertCounts: [Int] {
        eval(accumulator.counts)
        return accumulator.counts.asArray(Int32.self).map(Int.init)
    }
    var pendingExpertCounts: MLXArray { accumulator.counts }

    init(_ config: DeepSeekConfig) {
        self.numExperts = config.nRoutedExperts
        self.topK = config.numExpertsPerToken
        self.accumulator = ExpertCountAccumulator(numExperts: config.nRoutedExperts)

        _gate = ModuleInfo(wrappedValue: DeepSeekMoEGate(config), key: "gate")
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4
        _switchMLP = ModuleInfo(
            wrappedValue: MoESwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts,
                groupSize: groupSize, bits: bits,
                activation: .swiglu),
            key: "switch_mlp")
        if config.nSharedExperts > 0 {
            _sharedExperts = ModuleInfo(
                wrappedValue: DeepSeekMLP(
                    hiddenSize: config.hiddenSize,
                    intermediateSize: config.moeIntermediateSize * config.nSharedExperts),
                key: "shared_experts")
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let N = B * L
        let flat = x.reshaped(N, H)

        let (inds, weights) = gate(flat)  // [N, topK], [N, topK]

        let expertRange = MLXArray(Int32(0) ..< Int32(numExperts))
        let oneHot = inds.reshaped(N * topK, 1) .== expertRange
        accumulator.counts = accumulator.counts + oneHot.asType(.int32).sum(axis: 0)

        let expertOut = switchMLP(flat, indices: inds)        // [N, topK, H]
        var y = (expertOut * weights.asType(expertOut.dtype).reshaped(N, topK, 1))
            .sum(axis: 1)                                      // [N, H]
        if let sharedExperts {
            y = y + sharedExperts(flat)                        // shared added directly (no gate)
        }
        return y.reshaped(B, L, H)
    }

    func resetUtilizationStats() {
        accumulator.counts = MLXArray.zeros([numExperts], dtype: .int32)
    }
}

// MARK: - Decoder layer (dense prefix, then MoE)

class DeepSeekDecoderLayer: Module {
    // V2 uses `DeepSeekAttention` (kv_b_proj), V3 the absorbed
    // `DeepSeekV3Attention` (embed_q / unembed_out + latent cache). Declared as
    // `Module` so one layer type serves both; the concrete instance's own
    // @ModuleInfo keys bind the checkpoint, and `callAsFunction` casts.
    @ModuleInfo(key: "self_attn") var selfAttn: Module
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    let isMoE: Bool

    init(_ config: DeepSeekConfig, layerIndex: Int) {
        let attn: Module
        if config.usesAbsorbedMLA {
            attn = DeepSeekV3Attention(config)          // V3 absorbed MLA
        } else if config.useMLA {
            attn = DeepSeekAttention(config)            // V2 MLA (kv_b_proj)
        } else {
            attn = DeepSeekStandardAttention(config)    // plain MHA (DeepSeek-OCR)
        }
        _selfAttn = ModuleInfo(wrappedValue: attn, key: "self_attn")
        self.isMoE = config.isMoELayer(layerIndex)
        let mlpModule: Module = isMoE
            ? DeepSeekMoE(config)
            : DeepSeekMLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        _mlp = ModuleInfo(wrappedValue: mlpModule, key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
                        rowOffsets: [Int]? = nil) -> MLXArray {
        let normed = inputLayernorm(x)
        let attnOut: MLXArray
        if let v3 = selfAttn as? DeepSeekV3Attention {
            attnOut = v3(normed, mask: mask, cache: cache, rowOffsets: rowOffsets)
        } else if let v2 = selfAttn as? DeepSeekAttention {
            attnOut = v2(normed, mask: mask, cache: cache, rowOffsets: rowOffsets)
        } else if let std = selfAttn as? DeepSeekStandardAttention {
            attnOut = std(normed, mask: mask, cache: cache, rowOffsets: rowOffsets)
        } else {
            fatalError("DeepSeekDecoderLayer.selfAttn must be DeepSeekAttention, "
                + "DeepSeekV3Attention, or DeepSeekStandardAttention")
        }
        let h = x + attnOut
        let postAttn = postAttentionLayernorm(h)
        let mlpOut: MLXArray
        if let moe = mlp as? DeepSeekMoE {
            mlpOut = moe(postAttn)
        } else if let dense = mlp as? DeepSeekMLP {
            mlpOut = dense(postAttn)
        } else {
            fatalError("DeepSeekDecoderLayer.mlp must be DeepSeekMoE or DeepSeekMLP")
        }
        return h + mlpOut
    }
}

// MARK: - Inner model

class DeepSeekModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [DeepSeekDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: DeepSeekConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map {
                DeepSeekDecoderLayer(config, layerIndex: $0)
            },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil,
                        precomputedMask: MLXArray? = nil, rowOffsets: [Int]? = nil) -> MLXArray {
        var x = embedTokens(tokens)
        let mask: MLXArray?
        if let precomputedMask {
            mask = precomputedMask
        } else {
            let seqLen = x.dim(1)
            let cacheLen = caches?.first?.sequenceLength ?? 0
            mask = createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen, dtype: x.dtype)
        }
        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: caches?[i], rowOffsets: rowOffsets)
        }
        return norm(x)
    }

    var sparseExpertCounts: [MLXArray] {
        layers.compactMap { ($0.mlp as? DeepSeekMoE)?.pendingExpertCounts }
    }
}

// MARK: - ForCausalLM

public class DeepSeekForCausalLM: Module {
    @ModuleInfo(key: "model") var model: DeepSeekModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: DeepSeekConfig

    public init(_ config: DeepSeekConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: DeepSeekModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil, lastTokenOnly: Bool
    ) -> MLXArray {
        var hidden = model(tokens, caches: caches)
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        return lmHead(hidden)
    }

    public func batchedDecode(
        _ tokens: MLXArray, caches: [KVCache], mask: MLXArray, rowOffsets: [Int]
    ) -> MLXArray {
        let hidden = model(tokens, caches: caches, precomputedMask: mask, rowOffsets: rowOffsets)
        let logits = lmHead(hidden)
        MLX.eval([logits] + model.sparseExpertCounts)
        return logits
    }

    public func moeUtilization() -> MoEUtilization {
        var sparseLayers = 0
        var totalSlots = 0
        var activeSlots = 0
        var totalAssignments = 0
        var maxLoad = 0
        for block in model.layers {
            guard let moe = block.mlp as? DeepSeekMoE else { continue }
            sparseLayers += 1
            for count in moe.cumulativeExpertCounts {
                totalSlots += 1
                if count > 0 { activeSlots += 1 }
                totalAssignments += count
                if count > maxLoad { maxLoad = count }
            }
        }
        return MoEUtilization(
            sparseLayers: sparseLayers,
            expertsPerLayer: config.nRoutedExperts,
            totalExpertSlots: totalSlots,
            activeExpertSlots: activeSlots,
            totalAssignments: totalAssignments,
            maxExpertLoad: maxLoad)
    }

    public func resetMoEUtilizationStats() {
        for block in model.layers {
            (block.mlp as? DeepSeekMoE)?.resetUtilizationStats()
        }
    }
}
