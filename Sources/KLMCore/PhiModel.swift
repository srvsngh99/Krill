import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Phi Config

/// Configuration for Phi-3 / Phi-4 model family.
/// Key differences: fused qkv_proj, partial RoPE, LayerNorm (not RMSNorm in some variants).
public struct PhiConfig: ModelConfig, Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let quantization: QuantizationConfig?
    /// Fraction of each head's dimensions that receive RoPE. Phi-3-mini uses
    /// the full head (1.0); Phi-4-mini rotates only 0.75 of it and leaves the
    /// rest un-rotated. Applying full RoPE on a partial-rotary checkpoint
    /// corrupts attention and the model degenerates into garbage.
    public let partialRotaryFactor: Float
    /// When true the output projection shares the input embedding matrix and
    /// the checkpoint ships no `lm_head.weight` (Phi-4-mini). A separately
    /// allocated `lm_head` would then stay randomly initialized -> garbage.
    public let tieWordEmbeddings: Bool
    /// LongRoPE ("su") scaling, present on Phi-4-mini. nil for Phi-3-mini,
    /// which uses plain RoPE.
    public let ropeScaling: PhiRopeScaling?
    public let originalMaxPositionEmbeddings: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Number of head dimensions that actually receive RoPE.
    public var ropeDims: Int {
        max(2, Int((Float(headDim) * partialRotaryFactor).rounded(.down)))
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? (try c.decode(Int.self, forKey: .numAttentionHeads))
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 131_072
        partialRotaryFactor = try c.decodeIfPresent(
            Float.self, forKey: .partialRotaryFactor) ?? 1.0
        tieWordEmbeddings = try c.decodeIfPresent(
            Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeScaling = try c.decodeIfPresent(PhiRopeScaling.self, forKey: .ropeScaling)
        originalMaxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .originalMaxPositionEmbeddings) ?? maxPositionEmbeddings
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

/// Phi-4-mini's LongRoPE ("su"/"longrope") scaling parameters: per-frequency
/// rescale factors for short (<= original context) and long contexts.
public struct PhiRopeScaling: Codable, Sendable {
    public let shortFactor: [Float]
    public let longFactor: [Float]
    public let type: String

    enum CodingKeys: String, CodingKey {
        case shortFactor = "short_factor"
        case longFactor = "long_factor"
        case type
    }
}

// MARK: - Phi LongRoPE ("su"-scaled) rotary embedding

/// LongRoPE rotary embedding for Phi-4-mini. Standard NeoX (`rotate_half`)
/// RoPE over `dims` rotary features, with two refinements plain MLX `RoPE`
/// does not provide:
///   - per-frequency rescale factors chosen by context length (`short_factor`
///     within the original context, `long_factor` beyond it),
///   - a magnitude `scale` applied to cos/sin: the attention factor HF derives
///     from the context-extension ratio
///     (`sqrt(1 + ln(maxPos/origMax) / ln(origMax))`). Omitting it leaves the
///     rotary signal ~20% off and the model produces fluent-but-incoherent
///     text.
///
/// Plain class (not a `Module`): it holds only constant frequency tables, no
/// trainable parameters, so it stays out of the weight-loading tree.
final class PhiSuScaledRoPE {
    let dims: Int
    let freqBase: MLXArray      // base ** (2i/dims), [dims/2]
    let shortFactor: MLXArray   // [dims/2]
    let longFactor: MLXArray    // [dims/2]
    let originalMaxPos: Int
    let scale: Float

    init(dims: Int, base: Float, scaling: PhiRopeScaling,
         originalMaxPos: Int, maxPos: Int) {
        self.dims = dims
        let half = dims / 2
        self.freqBase = MLXArray((0 ..< half).map { powf(base, Float(2 * $0) / Float(dims)) })
        self.shortFactor = MLXArray(scaling.shortFactor)
        self.longFactor = MLXArray(scaling.longFactor)
        self.originalMaxPos = originalMaxPos
        let ratio = Float(maxPos) / Float(originalMaxPos)
        self.scale = ratio <= 1.0
            ? 1.0
            : (1.0 + Foundation.log(ratio) / Foundation.log(Float(originalMaxPos))).squareRoot()
    }

    /// `x` is `[B, H, L, dims]` (the rotary slice of each head). `offset` is
    /// the KV-cache position of the first new token.
    func callAsFunction(_ x: MLXArray, offset: Int) -> MLXArray {
        let L = x.dim(2)
        let factor = (offset + L) > originalMaxPos ? longFactor : shortFactor
        let invFreq = 1.0 / (factor * freqBase)                        // [dims/2]
        let positions = MLXArray((offset ..< (offset + L)).map { Float($0) })  // [L]
        let freqs = positions.reshaped(L, 1) * invFreq.reshaped(1, dims / 2)   // [L, dims/2]
        let emb = concatenated([freqs, freqs], axis: -1)               // [L, dims]
        let cos = MLX.cos(emb) * scale
        let sin = MLX.sin(emb) * scale
        let cosB = cos.reshaped(1, 1, L, dims)
        let sinB = sin.reshaped(1, 1, L, dims)
        return (x * cosB) + (Self.rotateHalf(x) * sinB)
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let d = x.dim(3)
        let x1 = x[0..., 0..., 0..., 0 ..< (d / 2)]
        let x2 = x[0..., 0..., 0..., (d / 2) ..< d]
        return concatenated([-x2, x1], axis: -1)
    }
}

// MARK: - Phi Attention (separate Q/K/V with bias)

class PhiAttention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let ropeDims: Int
    let scale: Float

    /// Phi fuses Q, K and V into one projection (`qkv_proj`), concatenated in
    /// that order; we split the output activation per head count below.
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    /// Plain RoPE (Phi-3-mini): MLX rotates the first `ropeDims` features of
    /// each head and passes the remainder through. nil when LongRoPE applies.
    let rope: RoPE?
    /// LongRoPE (Phi-4-mini). Applied to the rotary slice only. nil otherwise.
    let suRope: PhiSuScaledRoPE?

    init(_ config: PhiConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.ropeDims = config.ropeDims
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        let qSize = numHeads * headDim
        let kvSize = numKVHeads * headDim
        _qkvProj = ModuleInfo(
            wrappedValue: Linear(dim, qSize + 2 * kvSize, bias: false), key: "qkv_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")

        if let scaling = config.ropeScaling {
            self.suRope = PhiSuScaledRoPE(
                dims: config.ropeDims, base: config.ropeTheta, scaling: scaling,
                originalMaxPos: config.originalMaxPositionEmbeddings,
                maxPos: config.maxPositionEmbeddings)
            self.rope = nil
        } else {
            // MLX RoPE rotates the first `ropeDims` features and passes the
            // rest through, implementing partial rotary directly; for
            // Phi-3-mini ropeDims == headDim (full rotation).
            self.rope = RoPE(dimensions: config.ropeDims, traditional: false,
                             base: config.ropeTheta)
            self.suRope = nil
        }
    }

    /// Apply rotary embedding, handling the partial-rotary split when LongRoPE
    /// is active (plain `RoPE` does the split internally).
    private func applyRope(_ x: MLXArray, offset: Int) -> MLXArray {
        if let suRope {
            if ropeDims < headDim {
                let xRot = x[0..., 0..., 0..., 0 ..< ropeDims]
                let xPass = x[0..., 0..., 0..., ropeDims ..< headDim]
                return concatenated([suRope(xRot, offset: offset), xPass], axis: -1)
            }
            return suRope(x, offset: offset)
        }
        return rope!(x, offset: offset)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Split the fused qkv activation into Q | K | V along the last axis.
        let qSize = numHeads * headDim
        let kvSize = numKVHeads * headDim
        let qkv = qkvProj(x)
        let q = qkv[0..., 0..., 0 ..< qSize]
        let k = qkv[0..., 0..., qSize ..< (qSize + kvSize)]
        let v = qkv[0..., 0..., (qSize + kvSize) ..< (qSize + 2 * kvSize)]

        var queries = q.reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = k.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = v.reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.sequenceLength ?? 0
        queries = applyRope(queries, offset: offset)
        keys = applyRope(keys, offset: offset)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Phi MLP (SwiGLU, gate + up fused in some variants)

class PhiMLP: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let intermediateSize: Int

    init(_ config: PhiConfig) {
        self.intermediateSize = config.intermediateSize
        // Phi fuses gate and up into a single projection (2 * intermediate_size)
        _gateUpProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, 2 * config.intermediateSize, bias: false),
            key: "gate_up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: false),
            key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        // Split into gate and up halves along the last dimension
        let gate = gateUp[0..., 0..., ..<intermediateSize]
        let up = gateUp[0..., 0..., intermediateSize...]
        return downProj(silu(gate) * up)
    }
}

// MARK: - Phi Transformer Block

class PhiTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: PhiAttention
    @ModuleInfo(key: "mlp") var mlp: PhiMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: PhiConfig) {
        _selfAttn = ModuleInfo(wrappedValue: PhiAttention(config), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: PhiMLP(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - Phi Full Model

class PhiModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [PhiTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: PhiConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in PhiTransformerBlock(config) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        var x = embedTokens(tokens)
        let seqLen = x.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let mask = createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen)

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: caches?[i])
        }
        return norm(x)
    }
}

public class PhiForCausalLM: Module {
    @ModuleInfo(key: "model") var model: PhiModelInner
    /// Absent when `tie_word_embeddings` is set: the checkpoint then carries no
    /// `lm_head.weight` and logits are produced from the shared embedding
    /// matrix instead (see `callAsFunction`).
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: PhiConfig

    public init(_ config: PhiConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: PhiModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: config.tieWordEmbeddings
                ? nil
                : Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    /// `lastTokenOnly` slices hidden to the last position before
    /// the vocab projection. See `LlamaForCausalLM` for the
    /// rationale.
    public func callAsFunction(
        _ tokens: MLXArray,
        caches: [KVCache]? = nil,
        lastTokenOnly: Bool
    ) -> MLXArray {
        var hidden = model(tokens, caches: caches)
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        // Tied embeddings: project through the shared input embedding matrix.
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
