import Foundation
import MLX
import MLXNN
import MLXFast
import KrillCache

// Native Swift+MLX runtime for Meta's Muse Glimmer 30B
// (`MuseGlimmerForConditionalGeneration`, model_type "muse_glimmer") — a dense
// 52-layer decoder with an output-GATED attention, a 3:1 sliding/full layer
// mix where the FULL layers are NoPE, and a ~1.8B ViT-G/14 perception encoder
// (`MuseGlimmerVision.swift`).
//
// Spec + provenance: `.claude/skills/native-port/families/muse-glimmer.md`,
// derived from `transformers/models/muse_glimmer/modeling_muse_glimmer.py`.
//
// Six things here are NOT the dense template — see the spec for the full list,
// but the ones that silently produce garbage if dropped:
//
//   1. `attn_out * sigmoid(gate_proj(x))` BEFORE `o_proj`, where `x` is the
//      layer's normed input (not the attention output).
//   2. `layer_rope_theta[i] == 0` on every `full_attention` layer ⇒ NoPE. The
//      SLIDING layers are the ones carrying RoPE.
//   3. Two RMSNorm formulas: the four per-layer norms are `(1 + w)`-scaled
//      (Gemma-style, weights centred on zero), the final `norm` is plain `w`.
//   4. Three weightless norms with no checkpoint tensors: `qk_norm`,
//      `embed_norm`, and (vision side) `perception_emb_norm`.
//   5. `qk_scale_factor` (3.87) multiplies `q` IN ADDITION to the usual
//      `head_dim**-0.5` softmax scale; `k` is normed but not scaled.
//   6. Split eps: pre-norms use `rms_norm_eps` (1e-5), post-norms use
//      `post_norm_eps` (1e-8).
//
// PARITY STATUS: logit-parity green against the transformers reference on a
// synthetic checkpoint (`tools/verify_muse_glimmer_parity.py` +
// `MuseGlimmerParityTests`) — prefill, cached decode, vision tower and image
// features all at argmax + cosine > 0.9999 + max-abs < 2e-3.
//
// NOT verified on the real 30B weights: the smallest published MLX build is
// 19.4 GB and does not fit a 24 GB box, so there is no real-checkpoint smoke
// and no throughput number for this family.

// MARK: - Weightless RMS normalization

/// `MuseGlimmerRMSNorm(with_scale=False)` — the weightless norm used for
/// `qk_norm`, `embed_norm`, and `perception_emb_norm`. There are NO tensors for
/// these in the checkpoint, so this is a free function rather than a `Module`
/// (a parameterless `Module` would still claim a key path on `update`).
///
/// Reference computes in fp32 and casts back: `x * (mean(x²) + eps)^-0.5`.
@inline(__always)
func museGlimmerRMSNormalize(_ x: MLXArray, eps: Float) -> MLXArray {
    let dtype = x.dtype
    let xf = x.asType(.float32)
    let meanSq = MLX.mean(xf * xf, axis: -1, keepDims: true) + MLXArray(eps)
    return (xf * MLX.rsqrt(meanSq)).asType(dtype)
}

/// `MuseGlimmerTextCenteredRMSNorm` — `norm(x) * (1 + w)`, weights stored
/// centred on zero (Gemma-style). Used for all four per-layer norms.
///
/// Deliberately NOT `MLXNN.RMSNorm`: that computes `norm(x) * w`, which is the
/// formula the FINAL `language_model.norm` uses. Mixing the two up shifts every
/// residual by a factor of roughly `w` and is not visible as a crash.
final class MuseGlimmerCenteredRMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.zeros([dimensions])
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xf = x.asType(.float32)
        let meanSq = MLX.mean(xf * xf, axis: -1, keepDims: true) + MLXArray(eps)
        let normed = xf * MLX.rsqrt(meanSq)
        return (normed * (MLXArray(Float(1.0)) + weight.asType(.float32))).asType(dtype)
    }
}

// MARK: - Text config

public struct MuseGlimmerTextConfig: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let postNormEps: Float
    public let slidingWindow: Int
    /// Per-layer `"sliding_attention"` / `"full_attention"`. Authoritative —
    /// never infer the 3:1 pattern by modulo.
    public let layerTypes: [String]
    /// Per-layer RoPE base; `0` means the layer is NoPE.
    public let layerRopeTheta: [Float]
    public let ropeTheta: Float
    public let qkScaleFactor: Float
    public let outputMultiplier: Float
    public let finalLogitSoftcapping: Float
    public let attentionBias: Bool
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    public let hiddenActivation: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case postNormEps = "post_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case layerRopeTheta = "layer_rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeTheta = "rope_theta"
        case qkScaleFactor = "qk_scale_factor"
        case outputMultiplier = "output_multiplier"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenActivation = "hidden_activation"
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Float?
        let ropeType: String?
        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / numAttentionHeads)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        // Falls back to `rms_norm_eps`, matching the reference default when a
        // checkpoint omits the post-norm eps entirely.
        postNormEps = try c.decodeIfPresent(Float.self, forKey: .postNormEps) ?? rmsNormEps
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 2048
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: numHiddenLayers)
        // `rope_theta` lives under `rope_parameters` in the shipped config; the
        // flat key is accepted for hand-written and synthetic configs.
        let rp = try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
        let flatTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
        ropeTheta = rp?.ropeTheta ?? flatTheta ?? 500_000.0
        layerRopeTheta = try c.decodeIfPresent([Float].self, forKey: .layerRopeTheta)
            ?? Array(repeating: ropeTheta, count: numHiddenLayers)
        qkScaleFactor = try c.decodeIfPresent(Float.self, forKey: .qkScaleFactor) ?? 1.0
        outputMultiplier = try c.decodeIfPresent(Float.self, forKey: .outputMultiplier) ?? 1.0
        finalLogitSoftcapping = try c.decodeIfPresent(
            Float.self, forKey: .finalLogitSoftcapping) ?? 0.0
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        hiddenActivation = try c.decodeIfPresent(
            String.self, forKey: .hiddenActivation) ?? "silu"
    }

    /// Convenience initializer for tests and the synthetic parity checkpoint.
    public init(
        hiddenSize: Int, intermediateSize: Int, numHiddenLayers: Int,
        numAttentionHeads: Int, numKeyValueHeads: Int, headDim: Int, vocabSize: Int,
        layerTypes: [String], layerRopeTheta: [Float],
        rmsNormEps: Float = 1e-5, postNormEps: Float = 1e-8,
        slidingWindow: Int = 2048, ropeTheta: Float = 500_000.0,
        qkScaleFactor: Float = 3.87, outputMultiplier: Float = 0.19611613513818404,
        finalLogitSoftcapping: Float = 20.0, tieWordEmbeddings: Bool = false,
        maxPositionEmbeddings: Int = 131_072
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.vocabSize = vocabSize
        self.layerTypes = layerTypes
        self.layerRopeTheta = layerRopeTheta
        self.rmsNormEps = rmsNormEps
        self.postNormEps = postNormEps
        self.slidingWindow = slidingWindow
        self.ropeTheta = ropeTheta
        self.qkScaleFactor = qkScaleFactor
        self.outputMultiplier = outputMultiplier
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.attentionBias = false
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.hiddenActivation = "silu"
    }

    /// Whether layer `i` attends the whole causal context (vs the 2048 window).
    public func isFullAttention(layerIdx: Int) -> Bool {
        guard layerIdx < layerTypes.count else { return true }
        return layerTypes[layerIdx] != "sliding_attention"
    }

    /// Per-layer RoPE base, or nil when the layer is NoPE (`theta == 0`).
    public func ropeTheta(layerIdx: Int) -> Float? {
        guard layerIdx < layerRopeTheta.count else { return ropeTheta }
        let t = layerRopeTheta[layerIdx]
        return t == 0 ? nil : t
    }

    /// Per-layer cache kinds: sliding layers get a rotating (windowed) cache.
    public var cacheSpec: [KVCacheKind] {
        (0 ..< numHiddenLayers).map {
            isFullAttention(layerIdx: $0) ? .standard : .rotating(window: slidingWindow)
        }
    }

    /// Reject a checkpoint whose shape this runtime would silently mis-serve.
    public func validate() throws {
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw ModelLoadError.invalidConfig(
                "num_attention_heads (\(numAttentionHeads)) must be a multiple of "
                + "num_key_value_heads (\(numKeyValueHeads))")
        }
        guard layerTypes.count == numHiddenLayers else {
            throw ModelLoadError.invalidConfig(
                "layer_types has \(layerTypes.count) entries but num_hidden_layers "
                + "is \(numHiddenLayers); the sliding/full mix cannot be inferred.")
        }
        guard layerRopeTheta.count == numHiddenLayers else {
            throw ModelLoadError.invalidConfig(
                "layer_rope_theta has \(layerRopeTheta.count) entries but "
                + "num_hidden_layers is \(numHiddenLayers); which layers are NoPE "
                + "cannot be inferred.")
        }
        let unknown = Set(layerTypes).subtracting(["sliding_attention", "full_attention"])
        guard unknown.isEmpty else {
            throw ModelLoadError.unsupportedArchitecture(
                "Unknown layer_types \(unknown.sorted()) in muse_glimmer text config; "
                + "this runtime implements sliding_attention and full_attention.")
        }
    }
}

// MARK: - MLP

final class MuseGlimmerTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: MuseGlimmerTextConfig) {
        _gateProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: false), key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(c.intermediateSize, c.hiddenSize, bias: false), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Gated attention

/// GQA with a weightless per-head Q/K RMSNorm, a `qk_scale_factor` on Q only,
/// OPTIONAL RoPE (NoPE on `full_attention` layers), and a sigmoid output gate
/// applied before `o_proj`.
final class MuseGlimmerTextAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    /// The attention OUTPUT gate — reads the layer's normed input, not the
    /// attention output. Distinct from `mlp.gate_proj` despite the shared name.
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    /// The ordinary softmax scale. `qkScaleFactor` is applied ON TOP of this.
    let scale: Float
    let qkScaleFactor: Float
    let normEps: Float
    /// nil ⇒ this layer is NoPE.
    let rope: RoPE?

    init(_ c: MuseGlimmerTextConfig, layerIdx: Int) {
        numHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = 1.0 / Float(c.headDim).squareRoot()
        qkScaleFactor = c.qkScaleFactor
        normEps = c.rmsNormEps

        let qOut = c.numAttentionHeads * c.headDim
        let kvOut = c.numKeyValueHeads * c.headDim
        _qProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, qOut, bias: c.attentionBias), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, kvOut, bias: c.attentionBias), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, kvOut, bias: c.attentionBias), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(qOut, c.hiddenSize, bias: c.attentionBias), key: "o_proj")
        _gateProj = ModuleInfo(
            wrappedValue: Linear(c.hiddenSize, qOut, bias: false), key: "gate_proj")

        if let theta = c.ropeTheta(layerIdx: layerIdx) {
            rope = RoPE(dimensions: c.headDim, traditional: false, base: theta)
        } else {
            rope = nil
        }
    }

    func callAsFunction(
        _ x: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCacheProtocol?
    ) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)

        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        // Weightless Q/K norm over head_dim, then the extra Q-only scale.
        q = museGlimmerRMSNormalize(q, eps: normEps) * MLXArray(qkScaleFactor)
        k = museGlimmerRMSNormalize(k, eps: normEps)

        // NoPE layers skip this entirely (reference passes position_embeddings
        // = None when layer_rope_theta[i] == 0).
        if let rope {
            let offset = cache?.sequenceLength ?? 0
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }

        if let cache {
            (k, v) = cache.update(keys: k, values: v)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: maskMode)

        // [B, heads, L, headDim] -> [B, L, heads*headDim], gate, then project.
        var o = out.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim)
        o = o * MLX.sigmoid(gateProj(x))
        return oProj(o)
    }
}

// MARK: - Decoder layer

/// Gemma-style sandwich: norm → attn → post-norm → residual, then
/// norm → mlp → post-norm → residual. The two post-norms use `post_norm_eps`.
final class MuseGlimmerTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MuseGlimmerTextAttention
    @ModuleInfo(key: "mlp") var mlp: MuseGlimmerTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayernorm: MuseGlimmerCenteredRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayernorm: MuseGlimmerCenteredRMSNorm

    init(_ c: MuseGlimmerTextConfig, layerIdx: Int) {
        _selfAttn = ModuleInfo(
            wrappedValue: MuseGlimmerTextAttention(c, layerIdx: layerIdx), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: MuseGlimmerTextMLP(c), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: MuseGlimmerCenteredRMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: MuseGlimmerCenteredRMSNorm(dimensions: c.hiddenSize, eps: c.postNormEps),
            key: "post_attention_layernorm")
        _preFeedforwardLayernorm = ModuleInfo(
            wrappedValue: MuseGlimmerCenteredRMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps),
            key: "pre_feedforward_layernorm")
        _postFeedforwardLayernorm = ModuleInfo(
            wrappedValue: MuseGlimmerCenteredRMSNorm(dimensions: c.hiddenSize, eps: c.postNormEps),
            key: "post_feedforward_layernorm")
    }

    func callAsFunction(
        _ x: MLXArray,
        maskMode: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCacheProtocol?
    ) -> MLXArray {
        var residual = x
        var h = selfAttn(inputLayernorm(x), maskMode: maskMode, cache: cache)
        h = postAttentionLayernorm(h)
        h = residual + h

        residual = h
        var f = preFeedforwardLayernorm(h)
        f = mlp(f)
        f = postFeedforwardLayernorm(f)
        return residual + f
    }
}

// MARK: - Text model (`model.language_model.*`)

final class MuseGlimmerTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [MuseGlimmerTextDecoderLayer]
    /// Plain `norm(x) * w` — NOT the `(1 + w)` form the per-layer norms use.
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: MuseGlimmerTextConfig

    init(_ c: MuseGlimmerTextConfig) {
        config = c
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< c.numHiddenLayers).map {
                MuseGlimmerTextDecoderLayer(c, layerIdx: $0)
            },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "norm")
    }

    /// Token embedding + the weightless `embed_norm`. Kept separate so the
    /// multimodal path can splice vision features into embeddings that have
    /// ALREADY been normed (matching the reference, which norms inside
    /// `MuseGlimmerTextNormedEmbedding.forward` before any merge).
    func embed(_ tokens: MLXArray) -> MLXArray {
        museGlimmerRMSNormalize(embedTokens(tokens), eps: config.rmsNormEps)
    }

    /// Hidden states for pre-embedded inputs `[B, L, hidden]`.
    func hiddenStates(embeds: MLXArray, caches: [KVCacheProtocol]?) -> MLXArray {
        var h = embeds
        let L = h.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0

        // Two masks, picked per layer from `layer_types`. Feeding a sliding
        // layer the full context at long prompts is out-of-distribution.
        let fullMask = createCachedCausalMask(newLen: L, cacheLen: cacheLen, dtype: .bfloat16)
        let fullMode: MLXFast.ScaledDotProductAttentionMaskMode =
            fullMask != nil ? .array(fullMask!) : .none

        // With `RotatingKVCache` on the sliding layers the retained KV is
        // already trimmed to the window, so a decode step needs no mask at all
        // and a multi-token forward masks over the RETAINED length. Mirrors the
        // Gemma 4 path.
        let rotating = caches?.lazy.compactMap { $0 as? RotatingKVCache }.first
        let slidingMode: MLXFast.ScaledDotProductAttentionMaskMode
        if let rotating {
            if L == 1 {
                slidingMode = .none
            } else {
                let m = createSlidingWindowCausalMask(
                    newLen: L, cacheLen: rotating.maskCacheLength,
                    window: config.slidingWindow, dtype: .bfloat16)
                slidingMode = m != nil ? .array(m!) : .none
            }
        } else {
            let m = createSlidingWindowCausalMask(
                newLen: L, cacheLen: cacheLen, window: config.slidingWindow, dtype: .bfloat16)
            slidingMode = m != nil ? .array(m!) : .none
        }

        for (i, layer) in layers.enumerated() {
            let maskMode = config.isFullAttention(layerIdx: i) ? fullMode : slidingMode
            h = layer(h, maskMode: maskMode, cache: caches?[i])
            // Bound graph size during prefill.
            if L > 1 && (i + 1) % 5 == 0 { MLX.eval(h) }
        }
        return norm(h)
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCacheProtocol]?) -> MLXArray {
        hiddenStates(embeds: embed(tokens), caches: caches)
    }
}

// MARK: - Top-level config

public struct MuseGlimmerConfig: Decodable, Sendable {
    public let textConfig: MuseGlimmerTextConfig
    public let visionConfig: MuseGlimmerVisionConfig?
    public let imageTokenId: Int
    public let videoTokenId: Int
    public let outHiddenSize: Int
    public let projectorHiddenSize: Int
    public let projectorHiddenAct: String
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case outHiddenSize = "out_hidden_size"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorHiddenAct = "projector_hidden_act"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(MuseGlimmerTextConfig.self, forKey: .textConfig)
        visionConfig = try c.decodeIfPresent(MuseGlimmerVisionConfig.self, forKey: .visionConfig)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 200_092
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 200_091
        outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 6144
        projectorHiddenSize = try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 4096
        projectorHiddenAct = try c.decodeIfPresent(
            String.self, forKey: .projectorHiddenAct) ?? "gelu"
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - Inner model (`model.*`)

/// Mirrors the reference `MuseGlimmerModel`: the text decoder, the perception
/// encoder, and the two-stage projector, all under the `model.` key prefix.
final class MuseGlimmerModelInner: Module {
    @ModuleInfo(key: "language_model") var languageModel: MuseGlimmerTextModel
    @ModuleInfo(key: "vision_tower") var visionTower: MuseGlimmerVisionModel?
    @ModuleInfo(key: "vision_adapter") var visionAdapter: MuseGlimmerVisionAdapter?
    @ModuleInfo(key: "vision_projection") var visionProjection: Linear?

    let textEps: Float

    init(_ c: MuseGlimmerConfig) {
        textEps = c.textConfig.rmsNormEps
        _languageModel = ModuleInfo(
            wrappedValue: MuseGlimmerTextModel(c.textConfig), key: "language_model")
        if let vc = c.visionConfig {
            _visionTower = ModuleInfo(
                wrappedValue: MuseGlimmerVisionModel(vc), key: "vision_tower")
            _visionAdapter = ModuleInfo(
                wrappedValue: MuseGlimmerVisionAdapter(
                    inDim: c.outHiddenSize, hidden: c.projectorHiddenSize,
                    act: c.projectorHiddenAct),
                key: "vision_adapter")
            _visionProjection = ModuleInfo(
                wrappedValue: Linear(c.projectorHiddenSize, c.textConfig.hiddenSize, bias: false),
                key: "vision_projection")
        } else {
            _visionTower = ModuleInfo(wrappedValue: nil, key: "vision_tower")
            _visionAdapter = ModuleInfo(wrappedValue: nil, key: "vision_adapter")
            _visionProjection = ModuleInfo(wrappedValue: nil, key: "vision_projection")
        }
    }

    /// `get_image_features`: tower → adapter → projection → weightless norm.
    /// Returns `[nMergedTokens, textHidden]`.
    func imageFeatures(
        pixelValues: MLXArray, grids: [(t: Int, h: Int, w: Int)]
    ) -> MLXArray? {
        guard let visionTower, let visionAdapter, let visionProjection else { return nil }
        let towerOut = visionTower(pixelValues, grids: grids)   // [n, outHiddenSize]
        let projected = visionProjection(visionAdapter(towerOut))
        return museGlimmerRMSNormalize(projected, eps: textEps)
    }
}

// MARK: - MuseGlimmerForConditionalGeneration

/// Top-level Muse Glimmer model. Module keys map 1:1 onto the HF checkpoint:
/// `model.language_model.*`, `model.vision_tower.*`, `model.vision_adapter.*`,
/// `model.vision_projection.*`, `lm_head.*`.
public final class MuseGlimmerForConditionalGeneration: Module {
    @ModuleInfo(key: "model") var model: MuseGlimmerModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: MuseGlimmerConfig
    public let visionCache: VisionEncoderCache = VisionEncoderCache()

    public init(_ config: MuseGlimmerConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: MuseGlimmerModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(
                config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false),
            key: "lm_head")
    }

    /// Per-layer caches: `RotatingKVCache` on the sliding layers (so the
    /// retained KV is trimmed to the 2048 window), `KVCache` on the NoPE
    /// full-attention layers.
    public func makeCaches() -> [KVCacheProtocol] {
        makeKVCaches(
            spec: config.textConfig.cacheSpec,
            numLayers: config.textConfig.numHiddenLayers)
    }

    /// `logits * output_multiplier`, then tanh softcapping at
    /// `final_logit_softcapping`. Softcapping is skipped when the config sets
    /// it to 0 (the "disabled" convention).
    func project(_ hidden: MLXArray) -> MLXArray {
        var logits = lmHead(hidden)
        let m = config.textConfig.outputMultiplier
        if m != 1.0 { logits = logits * MLXArray(m) }
        let cap = config.textConfig.finalLogitSoftcapping
        if cap > 0 {
            logits = MLX.tanh(logits / MLXArray(cap)) * MLXArray(cap)
        }
        return logits
    }

    /// Text-only forward.
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCacheProtocol]? = nil, lastTokenOnly: Bool = false
    ) -> MLXArray {
        var h = model.languageModel(tokens, caches: caches)
        if lastTokenOnly {
            let last = h.dim(1) - 1
            h = h[0..., last ..< (last + 1), 0...]
        }
        return project(h)
    }

    /// Splice `[n, hidden]` vision features into the contiguous `<|patch|>`
    /// span of `[1, L, hidden]` input embeds.
    static func injectVisionEmbeds(
        inputEmbeds: MLXArray, visionEmbeds: MLXArray, start: Int
    ) -> MLXArray {
        let L = inputEmbeds.dim(1)
        let n = visionEmbeds.dim(0)
        precondition(start >= 0 && start + n <= L,
            "vision-embed span [\(start), \(start + n)) exceeds seq len \(L)")
        let before = inputEmbeds[0..., 0 ..< start, 0...]
        let after = inputEmbeds[0..., (start + n) ..< L, 0...]
        return MLX.concatenated(
            [before, visionEmbeds.expandedDimensions(axis: 0), after], axis: 1)
    }

    private func placeholderSpan(_ ids: [Int32], from start: Int, token: Int32) -> Int {
        var n = 0, i = start
        while i < ids.count && ids[i] == token { n += 1; i += 1 }
        return n
    }

    /// Multimodal forward. Muse Glimmer uses ordinary 1-D positions (no mRoPE),
    /// so once the features are spliced the decode path is the text path — no
    /// position offset needs threading through, unlike the Qwen-VL families.
    public func callAsFunction(
        _ tokens: MLXArray,
        pixelValues: MLXArray?,
        grids: [(t: Int, h: Int, w: Int)]?,
        caches: [KVCacheProtocol]? = nil,
        hostTokenIds: [Int32]? = nil,
        lastTokenOnly: Bool = false,
        mediaHash: String? = nil
    ) -> MLXArray {
        var embeds = model.languageModel.embed(tokens)

        if let pixelValues, let grids, !grids.isEmpty {
            let ids: [Int32]
            if let hostTokenIds {
                ids = hostTokenIds
            } else {
                eval(tokens)
                ids = tokens.asArray(Int32.self)
            }
            let features: MLXArray?
            if let hash = mediaHash, let cached = visionCache.lookup(hash) {
                features = cached
            } else {
                let computed = model.imageFeatures(pixelValues: pixelValues, grids: grids)
                if let hash = mediaHash, let computed {
                    MLX.eval(computed)
                    visionCache.store(hash, value: computed)
                }
                features = computed
            }
            if let features {
                let patchToken = Int32(config.imageTokenId)
                let videoToken = Int32(config.videoTokenId)
                let token = ids.contains(patchToken) ? patchToken : videoToken
                if let start = ids.firstIndex(of: token) {
                    let span = placeholderSpan(ids, from: start, token: token)
                    let n = features.dim(0)
                    precondition(span == n,
                        "placeholder span (\(span)) must equal merged vision-token count (\(n))")
                    embeds = Self.injectVisionEmbeds(
                        inputEmbeds: embeds, visionEmbeds: features, start: start)
                }
            }
        }

        var h = model.languageModel.hiddenStates(embeds: embeds, caches: caches)
        if lastTokenOnly {
            let last = h.dim(1) - 1
            h = h[0..., last ..< (last + 1), 0...]
        }
        return project(h)
    }
}
