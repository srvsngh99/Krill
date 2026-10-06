import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - EmbeddingGemma 2 (text path)
//
// `google/embeddinggemma-2` (`model_type: embedding_gemma2`): a Gemma 4
// derived BIDIRECTIONAL encoder. The checkpoint also carries a SigLIP-style
// vision tower and an audio conformer (Milestone 2); this file serves only
// the text path and explicitly skips those tensors (counted + logged by
// `loadEmbeddingGemma2`).
//
// It is NOT `Gemma4TextModel` run non-causally. Verified against
// transformers' `modeling_embedding_gemma2.py` and the checkpoint header:
//
//   * attention is bidirectional on every layer; sliding layers allow
//     |i - j| <= sliding_window (512), full layers allow everything
//   * full-attention layers (5, 11, 17, 23) use head_dim 512 and ONE kv head
//     (sliding: head_dim 256, 2 kv heads), selected by `per_layer_config`
//   * RoPE is plain rotate-half over the WHOLE head (theta 1e6 full / 1e4
//     sliding) - Gemma 4's proportional/partial RoPE does NOT apply
//   * PLE is projection-only: `per_layer_model_projection(embeds) * H^-0.5`
//     then RMSNorm; there is no `embed_tokens_per_layer` table and no
//     blending scale
//   * no KV sharing, no KV cache, no MoE, no lm_head
//   * v_norm (scale-free RMS) is applied to V like Gemma 4
//   * after the final norm a 512 -> 768 `embedding_projection`; sentence
//     vector = mean over real tokens, then L2-normalise (the projection
//     commutes with the mean, so projecting per token is equivalent)
//
// float16 is unsafe for this model; compute dtype is float32 (default) or
// bfloat16 only (`EmbeddingGemma2Model.setComputeDtype`).

public struct EmbeddingGemma2Config: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numLayers: Int
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let globalHeadDim: Int
    public let globalKVHeads: Int
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let pleDim: Int
    public let embeddingDim: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeThetaSliding: Float
    public let ropeThetaFull: Float
    /// Modality marker / placeholder ids from the top-level config.json
    /// (`boi_token_id`, `image_token_id`, ...). Used by `EG2SequenceBuilder`.
    public let modalityTokens: EG2ModalityTokens

    /// Context length of the model (the card: 8,192 tokens). The text
    /// config's `max_position_embeddings` (262144) is a training-time
    /// ceiling, not the supported context.
    public static let maxContext = 8192
    /// Matryoshka dimensions the model was trained for.
    public static let mrlDimensions = [768, 512, 256, 128]

    private struct Rope: Decodable { let ropeTheta: Float?
        enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" } }
    private struct PerLayer: Decodable {
        let headDim: Int?
        let numKeyValueHeads: Int?
        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim", numKeyValueHeads = "num_key_value_heads" }
    }
    private enum Root: String, CodingKey {
        case textConfig = "text_config"
        case boi = "boi_token_id", eoi = "eoi_token_id", image = "image_token_id"
        case boa = "boa_token_id", eoa = "eoa_token_index", audio = "audio_token_id"
        case video = "video_token_id"
    }
    private enum Text: String, CodingKey {
        case hiddenSize = "hidden_size", intermediateSize = "intermediate_size"
        case numLayers = "num_hidden_layers", numHeads = "num_attention_heads"
        case numKVHeads = "num_key_value_heads", headDim = "head_dim"
        case slidingWindow = "sliding_window", layerTypes = "layer_types"
        case pleDim = "hidden_size_per_layer_input", embeddingDim = "embedding_dim"
        case vocabSize = "vocab_size", rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters", perLayerConfig = "per_layer_config"
    }

    public init(from decoder: Decoder) throws {
        let root = try decoder.container(keyedBy: Root.self)
        let d = EG2ModalityTokens.checkpointDefaults
        modalityTokens = EG2ModalityTokens(
            boi: try root.decodeIfPresent(Int.self, forKey: .boi) ?? d.boi,
            eoi: try root.decodeIfPresent(Int.self, forKey: .eoi) ?? d.eoi,
            image: try root.decodeIfPresent(Int.self, forKey: .image) ?? d.image,
            boa: try root.decodeIfPresent(Int.self, forKey: .boa) ?? d.boa,
            eoa: try root.decodeIfPresent(Int.self, forKey: .eoa) ?? d.eoa,
            audio: try root.decodeIfPresent(Int.self, forKey: .audio) ?? d.audio,
            video: try root.decodeIfPresent(Int.self, forKey: .video) ?? d.video)
        let c = try root.nestedContainer(keyedBy: Text.self, forKey: .textConfig)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numLayers = try c.decode(Int.self, forKey: .numLayers)
        numHeads = try c.decode(Int.self, forKey: .numHeads)
        numKVHeads = try c.decode(Int.self, forKey: .numKVHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        pleDim = try c.decodeIfPresent(Int.self, forKey: .pleDim) ?? 512
        embeddingDim = try c.decodeIfPresent(Int.self, forKey: .embeddingDim) ?? 768
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        // HF default: every 6th layer is full attention, last layer forced full.
        var types = try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< numLayers).map { ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention" }
        if types.count != numLayers {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes, in: c,
                debugDescription: "layer_types has \(types.count) entries, expected \(numLayers)")
        }
        if types[numLayers - 1] != "full_attention" { types[numLayers - 1] = "full_attention" }
        layerTypes = types
        let rope = try c.decodeIfPresent([String: Rope].self, forKey: .ropeParameters)
        ropeThetaSliding = rope?["sliding_attention"]?.ropeTheta ?? 10_000
        ropeThetaFull = rope?["full_attention"]?.ropeTheta ?? 1_000_000
        // `per_layer_config` keys are layer indices ("05", "11", ...).
        let per = try c.decodeIfPresent([String: PerLayer].self, forKey: .perLayerConfig) ?? [:]
        let firstFull = per.values.first
        globalHeadDim = firstFull?.headDim ?? 512
        globalKVHeads = firstFull?.numKeyValueHeads ?? 1
        // Every full-attention layer must be described identically; the model
        // below assumes one (headDim, kvHeads) pair for all of them.
        for (k, v) in per where (v.headDim ?? globalHeadDim) != globalHeadDim
            || (v.numKeyValueHeads ?? globalKVHeads) != globalKVHeads {
            throw DecodingError.dataCorruptedError(
                forKey: .perLayerConfig, in: c,
                debugDescription: "per_layer_config[\(k)] differs from other full layers")
        }
    }

    public func isFull(_ layer: Int) -> Bool { layerTypes[layer] == "full_attention" }
    public func headDim(_ layer: Int) -> Int { isFull(layer) ? globalHeadDim : headDim }
    public func kvHeads(_ layer: Int) -> Int { isFull(layer) ? globalKVHeads : numKVHeads }
}

// MARK: - Layers

/// Scale-free RMS norm (`with_scale=False` in the reference), computed in
/// float32 then cast back, as the reference does.
private func rmsNormNoScale(_ x: MLXArray, eps: Float) -> MLXArray {
    let xf = x.asType(.float32)
    let v = MLX.mean(xf * xf, axis: -1, keepDims: true)
    return (xf * MLX.rsqrt(v + eps)).asType(x.dtype)
}

private func geluTanh(_ x: MLXArray) -> MLXArray { geluApproximate(x) }

final class EG2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numHeads: Int
    let numKV: Int
    let headDim: Int
    let eps: Float
    let rope: RoPE

    init(_ cfg: EmbeddingGemma2Config, layer: Int) {
        numHeads = cfg.numHeads
        numKV = cfg.kvHeads(layer)
        headDim = cfg.headDim(layer)
        eps = cfg.rmsNormEps
        let d = cfg.hiddenSize
        _qProj = ModuleInfo(wrappedValue: Linear(d, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(d, numKV * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(d, numKV * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, d, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: cfg.rmsNormEps), key: "k_norm")
        // Plain (non-proportional) RoPE over the whole head, rotate-half layout.
        rope = RoPE(dimensions: headDim, traditional: false,
                    base: cfg.isFull(layer) ? cfg.ropeThetaFull : cfg.ropeThetaSliding)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKV, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, numKV, headDim).transposed(0, 2, 1, 3)
        q = rope(qNorm(q))
        k = rope(kNorm(k))
        v = rmsNormNoScale(v, eps: eps)
        let o = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim))
    }
}

final class EG2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ cfg: EmbeddingGemma2Config) {
        _gate = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false), key: "gate_proj")
        _up = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.intermediateSize, bias: false), key: "up_proj")
        _down = ModuleInfo(wrappedValue: Linear(cfg.intermediateSize, cfg.hiddenSize, bias: false), key: "down_proj")
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluTanh(gate(x)) * up(x)) }
}

final class EG2PLEBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var gate: Linear
    @ModuleInfo(key: "per_layer_projection") var proj: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var norm: RMSNorm
    init(_ cfg: EmbeddingGemma2Config) {
        _gate = ModuleInfo(wrappedValue: Linear(cfg.hiddenSize, cfg.pleDim, bias: false), key: "per_layer_input_gate")
        _proj = ModuleInfo(wrappedValue: Linear(cfg.pleDim, cfg.hiddenSize, bias: false), key: "per_layer_projection")
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps), key: "post_per_layer_input_norm")
    }
    func callAsFunction(_ h: MLXArray, _ perLayerInput: MLXArray) -> MLXArray {
        h + norm(proj(geluTanh(gate(h)) * perLayerInput))
    }
}

final class EG2Layer: Module {
    @ModuleInfo(key: "self_attn") var attn: EG2Attention
    @ModuleInfo(key: "mlp") var mlp: EG2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFfnNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFfnNorm: RMSNorm
    @ModuleInfo(key: "ple_block") var ple: EG2PLEBlock
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ cfg: EmbeddingGemma2Config, layer: Int) {
        let d = cfg.hiddenSize
        _attn = ModuleInfo(wrappedValue: EG2Attention(cfg, layer: layer), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: EG2MLP(cfg), key: "mlp")
        _inputNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: d, eps: cfg.rmsNormEps), key: "input_layernorm")
        _postAttnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: d, eps: cfg.rmsNormEps), key: "post_attention_layernorm")
        _preFfnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: d, eps: cfg.rmsNormEps), key: "pre_feedforward_layernorm")
        _postFfnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: d, eps: cfg.rmsNormEps), key: "post_feedforward_layernorm")
        _ple = ModuleInfo(wrappedValue: EG2PLEBlock(cfg), key: "ple_block")
        _layerScalar = ParameterInfo(wrappedValue: MLXArray([Float(1)]), key: "layer_scalar")
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray,
                        mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        var h = x + postAttnNorm(attn(inputNorm(x), mask: mask))
        h = h + postFfnNorm(mlp(preFfnNorm(h)))
        h = ple(h, perLayerInput)
        return h * layerScalar
    }
}

/// Projection-only per-layer embeddings (`language_model.ple.*`).
final class EG2PLE: Module {
    @ModuleInfo(key: "per_layer_model_projection") var projection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var norm: RMSNorm
    let numLayers: Int
    let pleDim: Int
    let scale: Float
    init(_ cfg: EmbeddingGemma2Config) {
        numLayers = cfg.numLayers
        pleDim = cfg.pleDim
        scale = 1.0 / Float(cfg.hiddenSize).squareRoot()
        _projection = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.numLayers * cfg.pleDim, bias: false),
            key: "per_layer_model_projection")
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.pleDim, eps: cfg.rmsNormEps),
                           key: "per_layer_projection_norm")
    }
    /// `[B, T, H] -> [B, T, numLayers, pleDim]`
    func callAsFunction(_ embeds: MLXArray) -> MLXArray {
        let p = projection(embeds) * MLXArray(scale).asType(embeds.dtype)
        return norm(p.reshaped(embeds.dim(0), embeds.dim(1), numLayers, pleDim))
    }
}

// MARK: - Model

public final class EmbeddingGemma2Model: Module, SentenceEmbeddingEncoder {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [EG2Layer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "ple") var ple: EG2PLE
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    public let config: EmbeddingGemma2Config
    public private(set) var computeDtype: DType = .bfloat16

    public init(_ cfg: EmbeddingGemma2Config) {
        config = cfg
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: cfg.vocabSize, dimensions: cfg.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< cfg.numLayers).map { EG2Layer(cfg, layer: $0) }, key: "layers")
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: cfg.hiddenSize, eps: cfg.rmsNormEps), key: "norm")
        _ple = ModuleInfo(wrappedValue: EG2PLE(cfg), key: "ple")
        _embeddingProjection = ModuleInfo(
            wrappedValue: Linear(cfg.hiddenSize, cfg.embeddingDim, bias: false),
            key: "embedding_projection")
    }

    /// Cast every parameter to `dtype`. Only float32 and bfloat16 are
    /// accepted: float16 overflows / silently degrades this model.
    public func setComputeDtype(_ dtype: DType) {
        precondition(dtype == .float32 || dtype == .bfloat16,
                     "EmbeddingGemma2 supports only float32/bfloat16 compute")
        update(parameters: parameters().mapValues { $0.asType(dtype) })
        computeDtype = dtype
    }

    /// Text-token embeddings `[B, T, hidden]` in the compute dtype: table lookup
    /// times sqrt(hidden). sqrt(512) rounds to 22.625 in bf16; the reference
    /// casts the scale to the weight dtype, so do the same in both modes.
    /// Modality soft tokens are NOT scaled; `EG2SequenceBuilder` scatters them
    /// in after this (see `forward(inputsEmbeds:lengths:)`).
    public func embedText(_ tokens: MLXArray) -> MLXArray {
        let scale = MLXArray(Float(config.hiddenSize).squareRoot()).asType(computeDtype)
        return embedTokens(tokens) * scale
    }

    /// Token ids `[B, T]` (right-padded) plus real lengths -> per-token
    /// projected states `[B, T, embeddingDim]` in the compute dtype.
    func forward(_ tokens: MLXArray, lengths: [Int]) -> MLXArray {
        forward(inputsEmbeds: embedText(tokens), lengths: lengths)
    }

    /// The backbone on ALREADY-EMBEDDED inputs `[B, T, hidden]` (right-padded,
    /// compute dtype): text embeddings with any modality soft tokens scattered
    /// in. Per-layer inputs (PLE) are computed here, i.e. AFTER the merge, as in
    /// the reference (`language_model(inputs_embeds=...)`).
    public func forward(inputsEmbeds: MLXArray, lengths: [Int]) -> MLXArray {
        let B = inputsEmbeds.dim(0), T = inputsEmbeds.dim(1)
        var h = inputsEmbeds
        let perLayer = ple(h)  // [B, T, L, pleDim]

        // Key-padding mask (only when the batch is actually padded) combined
        // with the sliding band for sliding layers.
        let padded = lengths.contains { $0 != T }
        let neg = MLXArray(Float(-1e9)).asType(computeDtype)
        let zero = MLXArray(Float(0)).asType(computeDtype)
        var keyMask: MLXArray? = nil
        if padded {
            let pos = MLXArray(0 ..< Int32(T)).reshaped(1, T)
            let len = MLXArray(lengths.map { Int32($0) }).reshaped(B, 1)
            keyMask = MLX.where(pos .< len, zero, neg).reshaped(B, 1, 1, T)
        }
        let fullMode: MLXFast.ScaledDotProductAttentionMaskMode =
            keyMask.map { .array($0) } ?? .none
        var slidingMode = fullMode
        if T > config.slidingWindow + 1 {
            let i = MLXArray(0 ..< Int32(T)).reshaped(T, 1)
            let j = MLXArray(0 ..< Int32(T)).reshaped(1, T)
            let band = MLX.where(MLX.abs(i - j) .<= MLXArray(Int32(config.slidingWindow)), zero, neg)
                .reshaped(1, 1, T, T)
            slidingMode = .array(keyMask.map { $0 + band } ?? band)
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, perLayerInput: perLayer[0..., 0..., i, 0...],
                      mask: config.isFull(i) ? fullMode : slidingMode)
            if i % 6 == 5 { eval(h) }  // bound the lazy graph on long inputs
        }
        return embeddingProjection(norm(h))
    }

    /// `SentenceEmbeddingEncoder`: batch-1, unpadded, `[1, T, embeddingDim]`.
    public func lastHiddenState(_ tokens: MLXArray) -> MLXArray {
        forward(tokens, lengths: [tokens.dim(1)]).asType(.float32)
    }

    /// Mean-pooled (un-normalised) float32 sentence vectors `[B, embeddingDim]`
    /// for right-padded `tokens [B, T]` with real `lengths`.
    public func pooled(_ tokens: MLXArray, lengths: [Int]) -> MLXArray {
        meanPool(forward(tokens, lengths: lengths), lengths: lengths)
    }

    /// Same, for already-embedded inputs `[B, T, hidden]` (modality soft tokens
    /// merged in). The mean covers EXACTLY the first `lengths[b]` positions of
    /// row b: every real token, soft tokens and the `<boi>/<eoi>/<boa>/<eoa>`
    /// markers included (the reference pools with `include_prompt: true` over the
    /// full attention mask), so a task prefix is pooled too.
    public func pooled(inputsEmbeds: MLXArray, lengths: [Int]) -> MLXArray {
        meanPool(forward(inputsEmbeds: inputsEmbeds, lengths: lengths), lengths: lengths)
    }

    private func meanPool(_ hidden: MLXArray, lengths: [Int]) -> MLXArray {
        let states = hidden.asType(.float32)
        let T = hidden.dim(1)
        let pos = MLXArray(0 ..< Int32(T)).reshaped(1, T)
        let len = MLXArray(lengths.map { Int32($0) }).reshaped(lengths.count, 1)
        let m = (pos .< len).asType(.float32).reshaped(lengths.count, T, 1)
        let sum = MLX.sum(states * m, axis: 1)
        return sum / MLX.maximum(MLX.sum(m, axis: 1), MLXArray(Float(1)))
    }
}

// MARK: - Pure post-processing (unit-tested)

public enum EmbeddingMath {
    /// L2-normalise in place (zero vectors are left as zeros).
    public static func l2Normalize(_ v: [Float]) -> [Float] {
        var s: Float = 0
        for x in v { s += x * x }
        let n = s.squareRoot()
        return n > 0 ? v.map { $0 / n } : v
    }

    /// Matryoshka truncation: first `dim` components, then re-normalise.
    public static func truncate(_ v: [Float], to dim: Int) -> [Float] {
        l2Normalize(Array(v.prefix(dim)))
    }

    /// Masked mean over `[T][H]` states (the pooling the model uses).
    public static func maskedMean(_ states: [[Float]], length: Int) -> [Float] {
        guard length > 0, let h = states.first?.count else { return [] }
        var out = [Float](repeating: 0, count: h)
        for t in 0 ..< min(length, states.count) { for k in 0 ..< h { out[k] += states[t][k] } }
        return out.map { $0 / Float(length) }
    }
}

// MARK: - Loader (strict binding)

public struct EmbeddingGemma2LoadReport: Sendable, CustomStringConvertible {
    public let boundText: Int
    public let skippedVision: Int
    public let skippedAudio: Int
    public var skipped: Int { skippedVision + skippedAudio }
    public var description: String {
        "EmbeddingGemma2: bound \(boundText) text tensors (strict: every one consumed, none missing); "
        + "skipped \(skipped) multimodal tensors (\(skippedVision) vision, \(skippedAudio) audio)"
    }
}

/// Classify checkpoint keys: returns (text keys with the `language_model.`
/// prefix stripped, vision count, audio count). Throws on any key that is
/// neither text nor a recognised multimodal tensor, so a checkpoint layout
/// change cannot silently drop weights.
func partitionEmbeddingGemma2Keys<S: Sequence>(_ keys: S) throws
    -> (text: [String: String], vision: Int, audio: Int) where S.Element == String
{
    var text: [String: String] = [:]
    var vision = 0, audio = 0
    for k in keys {
        if k.hasPrefix("language_model.") {
            text[k] = String(k.dropFirst("language_model.".count))
        } else if k.hasPrefix("vision_tower.") || k.hasPrefix("embed_vision.") {
            vision += 1
        } else if k.hasPrefix("audio_tower.") || k.hasPrefix("embed_audio.") {
            audio += 1
        } else {
            throw EmbeddingGemma2LoadError.unrecognisedKey(k)
        }
    }
    return (text, vision, audio)
}

public enum EmbeddingGemma2LoadError: Error, CustomStringConvertible {
    case unrecognisedKey(String)
    public var description: String {
        switch self {
        case .unrecognisedKey(let k):
            return "EmbeddingGemma2 checkpoint has an unrecognised tensor '\(k)' "
                + "(not language_model.*, vision_tower.*, audio_tower.*, embed_vision.*, embed_audio.*)"
        }
    }
}

/// Load the text path with STRICT binding. Unlike the lax VL loaders, every
/// text tensor must land on a module parameter and every module parameter
/// must be covered with the right shape (`.allModelKeysSet`, `.shapeMismatch`,
/// `.noUnusedKeys`); vision/audio tensors are skipped and counted.
public func loadEmbeddingGemma2(
    directory: URL, dtype: DType = .float32
) throws -> (model: EmbeddingGemma2Model, report: EmbeddingGemma2LoadReport) {
    let cfgData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
    let cfg = try JSONDecoder().decode(EmbeddingGemma2Config.self, from: cfgData)
    let model = EmbeddingGemma2Model(cfg)

    let all = try loadWeightArrays(from: directory)
    let parts = try partitionEmbeddingGemma2Keys(all.keys)
    var text: [(String, MLXArray)] = []
    text.reserveCapacity(parts.text.count)
    for (full, stripped) in parts.text { text.append((stripped, all[full]!)) }

    try model.update(
        parameters: ModuleParameters.unflattened(text),
        verify: [.allModelKeysSet, .shapeMismatch, .noUnusedKeys])
    model.setComputeDtype(dtype)
    eval(model)

    let report = EmbeddingGemma2LoadReport(
        boundText: text.count, skippedVision: parts.vision, skippedAudio: parts.audio)
    FileHandle.standardError.write(Data((report.description + "\n").utf8))
    return (model, report)
}
