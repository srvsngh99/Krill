import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Llama-3.2-Vision (mllama) native runtime
//
// mllama = a tiled ViT vision tower (gated aspect-ratio + position embeddings,
// a local transformer + a gated global transformer, intermediate-layer
// concatenation) + a multi-modal projector + a Llama text decoder in which a
// few layers (`cross_attention_layers`) are CROSS-attention layers that attend
// to the projected vision features (the rest are standard self-attention Llama
// layers). This is structurally different from LLaVA's prefix-embed splice:
// vision enters via cross-attention, not by replacing token embeddings.
//
// Mirrors `mlx_vlm.models.mllama` (module attribute names match the HF weight
// keys so `update(parameters:)` binds directly). The text self-attention and
// MLP reuse the same math as the Llama runtime; the deltas are the tiled
// vision tower and the gated cross-attention layers.

// MARK: - Config

public struct Llama32VisionTextConfig: Codable, Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositionEmbeddings: Int
    public let crossAttentionLayers: [Int]

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case crossAttentionLayers = "cross_attention_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32_000
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 40
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        crossAttentionLayers = try c.decodeIfPresent([Int].self, forKey: .crossAttentionLayers)
            ?? [3, 8, 13, 18, 23, 28, 33, 38]
    }

    public var headDim: Int { hiddenSize / numAttentionHeads }
}

public struct Llama32VisionVisionConfig: Codable, Sendable {
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let maxNumTiles: Int
    public let maxAspectRatioId: Int
    public let numGlobalLayers: Int
    public let normEps: Float
    public let visionOutputDim: Int
    public let intermediateLayersIndices: [Int]

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case maxNumTiles = "max_num_tiles"
        case maxAspectRatioId = "max_aspect_ratio_id"
        case numGlobalLayers = "num_global_layers"
        case normEps = "norm_eps"
        case visionOutputDim = "vision_output_dim"
        case intermediateLayersIndices = "intermediate_layers_indices"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 560
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        maxNumTiles = try c.decodeIfPresent(Int.self, forKey: .maxNumTiles) ?? 4
        maxAspectRatioId = try c.decodeIfPresent(Int.self, forKey: .maxAspectRatioId) ?? 8
        numGlobalLayers = try c.decodeIfPresent(Int.self, forKey: .numGlobalLayers) ?? 8
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        visionOutputDim = try c.decodeIfPresent(Int.self, forKey: .visionOutputDim) ?? 7680
        intermediateLayersIndices = try c.decodeIfPresent([Int].self, forKey: .intermediateLayersIndices)
            ?? [3, 7, 15, 23, 30]
    }

    public var numPatches: Int { (imageSize / patchSize) * (imageSize / patchSize) + 1 }
}

public struct Llama32VisionConfig: Codable, Sendable {
    public let textConfig: Llama32VisionTextConfig
    public let visionConfig: Llama32VisionVisionConfig
    public let imageTokenIndex: Int
    public let quantization: QuantizationConfig?

    public var numHiddenLayers: Int { textConfig.numHiddenLayers }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenIndex = "image_token_index"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfig = try c.decode(Llama32VisionTextConfig.self, forKey: .textConfig)
        visionConfig = try c.decode(Llama32VisionVisionConfig.self, forKey: .visionConfig)
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 128_256
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
    }
}

// MARK: - Vision tower

/// Self-attention over vision patches (no causal mask; an optional additive
/// aspect-ratio padding mask hides padded tiles/patches).
class MllamaVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ c: Llama32VisionVisionConfig) {
        let dim = c.hiddenSize
        numHeads = c.numAttentionHeads
        headDim = dim / c.numAttentionHeads
        scale = Foundation.pow(Float(headDim), -0.5)
        _qProj = ModuleInfo(wrappedValue: Linear(dim, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(dim, numHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(dim, numHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, dim, bias: false), key: "o_proj")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class MllamaVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: Llama32VisionVisionConfig) {
        _fc1 = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(c.intermediateSize, c.hiddenSize, bias: true), key: "fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// A vision encoder layer. Global-transformer layers are gated (tanh(gate_attn)
/// / tanh(gate_ffn)); local-transformer layers are not.
class MllamaVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MllamaVisionAttention
    @ModuleInfo(key: "mlp") var mlp: MllamaVisionMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    let isGated: Bool
    // Only the global-transformer (gated) layers carry these; local layers must
    // NOT declare them or strict weight-load verify would flag them unset.
    @ParameterInfo(key: "gate_attn") var gateAttn: MLXArray?
    @ParameterInfo(key: "gate_ffn") var gateFFN: MLXArray?

    init(_ c: Llama32VisionVisionConfig, isGated: Bool) {
        self.isGated = isGated
        _selfAttn = ModuleInfo(wrappedValue: MllamaVisionAttention(c), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: MllamaVisionMLP(c), key: "mlp")
        _inputLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.normEps), key: "input_layernorm")
        _postAttentionLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.normEps), key: "post_attention_layernorm")
        if isGated {
            _gateAttn = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "gate_attn")
            _gateFFN = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "gate_ffn")
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        var attn = selfAttn(inputLayerNorm(h), mask: mask)
        if isGated, let g = gateAttn { attn = MLX.tanh(g) * attn }
        h = h + attn
        var ff = mlp(postAttentionLayerNorm(h))
        if isGated, let g = gateFFN { ff = MLX.tanh(g) * ff }
        return h + ff
    }
}

/// A stack of vision encoder layers that also returns every layer's output (the
/// local transformer's intermediate states feed the final concatenation).
class MllamaVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [MllamaVisionEncoderLayer]

    init(_ c: Llama32VisionVisionConfig, numLayers: Int, isGated: Bool) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< numLayers).map { _ in MllamaVisionEncoderLayer(c, isGated: isGated) },
            key: "layers")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> (MLXArray, [MLXArray]) {
        var h = x
        var states: [MLXArray] = []
        for layer in layers {
            h = layer(h, mask: mask)
            states.append(h)
        }
        return (h, states)
    }
}

/// Gated per-tile aspect-ratio embedding (added before/after the tiled
/// transformer). `embedding[aspect_ratio_id]` is reshaped to one vector per
/// tile and added, scaled by tanh(gate).
class MllamaPrecomputedAspectRatioEmbedding: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ParameterInfo(key: "gate") var gate: MLXArray

    let maxNumTiles: Int
    let hiddenSize: Int

    init(_ c: Llama32VisionVisionConfig) {
        maxNumTiles = c.maxNumTiles
        hiddenSize = c.hiddenSize
        _embedding = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: c.maxAspectRatioId + 1, dimensions: c.maxNumTiles * c.hiddenSize),
            key: "embedding")
        _gate = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "gate")
    }

    func callAsFunction(_ h: MLXArray, aspectRatioIds: MLXArray) -> MLXArray {
        var emb = embedding(aspectRatioIds)                       // [B, maxNumTiles*hidden]
        emb = emb.reshaped(-1, maxNumTiles, 1, hiddenSize)        // [B, tiles, 1, hidden]
        emb = emb * MLX.tanh(gate)
        return h + emb
    }
}

/// Gated position embedding: a per-patch base embedding (scaled by
/// `1 - tanh(gate)`) plus a per-(aspect-ratio, tile, patch) tile position
/// embedding (scaled by `tanh(gate)`).
class MllamaPrecomputedPositionEmbedding: Module {
    @ParameterInfo(key: "embedding") var embedding: MLXArray  // [numPatches, hidden]
    @ModuleInfo(key: "tile_embedding") var tileEmbedding: Embedding
    @ParameterInfo(key: "gate") var gate: MLXArray

    let maxNumTiles: Int
    let numPatches: Int
    let hiddenSize: Int

    init(_ c: Llama32VisionVisionConfig) {
        maxNumTiles = c.maxNumTiles
        numPatches = c.numPatches
        hiddenSize = c.hiddenSize
        _embedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([c.numPatches, c.hiddenSize]), key: "embedding")
        _tileEmbedding = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: c.maxAspectRatioId + 1,
                dimensions: c.maxNumTiles * c.numPatches * c.hiddenSize),
            key: "tile_embedding")
        _gate = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "gate")
    }

    func callAsFunction(_ h: MLXArray, aspectRatioIds: MLXArray) -> MLXArray {
        let gatedPos = (MLXArray(Float(1)) - MLX.tanh(gate)) * embedding
        var out = h + gatedPos.reshaped(1, 1, numPatches, hiddenSize)
        let B = h.dim(0)
        var tilePos = tileEmbedding(aspectRatioIds)               // [B, tiles*patches*hidden]
        tilePos = tilePos.reshaped(B, maxNumTiles, numPatches, hiddenSize)
        out = out + MLX.tanh(gate) * tilePos
        return out
    }
}

/// The full mllama vision tower. Input `pixelValues`
/// `[B, numMedia, numTiles, C, H, W]`; returns concatenated final + selected
/// intermediate hidden states `[B, numMedia, numTiles, numPatches, visionOutputDim]`.
public class MllamaVisionTower: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "gated_positional_embedding") var gatedPositionalEmbedding: MllamaPrecomputedPositionEmbedding
    @ModuleInfo(key: "pre_tile_positional_embedding") var preTile: MllamaPrecomputedAspectRatioEmbedding
    @ModuleInfo(key: "post_tile_positional_embedding") var postTile: MllamaPrecomputedAspectRatioEmbedding
    @ModuleInfo(key: "layernorm_pre") var layernormPre: LayerNorm
    @ModuleInfo(key: "layernorm_post") var layernormPost: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: MllamaVisionEncoder
    @ModuleInfo(key: "global_transformer") var globalTransformer: MllamaVisionEncoder

    let config: Llama32VisionVisionConfig
    let numPatches: Int

    init(_ c: Llama32VisionVisionConfig) {
        config = c
        numPatches = c.numPatches
        let scale = Foundation.pow(Float(c.hiddenSize), -0.5)
        _patchEmbedding = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: c.numChannels, outputChannels: c.hiddenSize,
                kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize), bias: false),
            key: "patch_embedding")
        _classEmbedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([c.hiddenSize]) * scale, key: "class_embedding")
        _gatedPositionalEmbedding = ModuleInfo(
            wrappedValue: MllamaPrecomputedPositionEmbedding(c), key: "gated_positional_embedding")
        _preTile = ModuleInfo(
            wrappedValue: MllamaPrecomputedAspectRatioEmbedding(c), key: "pre_tile_positional_embedding")
        _postTile = ModuleInfo(
            wrappedValue: MllamaPrecomputedAspectRatioEmbedding(c), key: "post_tile_positional_embedding")
        _layernormPre = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.normEps), key: "layernorm_pre")
        _layernormPost = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: c.hiddenSize, eps: c.normEps), key: "layernorm_post")
        _transformer = ModuleInfo(
            wrappedValue: MllamaVisionEncoder(c, numLayers: c.numHiddenLayers, isGated: false),
            key: "transformer")
        _globalTransformer = ModuleInfo(
            wrappedValue: MllamaVisionEncoder(c, numLayers: c.numGlobalLayers, isGated: true),
            key: "global_transformer")
    }

    public func callAsFunction(
        _ pixelValues: MLXArray, aspectRatioIds: MLXArray, aspectRatioMask: MLXArray
    ) -> MLXArray {
        let s = pixelValues.shape  // [B, numMedia, numTiles, C, H, W]
        let B = s[0], numMedia = s[1], numTiles = s[2]
        let C = s[3], H = s[4], W = s[5]
        let bm = B * numMedia
        let arIds = aspectRatioIds.reshaped(bm, -1)

        // Patch embedding: [bm*tiles, C, H, W] -> channels-last conv -> [bm*tiles, gh*gw, hidden]
        var pv = pixelValues.reshaped(bm * numTiles, C, H, W)
        let patches = patchEmbedding(pv.transposed(0, 2, 3, 1))   // [bm*tiles, gh, gw, hidden]
        let hidden = patches.dim(-1)
        var h = patches.reshaped(bm * numTiles, -1, hidden)       // [bm*tiles, gh*gw, hidden]
        var nPatch = h.dim(1)

        // Pre-tile aspect-ratio embedding (per tile).
        h = h.reshaped(bm, numTiles, nPatch, hidden)
        h = preTile(h, aspectRatioIds: arIds)

        // Prepend the CLS token per tile.
        h = h.reshaped(bm * numTiles, nPatch, hidden)
        let cls = broadcast(classEmbedding.reshaped(1, 1, hidden), to: [bm * numTiles, 1, hidden])
        h = concatenated([cls, h], axis: 1)
        nPatch += 1

        // Gated position embedding (per patch + per tile).
        h = h.reshaped(bm, numTiles, nPatch, hidden)
        h = gatedPositionalEmbedding(h, aspectRatioIds: arIds)
        h = layernormPre(h)

        // Pad patches to a multiple of 8.
        let numPad = (8 - (nPatch % 8)) % 8
        if numPad > 0 {
            h = padded(h, widths: [.init((0, 0)), .init((0, 0)), .init((0, numPad)), .init((0, 0))])
        }
        let paddedPatch = nPatch + numPad

        let attnMask = Self.aspectRatioAttentionMask(
            aspectRatioMask: aspectRatioMask.reshaped(bm, -1),
            numPatches: numPatches, targetLength: paddedPatch, dtype: h.dtype)

        // Local transformer over [bm, tiles*paddedPatch, hidden].
        h = h.reshaped(bm, numTiles * paddedPatch, hidden)
        let (localOut, localStates) = transformer(h, mask: attnMask)
        h = layernormPost(localOut)

        // Post-tile aspect-ratio embedding, then the gated global transformer.
        h = h.reshaped(bm, numTiles, paddedPatch, hidden)
        h = postTile(h, aspectRatioIds: arIds)
        h = h.reshaped(bm, numTiles * paddedPatch, hidden)
        let (globalOut, _) = globalTransformer(h, mask: attnMask)
        h = globalOut

        // Drop padding, restore tile layout: [bm, tiles, nPatch, hidden].
        h = h.reshaped(bm, numTiles, paddedPatch, hidden)
        h = h[0..., 0..., 0 ..< nPatch, 0...]

        // Selected intermediate local-transformer states, stacked on a new last
        // axis then gathered by `intermediate_layers_indices`, padding dropped,
        // and concatenated onto the final hidden state along the feature axis.
        var inter = MLX.stacked(localStates, axis: -1)            // [bm, tiles*paddedPatch, hidden, numLocal]
        inter = inter[.ellipsis, MLXArray(config.intermediateLayersIndices.map { Int32($0) })]
        let nInter = config.intermediateLayersIndices.count
        inter = inter.reshaped(bm, numTiles, paddedPatch, hidden * nInter)
        inter = inter[0..., 0..., 0 ..< nPatch, 0...]

        let out = concatenated([h, inter], axis: -1)             // [bm, tiles, nPatch, visionOutputDim]
        return out.reshaped(B, numMedia, numTiles, nPatch, out.dim(-1))
    }

    /// Additive attention mask from the per-tile aspect-ratio mask: padded
    /// tiles' patches are masked (-1e9) for all queries. Mirrors mlx-vlm's
    /// `_prepare_aspect_ratio_attention_mask` (outer-product form).
    static func aspectRatioAttentionMask(
        aspectRatioMask: MLXArray, numPatches: Int, targetLength: Int, dtype: DType
    ) -> MLXArray {
        let B = aspectRatioMask.dim(0), maxTiles = aspectRatioMask.dim(1)
        var m = aspectRatioMask.asType(.float32).reshaped(B, maxTiles, 1, 1)
        m = tiled(m, repetitions: [1, 1, targetLength, 1])       // [B, tiles, target, 1]
        // Zero the padding patches (the last targetLength-numPatches per tile).
        let padPatches = targetLength - numPatches
        if padPatches > 0 {
            let keep = m[0..., 0..., 0 ..< numPatches, 0...]
            let zeros = MLXArray.zeros([B, maxTiles, padPatches, 1])
            m = concatenated([keep, zeros], axis: 2)
        }
        m = MLXArray(Float(1)) - m                               // invert
        m = m.reshaped(B, maxTiles * targetLength, 1)
        var mask = MLX.matmul(m, m.transposed(0, 2, 1)) * MLXArray(Float(-1e9))
        mask = mask.expandedDimensions(axis: 1)                  // [B, 1, L, L]
        return mask.asType(dtype)
    }
}

// MARK: - Text decoder (self-attention + cross-attention layers)

class MllamaTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: Llama32VisionTextConfig) {
        _gateProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.intermediateSize, bias: false), key: "up_proj")
        _downProj = ModuleInfo(wrappedValue: Linear(c.intermediateSize, c.hiddenSize, bias: false), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// Standard Llama GQA self-attention with RoPE + KV cache.
class MllamaTextSelfAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float
    let rope: RoPE

    init(_ c: Llama32VisionTextConfig) {
        numHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = Foundation.pow(Float(headDim), -0.5)
        _qProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numKVHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numKVHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, c.hiddenSize, bias: false), key: "o_proj")
        rope = RoPE(dimensions: headDim, traditional: false, base: c.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        var q = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        if let cache {
            let offset = cache.sequenceLength
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
            (k, v) = cache.update(keys: k, values: v)
        } else {
            q = rope(q)
            k = rope(k)
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Cross-attention: queries from text, keys/values from the projected vision
/// features (`crossStates`). q/k are RMSNorm'd per head; no RoPE. The vision
/// K/V are static across decode, so they are computed once at prefill and
/// cached.
class MllamaTextCrossAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let numHeads: Int
    let numKVHeads: Int
    let headDim: Int
    let scale: Float

    init(_ c: Llama32VisionTextConfig) {
        numHeads = c.numAttentionHeads
        numKVHeads = c.numKeyValueHeads
        headDim = c.headDim
        scale = Foundation.pow(Float(headDim), -0.5)
        _qProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numHeads * headDim, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numKVHeads * headDim, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(c.hiddenSize, numKVHeads * headDim, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(numHeads * headDim, c.hiddenSize, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: c.rmsNormEps), key: "k_norm")
    }

    /// `crossStates` (vision K/V source) is provided on prefill; pass the cached
    /// `(k, v)` on decode steps. Returns the attention output.
    func callAsFunction(
        _ x: MLXArray, crossStates: MLXArray?, cachedKV: (MLXArray, MLXArray)?,
        mask: MLXArray?
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let B = x.dim(0), L = x.dim(1)
        let qRaw = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        let q = qNorm(qRaw)

        var k: MLXArray
        var v: MLXArray
        var producedKV: (MLXArray, MLXArray)? = nil
        if let crossStates {
            let S = crossStates.dim(1)
            k = kProj(crossStates).reshaped(B, S, numKVHeads, headDim).transposed(0, 2, 1, 3)
            v = vProj(crossStates).reshaped(B, S, numKVHeads, headDim).transposed(0, 2, 1, 3)
            k = kNorm(k)
            producedKV = (k, v)
        } else if let cachedKV {
            k = cachedKV.0
            v = cachedKV.1
        } else {
            // No image + no cache (text-only): mirror the reference's
            // `mx.split(query, 2, axis=1)` fallback -- split the PRE-norm query
            // along the HEAD axis into two halves (k from the first, v from the
            // second), then k_norm the key half. Cross-attention contributes
            // ~nothing through the (near-)zero gate on a trained checkpoint, but
            // the split must match the reference to stay bit-for-bit.
            let h2 = numHeads / 2
            k = kNorm(qRaw[0..., 0 ..< h2, 0..., 0...])
            v = qRaw[0..., h2 ..< numHeads, 0..., 0...]
        }
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: mask)
        return (oProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1)), producedKV)
    }
}

/// One text decoder layer. A layer is EITHER a self-attention layer (the
/// majority) OR a cross-attention layer (`cross_attention_layers` indices). A
/// single class with optional `self_attn` / `cross_attn` children keeps the
/// `layers.N.{self_attn|cross_attn}.*` weight keys flat (matching the HF
/// checkpoint) while letting the heterogeneous layers share one `layers` array.
class MllamaDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MllamaTextSelfAttention?
    @ModuleInfo(key: "cross_attn") var crossAttn: MllamaTextCrossAttention?
    @ModuleInfo(key: "mlp") var mlp: MllamaTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ParameterInfo(key: "cross_attn_attn_gate") var attnGate: MLXArray?
    @ParameterInfo(key: "cross_attn_mlp_gate") var mlpGate: MLXArray?

    let isCross: Bool

    init(_ c: Llama32VisionTextConfig, isCross: Bool) {
        self.isCross = isCross
        _mlp = ModuleInfo(wrappedValue: MllamaTextMLP(c), key: "mlp")
        _inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "input_layernorm")
        _postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "post_attention_layernorm")
        if isCross {
            _crossAttn = ModuleInfo(wrappedValue: MllamaTextCrossAttention(c), key: "cross_attn")
            _attnGate = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "cross_attn_attn_gate")
            _mlpGate = ParameterInfo(wrappedValue: MLXArray.zeros([1]), key: "cross_attn_mlp_gate")
        } else {
            _selfAttn = ModuleInfo(wrappedValue: MllamaTextSelfAttention(c), key: "self_attn")
        }
    }

    /// Self-attention layer forward.
    func callSelf(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let h = x + selfAttn!(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }

    /// Cross-attention layer forward; returns the produced vision K/V on prefill
    /// (for the static cross-KV cache).
    func callCross(
        _ x: MLXArray, crossStates: MLXArray?, cachedKV: (MLXArray, MLXArray)?,
        crossMask: MLXArray?, fullRowMask: MLXArray?
    ) -> (MLXArray, (MLXArray, MLXArray)?) {
        let (attn, producedKV) = crossAttn!(
            inputLayerNorm(x), crossStates: crossStates, cachedKV: cachedKV, mask: crossMask)
        var h = x + MLX.tanh(attnGate!) * attn
        var ff = mlp(postAttentionLayerNorm(h))
        if let fullRowMask { ff = fullRowMask * ff }
        h = h + MLX.tanh(mlpGate!) * ff
        return (h, producedKV)
    }
}

// MARK: - Cross-attention KV cache

/// Holds the per-cross-layer vision K/V produced once at image prefill and
/// reused on every decode step (the image is consumed in the prompt, so the
/// vision K/V never grow). Keyed by absolute decoder-layer index so only the
/// `cross_attention_layers` populate entries; self-attention layers use the
/// ordinary `[KVCache]`. An empty cache signals "prefill not yet run" — the
/// first forward with `crossStates` fills it.
public final class MllamaCrossKVCache {
    public var entries: [Int: (MLXArray, MLXArray)] = [:]
    public init() {}
    public var isPopulated: Bool { !entries.isEmpty }
}

// MARK: - Top-level model

/// Native Llama-3.2-Vision (mllama). Mirrors `mlx_vlm.models.mllama.Model`:
/// `vision_tower` + `multi_modal_projector` + a `language_model` whose
/// `cross_attention_layers` attend to the projected vision features.
public class Llama32VisionForCausalLM: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: MllamaVisionTower
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Linear
    @ModuleInfo(key: "language_model") var languageModel: MllamaLanguageModel

    public let config: Llama32VisionConfig
    let crossAttentionLayers: Set<Int>

    public init(_ config: Llama32VisionConfig) {
        self.config = config
        self.crossAttentionLayers = Set(config.textConfig.crossAttentionLayers)
        _visionTower = ModuleInfo(wrappedValue: MllamaVisionTower(config.visionConfig), key: "vision_tower")
        _multiModalProjector = ModuleInfo(
            wrappedValue: Linear(
                config.visionConfig.visionOutputDim, config.textConfig.hiddenSize, bias: true),
            key: "multi_modal_projector")
        _languageModel = ModuleInfo(
            wrappedValue: MllamaLanguageModel(config.textConfig), key: "language_model")
    }

    /// Project vision-tower output into the text hidden space, flattened to
    /// `[B, numTiles*numPatches, textHidden]` (the cross-attention K/V source).
    public func crossAttentionStates(
        pixelValues: MLXArray, aspectRatioIds: MLXArray, aspectRatioMask: MLXArray
    ) -> MLXArray {
        let vision = visionTower(
            pixelValues, aspectRatioIds: aspectRatioIds, aspectRatioMask: aspectRatioMask)
        let B = vision.dim(0)
        return multiModalProjector(vision).reshaped(B, -1, config.textConfig.hiddenSize)
    }

    /// Single full-sequence forward (prefill). When `pixelValues` is provided
    /// the cross-attention layers attend to the projected vision features.
    /// Returns logits `[B, L, vocab]`.
    public func callAsFunction(
        _ inputIds: MLXArray,
        pixelValues: MLXArray? = nil,
        aspectRatioIds: MLXArray? = nil,
        aspectRatioMask: MLXArray? = nil,
        caches: [KVCache]? = nil,
        crossKV: MllamaCrossKVCache? = nil,
        crossMask: MLXArray? = nil,
        fullRowMask: MLXArray? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        // Recompute the vision K/V source only when pixels are supplied AND the
        // cross-KV cache has not been filled yet (image prefill). On decode the
        // caller passes a populated `crossKV` and no pixels, so the vision tower
        // is skipped entirely.
        var crossStates: MLXArray? = nil
        if crossKV?.isPopulated != true,
           let pixelValues, let aspectRatioIds, let aspectRatioMask {
            crossStates = crossAttentionStates(
                pixelValues: pixelValues, aspectRatioIds: aspectRatioIds,
                aspectRatioMask: aspectRatioMask)
        }
        return languageModel(
            inputIds, crossStates: crossStates, caches: caches,
            crossAttentionLayers: crossAttentionLayers,
            crossKV: crossKV, crossMask: crossMask, fullRowMask: fullRowMask,
            lastTokenOnly: lastTokenOnly)
    }
}

/// The mllama text model + lm_head. `embed_tokens` carries 8 extra rows (the
/// reserved image/special tokens) beyond `vocab_size`; `lm_head` projects back
/// to `vocab_size`.
public class MllamaLanguageModel: Module {
    @ModuleInfo(key: "model") var model: MllamaTextInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: Llama32VisionTextConfig

    init(_ c: Llama32VisionTextConfig) {
        config = c
        _model = ModuleInfo(wrappedValue: MllamaTextInner(c), key: "model")
        _lmHead = ModuleInfo(wrappedValue: Linear(c.hiddenSize, c.vocabSize, bias: false), key: "lm_head")
    }

    func callAsFunction(
        _ inputIds: MLXArray, crossStates: MLXArray?, caches: [KVCache]?,
        crossAttentionLayers: Set<Int>,
        crossKV: MllamaCrossKVCache? = nil,
        crossMask: MLXArray? = nil, fullRowMask: MLXArray? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        let h = model(
            inputIds, crossStates: crossStates, caches: caches,
            crossAttentionLayers: crossAttentionLayers,
            crossKV: crossKV, crossMask: crossMask, fullRowMask: fullRowMask)
        // Project only the last row when the caller just needs the next-token
        // logits (prefill / decode), skipping the lm_head over the other rows.
        let hh = lastTokenOnly ? h[0..., (h.dim(1) - 1)..., 0...] : h
        return lmHead(hh)
    }
}

public class MllamaTextInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [MllamaDecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let numLayers: Int

    init(_ c: Llama32VisionTextConfig) {
        numLayers = c.numHiddenLayers
        let crossSet = Set(c.crossAttentionLayers)
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: c.vocabSize + 8, dimensions: c.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< c.numHiddenLayers).map { i in
                MllamaDecoderLayer(c, isCross: crossSet.contains(i))
            },
            key: "layers")
        _norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps), key: "norm")
    }

    func callAsFunction(
        _ inputIds: MLXArray, crossStates: MLXArray?, caches: [KVCache]?,
        crossAttentionLayers: Set<Int>,
        crossKV: MllamaCrossKVCache? = nil,
        crossMask: MLXArray? = nil, fullRowMask: MLXArray? = nil
    ) -> MLXArray {
        var h = embedTokens(inputIds)
        let seqLen = h.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let causal = createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen, dtype: h.dtype)
        for (i, layer) in layers.enumerated() {
            if layer.isCross {
                // A populated entry means prefill already produced this layer's
                // vision K/V; reuse it (decode step) rather than recomputing from
                // `crossStates`. On the prefill forward the entry is absent, so we
                // pass `crossStates`, capture the produced K/V, and store it.
                let cached = crossKV?.entries[i]
                let states = cached == nil ? crossStates : nil
                let (out, produced) = layer.callCross(
                    h, crossStates: states, cachedKV: cached,
                    crossMask: crossMask, fullRowMask: fullRowMask)
                if let produced, let crossKV { crossKV.entries[i] = produced }
                h = out
            } else {
                h = layer.callSelf(h, mask: causal, cache: caches?[i])
            }
        }
        return norm(h)
    }
}
