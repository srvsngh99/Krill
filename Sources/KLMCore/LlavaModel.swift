import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - LLaVA-1.5 native runtime
//
// LLaVA-1.5 = a CLIP ViT vision tower + a multi-modal projector (linear ->
// gelu -> linear) + a Llama text backbone. The vision features (the
// penultimate CLIP layer, CLS dropped) are projected to the text hidden size
// and spliced into the token embeddings at the `<image>` placeholder
// positions; the Llama stack then runs over the merged embeddings. Mirrors
// `mlx_vlm.models.llava`. The text backbone reuses `LlamaForCausalLM`.

public struct ClipVisionConfig: Codable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numHiddenLayers: Int
    public let imageSize: Int
    public let patchSize: Int
    public let numChannels: Int
    public let layerNormEps: Float
    public let modelType: String

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case numChannels = "num_channels"
        case layerNormEps = "layer_norm_eps"
        case modelType = "model_type"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 336
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "clip_vision_model"
    }

    public var numPatches: Int { (imageSize / patchSize) * (imageSize / patchSize) }
    /// CLIP prepends a CLS token (+1); SigLIP does not.
    public var numPositions: Int { modelType == "clip_vision_model" ? numPatches + 1 : numPatches }
}

public struct LlavaConfig: Codable, Sendable {
    public let textConfig: LlamaConfig
    public let visionConfig: ClipVisionConfig
    public let imageTokenIndex: Int
    public let visionFeatureLayer: Int
    public let visionFeatureSelectStrategy: String
    public let vocabSize: Int
    /// Top-level quantization (real LLaVA checkpoints carry one shared block);
    /// falls back to the text config's own block.
    public let quantization: QuantizationConfig?

    public var numHiddenLayers: Int { textConfig.numHiddenLayers }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case imageTokenIndex = "image_token_index"
        case visionFeatureLayer = "vision_feature_layer"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case vocabSize = "vocab_size"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // The canonical llava-1.5 checkpoints ship a MINIMAL `text_config`
        // (e.g. mlx-community/llava-1.5-7b-4bit omits hidden_size,
        // intermediate_size, num_attention_heads, num_hidden_layers,
        // rope_theta), relying on transformers' `LlamaConfig` class defaults --
        // which are exactly the vicuna-7b dims. `LlamaConfig`'s own decoder
        // requires those keys, so decode the text config through a defaulting
        // shim that fills the HF base-Llama defaults for whatever is absent.
        textConfig = try c.decode(LlavaTextConfig.self, forKey: .textConfig).toLlamaConfig()
        visionConfig = try c.decode(ClipVisionConfig.self, forKey: .visionConfig)
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 32_000
        visionFeatureLayer = try c.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) ?? -2
        visionFeatureSelectStrategy =
            try c.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy) ?? "default"
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? textConfig.vocabSize
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
            ?? textConfig.quantization
    }
}

/// Defaulting decoder for a llava `text_config`. Real llava-1.5 checkpoints
/// omit most Llama dims and lean on transformers' `LlamaConfig` class defaults
/// (which match vicuna-7b); a 13b checkpoint instead spells the dims out, and
/// `decodeIfPresent` honors those. The defaults below are the HF `LlamaConfig`
/// base defaults (NOT Krill's Llama-3-oriented `rope_theta=500000` /
/// `max_position_embeddings=131072`, which would be wrong for vicuna).
struct LlavaTextConfig: Codable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numHiddenLayers: Int
    let vocabSize: Int
    let rmsNormEps: Float
    let ropeTheta: Float
    let maxPositionEmbeddings: Int

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
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11_008
        numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? numAttentionHeads
        numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32_000
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
    }

    func toLlamaConfig() -> LlamaConfig {
        LlamaConfig(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numAttentionHeads: numAttentionHeads,
            numKeyValueHeads: numKeyValueHeads,
            numHiddenLayers: numHiddenLayers,
            vocabSize: vocabSize,
            rmsNormEps: rmsNormEps,
            ropeTheta: ropeTheta,
            maxPositionEmbeddings: maxPositionEmbeddings,
            quantization: nil)
    }
}

// MARK: - CLIP vision tower

/// Patch embed (`Conv2d`) + CLS token + learned position embedding. Input is
/// channels-last `[B, H, W, C]` (the caller transposes the `[B, C, H, W]`
/// pixel tensor first, matching mlx-vlm).
class ClipVisionEmbeddings: Module {
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let numPositions: Int

    init(_ config: ClipVisionConfig) {
        self.numPositions = config.numPositions
        _classEmbedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([config.hiddenSize]), key: "class_embedding")
        _patchEmbedding = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: config.numChannels, outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize), stride: IntOrPair(config.patchSize),
                bias: false),
            key: "patch_embedding")
        _positionEmbedding = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.numPositions, dimensions: config.hiddenSize),
            key: "position_embedding")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let patches = patchEmbedding(x)                       // [B, gh, gw, hidden]
        let hidden = patches.dim(-1)
        let flat = patches.reshaped(B, -1, hidden)            // [B, numPatches, hidden]
        let cls = broadcast(classEmbedding.reshaped(1, 1, hidden), to: [B, 1, hidden])
        var embeddings = concatenated([cls, flat], axis: 1)   // [B, numPatches+1, hidden]
        let positionIds = MLXArray(Int32(0) ..< Int32(numPositions))
        embeddings = embeddings + positionEmbedding(positionIds)
        return embeddings
    }
}

/// CLIP self-attention (q/k/v/out all biased, no causal mask).
class ClipVisionAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let numHeads: Int
    let scale: Float

    init(_ config: ClipVisionConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.scale = 1.0 / Float(dim / config.numAttentionHeads).squareRoot()
        _qProj = ModuleInfo(wrappedValue: Linear(dim, dim, bias: true), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(dim, dim, bias: true), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(dim, dim, bias: true), key: "v_proj")
        _outProj = ModuleInfo(wrappedValue: Linear(dim, dim, bias: true), key: "out_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0), L = x.dim(1)
        let q = qProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// CLIP MLP: fc1 -> fast-approx GELU -> fc2 (matches `nn.GELU(approx="fast")`).
class ClipVisionMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: ClipVisionConfig) {
        _fc1 = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: true), key: "fc1")
        _fc2 = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: true), key: "fc2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(geluFastApproximate(fc1(x)))
    }
}

class ClipVisionEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: ClipVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: ClipVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: ClipVisionConfig) {
        _selfAttn = ModuleInfo(wrappedValue: ClipVisionAttention(config), key: "self_attn")
        _layerNorm1 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm1")
        _mlp = ModuleInfo(wrappedValue: ClipVisionMLP(config), key: "mlp")
        _layerNorm2 = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttn(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

class ClipVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [ClipVisionEncoderLayer]
    init(_ config: ClipVisionConfig) {
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in ClipVisionEncoderLayer(config) },
            key: "layers")
    }
}

/// The CLIP transformer (`vision_model`). Returns the hidden states across
/// layers so the caller can pick `vision_feature_layer` (penultimate for
/// LLaVA). `hidden_states[0]` is the post-`pre_layrnorm` embedding, then one
/// per encoder layer -- matching mlx-vlm's `encoder_states` tuple.
class ClipVisionTransformer: Module {
    @ModuleInfo(key: "embeddings") var embeddings: ClipVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm   // sic: HF key is misspelled
    @ModuleInfo(key: "encoder") var encoder: ClipVisionEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    init(_ config: ClipVisionConfig) {
        _embeddings = ModuleInfo(wrappedValue: ClipVisionEmbeddings(config), key: "embeddings")
        _preLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "pre_layrnorm")
        _encoder = ModuleInfo(wrappedValue: ClipVisionEncoder(config), key: "encoder")
        _postLayerNorm = ModuleInfo(
            wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "post_layernorm")
    }

    /// Returns the list of hidden states (length `numLayers + 1`).
    func hiddenStates(_ x: MLXArray) -> [MLXArray] {
        var h = preLayerNorm(embeddings(x))
        var states = [h]
        for layer in encoder.layers {
            h = layer(h)
            states.append(h)
        }
        return states
    }
}

/// `vision_tower`: wraps the CLIP transformer under the `vision_model` key.
class ClipVisionModel: Module {
    @ModuleInfo(key: "vision_model") var visionModel: ClipVisionTransformer
    init(_ config: ClipVisionConfig) {
        _visionModel = ModuleInfo(wrappedValue: ClipVisionTransformer(config), key: "vision_model")
    }
}

// MARK: - Multi-modal projector

/// linear_1 -> exact GELU -> linear_2 (matches `nn.GELU()` with no approx).
class LlavaMultiModalProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: LlavaConfig) {
        _linear1 = ModuleInfo(
            wrappedValue: Linear(config.visionConfig.hiddenSize, config.textConfig.hiddenSize, bias: true),
            key: "linear_1")
        _linear2 = ModuleInfo(
            wrappedValue: Linear(config.textConfig.hiddenSize, config.textConfig.hiddenSize, bias: true),
            key: "linear_2")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Top-level LLaVA model

public class LlavaForCausalLM: Module {
    @ModuleInfo(key: "vision_tower") var visionTower: ClipVisionModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: LlavaMultiModalProjector
    @ModuleInfo(key: "language_model") var languageModel: LlamaForCausalLM

    public let config: LlavaConfig

    public init(_ config: LlavaConfig) {
        self.config = config
        _visionTower = ModuleInfo(wrappedValue: ClipVisionModel(config.visionConfig), key: "vision_tower")
        _multiModalProjector = ModuleInfo(
            wrappedValue: LlavaMultiModalProjector(config), key: "multi_modal_projector")
        _languageModel = ModuleInfo(wrappedValue: LlamaForCausalLM(config.textConfig), key: "language_model")
    }

    /// Compute the projected image features `[1, numFeatures, textHidden]` from
    /// channels-first `pixelValues` `[1, C, H, W]`.
    public func imageFeatures(_ pixelValues: MLXArray) -> MLXArray {
        // CLIP wants channels-last; the checkpoint's pixels are channels-first.
        let x = pixelValues.transposed(0, 2, 3, 1)               // [1, H, W, C]
        let states = visionTower.visionModel.hiddenStates(x)     // numLayers+1
        let idx = config.visionFeatureLayer >= 0
            ? config.visionFeatureLayer
            : states.count + config.visionFeatureLayer
        var feature = states[idx]                                // [1, numPositions, visionHidden]
        if config.visionFeatureSelectStrategy == "default" {
            feature = feature[0..., 1..., 0...]                  // drop the CLS token
        }
        return multiModalProjector(feature)                      // [1, numFeatures, textHidden]
    }

    /// Text-only forward (no image): straight through the Llama backbone.
    /// `lastTokenOnly` slices the prefill output to the final position before
    /// the vocab projection (the sampler reads only that row).
    public func callAsFunction(
        _ inputIds: MLXArray, caches: [KVCache]? = nil, lastTokenOnly: Bool = false
    ) -> MLXArray {
        languageModel(inputIds, caches: caches, lastTokenOnly: lastTokenOnly)
    }

    /// Splice the projected image features into the token embeddings at the
    /// `<image>` placeholder positions (contiguous, one image), then run the
    /// Llama text stack. Returns logits `[1, L, vocab]`.
    public func callAsFunction(
        _ inputIds: MLXArray, pixelValues: MLXArray, caches: [KVCache]? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        var inputEmbeds = languageModel.model.embedTokens(inputIds)   // [1, L, H]
        let features = imageFeatures(pixelValues)
        let n = features.dim(1)

        eval(inputIds)
        let ids = inputIds.reshaped(-1).asArray(Int32.self)
        let imagePositions = ids.indices.filter { ids[$0] == Int32(config.imageTokenIndex) }
        precondition(imagePositions.count == n,
            "LLaVA expects one image token per vision feature (\(imagePositions.count) tokens, \(n) features)")
        let start = imagePositions.first ?? 0
        let L = inputEmbeds.dim(1)
        let before = inputEmbeds[0..., 0 ..< start, 0...]
        let after = inputEmbeds[0..., (start + n) ..< L, 0...]
        inputEmbeds = concatenated(
            [before, features.asType(inputEmbeds.dtype), after], axis: 1)

        return languageModel(
            inputIds, inputsEmbeds: inputEmbeds, caches: caches,
            lastTokenOnly: lastTokenOnly)
    }
}
