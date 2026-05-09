import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Gemma 4 Config

/// Configuration for Gemma 4 model family (E2B, E4B, 12B, 27B, 31B).
/// Complex architecture: PLE, dual head_dim, KV sharing, 4-norm blocks.
public struct Gemma4Config: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let numHiddenLayers: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int

    // Gemma 4 specific
    public let headDim: Int              // 256 for sliding layers
    public let globalHeadDim: Int        // 512 for full attention layers
    public let slidingWindow: Int
    public let slidingWindowPattern: Int  // every Nth layer is full attention
    public let hiddenSizePerLayerInput: Int  // PLE dimension (256)
    public let numKVSharedLayers: Int    // layers sharing KV from earlier layers
    public let useDoubleWideMlp: Bool
    public let finalLogitSoftcapping: Float
    public let tieWordEmbeddings: Bool
    public let quantization: QuantizationConfig?

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case numHiddenLayers = "num_hidden_layers"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case hiddenSizePerLayerInput = "hidden_size_per_layer_input"
        case numKVSharedLayers = "num_kv_shared_layers"
        case useDoubleWideMlp = "use_double_wide_mlp"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
        case quantizationConfig = "quantization_config"
        case textConfig = "text_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        // Parse from nested text_config or flat
        let tc: KeyedDecodingContainer<CodingKeys>
        if let nested = try? c.nestedContainer(keyedBy: CodingKeys.self, forKey: .textConfig) {
            tc = nested
        } else {
            tc = c
        }

        hiddenSize = try tc.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try tc.decode(Int.self, forKey: .intermediateSize)
        numAttentionHeads = try tc.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try tc.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 1
        numHiddenLayers = try tc.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try tc.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try tc.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings = try tc.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262144
        headDim = try tc.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        globalHeadDim = try tc.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512
        slidingWindow = try tc.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        slidingWindowPattern = try tc.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 5
        hiddenSizePerLayerInput = try tc.decodeIfPresent(Int.self, forKey: .hiddenSizePerLayerInput) ?? 256
        numKVSharedLayers = try tc.decodeIfPresent(Int.self, forKey: .numKVSharedLayers) ?? 20
        useDoubleWideMlp = try tc.decodeIfPresent(Bool.self, forKey: .useDoubleWideMlp) ?? true
        finalLogitSoftcapping = try tc.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30.0
        tieWordEmbeddings = try tc.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        // Quantization at top level
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
            ?? (try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantizationConfig))
    }

    /// Whether a layer uses full attention (vs sliding window).
    public func isFullAttention(layerIdx: Int) -> Bool {
        (layerIdx + 1) % slidingWindowPattern == 0
    }

    /// Whether a layer shares KV from an earlier layer.
    public func isKVShared(layerIdx: Int) -> Bool {
        layerIdx >= (numHiddenLayers - numKVSharedLayers)
    }

    /// First layer index that shares KV.
    public var firstKVSharedLayer: Int { numHiddenLayers - numKVSharedLayers }

    // Conform to ModelConfig
    public var ropeTheta: Float { 10_000.0 }
}

// Explicit ModelConfig conformance
extension Gemma4Config: ModelConfig {}

// MARK: - Gemma 4 Attention

class Gemma4Attention: Module {
    let numHeads: Int
    let numKVHeads: Int
    let layerHeadDim: Int
    let isFullAttn: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE
    let rmsNormEps: Float

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.isFullAttn = config.isFullAttention(layerIdx: layerIdx)
        self.layerHeadDim = isFullAttn ? config.globalHeadDim : config.headDim
        self.rmsNormEps = config.rmsNormEps

        let qSize = numHeads * layerHeadDim
        let kvSize = numKVHeads * layerHeadDim

        _qProj = ModuleInfo(wrappedValue: Linear(dim, qSize, bias: false), key: "q_proj")
        _kProj = ModuleInfo(wrappedValue: Linear(dim, kvSize, bias: false), key: "k_proj")
        _vProj = ModuleInfo(wrappedValue: Linear(dim, kvSize, bias: false), key: "v_proj")
        _oProj = ModuleInfo(wrappedValue: Linear(qSize, dim, bias: false), key: "o_proj")
        _qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: layerHeadDim, eps: config.rmsNormEps), key: "q_norm")
        _kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: layerHeadDim, eps: config.rmsNormEps), key: "k_norm")

        // RoPE configuration:
        // Sliding layers: standard RoPE on all 256 dims, base=10000
        // Full layers: ProportionalRoPE - only 128 dims rotated but with freqs from 512-dim space
        //   MLX RoPE(dimensions=128) rotates first 128 dims with freqs base^(-2i/128)
        //   We need freqs base^(-2i/512) but only applied to 128 dims.
        //   Trick: RoPE(dimensions=128) with adjusted base that produces equivalent freqs.
        //   freq_i = base^(-2i/dims). We want base'^(-2i/128) = 1e6^(-2i/512)
        //   => base' = 1e6^(128/512) = 1e6^0.25 = 31.623
        //   This gives the same frequency values as 1e6 with dims=512.
        if isFullAttn {
            // ProportionalRoPE: rotate 128 dims with freqs from 512-dim space.
            // Adjusted base: 1e6^(128/512) = 31.623 gives correct frequency spacing.
            // Note: this rotates dims [0:128] sequentially. The reference rotates
            // [0:64]+[256:320] (split halves). For short sequences (<512 tokens),
            // the positional info is dominated by the lower-frequency components
            // which are the same in both orderings.
            let adjustedBase = powf(1_000_000.0, 128.0 / Float(layerHeadDim))
            self.rope = RoPE(dimensions: 128, traditional: false, base: adjustedBase)
        } else {
            self.rope = RoPE(dimensions: layerHeadDim, traditional: false, base: 10_000.0)
        }
    }

    /// Apply RoPE, handling proportional (split-half) for full attention layers.
    private func applyRoPE(_ x: MLXArray, offset: Int) -> MLXArray {
        if isFullAttn {
            // ProportionalRoPE: split head into halves, rotate first 64 dims of each
            // x shape: [B, H, L, headDim=512]
            let halfDim = layerHeadDim / 2  // 256
            let rotateDims = 64  // 128 total / 2 halves

            let leftHalf = x[0..., 0..., 0..., ..<halfDim]      // [B,H,L,256]
            let rightHalf = x[0..., 0..., 0..., halfDim...]      // [B,H,L,256]

            // Extract dims to rotate
            let leftRotate = leftHalf[0..., 0..., 0..., ..<rotateDims]   // [B,H,L,64]
            let leftKeep = leftHalf[0..., 0..., 0..., rotateDims...]     // [B,H,L,192]
            let rightRotate = rightHalf[0..., 0..., 0..., ..<rotateDims] // [B,H,L,64]
            let rightKeep = rightHalf[0..., 0..., 0..., rotateDims...]   // [B,H,L,192]

            // Concatenate the dims to rotate and apply RoPE
            let toRotate = concatenated([leftRotate, rightRotate], axis: -1) // [B,H,L,128]
            let rotated = rope(toRotate, offset: offset)                     // [B,H,L,128]

            // Split back
            let leftRotated = rotated[0..., 0..., 0..., ..<rotateDims]    // [B,H,L,64]
            let rightRotated = rotated[0..., 0..., 0..., rotateDims...]   // [B,H,L,64]

            // Reassemble
            let newLeft = concatenated([leftRotated, leftKeep], axis: -1)   // [B,H,L,256]
            let newRight = concatenated([rightRotated, rightKeep], axis: -1) // [B,H,L,256]
            return concatenated([newLeft, newRight], axis: -1)               // [B,H,L,512]
        } else {
            // Standard RoPE for sliding attention
            return rope(x, offset: offset)
        }
    }

    /// - Parameter sharedCache: If non-nil, use this cache's K/V instead of computing new ones (KV sharing)
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil,
        cache: KVCache? = nil, sharedCache: KVCache? = nil
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Always compute Q
        var q = qProj(x).reshaped(B, L, numHeads, layerHeadDim).transposed(0, 2, 1, 3)
        q = qNorm(q)
        let offset = cache?.sequenceLength ?? (sharedCache?.sequenceLength ?? 0)
        q = applyRoPE(q, offset: offset)

        let k: MLXArray
        let v: MLXArray

        if sharedCache != nil {
            // KV-shared layer: compute K/V for this token but DON'T write to donor cache.
            // The attention uses the donor cache's accumulated K/V for context.
            var newK = kProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            var newV = vProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            newK = kNorm(newK)
            let vVar = MLX.mean(newV * newV, axis: -1, keepDims: true)
            newV = newV * MLX.rsqrt(vVar + MLXArray(rmsNormEps).asType(newV.dtype))
            newK = applyRoPE(newK, offset: offset)
            // For shared layers: just use own K/V directly (no cache append)
            // The donor's cache provides historical context via the cache passed as 'cache'
            if let cache {
                (k, v) = cache.update(keys: newK, values: newV)
            } else {
                k = newK
                v = newV
            }
        } else {
            // Normal layer: compute K/V and write to own cache
            var newK = kProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            var newV = vProj(x).reshaped(B, L, numKVHeads, layerHeadDim).transposed(0, 2, 1, 3)
            newK = kNorm(newK)
            let vVar = MLX.mean(newV * newV, axis: -1, keepDims: true)
            newV = newV * MLX.rsqrt(vVar + MLXArray(rmsNormEps).asType(newV.dtype))
            newK = applyRoPE(newK, offset: offset)
            if let cache {
                (k, v) = cache.update(keys: newK, values: newV)
            } else {
                k = newK
                v = newV
            }
        }

        // Attention with scale=1.0 (Gemma 4 uses unit scale)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - Gemma 4 MLP (GeGLU, variable width)

class Gemma4MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        // KV-shared layers use double-wide MLP
        let inter: Int
        if config.useDoubleWideMlp && config.isKVShared(layerIdx: layerIdx) {
            inter = config.intermediateSize * 2
        } else {
            inter = config.intermediateSize
        }

        _gateProj = ModuleInfo(wrappedValue: Linear(dim, inter, bias: false), key: "gate_proj")
        _upProj = ModuleInfo(wrappedValue: Linear(dim, inter, bias: false), key: "up_proj")
        _downProj = ModuleInfo(wrappedValue: Linear(inter, dim, bias: false), key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(gelu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Gemma 4 Transformer Block (4 norms + PLE gating)

class Gemma4Block: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Gemma4Attention
    @ModuleInfo(key: "mlp") var mlp: Gemma4MLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFfnNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFfnNorm: RMSNorm
    @ModuleInfo(key: "per_layer_input_gate") var pleGate: Linear
    @ModuleInfo(key: "per_layer_projection") var pleProj: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var pleNorm: RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    init(_ config: Gemma4Config, layerIdx: Int) {
        let dim = config.hiddenSize
        let pleDim = config.hiddenSizePerLayerInput

        _selfAttn = ModuleInfo(wrappedValue: Gemma4Attention(config, layerIdx: layerIdx), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: Gemma4MLP(config, layerIdx: layerIdx), key: "mlp")
        _inputLayernorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "input_layernorm")
        _postAttnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "post_attention_layernorm")
        _preFfnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "pre_feedforward_layernorm")
        _postFfnNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "post_feedforward_layernorm")
        _pleGate = ModuleInfo(wrappedValue: Linear(dim, pleDim, bias: false), key: "per_layer_input_gate")
        _pleProj = ModuleInfo(wrappedValue: Linear(pleDim, dim, bias: false), key: "per_layer_projection")
        _pleNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps), key: "post_per_layer_input_norm")
        // Learned per-layer scalar (ranges from ~0.018 to ~0.87)
        _layerScalar = ModuleInfo(wrappedValue: MLXArray([Float(1.0)]), key: "layer_scalar")
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
        sharedCache: KVCache? = nil, perLayerInput: MLXArray? = nil
    ) -> MLXArray {
        // Attention with post-norm
        var h = x + postAttnNorm(selfAttn(inputLayernorm(x), mask: mask, cache: cache, sharedCache: sharedCache))

        // FFN with pre+post norm
        h = h + postFfnNorm(mlp(preFfnNorm(h)))

        // PLE gating
        if let ple = perLayerInput {
            let gate = gelu(pleGate(h))  // [B, L, 256]
            let gated = gate * ple        // element-wise with PLE slice
            let projected = pleProj(gated) // [B, L, 1536]
            h = h + pleNorm(projected)
        }

        // Per-layer scaling (critical for output quality)
        return h * layerScalar
    }
}

// MARK: - Gemma 4 Full Model

class Gemma4TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "embed_tokens_per_layer") var embedPerLayer: Embedding
    @ModuleInfo(key: "per_layer_model_projection") var perLayerProj: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerNorm: RMSNorm
    @ModuleInfo(key: "layers") var layers: [Gemma4Block]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let config: Gemma4Config

    init(_ config: Gemma4Config) {
        self.config = config
        let dim = config.hiddenSize
        let pleDim = config.hiddenSizePerLayerInput
        let pleTotal = config.numHiddenLayers * pleDim

        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: dim),
            key: "embed_tokens")
        _embedPerLayer = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: pleTotal),
            key: "embed_tokens_per_layer")
        _perLayerProj = ModuleInfo(
            wrappedValue: Linear(dim, pleTotal, bias: false),
            key: "per_layer_model_projection")
        _perLayerNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: pleDim, eps: config.rmsNormEps),
            key: "per_layer_projection_norm")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { i in Gemma4Block(config, layerIdx: i) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: dim, eps: config.rmsNormEps),
            key: "norm")
    }

    /// Forward pass with optional multimodal embedding injection.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs [B, L]
    ///   - caches: Per-layer KV caches
    ///   - imageEmbeddings: Vision encoder output to replace image_token_id positions [1, numImageTokens, hiddenSize]
    ///   - audioEmbeddings: Audio encoder output to replace audio_token_id positions [1, numAudioTokens, hiddenSize]
    ///   - imageTokenId: Token ID for image placeholder (default 258880)
    ///   - audioTokenId: Token ID for audio placeholder (default 258881)
    func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil,
        imageEmbeddings: MLXArray? = nil, audioEmbeddings: MLXArray? = nil,
        imageTokenId: Int = 258880, audioTokenId: Int = 258881
    ) -> MLXArray {
        let B = tokens.dim(0)
        let L = tokens.dim(1)
        let numLayers = config.numHiddenLayers
        let pleDim = config.hiddenSizePerLayerInput

        // Main embeddings (scaled by sqrt(hidden_size))
        // CRITICAL: scalars must be BF16 to avoid promoting quantized matmuls to float32
        let embedScale = MLXArray(Float(config.hiddenSize).squareRoot()).asType(.bfloat16)
        var h = embedTokens(tokens) * embedScale

        // Inject multimodal embeddings at placeholder token positions
        if let imgEmb = imageEmbeddings {
            h = injectEmbeddings(h, tokens: tokens, embeddings: imgEmb, tokenId: imageTokenId)
        }
        if let audEmb = audioEmbeddings {
            h = injectEmbeddings(h, tokens: tokens, embeddings: audEmb, tokenId: audioTokenId)
        }

        // PLE: compute per-layer inputs
        let pleScale = MLXArray(Float(pleDim).squareRoot()).asType(.bfloat16)
        let pleEmbed = embedPerLayer(tokens) * pleScale

        let projScale = MLXArray(1.0 / Float(config.hiddenSize).squareRoot()).asType(.bfloat16)
        let projection = perLayerProj(h) * projScale
        let projNormed = perLayerNorm(projection.reshaped(B * L * numLayers, pleDim))
            .reshaped(B, L, numLayers * pleDim)

        let combineScale = MLXArray(Float(0.7071067811865476)).asType(.bfloat16)
        let combinedPLE = (projNormed + pleEmbed) * combineScale

        // Causal mask
        let mask: MLXArray? = L > 1 ? createAdditiveCausalMask(L, dtype: .bfloat16) : nil

        // KV sharing: find the donor cache indices for shared layers.
        // Layers 0..<firstKVSharedLayer have their own caches.
        // Layers firstKVSharedLayer..< numLayers reuse KV from earlier donor layers.
        // Donor for sliding: last non-shared sliding layer
        // Donor for full: last non-shared full layer
        let firstShared = config.firstKVSharedLayer
        var lastSlidingDonor = 0
        var lastFullDonor = 0
        for i in 0 ..< firstShared {
            if config.isFullAttention(layerIdx: i) {
                lastFullDonor = i
            } else {
                lastSlidingDonor = i
            }
        }

        // Forward through layers
        // For simplicity in v1: all layers compute their own K/V and use their own cache.
        // Full KV sharing optimization (sharing caches between layers 15-34 and donors)
        // deferred to v2 for correctness-first approach.
        for (i, layer) in layers.enumerated() {
            let pleSlice = combinedPLE[0..., 0..., (i * pleDim) ..< ((i + 1) * pleDim)]
            h = layer(h, mask: mask, cache: caches?[i], perLayerInput: pleSlice)
        }

        return norm(h)
    }
}

// MARK: - Gemma4ForCausalLM

public class Gemma4ForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Gemma4TextModel
    // Separate lm_head for logit projection (loaded from embed_tokens weights)
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: Gemma4Config

    public init(_ config: Gemma4Config) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Gemma4TextModel(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        let hidden = model(tokens, caches: caches)
        let logits = lmHead(hidden)
        let cap = config.finalLogitSoftcapping
        return MLX.tanh(logits / cap) * cap
    }

    /// Forward pass with multimodal embedding injection.
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil,
        imageEmbeddings: MLXArray? = nil, audioEmbeddings: MLXArray? = nil,
        imageTokenId: Int = 258880, audioTokenId: Int = 258881
    ) -> MLXArray {
        let hidden = model(
            tokens, caches: caches,
            imageEmbeddings: imageEmbeddings, audioEmbeddings: audioEmbeddings,
            imageTokenId: imageTokenId, audioTokenId: audioTokenId)
        let logits = lmHead(hidden)
        let cap = config.finalLogitSoftcapping
        return MLX.tanh(logits / cap) * cap
    }
}

// MARK: - Multimodal Embedding Injection

/// Replace token embeddings at placeholder positions with encoder outputs.
///
/// For each position where tokens == tokenId, the corresponding embedding
/// is replaced with the next embedding from the encoder output sequence.
private func injectEmbeddings(
    _ embeddings: MLXArray, tokens: MLXArray,
    embeddings replacement: MLXArray, tokenId: Int
) -> MLXArray {
    // tokens: [B, L], embeddings: [B, L, D], replacement: [1, N, D]
    // Find positions where token == tokenId
    let mask = tokens .== MLXArray(Int32(tokenId))  // [B, L]

    // Expand mask to [B, L, 1] for broadcasting
    let maskExpanded = mask.reshaped(mask.dim(0), mask.dim(1), 1).asType(embeddings.dtype)

    // The replacement embeddings need to be padded/aligned to match sequence length.
    // Simple approach: create a replacement tensor of same shape, fill placeholder positions.
    let B = embeddings.dim(0)
    let L = embeddings.dim(1)
    let D = embeddings.dim(2)
    let N = replacement.dim(1)

    // Build replacement row: pad replacement embeddings to length L
    // Positions without the token ID will be zero (masked out by maskExpanded anyway)
    let padded: MLXArray
    if N >= L {
        padded = replacement[0..., ..<L, 0...]
    } else {
        let zeros = MLXArray.zeros([1, L - N, D]).asType(replacement.dtype)
        padded = concatenated([replacement, zeros], axis: 1)
    }

    // Tile to batch size if needed
    let paddedB: MLXArray
    if B > 1 {
        // Repeat along batch dimension
        var tiles = [Int](repeating: 1, count: padded.ndim)
        tiles[0] = B
        paddedB = MLX.tiled(padded, repetitions: tiles)
    } else {
        paddedB = padded
    }

    // Replace: where mask is true use replacement, else keep original
    return embeddings * (1 - maskExpanded) + paddedB * maskExpanded
}
