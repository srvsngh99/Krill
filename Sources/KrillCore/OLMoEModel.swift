import Foundation
import MLX
import MLXNN
import MLXFast
import KrillCache

// MARK: - OLMoE Config

/// Configuration for the OLMoE family
/// (`OlmoeForCausalLM`, `model_type: olmoe`). OLMoE-1B-7B is the reference.
///
/// Architecture notes (vs the other MoE families):
///   - Attention applies an RMSNorm to the WHOLE Q and K projections
///     (`q_norm` over `n_heads*head_dim`, `k_norm` over
///     `n_kv_heads*head_dim`) BEFORE reshaping into heads - distinct from
///     Qwen 3's per-head (`head_dim`) q/k-norm. So OLMoE needs a dedicated
///     attention rather than the shared `QwenAttention`.
///   - Every layer is sparse; there is NO shared expert (unlike Qwen2-MoE).
///   - The top-K router probabilities are renormalized only when
///     `norm_topk_prob` is set (default false for OLMoE-1B-7B).
///   - Embeddings are tied by default.
public struct OLMoEConfig: ModelConfig, Codable, Sendable {
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
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let explicitHeadDim: Int?

    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let normTopKProb: Bool

    public var headDim: Int {
        explicitHeadDim ?? (hiddenSize / numAttentionHeads)
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
        case quantization
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case explicitHeadDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopKProb = "norm_topk_prob"
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
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 4096
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        explicitHeadDim = try c.decodeIfPresent(Int.self, forKey: .explicitHeadDim)
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 64
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        normTopKProb = try c.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? false
    }
}

// MARK: - OLMoE Attention (whole-projection q/k RMSNorm)

/// OLMoE attention. Q/K/V projections (optionally biased) feed an RMSNorm
/// applied over the WHOLE projection (`n_heads*head_dim` for Q,
/// `n_kv_heads*head_dim` for K) BEFORE the reshape into heads, then RoPE +
/// GQA scaled-dot-product attention. The whole-projection norm is the OLMoE
/// delta vs Qwen 3's per-head q/k-norm, so this cannot reuse `QwenAttention`.
class OLMoEAttention: Module {
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

    init(_ config: OLMoEConfig) {
        let dim = config.hiddenSize
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.headDim
        self.scale = 1.0 / Float(config.headDim).squareRoot()

        let bias = config.attentionBias
        _qProj = ModuleInfo(
            wrappedValue: Linear(dim, numHeads * headDim, bias: bias), key: "q_proj")
        _kProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: bias), key: "k_proj")
        _vProj = ModuleInfo(
            wrappedValue: Linear(dim, numKVHeads * headDim, bias: bias), key: "v_proj")
        _oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, dim, bias: bias), key: "o_proj")

        // Whole-projection RMSNorm (over n_heads*head_dim / n_kv_heads*head_dim).
        _qNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: numHeads * headDim, eps: config.rmsNormEps),
            key: "q_norm")
        _kNorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: numKVHeads * headDim, eps: config.rmsNormEps),
            key: "k_norm")

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
                        rowOffsets: [Int]? = nil) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        // Norm the full projection BEFORE splitting into heads.
        var queries = qNorm(qProj(x)).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kNorm(kProj(x)).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        if let rowOffsets {
            queries = applyRoPEPerRow(rope, queries, offsets: rowOffsets)
            keys = applyRoPEPerRow(rope, keys, offsets: rowOffsets)
        } else {
            let offset = cache?.sequenceLength ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
        }

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

// MARK: - MoE experts: shared `MoESwitchGLU` (see MoESwitchGLU.swift)
//
// The stacked `gatherQuantizedMM` expert dispatch
// (`MoEQuantizedSwitchedLinear` + `MoESwitchGLU`, with the prefill
// `(token, expert)` sort path in `MoESortPath.swift`) is shared across all
// native MoE families. This family uses `MoEActivation.swiglu`.

// MARK: - Sparse MoE block (router + routed experts; no shared expert)

/// OLMoE `mlp` block: a router `gate` + `switch_mlp` routed experts, no
/// shared expert. Routing matches mlx-lm / HF OLMoE: `softmax(gate(x))` over
/// ALL experts (float32), top-K, and renormalize the selected probabilities
/// ONLY when `norm_topk_prob` is set (default false for OLMoE-1B-7B).
class OLMoESparseMLP: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: MoESwitchGLU

    let numExperts: Int
    let topK: Int
    let normTopK: Bool

    private let accumulator: ExpertCountAccumulator

    var cumulativeExpertCounts: [Int] {
        eval(accumulator.counts)
        return accumulator.counts.asArray(Int32.self).map(Int.init)
    }

    var pendingExpertCounts: MLXArray { accumulator.counts }

    init(_ config: OLMoEConfig) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerToken
        self.normTopK = config.normTopKProb
        self.accumulator = ExpertCountAccumulator(numExperts: config.numExperts)

        _gate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.numExperts, bias: false),
            key: "gate")

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4
        _switchMLP = ModuleInfo(
            wrappedValue: MoESwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.intermediateSize,
                numExperts: config.numExperts,
                groupSize: groupSize, bits: bits,
                activation: .swiglu),
            key: "switch_mlp")
    }

    private func route(_ flat: MLXArray) -> (topKExperts: MLXArray, weights: MLXArray) {
        let routerLogits = gate(flat)                                  // [N, E]
        let gates = softmax(routerLogits.asType(.float32), axis: -1)   // [N, E]
        let neg = MLXArray(0) - routerLogits
        let sortedByScore = argSort(neg, axis: -1)                     // [N, E]
        let topKExperts = sortedByScore[0..., 0 ..< topK]              // [N, topK]
        var weights = takeAlong(gates, topKExperts, axis: 1)          // [N, topK]
        if normTopK {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (topKExperts.asType(.int32), weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let N = B * L
        let flat = x.reshaped(N, H)

        let (topKExperts, weights) = route(flat)

        let expertRange = MLXArray(Int32(0) ..< Int32(numExperts))
        let oneHot = topKExperts.reshaped(N * topK, 1) .== expertRange
        accumulator.counts = accumulator.counts + oneHot.asType(.int32).sum(axis: 0)

        let expertOut = switchMLP(flat, indices: topKExperts)          // [N, topK, H]
        let result = (expertOut * weights.asType(expertOut.dtype).reshaped(N, topK, 1))
            .sum(axis: 1)                                              // [N, H]
        return result.reshaped(B, L, H)
    }

    func resetUtilizationStats() {
        accumulator.counts = MLXArray.zeros([numExperts], dtype: .int32)
    }
}

// MARK: - OLMoE Transformer Block

class OLMoETransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: OLMoEAttention
    @ModuleInfo(key: "mlp") var mlp: OLMoESparseMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: OLMoEConfig) {
        _selfAttn = ModuleInfo(wrappedValue: OLMoEAttention(config), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: OLMoESparseMLP(config), key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil,
                        rowOffsets: [Int]? = nil) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache, rowOffsets: rowOffsets)
        return h + mlp(postAttentionLayernorm(h))
    }
}

// MARK: - OLMoE Inner Model

class OLMoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [OLMoETransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: OLMoEConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in
                OLMoETransformerBlock(config)
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
        layers.map { $0.mlp.pendingExpertCounts }
    }
}

// MARK: - OLMoE ForCausalLM

public class OLMoEForCausalLM: Module {
    @ModuleInfo(key: "model") var model: OLMoEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: OLMoEConfig

    public init(_ config: OLMoEConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: OLMoEModelInner(config), key: "model")
        if !config.tieWordEmbeddings {
            _lmHead = ModuleInfo(
                wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
                key: "lm_head")
        }
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
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    public func batchedDecode(
        _ tokens: MLXArray, caches: [KVCache], mask: MLXArray, rowOffsets: [Int]
    ) -> MLXArray {
        let hidden = model(tokens, caches: caches, precomputedMask: mask, rowOffsets: rowOffsets)
        let logits = lmHead != nil ? lmHead!(hidden) : model.embedTokens.asLinear(hidden)
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
            sparseLayers += 1
            for count in block.mlp.cumulativeExpertCounts {
                totalSlots += 1
                if count > 0 { activeSlots += 1 }
                totalAssignments += count
                if count > maxLoad { maxLoad = count }
            }
        }
        return MoEUtilization(
            sparseLayers: sparseLayers,
            expertsPerLayer: config.numExperts,
            totalExpertSlots: totalSlots,
            activeExpertSlots: activeSlots,
            totalAssignments: totalAssignments,
            maxExpertLoad: maxLoad)
    }

    public func resetMoEUtilizationStats() {
        for block in model.layers {
            block.mlp.resetUtilizationStats()
        }
    }
}
