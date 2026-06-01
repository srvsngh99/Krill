import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Mixtral Config

/// Configuration for the Mixtral sparse-MoE family
/// (`MixtralForCausalLM`, `model_type: mixtral`).
///
/// Mixtral is Mistral attention (GQA, RoPE, RMSNorm, no QKV bias, no
/// q/k-norm) with the dense SwiGLU MLP replaced by a sparse mixture: a
/// router `block_sparse_moe.gate` projects hidden -> `num_local_experts`,
/// the top-`num_experts_per_tok` experts run, and their SwiGLU outputs are
/// summed weighted by the renormalized router probabilities.
///
/// Unlike Qwen 3 MoE there is no separate `moe_intermediate_size`: each
/// expert is a full-width SwiGLU at `intermediate_size`, and every layer is
/// sparse (no dense fallback layers).
public struct MixtralConfig: ModelConfig, Codable, Sendable {
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

    /// Total experts per MoE layer. Mixtral-8x7B: 8.
    public let numLocalExperts: Int
    /// Top-K experts activated per token. Mixtral-8x7B: 2.
    public let numExpertsPerToken: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Project onto a `LlamaConfig` to instantiate the shared `Attention`
    /// / `RMSNorm` modules. Mixtral attention is identical to Mistral/Llama
    /// (no bias, standard GQA + RoPE), so the dense base is reused verbatim;
    /// only the MLP differs.
    var attentionConfig: LlamaConfig {
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
            quantization: quantization)
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
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
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
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 32768
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        numLocalExperts = try c.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 8
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 2
    }
}

// MARK: - Quantized stacked switched linear (one projection, all experts)

/// One stacked quantized SwitchLinear inside the Mixtral `SwitchGLU` block.
///
/// Copy of `Qwen3QuantizedSwitchedLinear` (the cross-family consolidation
/// into one generic module is tracked in `docs/BACKLOG.md`). Holds the
/// `[numExperts, outputDims, inputDims_packed]` quantized weight plus
/// per-expert scales/biases and dispatches the chosen top-K experts in a
/// single `gatherQuantizedMM`. Parameter layout matches mlx-community's
/// packed Mixtral format (`block_sparse_moe.switch_mlp.{proj}.{weight,
/// scales,biases}`), so the loader binds it with no per-expert unpacking.
class MixtralQuantizedSwitchedLinear: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int
    let groupSize: Int
    let bits: Int

    init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        groupSize: Int, bits: Int
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts
        self.groupSize = groupSize
        self.bits = bits

        let packedIn = inputDims * bits / 32
        let groupsIn = inputDims / groupSize
        _weight = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, packedIn], dtype: .uint32),
            key: "weight")
        _scales = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, groupsIn], dtype: .bfloat16),
            key: "scales")
        _biases = ParameterInfo(
            wrappedValue: MLXArray.zeros([numExperts, outputDims, groupsIn], dtype: .bfloat16),
            key: "biases")
    }

    func callAsFunction(
        _ x: MLXArray, indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        return gatherQuantizedMM(
            x, weight,
            scales: scales, biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize, bits: bits, mode: .affine,
            sortedIndices: sortedIndices)
    }
}

// MARK: - SwitchGLU (router-dispatched SwiGLU experts)

/// Mixtral experts as three stacked quantized switched linears
/// (`gate_proj`/`up_proj`/`down_proj`) plus SwiGLU. SiLU activation, same
/// as Qwen 3 MoE (Mixtral and Qwen share SwiGLU; Gemma 4 uses GeGLU).
///
/// The weight key map from the original HF Mixtral expert tensors is
/// `w1 -> gate_proj`, `w3 -> up_proj`, `w2 -> down_proj`; mlx-community
/// checkpoints already ship the stacked `switch_mlp.{proj}` layout, so the
/// loader binds these directly. Decode/prefill dispatch + the `(token,
/// expert)` sort path are shared via `MoESortPath.swift`.
class MixtralSwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: MixtralQuantizedSwitchedLinear
    @ModuleInfo(key: "up_proj") var upProj: MixtralQuantizedSwitchedLinear
    @ModuleInfo(key: "down_proj") var downProj: MixtralQuantizedSwitchedLinear

    init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        groupSize: Int, bits: Int
    ) {
        _gateProj = ModuleInfo(
            wrappedValue: MixtralQuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: MixtralQuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: MixtralQuantizedSwitchedLinear(
                inputDims: hiddenDims, outputDims: inputDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "down_proj")
    }

    /// - Parameters:
    ///   - x: `[N, H]` flattened token activations.
    ///   - indices: `[N, topK]` Int32 expert ids per token.
    /// - Returns: `[N, topK, H]` per-expert outputs; caller does the topK
    ///   weighted sum.
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let N = x.dim(0)
        let H = x.dim(1)
        let topK = indices.dim(indices.ndim - 1)

        // Prefill: sort (token, expert) by expert id so each expert's gather
        // slice is contiguous and the gather_qmm sortedIndices fast path
        // applies (recovers long-prompt throughput). See `MoESortPath.swift`.
        if moeShouldSort(n: N, topK: topK) {
            let (xs, idx, invOrder) = moeGatherSort(x, indices: indices)
            let xGate = gateProj(xs, indices: idx, sortedIndices: true)
            let xUp = upProj(xs, indices: idx, sortedIndices: true)
            let activated = silu(xGate) * xUp
            let out = downProj(activated, indices: idx, sortedIndices: true)
            return moeScatterUnsort(out, invOrder: invOrder, n: N, topK: topK)
        }

        // Decode / short prompts: direct unsorted dispatch.
        let xExp = x.reshaped(N, 1, 1, H)
        let idx = indices.asType(.int32)
        let xGate = gateProj(xExp, indices: idx)
        let xUp = upProj(xExp, indices: idx)
        let activated = silu(xGate) * xUp
        let out = downProj(activated, indices: idx)
        return out.squeezed(axis: -2)
    }
}

// MARK: - Sparse MoE block (router + SwitchGLU experts)

/// Mixtral `block_sparse_moe`: a router `gate` + the `switch_mlp` experts.
///
/// Routing matches HF/mlx-lm Mixtral exactly, which differs from Qwen 3 MoE:
///   1. `gates = softmax(gate(x))` over **all** experts (in float32).
///   2. Take the top-K experts by score.
///   3. **Renormalize** the K selected probabilities to sum to 1.
/// (Qwen 3 MoE instead masks the non-top-K logits to -inf and softmaxes over
/// the survivors. The two are not equivalent; Mixtral softmaxes first.)
class MixtralSparseMLP: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: MixtralSwitchGLU

    let numExperts: Int
    let topK: Int

    /// On-device per-expert assignment tally (off the compute path; reused
    /// from the Qwen3 MoE machinery in `Qwen3MoEModel.swift`).
    private let accumulator: ExpertCountAccumulator

    var cumulativeExpertCounts: [Int] {
        eval(accumulator.counts)
        return accumulator.counts.asArray(Int32.self).map(Int.init)
    }

    var pendingExpertCounts: MLXArray { accumulator.counts }

    init(_ config: MixtralConfig) {
        self.numExperts = config.numLocalExperts
        self.topK = config.numExpertsPerToken
        self.accumulator = ExpertCountAccumulator(numExperts: config.numLocalExperts)

        _gate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.numLocalExperts, bias: false),
            key: "gate")

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4
        _switchMLP = ModuleInfo(
            wrappedValue: MixtralSwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.intermediateSize,
                numExperts: config.numLocalExperts,
                groupSize: groupSize, bits: bits),
            key: "switch_mlp")
    }

    /// Returns the per-token top-K expert ids `[N, topK]` (descending router
    /// score) and the renormalized top-K weights `[N, topK]` (float32).
    private func route(_ flat: MLXArray) -> (topKExperts: MLXArray, weights: MLXArray) {
        let routerLogits = gate(flat)                                  // [N, E]
        // Full softmax over all experts first (Mixtral/HF order), float32.
        let gates = softmax(routerLogits.asType(.float32), axis: -1)   // [N, E]
        // Top-K experts by score: argSort of (-logits) is ascending in -logit,
        // i.e. descending in score; softmax is monotonic so this also orders
        // `gates`. The first topK columns are the winners.
        let neg = MLXArray(0) - routerLogits
        let sortedByScore = argSort(neg, axis: -1)                     // [N, E]
        let topKExperts = sortedByScore[0..., 0 ..< topK]              // [N, topK]
        var weights = takeAlong(gates, topKExperts, axis: 1)           // [N, topK]
        weights = weights / weights.sum(axis: -1, keepDims: true)      // renormalize
        return (topKExperts.asType(.int32), weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let N = B * L
        let flat = x.reshaped(N, H)

        let (topKExperts, weights) = route(flat)  // [N, topK], [N, topK]

        // Off-path utilization: per-expert assignment counts, accumulated on
        // device (no host read here; materialized lazily for reporting).
        let expertRange = MLXArray(Int32(0) ..< Int32(numExperts))      // [E]
        let oneHot = topKExperts.reshaped(N * topK, 1) .== expertRange  // [N*topK, E]
        accumulator.counts = accumulator.counts + oneHot.asType(.int32).sum(axis: 0)

        let expertOut = switchMLP(flat, indices: topKExperts)          // [N, topK, H]
        let weighted = expertOut * weights.asType(expertOut.dtype).reshaped(N, topK, 1)
        let result = weighted.sum(axis: 1)                             // [N, H]
        return result.reshaped(B, L, H)
    }

    func resetUtilizationStats() {
        accumulator.counts = MLXArray.zeros([numExperts], dtype: .int32)
    }
}

// MARK: - Mixtral Transformer Block

/// Per-layer block: Mistral/Llama attention + the sparse MoE block. The MoE
/// block's in-checkpoint key is `block_sparse_moe` (mlx-lm naming), not
/// `mlp`, so it binds under that key.
class MixtralTransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoe: MixtralSparseMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: MixtralConfig) {
        let attnConfig = config.attentionConfig
        _selfAttn = ModuleInfo(wrappedValue: Attention(attnConfig), key: "self_attn")
        _blockSparseMoe = ModuleInfo(
            wrappedValue: MixtralSparseMLP(config), key: "block_sparse_moe")
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
        return h + blockSparseMoe(postAttentionLayernorm(h))
    }
}

// MARK: - Mixtral Inner Model

class MixtralModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [MixtralTransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: MixtralConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in
                MixtralTransformerBlock(config)
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
        layers.map { $0.blockSparseMoe.pendingExpertCounts }
    }
}

// MARK: - Mixtral ForCausalLM

public class MixtralForCausalLM: Module {
    @ModuleInfo(key: "model") var model: MixtralModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: MixtralConfig

    public init(_ config: MixtralConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: MixtralModelInner(config), key: "model")
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

    /// Batched ragged-decode step (Stage C): one new token per row, each
    /// rotated at its own next position under the explicit per-row mask. The
    /// sparse MoE block is token-count parametric, so the same router +
    /// SwitchGLU dispatch runs unchanged at `N = R`. Mirrors Qwen3 MoE.
    public func batchedDecode(
        _ tokens: MLXArray, caches: [KVCache], mask: MLXArray, rowOffsets: [Int]
    ) -> MLXArray {
        let hidden = model(tokens, caches: caches, precomputedMask: mask, rowOffsets: rowOffsets)
        let logits = lmHead(hidden)
        // Realize this step's logits together with every layer's expert tally
        // in one eval so the running-sum graphs stay depth-1 (see Qwen3 MoE).
        MLX.eval([logits] + model.sparseExpertCounts)
        return logits
    }

    /// Expert-utilization snapshot across every MoE layer since the last
    /// `resetMoEUtilizationStats()`. Off the compute path.
    public func moeUtilization() -> MoEUtilization {
        var sparseLayers = 0
        var totalSlots = 0
        var activeSlots = 0
        var totalAssignments = 0
        var maxLoad = 0
        for block in model.layers {
            sparseLayers += 1
            for count in block.blockSparseMoe.cumulativeExpertCounts {
                totalSlots += 1
                if count > 0 { activeSlots += 1 }
                totalAssignments += count
                if count > maxLoad { maxLoad = count }
            }
        }
        return MoEUtilization(
            sparseLayers: sparseLayers,
            expertsPerLayer: config.numLocalExperts,
            totalExpertSlots: totalSlots,
            activeExpertSlots: activeSlots,
            totalAssignments: totalAssignments,
            maxExpertLoad: maxLoad)
    }

    public func resetMoEUtilizationStats() {
        for block in model.layers {
            block.blockSparseMoe.resetUtilizationStats()
        }
    }
}
