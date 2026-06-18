import Foundation
import MLX
import MLXNN
import MLXFast
import KrillKernels
import KrillCache

// MARK: - Qwen 2 MoE Config

/// Configuration for the Qwen 2 MoE family
/// (`Qwen2MoeForCausalLM`, `model_type: qwen2_moe`). Qwen1.5-MoE-A2.7B is
/// the reference checkpoint.
///
/// Architecture vs Qwen 3 MoE:
///   - Attention is dense Qwen 2: QKV projections carry a bias, and there
///     is NO per-head q_norm / k_norm (that is a Qwen 3 delta). Reused via
///     a `QwenConfig` projection with `model_type: qwen2`.
///   - Every layer is sparse. In addition to the routed top-K experts each
///     MoE block has a single always-on SHARED expert (a dense SwiGLU MLP
///     at `shared_expert_intermediate_size`) gated by
///     `sigmoid(shared_expert_gate(x))`; its output is added to the routed
///     mixture.
///   - The top-K router probabilities are NOT renormalized (mlx-lm /
///     HF Qwen2MoE default `norm_topk_prob: false`): the routed weights are
///     the raw softmax probabilities of the selected experts.
public struct Qwen2MoEConfig: ModelConfig, Codable, Sendable {
    public let hiddenSize: Int
    /// Dense MLP intermediate size. Present in the config but unused by the
    /// layer (every layer is sparse); kept for completeness / projection.
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

    // MoE-specific fields.
    /// Total routed experts per MoE layer. Qwen1.5-MoE-A2.7B: 60.
    public let numExperts: Int
    /// Top-K routed experts per token. Qwen1.5-MoE-A2.7B: 4.
    public let numExpertsPerToken: Int
    /// Per-expert FFN intermediate size (the routed experts).
    public let moeIntermediateSize: Int
    /// Shared-expert FFN intermediate size (the always-on expert).
    public let sharedExpertIntermediateSize: Int

    public var headDim: Int { hiddenSize / numAttentionHeads }

    /// Project onto a `QwenConfig` for the shared `QwenAttention`. Qwen 2
    /// has QKV bias and no q/k-norm: `model_type: qwen2` makes
    /// `QwenConfig` set `hasQKNorm = false` and default `attentionBias =
    /// true`, matching mlx-lm's `bias=True` on q/k/v.
    var qwenAttentionConfig: QwenConfig {
        makeQwenConfig(intermediate: intermediateSize)
    }

    /// Project onto a `QwenConfig` whose `intermediate_size` is the
    /// shared-expert width, so the shared expert can be a stock `QwenMLP`
    /// (gate_proj/up_proj/down_proj at `shared_expert_intermediate_size`).
    var sharedExpertConfig: QwenConfig {
        makeQwenConfig(intermediate: sharedExpertIntermediateSize)
    }

    private func makeQwenConfig(intermediate: Int) -> QwenConfig {
        let dict: [String: Any] = [
            "hidden_size": hiddenSize,
            "intermediate_size": intermediate,
            "num_attention_heads": numAttentionHeads,
            "num_key_value_heads": numKeyValueHeads,
            "num_hidden_layers": numHiddenLayers,
            "vocab_size": vocabSize,
            "rms_norm_eps": rmsNormEps,
            "rope_theta": ropeTheta,
            "max_position_embeddings": maxPositionEmbeddings,
            "model_type": "qwen2",
            "tie_word_embeddings": tieWordEmbeddings,
            "head_dim": hiddenSize / numAttentionHeads,
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: dict)
            return try JSONDecoder().decode(QwenConfig.self, from: data)
        } catch {
            fatalError(
                "Qwen2MoE to QwenConfig projection failed: \(error). A "
                + "required QwenConfig field was added without a matching "
                + "entry in Qwen2MoEConfig.makeQwenConfig. Update the dict above.")
        }
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
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
            ?? (try c.decode(Int.self, forKey: .numAttentionHeads))
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 32768
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 60
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 4
        let moeIS = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        moeIntermediateSize = moeIS ?? 1408
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? moeIntermediateSize
        sharedExpertIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: .sharedExpertIntermediateSize) ?? 5632
    }
}

// MARK: - MoE experts: shared `MoESwitchGLU` (see MoESwitchGLU.swift)
//
// The stacked `gatherQuantizedMM` expert dispatch
// (`MoEQuantizedSwitchedLinear` + `MoESwitchGLU`, with the prefill
// `(token, expert)` sort path in `MoESortPath.swift`) is shared across all
// native MoE families. This family uses `MoEActivation.swiglu`.

// MARK: - Sparse MoE block (router + routed experts + shared expert)

/// Qwen 2 MoE `mlp` block: a router `gate`, the routed `switch_mlp`
/// experts, and a single always-on `shared_expert` (dense SwiGLU MLP) gated
/// by `sigmoid(shared_expert_gate(x))`.
///
/// Routing matches mlx-lm / HF Qwen2MoE: `softmax(gate(x))` over ALL experts
/// (float32), take top-K, and use the raw selected probabilities WITHOUT
/// renormalization (the family default `norm_topk_prob: false`). The shared
/// expert output is added to the routed mixture.
class Qwen2MoESparseMLP: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: MoESwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: QwenMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    let numExperts: Int
    let topK: Int

    private let accumulator: ExpertCountAccumulator

    var cumulativeExpertCounts: [Int] {
        eval(accumulator.counts)
        return accumulator.counts.asArray(Int32.self).map(Int.init)
    }

    var pendingExpertCounts: MLXArray { accumulator.counts }

    init(_ config: Qwen2MoEConfig) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerToken
        self.accumulator = ExpertCountAccumulator(numExperts: config.numExperts)

        _gate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.numExperts, bias: false),
            key: "gate")

        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4
        _switchMLP = ModuleInfo(
            wrappedValue: MoESwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                groupSize: groupSize, bits: bits,
                activation: .swiglu),
            key: "switch_mlp")
        _sharedExpert = ModuleInfo(
            wrappedValue: QwenMLP(config.sharedExpertConfig),
            key: "shared_expert")
        _sharedExpertGate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, 1, bias: false),
            key: "shared_expert_gate")
    }

    /// Top-K routed experts `[N, topK]` (descending router score) and their
    /// raw softmax probabilities `[N, topK]` (NOT renormalized).
    private func route(_ flat: MLXArray) -> (topKExperts: MLXArray, weights: MLXArray) {
        let routerLogits = gate(flat)                                  // [N, E]
        let gates = softmax(routerLogits.asType(.float32), axis: -1)   // [N, E]
        let neg = MLXArray(0) - routerLogits
        let sortedByScore = argSort(neg, axis: -1)                     // [N, E]
        let topKExperts = sortedByScore[0..., 0 ..< topK]              // [N, topK]
        let weights = takeAlong(gates, topKExperts, axis: 1)          // [N, topK] (no renorm)
        return (topKExperts.asType(.int32), weights)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let N = B * L
        let flat = x.reshaped(N, H)

        let (topKExperts, weights) = route(flat)  // [N, topK], [N, topK]

        let expertRange = MLXArray(Int32(0) ..< Int32(numExperts))      // [E]
        let oneHot = topKExperts.reshaped(N * topK, 1) .== expertRange  // [N*topK, E]
        accumulator.counts = accumulator.counts + oneHot.asType(.int32).sum(axis: 0)

        let expertOut = switchMLP(flat, indices: topKExperts)          // [N, topK, H]
        let routed = (expertOut * weights.asType(expertOut.dtype).reshaped(N, topK, 1))
            .sum(axis: 1)                                              // [N, H]

        // Always-on shared expert, sigmoid-gated.
        let shared = sharedExpert(flat)                                // [N, H]
        let sharedGate = sigmoid(sharedExpertGate(flat))               // [N, 1]
        let result = routed + sharedGate.asType(shared.dtype) * shared // [N, H]

        return result.reshaped(B, L, H)
    }

    func resetUtilizationStats() {
        accumulator.counts = MLXArray.zeros([numExperts], dtype: .int32)
    }
}

// MARK: - Qwen 2 MoE Transformer Block

class Qwen2MoETransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: QwenAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen2MoESparseMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: Qwen2MoEConfig) {
        _selfAttn = ModuleInfo(
            wrappedValue: QwenAttention(config.qwenAttentionConfig), key: "self_attn")
        _mlp = ModuleInfo(wrappedValue: Qwen2MoESparseMLP(config), key: "mlp")
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

// MARK: - Qwen 2 MoE Inner Model

class Qwen2MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen2MoETransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Qwen2MoEConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in
                Qwen2MoETransformerBlock(config)
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

// MARK: - Qwen 2 MoE ForCausalLM

public class Qwen2MoEForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Qwen2MoEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Qwen2MoEConfig

    public init(_ config: Qwen2MoEConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Qwen2MoEModelInner(config), key: "model")
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

    /// Batched ragged-decode step (Stage C). The sparse MoE block is
    /// token-count parametric, so the router + SwitchGLU + shared expert run
    /// unchanged at `N = R`. Mirrors Qwen3 MoE.
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
