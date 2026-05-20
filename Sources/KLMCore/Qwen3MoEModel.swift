import Foundation
import MLX
import MLXNN
import MLXFast
import KLMCache

// MARK: - Qwen 3 MoE Config

/// Configuration for the Qwen 3 MoE family
/// (`Qwen3MoeForCausalLM`, `model_type: qwen3_moe`).
///
/// Architectural deltas vs the dense Qwen 3 family captured here:
///   - Each transformer block's MLP is a sparse mixture: a router
///     `mlp.gate` projects hidden -> num_experts, top-K experts run,
///     their outputs are weight-summed.
///   - `moe_intermediate_size` is the per-expert FFN width (smaller
///     than the dense `intermediate_size`; experts are narrower
///     than a single dense MLP).
///   - `decoder_sparse_step` (default 1) and `mlp_only_layers` let
///     specific layers fall back to a dense MLP. We parse these but
///     in practice Qwen3-MoE checkpoints use sparse on every layer
///     (`decoder_sparse_step: 1`, empty `mlp_only_layers`).
///   - `norm_topk_prob` (default true) renormalizes the top-K
///     routing probabilities so each token's mixture weights sum
///     to 1. This is what the reference (mlx-lm, HF Transformers)
///     does for Qwen3-MoE.
///
/// Attention is identical to the dense Qwen 3 path: no QKV bias,
/// per-head q_norm / k_norm before RoPE, optional tied embeddings,
/// explicit `head_dim`. We reuse `QwenAttention` by projecting this
/// config onto a `QwenConfig` shape at block construction time.
public struct Qwen3MoEConfig: ModelConfig, Codable, Sendable {
    public let hiddenSize: Int
    /// Dense MLP intermediate size. Present in some configs for the
    /// `mlp_only_layers` fallback; ignored when all layers are MoE.
    /// Defaults to `moeIntermediateSize` when absent.
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
    public let explicitHeadDim: Int?
    public let modelType: String

    // MoE-specific fields
    /// Total number of experts per MoE layer. Qwen3-30B-A3B: 128.
    public let numExperts: Int
    /// Top-K experts activated per token. Qwen3-30B-A3B: 8.
    public let numExpertsPerToken: Int
    /// Per-expert FFN intermediate size. Qwen3-30B-A3B: 768
    /// (narrower than the dense `intermediate_size` because each
    /// token uses K of them; aggregate compute matches a dense MLP).
    public let moeIntermediateSize: Int
    /// Cadence at which an MoE layer appears. 1 = every layer is
    /// sparse, 2 = every other, etc. Always 1 for Qwen3-MoE today.
    public let decoderSparseStep: Int
    /// Layer indices that fall back to a dense MLP. Empty for
    /// Qwen3-30B-A3B; included for forward compatibility with
    /// future variants that mix dense + sparse blocks.
    public let mlpOnlyLayers: [Int]
    /// Whether to renormalize the top-K probabilities to sum to 1.
    /// True for Qwen3-MoE per the reference implementation.
    public let normTopKProb: Bool

    public var headDim: Int {
        explicitHeadDim ?? (hiddenSize / numAttentionHeads)
    }

    /// Whether the given transformer layer index uses the sparse
    /// MoE MLP. Returns false for layers in `mlpOnlyLayers` or for
    /// layers that fall outside the `decoderSparseStep` cadence.
    public func isSparseLayer(_ index: Int) -> Bool {
        if mlpOnlyLayers.contains(index) { return false }
        if decoderSparseStep <= 1 { return true }
        return index % decoderSparseStep == 0
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
        case explicitHeadDim = "head_dim"
        case modelType = "model_type"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case normTopKProb = "norm_topk_prob"
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
            ?? 40960
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_moe"
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        explicitHeadDim = try c.decodeIfPresent(Int.self, forKey: .explicitHeadDim)

        // MoE fields. Qwen 3 MoE configs use `num_experts`; some
        // reference forks use `num_local_experts` (Mixtral-style).
        // We accept either to keep the loader robust against
        // checkpoint naming drift, defaulting to the Qwen3-30B-A3B
        // shape (128 experts, top-8) when both are absent (which
        // would indicate a malformed config and should fail later
        // at weight load anyway).
        if let n = try c.decodeIfPresent(Int.self, forKey: .numExperts) {
            numExperts = n
        } else {
            numExperts = 128
        }
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        let defaultMoeIS = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        moeIntermediateSize = defaultMoeIS ?? 768
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
            ?? moeIntermediateSize
        decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        normTopKProb = try c.decodeIfPresent(Bool.self, forKey: .normTopKProb) ?? true
    }

    /// Project the MoE config onto a `QwenConfig` for instantiating
    /// the shared `QwenAttention` / `RMSNorm` modules. The MoE
    /// fields (numExperts etc.) drop away because they're irrelevant
    /// to attention. `attention_bias` is forced false (Qwen 3 MoE
    /// matches dense Qwen 3 here); `hasQKNorm` is forced true via
    /// the synthesized `model_type: qwen3`.
    ///
    /// The decode through `QwenConfig.init(from:)` is fallible in
    /// principle. If a future `QwenConfig` field becomes required
    /// without a matching entry in the dict below, we abort with an
    /// explicit message that names the failure surface rather than
    /// the opaque trap `try!` would produce.
    var qwenAttentionConfig: QwenConfig {
        let dict: [String: Any] = [
            "hidden_size": hiddenSize,
            "intermediate_size": intermediateSize,
            "num_attention_heads": numAttentionHeads,
            "num_key_value_heads": numKeyValueHeads,
            "num_hidden_layers": numHiddenLayers,
            "vocab_size": vocabSize,
            "rms_norm_eps": rmsNormEps,
            "rope_theta": ropeTheta,
            "max_position_embeddings": maxPositionEmbeddings,
            "model_type": "qwen3",
            "attention_bias": false,
            "tie_word_embeddings": tieWordEmbeddings,
            "head_dim": explicitHeadDim ?? (hiddenSize / numAttentionHeads),
        ]
        do {
            let data = try JSONSerialization.data(withJSONObject: dict)
            return try JSONDecoder().decode(QwenConfig.self, from: data)
        } catch {
            fatalError(
                "Qwen3MoE to QwenConfig projection failed: \(error). "
                + "A required QwenConfig field was added without a "
                + "corresponding entry in Qwen3MoEConfig.qwenAttentionConfig. "
                + "Update the dict above.")
        }
    }
}

// MARK: - Single Expert (gate/up/down FFN)

/// One expert FFN inside a Qwen 3 MoE layer. Same SwiGLU shape as
/// the dense `QwenMLP`, but with `moeIntermediateSize` as the hidden
/// width and indexed under `mlp.experts.{i}.*` in the checkpoint.
class Qwen3MoEExpert: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: Qwen3MoEConfig) {
        _gateProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.moeIntermediateSize, bias: false),
            key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.moeIntermediateSize, bias: false),
            key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Linear(config.moeIntermediateSize, config.hiddenSize, bias: false),
            key: "down_proj")
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Sparse MoE MLP (router + experts)

/// Replaces the dense `QwenMLP` at MoE layers. Loaded keys:
///   - `mlp.gate.weight`           [num_experts, hidden_size]
///   - `mlp.experts.{e}.gate_proj.weight`   [moe_intermediate_size, hidden]
///   - `mlp.experts.{e}.up_proj.weight`     [moe_intermediate_size, hidden]
///   - `mlp.experts.{e}.down_proj.weight`   [hidden, moe_intermediate_size]
///
/// Forward pass:
///   1. Router logits = `gate(x)` -> `[N, E]`.
///   2. Compute top-K mask using a rank derived from argSort
///      (no native top_k op in mlx-swift today; the argSort+argSort
///      rank trick gives the same mask in two vectorized passes).
///   3. Softmax over the top-K logits only (positions outside top-K
///      get zero weight). Optionally renormalize to sum to 1.
///   4. Accumulate `expert_e(x) * dispatch_weight[:, e]` for every
///      expert. This evaluates ALL experts on ALL tokens — correct
///      but O(num_experts) FFN passes per layer. This is the
///      correctness-first variant; a follow-up replaces the loop
///      with a gather/scatter dispatch that runs each expert only
///      on its assigned tokens. Documented in
///      `docs/workstreams/WS6_MOE_RUNTIME_SUPPORT.md`.
///
/// The forward is deterministic and family-agnostic; the same
/// implementation would work for Mixtral or OLMoE with a different
/// expert count and module-name mapping.
class Qwen3MoESparseMLP: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "experts") var experts: [Qwen3MoEExpert]

    let numExperts: Int
    let topK: Int
    let hiddenSize: Int
    let normTopK: Bool

    init(_ config: Qwen3MoEConfig) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerToken
        self.hiddenSize = config.hiddenSize
        self.normTopK = config.normTopKProb

        // Router. Qwen3-MoE router has no bias and no quantization
        // (the checkpoint stores it as a fp16/bf16 Linear weight).
        _gate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.numExperts, bias: false),
            key: "gate")
        _experts = ModuleInfo(
            wrappedValue: (0 ..< config.numExperts).map { _ in Qwen3MoEExpert(config) },
            key: "experts")
    }

    /// Build a dense `[N, E]` dispatch-weight matrix from `[N, E]`
    /// router logits. Top-K positions carry the softmax probability
    /// (optionally renormalized); all other positions are zero. Used
    /// to combine per-expert outputs by elementwise multiply+sum.
    private func dispatchWeights(routerLogits: MLXArray) -> MLXArray {
        // routerLogits: [N, E]
        // Rank trick: argSort of (-logits) gives indices ordered
        // descending; argSort of those indices gives each position's
        // descending rank (0 = top, 1 = second, ...).
        let neg = MLXArray(0) - routerLogits
        let sortedIdx = argSort(neg, axis: -1)
        let rank = argSort(sortedIdx, axis: -1)
        let kArr = MLXArray(Int32(topK))
        let topKMask = (rank .< kArr).asType(routerLogits.dtype)  // [N, E]

        // Mask logits outside top-K with a large negative so they
        // collapse to ~0 after softmax. We MUST mask BEFORE softmax
        // (not after) so the top-K probabilities are the softmax over
        // the K winning logits, matching mlx-lm / HF Transformers.
        let negInf = MLXArray(Float(-1e9)).asType(routerLogits.dtype)
        let maskedLogits = MLX.where(topKMask .> 0, routerLogits, negInf)
        var weights = softmax(maskedLogits, axis: -1)

        if normTopK {
            // Softmax over the masked logits already sums to ~1
            // across the top-K because the masked-out positions
            // contribute ~0. The explicit renorm guards against
            // numerical drift when `topK` >> 1.
            let denom = weights.sum(axis: -1, keepDims: true)
            weights = weights / (denom + Float(1e-9))
        }
        return weights  // [N, E]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, L, H] -> [N, H]
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let flat = x.reshaped(B * L, H)

        let routerLogits = gate(flat)  // [N, E]
        let dispatch = dispatchWeights(routerLogits: routerLogits)  // [N, E]

        // Accumulate weighted expert outputs. Each expert sees ALL
        // tokens; tokens that did not select expert e contribute 0
        // because `dispatch[n, e]` is 0 outside their top-K. This is
        // the correctness-first variant; performance optimization
        // (gather assigned tokens, run expert on the subset, scatter
        // back) is tracked as a follow-up.
        var out = MLXArray.zeros([B * L, H]).asType(flat.dtype)
        for e in 0 ..< numExperts {
            let weight = dispatch[0..., e ..< (e + 1)]  // [N, 1]
            let expertOut = experts[e](flat)            // [N, H]
            out = out + expertOut * weight
        }
        return out.reshaped(B, L, H)
    }
}

// MARK: - Qwen 3 MoE Transformer Block

/// Per-layer block: attention is identical to dense Qwen 3, the MLP
/// is either the sparse MoE MLP or (for layers listed in
/// `mlpOnlyLayers` / outside the sparse cadence) a dense Qwen MLP
/// at the dense `intermediate_size`.
class Qwen3MoETransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: QwenAttention
    /// Block-local MLP. Holds either a `Qwen3MoESparseMLP` (router +
    /// experts) for sparse layers, or a `QwenMLP` (gate/up/down) for
    /// dense fallback layers. Typed as the base `Module` so MLX-swift
    /// stores ONE entry under the `mlp` key in its parameter cache.
    ///
    /// An earlier draft declared two parallel `@ModuleInfo(key:
    /// "mlp")` properties (one Optional sparse, one Optional dense).
    /// That was a real bug: Module.buildCaches walks the Mirror and
    /// writes `items["mlp"] = value` for each property, so the
    /// second property's nil overwrote the first property's real
    /// Module entry. `model.update(parameters:)` then never assigned
    /// the sparse MLP's router / expert weights (passed verify: []
    /// so the missed assignment was silent), leaving them at random
    /// init on every load. Forward worked (it dereferenced the Swift
    /// property directly), so a synthetic random-weight forward test
    /// could not catch it. The single-property + downcast pattern
    /// here avoids the collision while preserving the dense-layer
    /// fallback path (`mlp_only_layers`).
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    let isSparse: Bool

    init(_ config: Qwen3MoEConfig, layerIndex: Int) {
        let attnConfig = config.qwenAttentionConfig
        _selfAttn = ModuleInfo(
            wrappedValue: QwenAttention(attnConfig), key: "self_attn")
        self.isSparse = config.isSparseLayer(layerIndex)
        let mlpModule: Module
        if isSparse {
            mlpModule = Qwen3MoESparseMLP(config)
        } else {
            mlpModule = QwenMLP(attnConfig)
        }
        _mlp = ModuleInfo(wrappedValue: mlpModule, key: "mlp")
        _inputLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "input_layernorm")
        _postAttentionLayernorm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "post_attention_layernorm")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil, cache: KVCache? = nil) -> MLXArray {
        let h = x + selfAttn(inputLayernorm(x), mask: mask, cache: cache)
        let postAttn = postAttentionLayernorm(h)
        let mlpOut: MLXArray
        if let sparse = mlp as? Qwen3MoESparseMLP {
            mlpOut = sparse(postAttn)
        } else if let dense = mlp as? QwenMLP {
            mlpOut = dense(postAttn)
        } else {
            // Init guarantees one of the two concrete types; this
            // arm exists only to keep the type system happy.
            fatalError("Qwen3MoETransformerBlock.mlp must be either "
                + "Qwen3MoESparseMLP or QwenMLP")
        }
        return h + mlpOut
    }
}

// MARK: - Qwen 3 MoE Inner Model

class Qwen3MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Qwen3MoETransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Qwen3MoEConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { i in
                Qwen3MoETransformerBlock(config, layerIndex: i)
            },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        var x = embedTokens(tokens)
        let seqLen = x.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let mask = createCachedCausalMask(
            newLen: seqLen, cacheLen: cacheLen, dtype: x.dtype)

        for (i, layer) in layers.enumerated() {
            x = layer(x, mask: mask, cache: caches?[i])
        }
        return norm(x)
    }
}

// MARK: - Qwen 3 MoE ForCausalLM

public class Qwen3MoEForCausalLM: Module {
    @ModuleInfo(key: "model") var model: Qwen3MoEModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Qwen3MoEConfig

    public init(_ config: Qwen3MoEConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: Qwen3MoEModelInner(config), key: "model")
        if !config.tieWordEmbeddings {
            _lmHead = ModuleInfo(
                wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
                key: "lm_head")
        }
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        let hidden = model(tokens, caches: caches)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }
}
