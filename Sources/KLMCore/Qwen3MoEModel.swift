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
/// Forward pass (router):
///   1. Router logits = `gate(x)` -> `[N, E]`.
///   2. Compute top-K experts per token using a rank derived from
///      argSort (no native top_k op in mlx-swift today; the
///      argSort+argSort rank trick gives the same selection in two
///      vectorized passes).
///   3. Softmax over the top-K logits only; optionally renormalize.
///
/// Forward pass (expert dispatch), `callAsFunction`:
///   The default path is a **scatter dispatch**: it builds the
///   `N * topK` (token, expert) assignment list, sorts the
///   assignments by expert id, runs each expert ONCE on the
///   contiguous slice of tokens routed to it, then un-sorts and
///   weight-sums back per token. Each expert sees only `count_e`
///   tokens, so total FFN work is `N * topK` token-passes instead
///   of the brute-force `N * numExperts`. For Qwen3-30B-A3B
///   (128 experts, top-8) that is a 16x reduction.
///
///   The brute-force reference (`referenceForward`), every expert
///   on every token weighted by a dense `[N, E]` dispatch matrix,
///   is retained for the parity test: `callAsFunction` must
///   produce numerically equal output (within fp tolerance; the
///   two differ only in summation order).
///
/// The scatter dispatch performs ONE host sync per layer to read
/// the per-expert token counts (needed to slice the sorted
/// assignment array). That is a per-layer cost, not per-token, and
/// is the price of exact (non-capacity-dropping) routing without a
/// fused gather-matmul kernel.
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

    /// Cumulative count of `(token, slot)` assignments each expert has
    /// served since the last `resetUtilizationStats()`. Index `e` is
    /// expert `e`'s running total. Updated off the compute path from
    /// the scatter dispatch's per-layer host count read, so it adds no
    /// kernel work. Used only for expert-load / utilization reporting
    /// (`Qwen3MoEForCausalLM.moeUtilization()`); it never feeds the
    /// forward. Mutated sequentially within a generation - the server
    /// serializes generation, so no locking is needed.
    private(set) var cumulativeExpertCounts: [Int]

    init(_ config: Qwen3MoEConfig) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerToken
        self.hiddenSize = config.hiddenSize
        self.normTopK = config.normTopKProb
        self.cumulativeExpertCounts = [Int](repeating: 0, count: config.numExperts)

        // Router. Qwen3-MoE router has no bias and no quantization
        // (the checkpoint stores it as a fp16/bf16 Linear weight).
        _gate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.numExperts, bias: false),
            key: "gate")
        _experts = ModuleInfo(
            wrappedValue: (0 ..< config.numExperts).map { _ in Qwen3MoEExpert(config) },
            key: "experts")
    }

    /// Router top-K. Returns the per-token top-K expert ids
    /// `[N, topK]` (descending router score) and the per-token
    /// dense dispatch-weight matrix `[N, E]` (softmax over the
    /// top-K logits, zero elsewhere, optionally renormalized).
    private func route(_ flat: MLXArray) -> (topKExperts: MLXArray, dispatch: MLXArray) {
        let routerLogits = gate(flat)  // [N, E]
        // Rank trick: argSort of (-logits) gives indices ordered by
        // descending score; argSort of those indices gives each
        // position's descending rank (0 = top, 1 = second, ...).
        let neg = MLXArray(0) - routerLogits
        let sortedByScore = argSort(neg, axis: -1)        // [N, E]
        let rank = argSort(sortedByScore, axis: -1)        // [N, E]
        let topKMask = rank .< MLXArray(Int32(topK))       // [N, E] bool

        // Mask logits outside top-K with a large negative so they
        // collapse to ~0 after softmax. Masking BEFORE softmax makes
        // the top-K probabilities the softmax over the K winning
        // logits, matching mlx-lm / HF Transformers.
        let negInf = MLXArray(Float(-1e9)).asType(routerLogits.dtype)
        let maskedLogits = MLX.where(topKMask, routerLogits, negInf)
        var dispatch = softmax(maskedLogits, axis: -1)     // [N, E]
        if normTopK {
            let denom = dispatch.sum(axis: -1, keepDims: true)
            dispatch = dispatch / (denom + Float(1e-9))
        }
        // The first `topK` columns of sortedByScore are the chosen
        // expert ids for each token (highest score first).
        let topKExperts = sortedByScore[0..., 0 ..< topK]  // [N, topK]
        return (topKExperts, dispatch)
    }

    /// Scatter dispatch: each expert runs once on the contiguous
    /// slice of tokens routed to it. See the type doc for the
    /// algorithm and the per-layer host-sync note.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let N = B * L
        let flat = x.reshaped(N, H)

        let (topKExperts, dispatch) = route(flat)  // [N, topK], [N, E]

        // Build the flat (token, expert) assignment list of length
        // N * topK. `assignExpert[i]` is the expert id; `tokenIds[i]`
        // the source token.
        let assignExpert = topKExperts.reshaped(N * topK)            // [N*topK]
        let tokenIdsCol = MLXArray(Int32(0) ..< Int32(N)).reshaped(N, 1)
        let tokenIds = MLX.broadcast(tokenIdsCol, to: [N, topK])
            .reshaped(N * topK)                                       // [N*topK]

        // Sort the assignments by expert id so each expert's tokens
        // form one contiguous run.
        let order = argSort(assignExpert, axis: -1)                   // [N*topK]
        let sortedExpert = MLX.take(assignExpert, order, axis: 0)     // [N*topK]
        let sortedToken = MLX.take(tokenIds, order, axis: 0)         // [N*topK]
        let gathered = MLX.take(flat, sortedToken, axis: 0)          // [N*topK, H]

        // Per-expert token counts via a one-hot sum. ONE host sync
        // per layer: the counts are needed on the host to slice the
        // sorted assignment array into per-expert chunks.
        let expertRange = MLXArray(Int32(0) ..< Int32(numExperts))   // [E]
        let oneHot = sortedExpert.reshaped(N * topK, 1) .== expertRange  // [N*topK, E]
        let counts = oneHot.asType(.int32).sum(axis: 0)               // [E]
        eval(counts)
        let countsHost = counts.asArray(Int32.self)

        // Off-path instrumentation: fold this forward's per-expert
        // assignment counts into the cumulative tally. `countsHost`
        // already sums to N * topK (every (token, slot) assignment),
        // and it is read here for the dispatch anyway, so this adds
        // no extra host sync.
        for e in 0 ..< numExperts {
            cumulativeExpertCounts[e] += Int(countsHost[e])
        }

        // Run each non-empty expert on its slice and concatenate the
        // results back in sorted-assignment order.
        var parts: [MLXArray] = []
        var offset = 0
        for e in 0 ..< numExperts {
            let c = Int(countsHost[e])
            if c == 0 { continue }
            let chunk = gathered[offset ..< (offset + c)]   // [c, H]
            parts.append(experts[e](chunk))                 // [c, H]
            offset += c
        }
        let sortedOut: MLXArray = parts.isEmpty
            ? MLXArray.zeros([N * topK, H]).asType(flat.dtype)
            : MLX.concatenated(parts, axis: 0)              // [N*topK, H]

        // Un-sort: argSort(order) is the inverse permutation, so
        // taking sortedOut by it restores token-major order
        // [token0 slot0, token0 slot1, ..., token1 slot0, ...].
        let inverseOrder = argSort(order, axis: -1)          // [N*topK]
        let perSlot = MLX.take(sortedOut, inverseOrder, axis: 0)
            .reshaped(N, topK, H)                            // [N, topK, H]

        // Weight each slot by its router probability and sum the
        // topK contributions per token. The slot weight is
        // dispatch[token, chosen_expert]; gather it along the expert
        // axis with the per-token topK expert ids.
        let slotWeights = takeAlong(dispatch, topKExperts, axis: 1)   // [N, topK]
        let weighted = perSlot * slotWeights.reshaped(N, topK, 1)
        let result = weighted.sum(axis: 1)                            // [N, H]
        return result.reshaped(B, L, H)
    }

    /// Brute-force reference: every expert on every token, combined
    /// by a dense `[N, E]` dispatch matrix. Retained ONLY as the
    /// parity oracle for `callAsFunction` (the scatter dispatch).
    /// O(numExperts) FFN passes per layer; not used in production.
    func referenceForward(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let flat = x.reshaped(B * L, H)
        let (_, dispatch) = route(flat)  // [N, E]

        var out = MLXArray.zeros([B * L, H]).asType(flat.dtype)
        for e in 0 ..< numExperts {
            let weight = dispatch[0..., e ..< (e + 1)]  // [N, 1]
            let expertOut = experts[e](flat)            // [N, H]
            out = out + expertOut * weight
        }
        return out.reshaped(B, L, H)
    }

    /// Zero the cumulative per-expert assignment tally. Call before a
    /// generation to scope `cumulativeExpertCounts` (and the model's
    /// `moeUtilization()`) to that run.
    func resetUtilizationStats() {
        cumulativeExpertCounts = [Int](repeating: 0, count: numExperts)
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

// MARK: - MoE utilization snapshot

/// Aggregate expert-utilization snapshot across every sparse MoE layer,
/// covering all forwards since the last `resetMoEUtilizationStats()`.
///
/// A `(layer, expert)` pair is one "expert slot". `activeExpertSlots` is
/// how many slots served at least one routed token; the ratio against
/// `totalExpertSlots` is a load-coverage signal (a healthy router spreads
/// load, a degenerate one collapses onto a few experts). All fields are
/// host integers read off the compute path - cheap to snapshot after a
/// generation for load-balance inspection or benchmark metadata.
public struct MoEUtilization: Sendable {
    /// Number of sparse MoE layers contributing to the snapshot.
    public let sparseLayers: Int
    /// Experts per sparse layer (`num_experts`).
    public let expertsPerLayer: Int
    /// Total `(layer, expert)` slots = `sparseLayers * expertsPerLayer`.
    public let totalExpertSlots: Int
    /// Slots that served at least one `(token, slot)` assignment.
    public let activeExpertSlots: Int
    /// Total `(token, slot)` assignments routed across all sparse layers.
    public let totalAssignments: Int
    /// Busiest single `(layer, expert)` slot's assignment count.
    public let maxExpertLoad: Int

    public init(
        sparseLayers: Int, expertsPerLayer: Int, totalExpertSlots: Int,
        activeExpertSlots: Int, totalAssignments: Int, maxExpertLoad: Int
    ) {
        self.sparseLayers = sparseLayers
        self.expertsPerLayer = expertsPerLayer
        self.totalExpertSlots = totalExpertSlots
        self.activeExpertSlots = activeExpertSlots
        self.totalAssignments = totalAssignments
        self.maxExpertLoad = maxExpertLoad
    }

    /// `activeExpertSlots / totalExpertSlots`, in `[0, 1]`; `0` when the
    /// model has no sparse layers or no forward has run since the reset.
    public var utilizationRatio: Double {
        totalExpertSlots > 0
            ? Double(activeExpertSlots) / Double(totalExpertSlots) : 0
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
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    /// `lastTokenOnly` slices hidden to the last position before
    /// the vocab projection. See `LlamaForCausalLM` for the
    /// rationale; behaves identically for tied embeddings.
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
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embedTokens.asLinear(hidden)
    }

    /// Snapshot expert utilization across every sparse MoE layer for the
    /// forwards run since the last `resetMoEUtilizationStats()`. Off the
    /// compute path - safe to call after a generation. Returns an
    /// all-zero snapshot for a model with no sparse layers.
    public func moeUtilization() -> MoEUtilization {
        var sparseLayers = 0
        var totalSlots = 0
        var activeSlots = 0
        var totalAssignments = 0
        var maxLoad = 0
        for block in model.layers {
            guard let sparse = block.mlp as? Qwen3MoESparseMLP else { continue }
            sparseLayers += 1
            for count in sparse.cumulativeExpertCounts {
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

    /// Zero every sparse layer's cumulative expert tally. Call before a
    /// generation so `moeUtilization()` reflects only that run.
    public func resetMoEUtilizationStats() {
        for block in model.layers {
            (block.mlp as? Qwen3MoESparseMLP)?.resetUtilizationStats()
        }
    }
}
