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

// MARK: - Quantized stacked switched linear (one projection, all experts)

/// One stacked quantized SwitchLinear inside the Qwen3 `SwitchGLU`
/// block. Holds the `[numExperts, outputDims, inputDims_packed]`
/// quantized weight plus per-expert scales and biases, and dispatches
/// across the chosen top-K experts in a single `gatherQuantizedMM`
/// call instead of a Swift `for` loop over per-expert matmuls.
///
/// This mirrors `Gemma4QuantizedSwitchedLinear` (PR #82). The earlier
/// Qwen3-MoE runtime used a scatter dispatch that walked the experts
/// in a Swift loop, forcing a per-layer host sync (the loop bounds
/// came from a CPU read of per-expert token counts); decoding one
/// token per step paid that sync once per layer and dominated the FFN
/// math, leaving 30B-A3B behind Ollama. `gather_qmm` keeps the whole
/// dispatch on the GPU and matches `mlx_lm/models/switch_layers.
/// QuantizedSwitchLinear` bit for bit.
///
/// Parameter layout matches mlx-community's packed Qwen3-MoE format
/// directly, so the loader binds `mlp.switch_mlp.{proj}.*` with no
/// per-expert unpacking:
///   - `weight: [E, O, I/(32/bits)]` int-packed
///   - `scales: [E, O, I/groupSize]`
///   - `biases: [E, O, I/groupSize]`
class Qwen3QuantizedSwitchedLinear: Module {
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

        // Pre-allocate the parameter tensors with the SAME shape the
        // mlx-community checkpoint ships so the loader's
        // `model.update(parameters:)` binds them by shape match. The
        // fill values are placeholders overwritten at load time.
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

    /// Per-token expert dispatch. `x` is shaped so the last two dims
    /// feed `gather_qmm`'s `[..., M, K]` matmul slot (the SwitchGLU
    /// caller expands to `[..., 1, 1, I]`); `indices` is `[..., K]`
    /// Int32 expert ids into the weight tensor's leading batch dim.
    /// - Parameter sortedIndices: When true the caller has pre-sorted
    ///   `indices` by expert id so MLX's gather kernel can use the
    ///   faster sorted-indices path (the prefill sort path).
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

/// Qwen3 experts as three stacked quantized switched linears
/// (`gate_proj`, `up_proj`, `down_proj`) plus the SwiGLU activation.
/// Mirrors mlx-lm's `switch_layers.SwitchGLU` (with `silu`, unlike
/// Gemma 4's GeGLU). The in-checkpoint key path
/// `switch_mlp.{proj}.{weight,scales,biases}` lines up with this
/// module hierarchy directly.
///
/// Forward:
///   1. Reshape `[N, H]` to `[N, 1, 1, H]` so each row participates in
///      `topK` expert matmuls (one per chosen expert).
///   2. `gate_proj` / `up_proj` via `gatherQuantizedMM` -> `[N, topK,
///      1, moeIntermediate]` in a single device kernel each.
///   3. SwiGLU activation: `silu(gate) * up`.
///   4. `down_proj` back to `[N, topK, 1, H]`.
///   5. Squeeze the M=1 axis to `[N, topK, H]`. The caller does the
///      topK weighted sum.
class Qwen3SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Qwen3QuantizedSwitchedLinear
    @ModuleInfo(key: "up_proj") var upProj: Qwen3QuantizedSwitchedLinear
    @ModuleInfo(key: "down_proj") var downProj: Qwen3QuantizedSwitchedLinear

    init(
        inputDims: Int, hiddenDims: Int, numExperts: Int,
        groupSize: Int, bits: Int
    ) {
        _gateProj = ModuleInfo(
            wrappedValue: Qwen3QuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "gate_proj")
        _upProj = ModuleInfo(
            wrappedValue: Qwen3QuantizedSwitchedLinear(
                inputDims: inputDims, outputDims: hiddenDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "up_proj")
        _downProj = ModuleInfo(
            wrappedValue: Qwen3QuantizedSwitchedLinear(
                inputDims: hiddenDims, outputDims: inputDims,
                numExperts: numExperts, groupSize: groupSize, bits: bits),
            key: "down_proj")
    }

    /// - Parameters:
    ///   - x: `[N, H]` flattened token activations.
    ///   - indices: `[N, topK]` Int32 expert ids per token (router
    ///     score order).
    /// - Returns: `[N, topK, H]` per-expert outputs; the caller does
    ///   the topK weighted sum.
    func callAsFunction(_ x: MLXArray, indices: MLXArray) -> MLXArray {
        let N = x.dim(0)
        let H = x.dim(1)
        let topK = indices.dim(indices.ndim - 1)

        // Prefill (many assignments): sort (token, expert) by expert id
        // so each expert's gather slice is contiguous and the gather_qmm
        // `sortedIndices` fast path applies, recovering long-prompt
        // throughput the unsorted dispatch regresses. Output is unsorted
        // back to (token, slot) order so the caller's weighted sum is
        // unchanged. See `MoESortPath.swift`.
        if moeShouldSort(n: N, topK: topK) {
            let (xs, idx, invOrder) = moeGatherSort(x, indices: indices)
            // xs: [N*topK, 1, H] -- one M=1 row per assignment, sorted.
            let xGate = gateProj(xs, indices: idx, sortedIndices: true)
            let xUp = upProj(xs, indices: idx, sortedIndices: true)
            let activated = silu(xGate) * xUp
            let out = downProj(activated, indices: idx, sortedIndices: true)
            // out: [N*topK, 1, H_out] -- unsort to [N, topK, H].
            return moeScatterUnsort(out, invOrder: invOrder, n: N, topK: topK)
        }

        // Decode / short prompts (assignments below the sort threshold):
        // expand to [N, 1, 1, H] so each (token, slot) sees an M=1 row
        // inside the gather. [N, 1] outer batch x [N, topK] indices ->
        // [N, topK, 1, H_out] per projection: no Swift loop, no
        // per-layer host sync, one device kernel per projection.
        let xExp = x.reshaped(N, 1, 1, H)
        let idx = indices.asType(.int32)

        let xGate = gateProj(xExp, indices: idx)
        let xUp = upProj(xExp, indices: idx)
        let activated = silu(xGate) * xUp
        let out = downProj(activated, indices: idx)

        // out: [N, topK, 1, H_out] -- squeeze the M=1 inner axis.
        return out.squeezed(axis: -2)
    }
}

// MARK: - Expert-utilization accumulator (off the compute path)

/// Reference box for the on-device per-expert assignment tally. Held
/// as a plain class (not an `MLXArray` property) so MLXNN's Mirror
/// walk does NOT treat it as a model parameter -- the counts must
/// never appear in the parameter cache or the strict-verify loader
/// would demand a checkpoint key for them.
final class ExpertCountAccumulator {
    var counts: MLXArray
    init(numExperts: Int) {
        counts = MLXArray.zeros([numExperts], dtype: .int32)
    }
}

// MARK: - Sparse MoE MLP (router + SwitchGLU experts)

/// Replaces the dense `QwenMLP` at MoE layers. Loaded keys:
///   - `mlp.gate.weight` (+ scales/biases when quantized) [E, hidden]
///   - `mlp.switch_mlp.{gate_proj,up_proj,down_proj}.{weight,scales,biases}`
///     stacked `[E, O, I_packed]` quantized expert tensors.
///
/// Forward pass (router):
///   1. Router logits = `gate(x)` -> `[N, E]`.
///   2. Top-K experts per token via the argSort+argSort rank trick
///      (no native top_k op in mlx-swift).
///   3. Softmax over the top-K logits only; optionally renormalize.
///
/// Forward pass (expert dispatch):
///   `Qwen3SwitchGLU` runs all top-K experts for every token in a
///   single `gatherQuantizedMM` per projection -- no Swift loop, no
///   per-layer host sync. The earlier scatter dispatch read per-expert
///   token counts on the host each layer to slice a sorted assignment
///   array; that host round-trip dominated decode and is gone (PR
///   mirroring Gemma 4's #82 SwitchGLU rewrite).
class Qwen3MoESparseMLP: Module {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: Qwen3SwitchGLU

    let numExperts: Int
    let topK: Int
    let hiddenSize: Int
    let normTopK: Bool

    /// On-device cumulative `(token, slot)` assignment tally `[E]`.
    /// Accumulated each forward from the routing output WITHOUT a host
    /// sync -- the SwitchGLU dispatch no longer needs the counts on the
    /// host, so reading them per layer would re-introduce exactly the
    /// stall this rewrite removed. Materialized to `[Int]` lazily in
    /// `cumulativeExpertCounts`, off the compute path. The engine
    /// resets before a generation and reads once after, so the
    /// accumulator's graph is bounded by the generation length.
    private let accumulator: ExpertCountAccumulator

    /// Per-expert assignment totals since the last
    /// `resetUtilizationStats()`. Reading this forces a single
    /// host sync of the small `[E]` accumulator; used only for
    /// expert-load reporting (`moeUtilization()`), never the forward.
    var cumulativeExpertCounts: [Int] {
        eval(accumulator.counts)
        return accumulator.counts.asArray(Int32.self).map(Int.init)
    }

    init(_ config: Qwen3MoEConfig) {
        self.numExperts = config.numExperts
        self.topK = config.numExpertsPerToken
        self.hiddenSize = config.hiddenSize
        self.normTopK = config.normTopKProb
        self.accumulator = ExpertCountAccumulator(numExperts: config.numExperts)

        // Router. A Linear; the loader's quantize pass converts it to
        // a QuantizedLinear when the checkpoint quantizes it (the
        // Qwen3-Coder checkpoint ships an 8-bit router via a per-module
        // override). The experts are born quantized below.
        _gate = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.numExperts, bias: false),
            key: "gate")

        // Expert quantization comes from the config's top-level
        // (groupSize, bits). On 30B-A3B that is 4-bit / group-64; the
        // 8-bit override applies only to `mlp.gate` (the router),
        // never the experts. Fall back to (64, 4) so unit-test
        // fixtures without a quantization block stay instantiable.
        let groupSize = config.quantization?.groupSize ?? 64
        let bits = config.quantization?.bits ?? 4
        _switchMLP = ModuleInfo(
            wrappedValue: Qwen3SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                groupSize: groupSize, bits: bits),
            key: "switch_mlp")
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

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let H = x.dim(2)
        let N = B * L
        let flat = x.reshaped(N, H)

        let (topKExperts, dispatch) = route(flat)  // [N, topK], [N, E]

        // Off-path utilization: per-expert assignment counts via a
        // one-hot sum over the routing output, accumulated ON DEVICE.
        // No `eval` / host read here -- materialized lazily in
        // `cumulativeExpertCounts`, so the decode path pays no sync.
        let expertRange = MLXArray(Int32(0) ..< Int32(numExperts))      // [E]
        let oneHot = topKExperts.reshaped(N * topK, 1) .== expertRange  // [N*topK, E]
        accumulator.counts = accumulator.counts + oneHot.asType(.int32).sum(axis: 0)

        // SwitchGLU: all topK experts per token in one gather_qmm per
        // projection. Returns [N, topK, H]; weight each slot by its
        // router probability and sum the topK contributions per token.
        let expertOut = switchMLP(flat, indices: topKExperts)          // [N, topK, H]
        let slotWeights = takeAlong(dispatch, topKExperts, axis: 1)     // [N, topK]
        let weighted = expertOut * slotWeights.reshaped(N, topK, 1)
        let result = weighted.sum(axis: 1)                             // [N, H]
        return result.reshaped(B, L, H)
    }

    /// Zero the cumulative per-expert assignment tally. Call before a
    /// generation to scope `cumulativeExpertCounts` (and the model's
    /// `moeUtilization()`) to that run.
    func resetUtilizationStats() {
        accumulator.counts = MLXArray.zeros([numExperts], dtype: .int32)
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
