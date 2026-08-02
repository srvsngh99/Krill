import Foundation
import MLX
import MLXNN
import KrillCache

// MARK: - Nanbeige Config

/// Configuration for the Nanbeige 4.2 family (`NanbeigeForCausalLM`, model_type
/// "nanbeige").
///
/// Block-for-block this is a Llama-shaped dense decoder - separate q/k/v/o
/// projections with no bias, a SwiGLU MLP, and the standard two-RMSNorm pre-norm
/// sandwich - with two deltas that matter:
///
/// 1. **`head_dim` is decoupled from `hidden_size`.** Nanbeige4.2-3B runs 48
///    query heads of 128 dims against a 3072-wide residual, so `q_proj` is
///    3072 -> 6144 and `o_proj` is 6144 -> 3072. The usual
///    `hidden_size / num_attention_heads` derivation gives 64 and mis-shapes
///    every attention projection, so `head_dim` is read explicitly.
///
/// 2. **`num_loops` > 1: a LOOPED transformer.** The full stack of
///    `num_hidden_layers` blocks is executed `num_loops` times over the same
///    weights, which buys depth (22 blocks x 2 = 44 effective layers) at 22
///    blocks' worth of parameters. Each execution gets its OWN KV cache slot -
///    the reference indexes the cache as `layer_idx + loop_idx * num_hidden_layers`
///    - so a looped model holds `num_loops * num_hidden_layers` caches, and that
///    (not `num_hidden_layers`) is what the engine must allocate. See
///    `NanbeigeForCausalLM.cacheCount`.
///
/// The upstream `configuration_nanbeige.py` also describes optional research
/// features - multi-head hyper-connections with depth attention, concatenated
/// n-gram embeddings, LoopSplit, and loop-shared KV. All of them default to off
/// and NONE are enabled in the released Nanbeige4.2-3B checkpoint (its weight map
/// carries only the standard Llama tensor set). This runtime implements the
/// shipped configuration and `validate()` REFUSES any checkpoint that turns one
/// of them on, rather than silently ignoring a flag and emitting garbage logits.
/// Decodable-only (not `Codable`): several stored flags exist purely so
/// `validate()` can refuse an unsupported checkpoint and do not map 1:1 onto
/// their CodingKeys, and nothing re-serializes a Nanbeige config.
public struct NanbeigeConfig: ModelConfig, Decodable, Sendable {
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

    /// Explicit per-head width. Nanbeige decouples this from `hiddenSize`.
    public let headDim: Int

    /// Number of times the whole layer stack is executed. 1 = a plain dense stack.
    public let numLoops: Int

    /// When false (the shipped default) the final `model.norm` is applied at the
    /// END OF EVERY loop, so its output feeds the next loop's first block. When
    /// true the norm is applied once, after the last loop. Identical when
    /// `numLoops == 1`.
    public let skipLoopFinalNorm: Bool

    /// `lm_head` reuses `embed_tokens` when true. False in the shipped checkpoint
    /// (it ships a distinct `lm_head.weight`).
    public let tieWordEmbeddings: Bool

    // Research-feature flags, decoded only so `validate()` can refuse them.
    let attentionBias: Bool
    let qkLayernorm: Bool
    let enableHyperConnection: Bool
    let enableDoubleLoopSplit: Bool
    let enableDepthAttention: Bool
    let loopShareKV: Bool
    let embNeighborNum: Int?
    let loopLossWeights: [Float]
    let hasRopeScaling: Bool

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
        case headDim = "head_dim"
        case numLoops = "num_loops"
        case skipLoopFinalNorm = "skip_loop_final_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case qkLayernorm = "qk_layernorm"
        case enableHyperConnection = "enable_hyper_connection"
        case enableDoubleLoopSplit = "enable_double_loop_split"
        case enableDepthAttention = "enable_depth_attention"
        case loopShareKV = "loop_share_kv"
        case embNeighborNum = "emb_neighbor_num"
        case loopLossWeights = "loop_loss_weights"
        case ropeScaling = "rope_scaling"
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
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000.0
        maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? 262_144
        quantization = try c.decodeIfPresent(QuantizationConfig.self, forKey: .quantization)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / numAttentionHeads)
        numLoops = try c.decodeIfPresent(Int.self, forKey: .numLoops) ?? 1
        skipLoopFinalNorm = try c.decodeIfPresent(Bool.self, forKey: .skipLoopFinalNorm) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        qkLayernorm = try c.decodeIfPresent(Bool.self, forKey: .qkLayernorm) ?? false
        enableHyperConnection = try c.decodeIfPresent(Bool.self, forKey: .enableHyperConnection)
            ?? false
        enableDoubleLoopSplit = try c.decodeIfPresent(Bool.self, forKey: .enableDoubleLoopSplit)
            ?? false
        enableDepthAttention = try c.decodeIfPresent(Bool.self, forKey: .enableDepthAttention)
            ?? false
        loopShareKV = try c.decodeIfPresent(Bool.self, forKey: .loopShareKV) ?? false
        embNeighborNum = try c.decodeIfPresent(Int.self, forKey: .embNeighborNum)
        loopLossWeights = try c.decodeIfPresent([Float].self, forKey: .loopLossWeights) ?? []
        // `rope_scaling` is explicitly null in the shipped config; any non-null
        // value selects a scaled RoPE variant this runtime does not implement.
        hasRopeScaling = c.contains(.ropeScaling)
            && !((try? c.decodeNil(forKey: .ropeScaling)) ?? true)
    }

    /// Convenience initializer for tests and manual construction.
    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        numAttentionHeads: Int,
        numKeyValueHeads: Int,
        numHiddenLayers: Int,
        vocabSize: Int,
        headDim: Int? = nil,
        numLoops: Int = 1,
        skipLoopFinalNorm: Bool = false,
        tieWordEmbeddings: Bool = false,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000.0,
        maxPositionEmbeddings: Int = 262_144,
        quantization: QuantizationConfig? = nil
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.numHiddenLayers = numHiddenLayers
        self.vocabSize = vocabSize
        self.headDim = headDim ?? (hiddenSize / numAttentionHeads)
        self.numLoops = numLoops
        self.skipLoopFinalNorm = skipLoopFinalNorm
        self.tieWordEmbeddings = tieWordEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.quantization = quantization
        self.attentionBias = false
        self.qkLayernorm = false
        self.enableHyperConnection = false
        self.enableDoubleLoopSplit = false
        self.enableDepthAttention = false
        self.loopShareKV = false
        self.embNeighborNum = nil
        self.loopLossWeights = []
        self.hasRopeScaling = false
    }

    /// The Llama-shaped view used to build the shared `TransformerBlock` /
    /// `Attention` / `FeedForward` leaves. Those are the same parity-gated blocks
    /// the Llama and Mistral runtimes use; carrying `headDim` through is what
    /// makes the decoupled-head geometry come out right.
    var blockConfig: LlamaConfig {
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
            quantization: quantization,
            headDim: headDim)
    }

    /// Reject a checkpoint that enables an upstream research feature this runtime
    /// does not implement. Loading such a checkpoint anyway would produce
    /// confidently wrong logits with no visible failure, so we fail at load.
    public func validate() throws {
        var unsupported: [String] = []
        if enableHyperConnection { unsupported.append("enable_hyper_connection (mHC)") }
        if enableDepthAttention { unsupported.append("enable_depth_attention") }
        if enableDoubleLoopSplit { unsupported.append("enable_double_loop_split (LoopSplit)") }
        if loopShareKV { unsupported.append("loop_share_kv") }
        if embNeighborNum != nil { unsupported.append("emb_neighbor_num (n-gram embeddings)") }
        if qkLayernorm { unsupported.append("qk_layernorm") }
        if attentionBias { unsupported.append("attention_bias") }
        if hasRopeScaling { unsupported.append("rope_scaling") }
        // A non-empty `loop_loss_weights` overrides `num_loops` upstream
        // (`num_loops = len(weights) + 1`). Refuse rather than guess.
        if !loopLossWeights.isEmpty { unsupported.append("loop_loss_weights") }

        guard unsupported.isEmpty else {
            throw ModelLoadError.unsupportedArchitecture(
                "Nanbeige checkpoint enables feature(s) with no native runtime: "
                + unsupported.joined(separator: ", ")
                + ". Krill implements the shipped Nanbeige 4.2 configuration "
                + "(looped dense decoder); these flags are off in that checkpoint.")
        }
        guard numLoops >= 1 else {
            throw ModelLoadError.invalidConfig("num_loops must be >= 1, got \(numLoops)")
        }
        guard numAttentionHeads % numKeyValueHeads == 0 else {
            throw ModelLoadError.invalidConfig(
                "num_attention_heads (\(numAttentionHeads)) must be a multiple of "
                + "num_key_value_heads (\(numKeyValueHeads))")
        }
    }
}

// MARK: - Inner Model (embed_tokens + looped layers + norm)

/// Inner Nanbeige model: embedding, the looped transformer stack, and the final
/// norm. Corresponds to the `model` key in HuggingFace weight naming - the layer
/// stack is stored ONCE (`model.layers.0..N-1`) and re-executed, so the weight
/// names are identical to a non-looped Llama of the same depth.
class NanbeigeModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let numLoops: Int
    let skipLoopFinalNorm: Bool

    init(_ config: NanbeigeConfig) {
        self.numLoops = config.numLoops
        self.skipLoopFinalNorm = config.skipLoopFinalNorm
        let block = config.blockConfig
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in TransformerBlock(block) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    /// Run the stack `numLoops` times.
    ///
    /// `caches` is indexed `loop * layerCount + layerIdx`, matching the reference's
    /// `layer_idx + loop_idx * num_hidden_layers`. Every cache advances by the same
    /// number of tokens on every forward, so they all share one sequence length -
    /// which is why a single mask (and the single RoPE offset `Attention` derives
    /// from its own cache) is correct for all loops.
    func callAsFunction(
        _ tokens: MLXArray,
        mask: MLXArray? = nil,
        caches: [KVCache]? = nil,
        rowOffsets: [Int]? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        var h = inputsEmbeds ?? embedTokens(tokens)

        let seqLen = h.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        // Additive mask in the ACTIVATION dtype: affine 4-bit dequants to fp16 and
        // nvfp4 to bf16, and SDPA requires the mask to promote to the output type.
        // A hardcoded fp16 mask fails the nvfp4 build with "Mask type must promote
        // to output type bfloat16" - and nvfp4 is the shipped Krill build.
        let effectiveMask = mask
            ?? createCachedCausalMask(newLen: seqLen, cacheLen: cacheLen, dtype: h.dtype)

        let layerCount = layers.count
        for loop in 0 ..< numLoops {
            for (i, layer) in layers.enumerated() {
                h = layer(
                    h, mask: effectiveMask, cache: caches?[loop * layerCount + i],
                    rowOffsets: rowOffsets)
            }
            // The reference norms at the end of EVERY loop unless
            // `skip_loop_final_norm`, so loop N+1 consumes a normed residual.
            if !skipLoopFinalNorm { h = norm(h) }
        }
        if skipLoopFinalNorm { h = norm(h) }
        return h
    }
}

// MARK: - NanbeigeForCausalLM (top-level: model + lm_head)

/// Complete Nanbeige model for causal language modeling.
///
/// Matches HuggingFace `NanbeigeForCausalLM`:
///   - `model.*` -> `NanbeigeModelInner`
///   - `lm_head.*` -> the vocab projection (untied in the shipped checkpoint)
public class NanbeigeForCausalLM: Module {
    @ModuleInfo(key: "model") var model: NanbeigeModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: NanbeigeConfig

    /// Number of KV caches the engine must allocate: one per layer PER LOOP.
    /// Allocating only `numHiddenLayers` would make loop 1 overwrite loop 0's
    /// keys and values and silently corrupt every decode step past the first.
    public var cacheCount: Int { config.numLoops * config.numHiddenLayers }

    public init(_ config: NanbeigeConfig) {
        self.config = config
        _model = ModuleInfo(wrappedValue: NanbeigeModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    public func callAsFunction(_ tokens: MLXArray, caches: [KVCache]? = nil) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    /// `lastTokenOnly` slices the hidden state to the final position before the
    /// vocab projection, dropping the `vocabSize * hidden` matmul over the unused
    /// prefix rows on prefill. Bit-exact for the sampled token - the KV caches are
    /// filled by the attention layers below the head. See `LlamaForCausalLM`.
    public func callAsFunction(
        _ tokens: MLXArray, caches: [KVCache]? = nil, lastTokenOnly: Bool
    ) -> MLXArray {
        var h = model(tokens, caches: caches)
        if lastTokenOnly {
            let last = h.dim(1) - 1
            h = h[0..., last ..< (last + 1), 0...]
        }
        return lmHead(h)
    }

    /// Batched ragged-decode step: one new token per row (`tokens` is `[R, 1]`),
    /// each rotated at its own next position (`rowOffsets[r]`), under the explicit
    /// per-row additive `mask`. Returns logits `[R, 1, vocab]`.
    public func batchedDecode(
        _ tokens: MLXArray, caches: [KVCache], mask: MLXArray, rowOffsets: [Int]
    ) -> MLXArray {
        lmHead(model(tokens, mask: mask, caches: caches, rowOffsets: rowOffsets))
    }
}
