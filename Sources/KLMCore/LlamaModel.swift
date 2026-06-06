import MLX
import MLXNN
import KLMCache

// MARK: - Inner Model (embed_tokens + layers + norm)

/// Inner Llama model: embedding, transformer layers, and final norm.
/// Corresponds to the `model` key in HuggingFace weight naming.
class LlamaModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [TransformerBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: LlamaConfig) {
        _embedTokens = ModuleInfo(
            wrappedValue: Embedding(
                embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        _layers = ModuleInfo(
            wrappedValue: (0 ..< config.numHiddenLayers).map { _ in TransformerBlock(config) },
            key: "layers")
        _norm = ModuleInfo(
            wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")
    }

    func callAsFunction(
        _ tokens: MLXArray,
        mask: MLXArray? = nil,
        caches: [KVCache]? = nil,
        rowOffsets: [Int]? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        // VLMs on a Llama backbone (e.g. LLaVA) merge projected image features
        // into the token embeddings and forward from there; `inputsEmbeds`
        // bypasses the token embedding lookup. Default nil keeps the standard
        // token-id path unchanged.
        var x = inputsEmbeds ?? embedTokens(tokens)

        // Create causal mask. For an empty-cache prefill this is the
        // classic (seqLen, seqLen) lower-triangular mask. For multi-token
        // forward against a non-empty cache (e.g. speculative-decode
        // verify) the mask must extend across the cached prefix too.
        // On the batched ragged-decode path an explicit per-row mask is
        // passed in (and rowOffsets carries each row's position).
        let seqLen = x.dim(1)
        let cacheLen = caches?.first?.sequenceLength ?? 0
        let effectiveMask: MLXArray?
        if let mask {
            effectiveMask = mask
        } else {
            effectiveMask = createCachedCausalMask(
                newLen: seqLen, cacheLen: cacheLen)
        }

        for (i, layer) in layers.enumerated() {
            let cache = caches?[i]
            x = layer(x, mask: effectiveMask, cache: cache, rowOffsets: rowOffsets)
        }

        return norm(x)
    }
}

// MARK: - LlamaForCausalLM (top-level: model + lm_head)

/// Complete Llama model for causal language modeling.
///
/// Matches HuggingFace LlamaForCausalLM structure:
///   - `model.*` keys map to LlamaModelInner
///   - `lm_head.*` maps to the projection head
public class LlamaForCausalLM: Module {
    @ModuleInfo(key: "model") var model: LlamaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: LlamaConfig

    public init(_ config: LlamaConfig) {
        self.config = config
        _model = ModuleInfo(
            wrappedValue: LlamaModelInner(config), key: "model")
        _lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false),
            key: "lm_head")
    }

    /// Forward pass returning logits.
    ///
    /// - Parameters:
    ///   - tokens: Input token IDs, shape `[B, seqLen]`
    ///   - caches: Optional array of KVCache (one per layer)
    /// - Returns: Logits, shape `[B, seqLen, vocabSize]`
    public func callAsFunction(
        _ tokens: MLXArray,
        caches: [KVCache]? = nil
    ) -> MLXArray {
        callAsFunction(tokens, caches: caches, lastTokenOnly: false)
    }

    /// Same forward pass, but slices the transformer output to the
    /// last position before the vocab projection when
    /// `lastTokenOnly` is true. The sampler reads from the last
    /// position only, so on prefill the
    /// `[1, L, hidden] -> [1, L, vocab]` matmul wastes work over
    /// ~L-1 unused rows. The KV cache is filled by the attention
    /// layers above the head, so the sampled token is bit-exact.
    /// Decode steps forward a single token so the slice is a no-op.
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
        return lmHead(hidden)
    }

    /// Forward from pre-merged input embeddings (a VLM path: e.g. LLaVA splices
    /// projected image features into the token embeddings, then runs the text
    /// stack from those). `tokens` is still passed for cache/mask shaping but
    /// its embeddings are not used. Returns logits `[B, L, vocab]`.
    public func callAsFunction(
        _ tokens: MLXArray, inputsEmbeds: MLXArray, caches: [KVCache]? = nil,
        lastTokenOnly: Bool = false
    ) -> MLXArray {
        var hidden = model(tokens, caches: caches, inputsEmbeds: inputsEmbeds)
        if lastTokenOnly {
            let last = hidden.dim(1) - 1
            hidden = hidden[0..., last ..< (last + 1), 0...]
        }
        return lmHead(hidden)
    }

    /// Batched ragged-decode step (Stage B): one new token per row
    /// (`tokens` is `[R, 1]`), each rotated at its own next position
    /// (`rowOffsets[r]`), attending under the explicit per-row additive
    /// `mask` that hides each row's left-padded prefix in the stacked cache.
    /// Returns logits `[R, 1, vocab]`.
    public func batchedDecode(
        _ tokens: MLXArray, caches: [KVCache], mask: MLXArray, rowOffsets: [Int]
    ) -> MLXArray {
        let hidden = model(tokens, mask: mask, caches: caches, rowOffsets: rowOffsets)
        return lmHead(hidden)
    }
}

// MARK: - Causal Mask

/// Create an additive causal mask for self-attention during prefill.
///
/// Returns a mask of shape `[1, 1, N, N]` where positions that should
/// NOT be attended to have a large negative value and valid positions have 0.
public func createAdditiveCausalMask(_ n: Int, dtype: DType = .float16) -> MLXArray {
    let indices = MLXArray(0 ..< Int32(n))
    // mask[i, j] = true when j > i (future positions)
    let mask = expandedDimensions(indices, axis: 1) .< expandedDimensions(indices, axis: 0)
    // Convert bool mask to additive: true -> large negative, false -> 0
    // Dtype matches the model's compute type (fp16 for Llama, bf16 for Gemma 4)
    let additive = mask.asType(dtype) * Float(-10000.0)
    // Add batch and head dimensions: [1, 1, N, N]
    return expandedDimensions(expandedDimensions(additive, axis: 0), axis: 0)
}

/// Build the right additive causal mask for a forward of `newLen` tokens
/// against a KV cache that already holds `cacheLen` tokens.
///
/// Returns `nil` for the single-token decode case (no mask needed: the one
/// new query attends freely to all cached positions, and there is nothing
/// to mask within the new slice).
///
/// For empty-cache multi-token prefill (`cacheLen == 0`), returns the
/// classic square `(1, 1, newLen, newLen)` causal mask.
///
/// For non-empty-cache multi-token forward (`cacheLen > 0`, `newLen > 1`)
/// (the case that occurs during speculative-decode verify and after a
/// partial prefix-cache resume) returns a `(1, 1, newLen, cacheLen + newLen)`
/// mask. The first `cacheLen` columns are zero (new queries can attend to
/// every cached position) and the last `newLen` columns are upper-
/// triangular (row `i` may attend within the new slice only up to its own
/// position). Without this shape, the attention softmax tries to broadcast
/// an `(newLen, newLen)` mask against an `(newLen, cacheLen + newLen)`
/// score matrix and the runtime errors with a shape mismatch.
public func createCachedCausalMask(
    newLen: Int, cacheLen: Int, dtype: DType = .float16
) -> MLXArray? {
    if newLen <= 1 { return nil }
    if cacheLen == 0 {
        return createAdditiveCausalMask(newLen, dtype: dtype)
    }
    let queryIdx = MLXArray(Int32(cacheLen) ..< Int32(cacheLen + newLen))
    let keyIdx = MLXArray(0 ..< Int32(cacheLen + newLen))
    // mask[i, j] = true when key position j is in the future of query
    // position (cacheLen + i). For cached keys j < cacheLen this is
    // always false; for j >= cacheLen it forms the within-slice causal
    // upper triangle.
    let mask = expandedDimensions(queryIdx, axis: 1) .< expandedDimensions(keyIdx, axis: 0)
    let additive = mask.asType(dtype) * Float(-10000.0)
    return expandedDimensions(expandedDimensions(additive, axis: 0), axis: 0)
}

/// Additive `[1, 1, newLen, cacheLen+newLen]` mask for a CAUSAL SLIDING-WINDOW
/// layer (Gemma family): a query at absolute position `q = cacheLen + i` may
/// attend a key at absolute position `k` iff it is causal (`k <= q`) AND inside
/// the window (`q - k < window`) - mlx-lm's convention (each token sees itself
/// plus the previous `window - 1`).
///
/// When the farthest causal lookback in this forward (`cacheLen + newLen - 1`)
/// already fits inside the window, the window never bites, so this is exactly
/// the plain causal mask - we delegate to `createCachedCausalMask` so SHORT
/// prompts (and a single decode token before the cache reaches `window`) are
/// byte-identical to the non-windowed path. Returns nil only in that delegated
/// `newLen <= 1` case.
public func createSlidingWindowCausalMask(
    newLen: Int, cacheLen: Int, window: Int, dtype: DType = .float16
) -> MLXArray? {
    if cacheLen + newLen <= window {
        return createCachedCausalMask(newLen: newLen, cacheLen: cacheLen, dtype: dtype)
    }
    let total = cacheLen + newLen
    let q = MLXArray(Int32(cacheLen) ..< Int32(total)).reshaped(newLen, 1)  // [L, 1]
    let k = MLXArray(Int32(0) ..< Int32(total)).reshaped(1, total)          // [1, total]
    let causal = k .<= q                                  // k <= q
    let within = (q - k) .< MLXArray(Int32(window))       // q - k < window
    let allowed = causal .&& within
    return MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-10000.0)))
        .reshaped(1, 1, newLen, total).asType(dtype)
}
