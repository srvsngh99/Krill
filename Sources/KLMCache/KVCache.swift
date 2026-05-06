import MLX

/// Per-layer Key-Value cache for transformer attention.
///
/// Stores accumulated K/V tensors and grows along the sequence dimension
/// as tokens are generated. One KVCache instance per transformer layer.
public final class KVCache: @unchecked Sendable {
    private var _keys: MLXArray?
    private var _values: MLXArray?

    public init() {}

    /// Number of tokens currently cached.
    public var sequenceLength: Int {
        _keys?.dim(2) ?? 0
    }

    /// Append new key/value tensors and return the full accumulated K/V.
    ///
    /// - Parameters:
    ///   - keys:   New keys,  shape `[B, numKVHeads, seqLen, headDim]`
    ///   - values: New values, shape `[B, numKVHeads, seqLen, headDim]`
    /// - Returns: Tuple of full (keys, values) including the new tokens.
    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        if let existingK = _keys, let existingV = _values {
            let k = concatenated([existingK, newK], axis: 2)
            let v = concatenated([existingV, newV], axis: 2)
            _keys = k
            _values = v
            return (k, v)
        }
        _keys = newK
        _values = newV
        return (newK, newV)
    }

    /// Return a snapshot of the current KV arrays, or nil if no state has been cached yet.
    public func snapshot() -> (keys: MLXArray, values: MLXArray)? {
        guard let k = _keys, let v = _values else { return nil }
        return (k, v)
    }

    /// Restore KV state directly (used when replaying a prefix-cache hit).
    public func restore(keys: MLXArray, values: MLXArray) {
        _keys = keys
        _values = values
    }

    /// Truncate cached KV state to the given sequence length.
    ///
    /// Both keys and values use shape `[B, heads, seq, head_dim]` (seq on axis 2),
    /// matching the layout assumed by `update`.
    public func truncate(to sequenceLength: Int) {
        guard let k = _keys, let v = _values else { return }
        let currentLen = k.dim(2)
        guard sequenceLength < currentLen else { return }
        _keys = k[0..., 0..., 0 ..< sequenceLength, 0...]
        _values = v[0..., 0..., 0 ..< sequenceLength, 0...]
    }

    /// Discard all cached state for a new generation.
    public func reset() {
        _keys = nil
        _values = nil
    }
}

/// Create an array of empty KV caches, one per transformer layer.
public func makeKVCaches(numLayers: Int) -> [KVCache] {
    (0 ..< numLayers).map { _ in KVCache() }
}
