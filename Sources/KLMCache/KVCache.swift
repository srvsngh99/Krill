import MLX

/// Per-layer Key-Value cache for transformer attention.
///
/// Stores accumulated K/V tensors and grows along the sequence dimension
/// as tokens are generated. One KVCache instance per transformer layer.
///
/// Performance: uses step-based concatenation with periodic compaction to
/// avoid creating a new array on every single decode step. During decode,
/// new K/V slices are accumulated in a small buffer and the full history
/// is rebuilt only when needed for attention.
public final class KVCache: @unchecked Sendable {
    private var _keys: MLXArray?
    private var _values: MLXArray?

    /// Pending K/V slices not yet merged into _keys/_values.
    /// Batching concatenation reduces per-token allocation overhead.
    private var _pendingKeys: [MLXArray] = []
    private var _pendingValues: [MLXArray] = []

    /// Compact pending slices after this many accumulations.
    private static let compactThreshold = 8

    public init() {}

    /// Number of tokens currently cached (including pending).
    public var sequenceLength: Int {
        let baseLen = _keys?.dim(2) ?? 0
        let pendingLen = _pendingKeys.reduce(0) { $0 + $1.dim(2) }
        return baseLen + pendingLen
    }

    /// Merge any pending slices into the main K/V arrays.
    private func compact() {
        guard !_pendingKeys.isEmpty else { return }

        if let existingK = _keys, let existingV = _values {
            var allK = [existingK] + _pendingKeys
            var allV = [existingV] + _pendingValues
            _keys = concatenated(allK, axis: 2)
            _values = concatenated(allV, axis: 2)
        } else if _pendingKeys.count == 1 {
            _keys = _pendingKeys[0]
            _values = _pendingValues[0]
        } else {
            _keys = concatenated(_pendingKeys, axis: 2)
            _values = concatenated(_pendingValues, axis: 2)
        }
        _pendingKeys.removeAll(keepingCapacity: true)
        _pendingValues.removeAll(keepingCapacity: true)
    }

    /// Append new key/value tensors and return the full accumulated K/V.
    ///
    /// - Parameters:
    ///   - keys:   New keys,  shape `[B, numKVHeads, seqLen, headDim]`
    ///   - values: New values, shape `[B, numKVHeads, seqLen, headDim]`
    /// - Returns: Tuple of full (keys, values) including the new tokens.
    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        _pendingKeys.append(newK)
        _pendingValues.append(newV)

        // Compact periodically to bound memory from dangling references
        if _pendingKeys.count >= Self.compactThreshold {
            compact()
        }

        // Return the full K/V (compacted + pending) for attention
        if let existingK = _keys, let existingV = _values {
            if _pendingKeys.isEmpty {
                return (existingK, existingV)
            }
            let allK = concatenated([existingK] + _pendingKeys, axis: 2)
            let allV = concatenated([existingV] + _pendingValues, axis: 2)
            return (allK, allV)
        } else {
            if _pendingKeys.count == 1 {
                return (_pendingKeys[0], _pendingValues[0])
            }
            let allK = concatenated(_pendingKeys, axis: 2)
            let allV = concatenated(_pendingValues, axis: 2)
            return (allK, allV)
        }
    }

    /// Return a snapshot of the current KV arrays, or nil if no state has been cached yet.
    public func snapshot() -> (keys: MLXArray, values: MLXArray)? {
        compact()
        guard let k = _keys, let v = _values else { return nil }
        return (k, v)
    }

    /// Restore KV state directly (used when replaying a prefix-cache hit).
    public func restore(keys: MLXArray, values: MLXArray) {
        _keys = keys
        _values = values
        _pendingKeys.removeAll(keepingCapacity: true)
        _pendingValues.removeAll(keepingCapacity: true)
    }

    /// Truncate cached KV state to the given sequence length.
    ///
    /// Both keys and values use shape `[B, heads, seq, head_dim]` (seq on axis 2),
    /// matching the layout assumed by `update`.
    public func truncate(to sequenceLength: Int) {
        compact()
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
        _pendingKeys.removeAll(keepingCapacity: true)
        _pendingValues.removeAll(keepingCapacity: true)
    }
}

/// Create an array of empty KV caches, one per transformer layer.
public func makeKVCaches(numLayers: Int) -> [KVCache] {
    (0 ..< numLayers).map { _ in KVCache() }
}
