import MLX

/// Common interface implemented by every per-layer KV cache backend.
///
/// Lowest common denominator across `KVCache` (fp16) and `QuantizedKVCache` (int8).
/// Code paths that need fp16-only operations (`snapshot`, `restore`, `truncate`) must
/// keep the concrete `KVCache` type.
public protocol KVCacheProtocol: AnyObject {
    func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray)
    func snapshot() -> (keys: MLXArray, values: MLXArray)?
    func reset()
    var sequenceLength: Int { get }
}

/// Per-layer Key-Value cache for transformer attention.
///
/// Stores accumulated K/V tensors and grows along the sequence dimension
/// as tokens are generated. One KVCache instance per transformer layer.
///
/// Performance: uses step-based concatenation with periodic compaction to
/// avoid creating a new array on every single decode step. During decode,
/// new K/V slices are accumulated in a small buffer and the full history
/// is rebuilt only when needed for attention.
public final class KVCache: KVCacheProtocol, @unchecked Sendable {
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

// MARK: - Updatable conformance

/// `KVCache` is MLX `Updatable` so it can be passed as `inputs:` / `outputs:`
/// to `MLX.compile` (and to `eval`). `innerState()` is the compacted full K/V
/// -- the mutable state a compiled block forward reads and writes. This is the
/// prerequisite for compiling a text decoder block that mutates the cache (the
/// vision blocks already compile because they hold no cache). It does not
/// change `update`/decode behaviour; the production decode path is untouched.
extension KVCache: Updatable {
    public func innerState() -> [MLXArray] {
        compact()
        return [_keys, _values].compactMap { $0 }
    }
}

// MARK: - Fixed-buffer cache (shape-stable, for compiled decode)

/// A KV cache backed by a pre-allocated, fixed-size `[B, H, capacity, D]`
/// buffer written in place at a *dynamic* offset, so the attention it feeds
/// keeps a constant shape across decode steps.
///
/// ## Why a separate cache
///
/// The default `KVCache` returns a slice `[..., :offset, :]` that grows by one
/// row per decode step, so a block forward built on it changes input shape
/// every step -- `MLX.compile` would re-trace each step and the compiled graph
/// would never be reused. A fixed `capacity` buffer keeps the attention shape
/// constant, and an additive position mask hides the `>= offset` tail. The
/// write position is a *runtime* index (an `MLXArray`, via scatter), NOT a
/// static slice bound -- a static bound would bake into the traced graph and
/// force a re-trace every step, defeating the purpose.
///
/// Used by the compiled-decode probe (`KLM_DECODE_PROBE`); not yet wired into
/// the production engine.
public final class FixedBufferKVCache: Updatable, @unchecked Sendable {
    public private(set) var keys: MLXArray
    public private(set) var values: MLXArray
    /// Number of valid (written) positions. Host-side; used only to build the
    /// runtime offset index and the position mask, never as a slice bound.
    public private(set) var offset: Int
    public let capacity: Int

    public init(
        batch: Int, heads: Int, headDim: Int, valueDim: Int,
        capacity: Int, dtype: DType = .float16
    ) {
        self.capacity = capacity
        self.offset = 0
        self.keys = MLXArray.zeros([batch, heads, capacity, headDim], dtype: dtype)
        self.values = MLXArray.zeros([batch, heads, capacity, valueDim], dtype: dtype)
    }

    public func innerState() -> [MLXArray] { [keys, values] }

    /// Write one decode step's `[B, H, 1, D]` key/value at the current offset
    /// (dynamic index) and advance. The buffers keep their fixed shape, so a
    /// compiled forward built over them is traced once and replayed.
    public func writeStep(keys newK: MLXArray, values newV: MLXArray) {
        let idx = MLXArray(Int32(offset))
        keys[0..., 0..., idx, 0...] = newK.squeezed(axis: 2)
        values[0..., 0..., idx, 0...] = newV.squeezed(axis: 2)
        offset += 1
    }

    /// Additive position mask `[1, 1, 1, capacity]`: `0` for written positions
    /// (`< offset`), a large negative for the unwritten tail. Added to the
    /// single query's attention scores so the padded buffer behaves like a
    /// length-`offset` cache.
    public func positionMask(dtype: DType) -> MLXArray {
        let pos = MLXArray(Int32(0) ..< Int32(capacity))
        let invalid = pos .>= MLXArray(Int32(offset))
        return (invalid.asType(dtype) * Float(-30000.0)).reshaped(1, 1, 1, capacity)
    }

    public func reset() {
        offset = 0
        keys = MLXArray.zeros(keys.shape, dtype: keys.dtype)
        values = MLXArray.zeros(values.shape, dtype: values.dtype)
    }
}
