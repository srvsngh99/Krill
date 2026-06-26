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

/// The fp16 cache operations the engine's prefix-cache and speculative-decode
/// paths need beyond `KVCacheProtocol`: restoring a stored span and dropping a
/// tail. Conformers: `KVCache` (full history - restore/truncate are total) and
/// `RotatingKVCache` (windowed - only the retained tail can be truncated, so
/// callers must consult `canTruncate` and fall back to a cold prefill).
public protocol RestorableKVCache: KVCacheProtocol {
    /// Restore a stored span whose LAST row sat at absolute position
    /// `totalSeen - 1`. For a full-history cache `keys.dim(2) == totalSeen`;
    /// for a rotating cache the span may be just the retained window tail.
    func restore(keys: MLXArray, values: MLXArray, totalSeen: Int)
    /// Whether `truncate(to: n)` can be honored exactly.
    func canTruncate(to n: Int) -> Bool
    /// Drop rows so the cache covers absolute positions `0 ..< n`.
    func truncate(to n: Int)
}

/// Which cache implementation a model layer needs. Families with uniform
/// full-history attention use `.standard` everywhere (the default); families
/// mixing sliding-window layers (Gemma 4) mark those `.rotating` so decode
/// reads O(window) KV instead of O(context).
public enum KVCacheKind: Sendable, Equatable {
    case standard
    case rotating(window: Int)
    /// GatedDeltaNet linear-attention layer: a conv-state + SSM recurrent-state
    /// cache (`GatedDeltaCache`), not a key/value cache. Used by qwen3_5-class
    /// hybrid models for their linear layers; full-attention layers stay `.standard`.
    case ssm
}

/// Per-layer cache for a GatedDeltaNet (linear-attention) layer. Holds the
/// causal-conv state and the `[B, Hv, Dv, Dk]` SSM recurrent state — NOT keys
/// and values. Conforms to `RestorableKVCache` only so it can ride in the
/// engine's `[RestorableKVCache]` array; the KV-style operations are inert
/// (`snapshot` nil, not truncatable) so the engine cold-prefills these layers
/// rather than seeding them from the prefix cache. The model layer reads/writes
/// `convState`/`ssmState` directly after downcasting.
public final class GatedDeltaCache: RestorableKVCache, @unchecked Sendable {
    public var convState: MLXArray?
    public var ssmState: MLXArray?
    private var seq = 0

    public init() {}

    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("GatedDeltaCache: linear-attention layers do not use KV update")
    }
    public func snapshot() -> (keys: MLXArray, values: MLXArray)? { nil }
    public func reset() { convState = nil; ssmState = nil; seq = 0 }
    public var sequenceLength: Int { seq }
    public func advance(_ n: Int) { seq += n }

    // RestorableKVCache: inert — SSM state is not addressable by position, so the
    // engine treats these layers as cold-prefill-only.
    public func restore(keys: MLXArray, values: MLXArray, totalSeen: Int) {}
    public func canTruncate(to n: Int) -> Bool { false }
    public func truncate(to n: Int) {}
}

/// Per-layer Key-Value cache for transformer attention.
///
/// Stores accumulated K/V tensors and grows along the sequence dimension
/// as tokens are generated. One KVCache instance per transformer layer.
///
/// Performance: a preallocated `[B, H, capacity, D]` buffer written in place
/// at `offset` (mlx-lm's KVCache pattern). `update()` slice-assigns the new
/// rows and returns `[0 ..< offset]` sliced views - amortized O(1) per step.
/// The previous design concatenated `[existing] + pending` on EVERY step (a
/// full O(context) copy per layer per step), which dominated decode time at
/// long context. `truncate(to:)` is now O(1) (it just lowers `offset`).
///
/// Aliasing contract (MLX subscript-assign rebinds this instance's handle, so
/// writes never mutate arrays other code holds - but a held reference blocks
/// MLX's buffer donation and forces full-buffer copies):
///  - `snapshot()` returns cheap lazy slice VIEWS. Hot per-step readers (the
///    Gemma KV-shared donor path, batched stacking) consume them immediately.
///    Long-lived holders (the prefix cache) must materialize - see
///    `PrefixCache.store`, which wraps entries in `contiguous(...)`.
///  - `restore()` copies into a fresh buffer rather than adopting the caller's
///    instance, so later in-place writes can never touch the cache entry.
public final class KVCache: KVCacheProtocol, @unchecked Sendable {
    private var _keys: MLXArray?    // [B, H, capacity, Dk]
    private var _values: MLXArray?  // [B, H, capacity, Dv]
    /// Valid rows in the buffers; rows past `offset` are stale/zero.
    private var offset = 0

    /// Buffers grow in multiples of this (plus a proportional term in
    /// `grownCapacity` so very long contexts stay amortized-cheap).
    private static let growthStep = 256

    public init() {}

    /// Number of tokens currently cached.
    public var sequenceLength: Int { offset }

    private static func grownCapacity(needed: Int) -> Int {
        // At least +25% headroom over what's needed, rounded up to the step.
        let withHeadroom = needed + Swift.max(Self.growthStep, needed / 4)
        return ((withHeadroom + Self.growthStep - 1) / Self.growthStep) * Self.growthStep
    }

    /// Reallocate to `capacity >= needed`, copying the valid prefix.
    private func grow(toAtLeast needed: Int, likeK: MLXArray, likeV: MLXArray) {
        let cap = Self.grownCapacity(needed: needed)
        let newK = MLXArray.zeros(
            [likeK.dim(0), likeK.dim(1), cap, likeK.dim(3)], dtype: likeK.dtype)
        let newV = MLXArray.zeros(
            [likeV.dim(0), likeV.dim(1), cap, likeV.dim(3)], dtype: likeV.dtype)
        if let oldK = _keys, let oldV = _values, offset > 0 {
            newK[0..., 0..., 0 ..< offset, 0...] = oldK[0..., 0..., 0 ..< offset, 0...]
            newV[0..., 0..., 0 ..< offset, 0...] = oldV[0..., 0..., 0 ..< offset, 0...]
        }
        _keys = newK
        _values = newV
    }

    /// Append new key/value tensors and return the full accumulated K/V.
    ///
    /// - Parameters:
    ///   - keys:   New keys,  shape `[B, numKVHeads, seqLen, headDim]`
    ///   - values: New values, shape `[B, numKVHeads, seqLen, headDim]`
    /// - Returns: Tuple of full (keys, values) including the new tokens
    ///            (sliced views over the backing buffer).
    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        let L = newK.dim(2)
        let needed = offset + L
        if _keys == nil || needed > _keys!.dim(2) {
            grow(toAtLeast: needed, likeK: newK, likeV: newV)
        }
        _keys![0..., 0..., offset ..< needed, 0...] = newK
        _values![0..., 0..., offset ..< needed, 0...] = newV
        offset = needed
        return (_keys![0..., 0..., 0 ..< offset, 0...],
                _values![0..., 0..., 0 ..< offset, 0...])
    }

    /// Return a snapshot of the current KV arrays, or nil if no state has been
    /// cached yet. The returned arrays are LAZY SLICE VIEWS over the backing
    /// buffer - cheap for per-step readers, but long-lived holders must
    /// materialize (`contiguous`) their own copy (see class doc).
    public func snapshot() -> (keys: MLXArray, values: MLXArray)? {
        guard offset > 0, let k = _keys, let v = _values else { return nil }
        return (k[0..., 0..., 0 ..< offset, 0...],
                v[0..., 0..., 0 ..< offset, 0...])
    }

    /// Restore KV state (used when replaying a prefix-cache hit). Copies into
    /// a fresh buffer: adopting the caller's MLXArray instance would let later
    /// in-place `update()` writes mutate the prefix-cache entry's array.
    public func restore(keys: MLXArray, values: MLXArray) {
        let len = keys.dim(2)
        offset = 0
        _keys = nil
        _values = nil
        guard len > 0 else { return }
        grow(toAtLeast: len, likeK: keys, likeV: values)
        _keys![0..., 0..., 0 ..< len, 0...] = keys
        _values![0..., 0..., 0 ..< len, 0...] = values
        offset = len
    }

    /// Truncate cached KV state to the given sequence length. O(1): rows past
    /// `offset` are simply treated as stale and overwritten by later updates.
    public func truncate(to sequenceLength: Int) {
        guard sequenceLength < offset else { return }
        offset = Swift.max(0, sequenceLength)
    }

    /// Discard all cached state for a new generation.
    public func reset() {
        _keys = nil
        _values = nil
        offset = 0
    }
}

/// Create an array of empty KV caches, one per transformer layer.
public func makeKVCaches(numLayers: Int) -> [KVCache] {
    (0 ..< numLayers).map { _ in KVCache() }
}

/// Create per-layer caches from a per-layer spec. `nil` (or a spec of the
/// wrong length) falls back to uniform `.standard`, so families that never
/// declare a spec are byte-for-byte unchanged.
public func makeKVCaches(spec: [KVCacheKind]?, numLayers: Int) -> [RestorableKVCache] {
    guard let spec, spec.count == numLayers else {
        return makeKVCaches(numLayers: numLayers)
    }
    return spec.map { kind in
        switch kind {
        case .standard: return KVCache()
        case .rotating(let window): return RotatingKVCache(window: window)
        case .ssm: return GatedDeltaCache()
        }
    }
}

extension KVCache: RestorableKVCache {
    /// Full-history restore: the stored span IS positions `0 ..< totalSeen`.
    public func restore(keys: MLXArray, values: MLXArray, totalSeen: Int) {
        restore(keys: keys, values: values)
    }

    /// A full-history cache can truncate to any non-negative length.
    public func canTruncate(to n: Int) -> Bool { n >= 0 }
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
        guard let k = snapshot() else { return [] }
        return [k.keys, k.values]
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
/// Used by the compiled-decode probe (`KRILL_DECODE_PROBE`); not yet wired into
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
