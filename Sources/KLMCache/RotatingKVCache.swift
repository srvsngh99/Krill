import MLX

/// Per-layer KV cache for SLIDING-WINDOW attention layers: retains only the
/// last `window - 1` tokens (plus the current step's appends) instead of the
/// full context.
///
/// ## Why
///
/// A sliding-window layer's query attends itself + the previous `window - 1`
/// keys (`q - k < window` in `createSlidingWindowCausalMask`). Storing the
/// full context and masking the rest out - what `KVCache` does - makes every
/// decode step read O(context) KV per layer when O(window) suffices. On
/// Gemma 4 12B that is 40 of 48 layers reading the whole history for nothing:
/// the dominant term in the long-context decode slowdown.
///
/// ## Retention invariant
///
/// `update()` lazily trims the retained span to `window - 1` tokens BEFORE
/// appending. After appending L tokens the retained set is exactly the keys
/// the newest query may attend (`window` total at L=1), so decode needs NO
/// sliding mask at all - and a chunked-prefill update (L possibly > window)
/// needs only the standard sliding mask over `[retained_old + L]` with
/// `cacheLen = maskCacheLength`, because relative distances are preserved.
///
/// ## Layout
///
/// Temporal-order contiguous buffer (NOT a ring): `[B, H, capacity, D]` with
/// `start ..< end` valid, grown/compacted like `KVCache`. Temporal order keeps
/// `snapshot()` directly usable by the Gemma KV-shared donor read and the
/// prefix cache without an unrotate step; the cost is an amortized O(window)
/// front-compaction every ~capacity rollover, which is negligible.
///
/// RoPE positions are absolute and applied before caching, so dropping old
/// rows never invalidates retained ones; `sequenceLength` reports TOTAL tokens
/// seen (not retained) because RoPE offsets, the full-attention mask, and the
/// KV-shared `donorLen - L` fix all key off it.
///
/// Aliasing contract matches `KVCache`: `snapshot()` returns lazy views for
/// hot per-step readers; `restore` copies into a fresh buffer.
public final class RotatingKVCache: RestorableKVCache, @unchecked Sendable {
    /// The layer's sliding window (e.g. Gemma 4's 1024).
    public let window: Int

    private var _keys: MLXArray?    // [B, H, capacity, Dk]
    private var _values: MLXArray?  // [B, H, capacity, Dv]
    private var start = 0           // first valid row
    private var end = 0             // one past last valid row
    private var totalSeen = 0       // absolute tokens ever appended

    private static let growthStep = 256

    public init(window: Int) {
        precondition(window > 1, "RotatingKVCache needs window > 1")
        self.window = window
    }

    /// TOTAL tokens ever appended - NOT the retained count. RoPE offsets, the
    /// full-attention causal mask, and the Gemma KV-shared donor offset all
    /// derive absolute positions from this.
    public var sequenceLength: Int { totalSeen }

    /// Rows currently retained in the buffer.
    public var retainedLength: Int { end - start }

    /// The `cacheLen` the sliding mask must use for an L>1 forward: the
    /// retained length AFTER the trim the next `update()` performs.
    public var maskCacheLength: Int { Swift.min(retainedLength, window - 1) }

    /// Trim the retained span so at most `window - 1` rows precede the next
    /// append. O(1): just advances `start`.
    private func trimForAppend() {
        let excess = retainedLength - (window - 1)
        if excess > 0 { start += excess }
    }

    /// Ensure capacity for `extra` more rows past `end`, compacting the
    /// retained span to the front of a (possibly larger) buffer when needed.
    private func ensureCapacity(extra: Int, likeK: MLXArray, likeV: MLXArray) {
        let capacity = _keys?.dim(2) ?? 0
        if end + extra <= capacity { return }
        let retained = retainedLength
        let needed = retained + extra
        // Keep headroom so steady-state decode compacts only every ~growthStep
        // appends, not every step once the buffer first fills.
        let newCap = ((needed + Self.growthStep + Self.growthStep - 1)
            / Self.growthStep) * Self.growthStep
        let newK = MLXArray.zeros(
            [likeK.dim(0), likeK.dim(1), newCap, likeK.dim(3)], dtype: likeK.dtype)
        let newV = MLXArray.zeros(
            [likeV.dim(0), likeV.dim(1), newCap, likeV.dim(3)], dtype: likeV.dtype)
        if let oldK = _keys, let oldV = _values, retained > 0 {
            newK[0..., 0..., 0 ..< retained, 0...] = oldK[0..., 0..., start ..< end, 0...]
            newV[0..., 0..., 0 ..< retained, 0...] = oldV[0..., 0..., start ..< end, 0...]
        }
        _keys = newK
        _values = newV
        end = retained
        start = 0
    }

    /// Append new K/V and return the retained window INCLUDING the new rows
    /// (`[B, H, retained_old + L, D]` sliced views, temporal order).
    public func update(keys newK: MLXArray, values newV: MLXArray) -> (MLXArray, MLXArray) {
        trimForAppend()
        let L = newK.dim(2)
        ensureCapacity(extra: L, likeK: newK, likeV: newV)
        _keys![0..., 0..., end ..< (end + L), 0...] = newK
        _values![0..., 0..., end ..< (end + L), 0...] = newV
        end += L
        totalSeen += L
        return (_keys![0..., 0..., start ..< end, 0...],
                _values![0..., 0..., start ..< end, 0...])
    }

    /// Retained rows (lazy views, temporal order), or nil when empty. Cheap -
    /// the Gemma KV-shared donor path calls this every forward. Long-lived
    /// holders (prefix cache) materialize their own copy in `PrefixCache.store`.
    public func snapshot() -> (keys: MLXArray, values: MLXArray)? {
        guard retainedLength > 0, let k = _keys, let v = _values else { return nil }
        return (k[0..., 0..., start ..< end, 0...],
                v[0..., 0..., start ..< end, 0...])
    }

    /// Whether `truncate(to: n)` can be honored: only the retained tail can be
    /// dropped; positions older than `totalSeen - retainedLength` are gone.
    /// Callers (prefix-cache partial reuse) must check this and fall back to a
    /// cold prefill when it is false.
    public func canTruncate(to n: Int) -> Bool {
        n >= totalSeen - retainedLength && n <= totalSeen
    }

    /// Drop the newest `totalSeen - n` rows (the spec-decode rollback /
    /// prefix-restore trim pattern). Precondition: `canTruncate(to: n)`.
    public func truncate(to n: Int) {
        guard n < totalSeen else { return }
        precondition(canTruncate(to: n),
            "RotatingKVCache.truncate(to: \(n)) below retained range "
            + "(retained \(totalSeen - retainedLength) ..< \(totalSeen))")
        let drop = totalSeen - n
        end -= drop
        totalSeen = n
    }

    /// Restore a retained span whose LAST row sat at absolute position
    /// `totalSeen - 1`. Copies into a fresh buffer (never adopts the caller's
    /// arrays - see `KVCache.restore`). `keys.dim(2)` may be smaller than
    /// `totalSeen` (a rotated snapshot stores only the window tail).
    public func restore(keys: MLXArray, values: MLXArray, totalSeen: Int) {
        let len = keys.dim(2)
        precondition(len <= totalSeen,
            "restored span (\(len)) longer than totalSeen (\(totalSeen))")
        reset()
        guard len > 0 else {
            self.totalSeen = totalSeen
            return
        }
        ensureCapacity(extra: len, likeK: keys, likeV: values)
        _keys![0..., 0..., 0 ..< len, 0...] = keys
        _values![0..., 0..., 0 ..< len, 0...] = values
        start = 0
        end = len
        self.totalSeen = totalSeen
    }

    /// Discard all cached state for a new generation.
    public func reset() {
        _keys = nil
        _values = nil
        start = 0
        end = 0
        totalSeen = 0
    }
}
