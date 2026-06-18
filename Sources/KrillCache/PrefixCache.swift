import Foundation
import MLX

/// Persistent prefix cache for reusing KV state across requests.
///
/// When a prompt shares a prefix with a previous request (e.g., the same system
/// prompt), the KV cache for that prefix is restored instead of re-computing it.
/// This reduces TTFT from seconds to <100ms for repeated prefixes.
///
/// Two tiers:
/// - In-memory LRU (fast, limited by `maxMemoryEntries`)
/// - On-disk safetensors at ~/.krill/cache/ (persistent across restarts)
///
/// Cache key = SHA256 of the token prefix bytes + model identifier.
public final class PrefixCache: @unchecked Sendable {
    /// Background queue for write-behind disk persistence. Shared
    /// across PrefixCache instances; QoS `.utility` keeps it off
    /// the user-interactive band but still progresses in a
    /// reasonable timeframe.
    private static let diskQueue = DispatchQueue(
        label: "krill.prefix-cache.disk", qos: .utility)

    private let cacheDir: URL
    private let maxMemoryEntries: Int
    private let minPrefixLength: Int

    /// Byte budget for the on-disk tier (`~/.krill/cache/`). Enforced by LRU
    /// eviction after every disk write so a run of unique prompts (each writing
    /// a full, never-reused KV state) cannot grow the cache without bound and
    /// ENOSPC-crash `serve` (issue #177). Semantics:
    ///   - `> 0`: keep total on-disk bytes at or under this budget.
    ///   - `== 0`: disk tier disabled — no writes (in-memory LRU still works).
    ///   - `< 0`: unbounded (legacy behavior; nothing enforces a cap).
    private let diskBudgetBytes: Int64

    /// Per-entry KV byte cap for the in-memory tier. A `store` whose KV state
    /// exceeds this is skipped entirely (no in-memory copy, no disk write), so
    /// a single huge prefix cannot spike memory and push the box into swap.
    /// This bites FULL-ATTENTION families at long context (their KV grows on
    /// every layer - llama-3.2-3b at ~94k is ~10GB. A sliding-window model
    /// (Gemma 4) is hit far later: its 40 windowed layers stay tiny via
    /// `RotatingKVCache.snapshot()`, but its 8 GLOBAL (full-attention) layers
    /// still grow, so a 12B entry crosses a 4GB cap only around ~61k context
    /// (vs a few-thousand-token full-attention prefix). The cap is therefore
    /// expressed in bytes, not as a family flag - the size IS the signal, and
    /// realistic reuse prefixes (system prompts, RAG, tool schemas; a few
    /// thousand tokens) sit far under it on every family. Semantics:
    ///   - `> 0`: skip storing any entry whose KV exceeds this many bytes.
    ///   - `<= 0`: no cap (legacy behavior; store entries of any size).
    private let maxEntryBytes: Int64

    /// One-shot guard so the "skipped a huge entry" notice prints once per
    /// process, not once per oversized request.
    private var didNoteEntryTooLarge = false
    private let entryNoteLock = NSLock()

    /// Running estimate of bytes occupied by the on-disk tier, so the common
    /// case (a write that leaves us under budget) costs O(1) instead of an
    /// O(N) directory rescan. Mutated only under ``diskCounterLock``: writes
    /// run on the serial ``diskQueue``, but ``clear()`` resets it from an
    /// arbitrary thread. Lazily seeded from a one-time scan on first write so
    /// a pre-existing cache (e.g. a 109 GB dir from a prior crash) is counted
    /// and trimmed. A full scan still runs, but ONLY when the counter crosses
    /// budget — and it then evicts down to a low-water mark so the next batch
    /// of writes stays scan-free. The scan also recomputes the true total,
    /// self-correcting any drift in the estimate.
    private var diskBytesCounter: Int64 = 0
    private var diskCounterSeeded = false
    private let diskCounterLock = NSLock()

    /// In-memory LRU: key -> (kvState, accessTime).
    /// fp16 and int8 entries share the LRU; their key strings are namespaced
    /// by dtype so a lookup from one path cannot read the other's tensors.
    ///
    /// Guarded by ``memoryLock``: a single ``PrefixCache`` is shared across the
    /// serial ``generate`` path AND the batched decode path (Stage C4), and each
    /// runs its prefill on its own Task, so concurrent `lookup`/`store` calls
    /// race the dictionary + LRU array. Without the lock, two stores can
    /// interleave a `memoryCache` insert with an `accessOrder` mutation and
    /// corrupt the LRU bookkeeping (or trap on the array). The lock is only ever
    /// held around the in-memory bookkeeping itself - never across disk I/O or
    /// the pure `build` closure - so it adds no contention to the hot path.
    private var memoryCache: [String: MemoryCacheEntry] = [:]
    private var accessOrder: [String] = []
    private let memoryLock = NSLock()

    /// - Parameter diskBudgetGB: on-disk byte budget in gigabytes. `nil` (the
    ///   default) resolves from `KRILL_PREFIX_CACHE_GB`, falling back to 2.0 GB
    ///   — so any `PrefixCache()` is bounded even on paths that don't thread
    ///   `KrillConfig` through. `0` disables the disk tier; a negative value
    ///   means unbounded.
    /// - Parameter maxEntryGB: per-entry in-memory KV cap in gigabytes. `nil`
    ///   resolves from `KRILL_PREFIX_CACHE_MAX_ENTRY_GB`, falling back to 4.0 GB
    ///   chosen for a 24 GB box: a 4 GB resident entry plus its ~4 GB
    ///   transient materialization on top of a ~9 GB model stays under the swap
    ///   line, where an 8 GB entry would not. `0` (or negative) disables the
    ///   cap. Raise it on a larger-RAM machine.
    public init(
        cacheDir: URL? = nil,
        maxMemoryEntries: Int = 8,
        minPrefixLength: Int = 8,
        diskBudgetGB: Double? = nil,
        maxEntryGB: Double? = nil
    ) {
        let dir = cacheDir ?? PrefixCache.defaultCacheDir()
        self.cacheDir = dir
        self.maxMemoryEntries = maxMemoryEntries
        self.minPrefixLength = minPrefixLength
        let gb = diskBudgetGB ?? PrefixCache.envBudgetGB() ?? 2.0
        self.diskBudgetBytes = gb < 0 ? -1 : Int64(gb * 1_000_000_000)
        let entryGB = maxEntryGB ?? PrefixCache.envMaxEntryGB() ?? 4.0
        self.maxEntryBytes = entryGB <= 0 ? -1 : Int64(entryGB * 1_000_000_000)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    /// `KRILL_PREFIX_CACHE_GB` parsed as a Double, or nil if unset/unparseable.
    private static func envBudgetGB() -> Double? {
        guard let v = ProcessInfo.processInfo.environment["KRILL_PREFIX_CACHE_GB"] else { return nil }
        return Double(v)
    }

    /// `KRILL_PREFIX_CACHE_MAX_ENTRY_GB` parsed as a Double, or nil if
    /// unset/unparseable.
    private static func envMaxEntryGB() -> Double? {
        guard let v = ProcessInfo.processInfo.environment["KRILL_PREFIX_CACHE_MAX_ENTRY_GB"] else { return nil }
        return Double(v)
    }

    static func defaultCacheDir() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".krill")
            .appendingPathComponent("cache")
    }

    /// Look up the longest cached prefix matching the given tokens.
    ///
    /// Returns the cached KV arrays and how many tokens they cover,
    /// or nil if no matching prefix is found.
    ///
    /// `mediaHash` MUST encode all non-text conditioning (image/audio bytes).
    /// Two requests with the same tokens but different media MUST pass
    /// different `mediaHash` strings — otherwise a hit would serve KV state
    /// conditioned on the wrong media. Pass nil/"" only for pure text.
    public func lookup(
        tokens: [Int],
        modelId: String,
        mediaHash: String? = nil
    ) -> PrefixCacheHit? {
        return scan(tokens: tokens, modelId: modelId, mediaHash: mediaHash, dtype: .fp16) { entry, checkLen in
            guard case let .fp16(keys, values) = entry.storage else { return nil }
            return PrefixCacheHit(keys: keys, values: values, prefixLength: checkLen)
        }
    }

    /// Look up a quantized (int8) prefix cache entry.
    ///
    /// Returns the per-layer `QuantizedKVSnapshot`s and the prefix length they
    /// cover, or nil on miss. Entries written via `storeQuantized` are stored
    /// in their uint8 form alongside fp16 scales/zeros, so a hit avoids the
    /// dequant→requant round trip that would otherwise compound error.
    public func lookupQuantized(
        tokens: [Int],
        modelId: String,
        mediaHash: String? = nil
    ) -> QuantizedPrefixCacheHit? {
        return scan(tokens: tokens, modelId: modelId, mediaHash: mediaHash, dtype: .int8) { entry, checkLen in
            guard case let .int8(snapshots) = entry.storage else { return nil }
            return QuantizedPrefixCacheHit(layers: snapshots, prefixLength: checkLen)
        }
    }

    /// Look up the in-memory entry sharing the LONGEST common prefix with
    /// `tokens` (fp16 only). Unlike `lookup`, this does not require the cached
    /// tokens to equal the request exactly: it returns the entry whose tokens
    /// share the most leading tokens with the request, so a long shared context
    /// (system prompt + RAG docs + tool schema) reused across requests with
    /// different tails is restored once and only the diverging suffix is
    /// prefilled. The returned `prefixLength` is the shared length; the carried
    /// KV tensors are the entry's FULL stored state, so the caller restores them
    /// and then truncates the per-layer caches to `prefixLength`.
    ///
    /// Matches are constrained to the same `modelId` AND `mediaHash` as the
    /// full-match key, so a partial hit can never serve KV conditioned on a
    /// different model or a different image/audio context. Disk-hydrated entries
    /// (empty `tokens`) are skipped: the persistent tier serves full hits only.
    ///
    /// Returns nil when no in-memory entry shares at least `minPrefixLength`
    /// leading tokens.
    public func lookupLongestPrefix(
        tokens: [Int],
        modelId: String,
        mediaHash: String? = nil
    ) -> PrefixCacheHit? {
        guard tokens.count >= minPrefixLength else { return nil }
        memoryLock.lock(); defer { memoryLock.unlock() }

        var bestKey: String? = nil
        var bestLen = 0
        var bestStored = 0
        var bestKeys: [[MLXArray]] = []
        var bestValues: [[MLXArray]] = []
        let wantMedia = mediaHash ?? ""
        for (key, entry) in memoryCache {
            guard entry.modelId == modelId,
                  (entry.mediaHash ?? "") == wantMedia,
                  case let .fp16(keys, values) = entry.storage,
                  !entry.tokens.isEmpty
            else { continue }
            let lcp = Self.commonPrefixLength(tokens, entry.tokens)
            // A usable match shares at least `minPrefixLength` leading tokens.
            // `lcp` is bounded by both token counts, so `lcp <= entry length`
            // always holds and the caller's truncate(to: lcp) is in range.
            if lcp >= minPrefixLength, lcp > bestLen {
                bestLen = lcp
                bestStored = entry.tokens.count
                bestKey = key
                bestKeys = keys
                bestValues = values
            }
        }
        guard let hitKey = bestKey else { return nil }
        touchEntryLocked(hitKey)
        return PrefixCacheHit(keys: bestKeys, values: bestValues,
                              prefixLength: bestLen, storedLength: bestStored)
    }

    /// int8 analogue of `lookupLongestPrefix`: returns the per-layer quantized
    /// snapshots of the in-memory entry that shares the longest leading-token
    /// run with `tokens`, plus that shared length. The snapshots are the
    /// entry's FULL stored state, so the caller restores them and then
    /// truncates the per-layer caches to `prefixLength` (mirrors the fp16
    /// partial path). Same `modelId` + `mediaHash` constraint as the full
    /// match. Disk-hydrated entries (empty `tokens`) are skipped. Returns nil
    /// when no in-memory entry shares at least `minPrefixLength` leading tokens.
    public func lookupLongestPrefixQuantized(
        tokens: [Int],
        modelId: String,
        mediaHash: String? = nil
    ) -> QuantizedPrefixCacheHit? {
        guard tokens.count >= minPrefixLength else { return nil }
        memoryLock.lock(); defer { memoryLock.unlock() }

        var bestKey: String? = nil
        var bestLen = 0
        var bestSnaps: [QuantizedKVSnapshot] = []
        let wantMedia = mediaHash ?? ""
        for (key, entry) in memoryCache {
            guard entry.modelId == modelId,
                  (entry.mediaHash ?? "") == wantMedia,
                  case let .int8(snapshots) = entry.storage,
                  !entry.tokens.isEmpty
            else { continue }
            let lcp = Self.commonPrefixLength(tokens, entry.tokens)
            if lcp >= minPrefixLength, lcp > bestLen {
                bestLen = lcp
                bestKey = key
                bestSnaps = snapshots
            }
        }
        guard let hitKey = bestKey else { return nil }
        touchEntryLocked(hitKey)
        return QuantizedPrefixCacheHit(layers: bestSnaps, prefixLength: bestLen)
    }

    /// Number of leading tokens `a` and `b` share.
    static func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let n = min(a.count, b.count)
        var i = 0
        while i < n, a[i] == b[i] { i += 1 }
        return i
    }

    private func scan<T>(
        tokens: [Int],
        modelId: String,
        mediaHash: String?,
        dtype: KVDtype,
        build: (MemoryCacheEntry, Int) -> T?
    ) -> T? {
        let maxLen = tokens.count
        guard maxLen >= minPrefixLength else { return nil }

        let step = max(1, minPrefixLength / 2)
        var checkLen = maxLen

        while checkLen >= minPrefixLength {
            let prefix = Array(tokens[0 ..< checkLen])
            let key = cacheKey(tokens: prefix, modelId: modelId, mediaHash: mediaHash, dtype: dtype)

            // Read + LRU-touch atomically under the lock, then run the (pure)
            // build closure outside it so the lock never spans non-map work.
            memoryLock.lock()
            let memEntry = memoryCache[key]
            if memEntry != nil { touchEntryLocked(key) }
            memoryLock.unlock()
            if let memEntry, let hit = build(memEntry, checkLen) { return hit }

            // Disk I/O happens with the lock RELEASED; storeInMemory re-takes it.
            if let entry = loadFromDisk(key: key, dtype: dtype) {
                storeInMemory(key: key, entry: entry)
                if let hit = build(entry, checkLen) { return hit }
            }

            checkLen -= step
        }

        return nil
    }

    /// Store a prefix's KV state after prefill completes.
    ///
    /// Called asynchronously (write-behind) so it never blocks generation.
    /// `mediaHash` MUST match the value used at lookup time and must encode
    /// any image/audio conditioning the KV state was computed under.
    public func store(
        tokens: [Int],
        modelId: String,
        keys: [[MLXArray]],
        values: [[MLXArray]],
        mediaHash: String? = nil
    ) {
        guard tokens.count >= minPrefixLength else { return }

        // Skip a single oversized entry before it is materialized: `nbytes` on
        // the lazy snapshot views is metadata-only (no evaluation), so this
        // costs nothing on the common (small) path and avoids the full
        // `contiguous` copy below for a giant full-attention prefix.
        if maxEntryBytes > 0 {
            let bytes = fp16KVBytes(keys: keys, values: values)
            if bytes > maxEntryBytes { noteEntryTooLargeOnce(bytes: bytes); return }
        }

        let key = cacheKey(tokens: tokens, modelId: modelId, mediaHash: mediaHash, dtype: .fp16)
        // Materialize the snapshots before retaining them: `KVCache.snapshot()`
        // returns lazy slice views over the cache's live backing buffer, and a
        // long-lived reference to that buffer blocks MLX's donation on every
        // subsequent in-place cache write (forcing full-buffer copies each
        // decode step). `contiguous` detaches the entry onto its own storage.
        let detachedKeys = keys.map { layer in layer.map { contiguous($0) } }
        let detachedValues = values.map { layer in layer.map { contiguous($0) } }
        // Retain the tokens + identity so a later request sharing this prefix can
        // LCP-match it (see `lookupLongestPrefix`). The KV tensors are kept by the
        // LRU regardless; the token array is a few KB of Ints on top.
        let entry = MemoryCacheEntry(
            storage: .fp16(keys: detachedKeys, values: detachedValues),
            tokens: tokens, modelId: modelId, mediaHash: mediaHash)

        storeInMemory(key: key, entry: entry)

        // Hand off the disk write to a background queue. Used to
        // be Task.detached, but Swift 6.2's strict-concurrency
        // checker rejects the capture of locals into a sending
        // closure even with nonisolated(unsafe) (the captures
        // are "accessible to the current task"). DispatchQueue's
        // async is not bound by the same sending-parameter rules,
        // and the semantics are identical: a one-shot
        // fire-and-forget write on a background thread.
        Self.diskQueue.async { [self] in
            self.writeToDisk(key: key, entry: entry, dtype: .fp16)
        }
    }

    /// Store quantized per-layer snapshots for later replay.
    ///
    /// `snapshots` is indexed by transformer layer. Empty snapshots (nil
    /// entries when a layer has no cached state) are not supported — the
    /// caller should only invoke this after a full prefill that populates
    /// every layer.
    public func storeQuantized(
        tokens: [Int],
        modelId: String,
        snapshots: [QuantizedKVSnapshot],
        mediaHash: String? = nil
    ) {
        guard tokens.count >= minPrefixLength, !snapshots.isEmpty else { return }

        // Same per-entry size guard as the fp16 path. int8 KV is ~4x smaller
        // per token, so this rarely trips, but a long enough full-attention
        // prefix can still cross the cap.
        if maxEntryBytes > 0 {
            let bytes = int8KVBytes(snapshots: snapshots)
            if bytes > maxEntryBytes { noteEntryTooLargeOnce(bytes: bytes); return }
        }

        let key = cacheKey(tokens: tokens, modelId: modelId, mediaHash: mediaHash, dtype: .int8)
        // Retain tokens + identity so a later request sharing this prefix can
        // LCP-match it (see `lookupLongestPrefixQuantized`), exactly as the
        // fp16 `store` does. Without this, quantized entries could only ever
        // serve byte-identical full hits, never a shared-prefix partial reuse.
        let entry = MemoryCacheEntry(
            storage: .int8(snapshots: snapshots),
            tokens: tokens, modelId: modelId, mediaHash: mediaHash)

        storeInMemory(key: key, entry: entry)

        // See note on the equivalent fp16 path above.
        Self.diskQueue.async { [self] in
            self.writeToDisk(key: key, entry: entry, dtype: .int8)
        }
    }

    /// Clear all cached entries (memory + disk).
    public func clear() {
        memoryLock.lock()
        memoryCache.removeAll()
        accessOrder.removeAll()
        memoryLock.unlock()
        try? FileManager.default.removeItem(at: cacheDir)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        diskCounterLock.lock()
        diskBytesCounter = 0
        diskCounterSeeded = true  // dir is now empty; no re-scan needed
        diskCounterLock.unlock()
    }

    /// Number of entries in memory cache.
    public var memoryCount: Int {
        memoryLock.lock(); defer { memoryLock.unlock() }
        return memoryCache.count
    }

    /// Number of entries on disk.
    public var diskCount: Int {
        (try? FileManager.default.contentsOfDirectory(at: cacheDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.count) ?? 0
    }
}

// MARK: - Types

/// A successful prefix cache hit.
public struct PrefixCacheHit {
    /// Per-layer cached keys: [numLayers][numKVHeads, prefixLen, headDim]
    public let keys: [[MLXArray]]

    /// Per-layer cached values: [numLayers][numKVHeads, prefixLen, headDim]
    public let values: [[MLXArray]]

    /// How many leading tokens of the REQUEST this hit covers (the shared
    /// length for an LCP hit; the full prompt for an exact hit).
    public let prefixLength: Int

    /// The stored ENTRY's total token count - the absolute position one past
    /// the last stored KV row. Equal to `prefixLength` for an exact hit;
    /// >= `prefixLength` for an LCP hit (the carried KV is the entry's full
    /// stored state). Rotating (windowed) caches need this to place a
    /// window-trimmed span at its true absolute position on restore.
    public let storedLength: Int

    init(keys: [[MLXArray]], values: [[MLXArray]], prefixLength: Int,
         storedLength: Int? = nil) {
        self.keys = keys
        self.values = values
        self.prefixLength = prefixLength
        self.storedLength = storedLength ?? prefixLength
    }
}

/// A successful int8 prefix cache hit.
///
/// Carries the raw quantized state for each layer; callers feed each
/// `QuantizedKVSnapshot` into a `QuantizedKVCache.restoreQuantized(_:)`.
public struct QuantizedPrefixCacheHit {
    public let layers: [QuantizedKVSnapshot]
    public let prefixLength: Int
}

// MARK: - Internal

/// Internal storage form for a cached entry. fp16 and int8 are disjoint so
/// a lookup from one path cannot read tensors written by the other.
enum CachedStorage {
    case fp16(keys: [[MLXArray]], values: [[MLXArray]])
    case int8(snapshots: [QuantizedKVSnapshot])
}

struct MemoryCacheEntry {
    let storage: CachedStorage
    /// The exact token prefix this entry's KV was computed under. Retained for
    /// in-memory longest-common-prefix (LCP) matching so a request that SHARES
    /// a prefix with this entry (same system prompt / context, different tail)
    /// can reuse the prefix KV instead of re-prefilling it. Empty for entries
    /// hydrated from disk (the disk tier serves full-match hits only).
    var tokens: [Int] = []
    /// Model + media identity the KV was conditioned under. LCP matches must
    /// agree on both, exactly as the full-match hash key does, so a partial hit
    /// can never serve KV from another model or another image/audio context.
    var modelId: String = ""
    var mediaHash: String? = nil
}

enum KVDtype: UInt8 {
    case fp16 = 1
    case int8 = 2

    var fileSuffix: String {
        switch self {
        case .fp16: return "safetensors"
        case .int8: return "q8.safetensors"
        }
    }

    var keyTag: UInt8 { rawValue }
}

extension PrefixCache {
    /// Cache key schema version. Bump on backward-incompatible key changes
    /// so stale on-disk entries become unreachable instead of mis-served.
    /// v2: includes mediaHash to prevent multimodal cross-contamination.
    /// v3: includes dtype tag so int8 and fp16 entries cannot collide.
    private static let keySchemaVersion: UInt8 = 3

    private func cacheKey(tokens: [Int], modelId: String, mediaHash: String?, dtype: KVDtype) -> String {
        // FNV-1a hash of: schema version || dtype || modelId || mediaHash || token bytes.
        var data = Data()
        data.append(PrefixCache.keySchemaVersion)
        data.append(dtype.keyTag)
        data.append(modelId.data(using: .utf8) ?? Data())
        data.append(0xFF)
        if let mediaHash, !mediaHash.isEmpty {
            data.append(mediaHash.data(using: .utf8) ?? Data())
        }
        data.append(0xFF)
        data.append(Data(bytes: tokens, count: tokens.count * MemoryLayout<Int>.size))

        var hash: UInt64 = 14695981039346656037 // FNV-1a offset basis
        for byte in data {
            hash ^= UInt64(byte)
            hash &*= 1099511628211 // FNV-1a prime
        }
        return String(format: "%016llx", hash)
    }

    /// Move `key` to the most-recently-used end. Caller MUST hold `memoryLock`.
    private func touchEntryLocked(_ key: String) {
        accessOrder.removeAll { $0 == key }
        accessOrder.append(key)
    }

    private func storeInMemory(key: String, entry: MemoryCacheEntry) {
        memoryLock.lock()
        defer { memoryLock.unlock() }
        // Evict LRU if at capacity
        while memoryCache.count >= maxMemoryEntries, let oldest = accessOrder.first {
            memoryCache.removeValue(forKey: oldest)
            accessOrder.removeFirst()
        }

        memoryCache[key] = entry
        touchEntryLocked(key)
    }

    private func loadFromDisk(key: String, dtype: KVDtype) -> MemoryCacheEntry? {
        let fileURL = cacheDir.appendingPathComponent("\(key).\(dtype.fileSuffix)")
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        guard let arrays = try? loadArrays(url: fileURL) else { return nil }

        // Refresh modification time so a hit marks this entry recently-used and
        // the disk-budget LRU keeps hot prefixes over cold ones (issue #177).
        try? FileManager.default.setAttributes(
            [.modificationDate: Date()], ofItemAtPath: fileURL.path)

        switch dtype {
        case .fp16:
            // Convention: "layer_0_k", "layer_0_v", "layer_1_k", etc.
            var layerKeys: [[MLXArray]] = []
            var layerValues: [[MLXArray]] = []
            var i = 0
            while let k = arrays["layer_\(i)_k"], let v = arrays["layer_\(i)_v"] {
                layerKeys.append([k])
                layerValues.append([v])
                i += 1
            }
            guard !layerKeys.isEmpty else { return nil }
            return MemoryCacheEntry(storage: .fp16(keys: layerKeys, values: layerValues))

        case .int8:
            // Convention per layer: "layer_{i}_qk", "_qv", "_ks", "_kz", "_vs", "_vz".
            var snapshots: [QuantizedKVSnapshot] = []
            var i = 0
            while let qk = arrays["layer_\(i)_qk"],
                  let qv = arrays["layer_\(i)_qv"],
                  let ks = arrays["layer_\(i)_ks"],
                  let kz = arrays["layer_\(i)_kz"],
                  let vs = arrays["layer_\(i)_vs"],
                  let vz = arrays["layer_\(i)_vz"]
            {
                snapshots.append(QuantizedKVSnapshot(
                    keys: qk, values: qv,
                    keyScales: ks, keyZeros: kz,
                    valueScales: vs, valueZeros: vz
                ))
                i += 1
            }
            guard !snapshots.isEmpty else { return nil }
            return MemoryCacheEntry(storage: .int8(snapshots: snapshots))
        }
    }

    private func writeToDisk(key: String, entry: MemoryCacheEntry, dtype: KVDtype) {
        // diskBudgetBytes == 0 disables the disk tier entirely (issue #177).
        guard diskBudgetBytes != 0 else { return }

        let fileURL = cacheDir.appendingPathComponent("\(key).\(dtype.fileSuffix)")
        var arrays: [String: MLXArray] = [:]

        switch entry.storage {
        case let .fp16(keys, values):
            for (i, layerK) in keys.enumerated() {
                for k in layerK { arrays["layer_\(i)_k"] = k }
            }
            for (i, layerV) in values.enumerated() {
                for v in layerV { arrays["layer_\(i)_v"] = v }
            }
        case let .int8(snapshots):
            for (i, snap) in snapshots.enumerated() {
                arrays["layer_\(i)_qk"] = snap.keys
                arrays["layer_\(i)_qv"] = snap.values
                arrays["layer_\(i)_ks"] = snap.keyScales
                arrays["layer_\(i)_kz"] = snap.keyZeros
                arrays["layer_\(i)_vs"] = snap.valueScales
                arrays["layer_\(i)_vz"] = snap.valueZeros
            }
        }

        // Account the new file against the running byte counter. The common
        // case (still under budget) ends here in O(1); only crossing budget
        // pays for a directory scan + eviction.
        let priorSize = fileSizeOnDisk(fileURL)
        try? save(arrays: arrays, url: fileURL)
        let newSize = fileSizeOnDisk(fileURL)

        let overBudget: Bool
        diskCounterLock.lock()
        if !diskCounterSeeded {
            // First write of this process: seed the counter from a one-time
            // scan so any pre-existing on-disk cache is counted (and trimmed
            // below if it already exceeds budget). The scan total includes the
            // file we just wrote.
            diskBytesCounter = scanDiskBytes()
            diskCounterSeeded = true
        } else {
            diskBytesCounter += newSize - priorSize
        }
        overBudget = diskBudgetBytes > 0 && diskBytesCounter > diskBudgetBytes
        diskCounterLock.unlock()

        // Evict only when we have actually crossed budget. Runs on the serial
        // `diskQueue` (all writes do), so evictions never race other writes; a
        // concurrent reader at worst sees a file vanish between its
        // exists-check and load and falls through to a clean miss.
        if overBudget { evictToLowWater() }
    }

    /// Evict least-recently-used on-disk entries until the cache drops to a
    /// low-water mark (90% of budget). Recency is the file modification date,
    /// which a write sets to "now" and a disk hit refreshes (see
    /// `loadFromDisk`), so a long run of distinct prompts evicts its own cold
    /// KV states instead of growing without bound (issue #177). Evicting to a
    /// low-water mark (rather than exactly to budget) leaves headroom so the
    /// next ~10%-of-budget worth of writes stay scan-free. The post-eviction
    /// true total is written back to the counter, self-correcting drift.
    private func evictToLowWater() {
        guard diskBudgetBytes > 0 else { return }
        let lowWater = diskBudgetBytes - diskBudgetBytes / 10  // 90% of budget
        let fm = FileManager.default
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        guard let urls = try? fm.contentsOfDirectory(
            at: cacheDir, includingPropertiesForKeys: keys, options: [.skipsHiddenFiles])
        else { return }

        var entries: [(url: URL, size: Int64, mtime: Date)] = []
        var total: Int64 = 0
        for url in urls where url.pathExtension == "safetensors" {
            let vals = try? url.resourceValues(forKeys: Set(keys))
            let size = Int64(vals?.fileSize ?? 0)
            let mtime = vals?.contentModificationDate ?? Date.distantPast
            entries.append((url, size, mtime))
            total += size
        }

        if total > diskBudgetBytes {
            // Oldest first; drop until at or below the low-water mark.
            entries.sort { $0.mtime < $1.mtime }
            for e in entries {
                if total <= lowWater { break }
                if (try? fm.removeItem(at: e.url)) != nil { total -= e.size }
            }
        }

        diskCounterLock.lock()
        diskBytesCounter = total
        diskCounterSeeded = true
        diskCounterLock.unlock()
    }

    /// Total bytes of an fp16 entry's KV tensors, summed from `nbytes` (shape
    /// and dtype metadata only - does not evaluate the lazy snapshot views).
    private func fp16KVBytes(keys: [[MLXArray]], values: [[MLXArray]]) -> Int64 {
        var total: Int64 = 0
        for layer in keys { for a in layer { total += Int64(a.nbytes) } }
        for layer in values { for a in layer { total += Int64(a.nbytes) } }
        return total
    }

    /// Total bytes of an int8 entry's per-layer snapshots (quantized payload
    /// plus the fp16 scales/zeros), summed from `nbytes`.
    private func int8KVBytes(snapshots: [QuantizedKVSnapshot]) -> Int64 {
        var total: Int64 = 0
        for s in snapshots {
            total += Int64(s.keys.nbytes + s.values.nbytes
                + s.keyScales.nbytes + s.keyZeros.nbytes
                + s.valueScales.nbytes + s.valueZeros.nbytes)
        }
        return total
    }

    /// Emit the "skipped an oversized entry" notice once per process. The cap
    /// is a memory-safety floor, so an operator who actually wants to cache
    /// huge prefixes needs to know the knob exists.
    private func noteEntryTooLargeOnce(bytes: Int64) {
        entryNoteLock.lock(); defer { entryNoteLock.unlock() }
        guard !didNoteEntryTooLarge else { return }
        didNoteEntryTooLarge = true
        let gb = Double(bytes) / 1_000_000_000
        let capGB = Double(maxEntryBytes) / 1_000_000_000
        FileHandle.standardError.write(Data((
            "[Krill] prefix-cache: not caching a "
            + String(format: "%.1f", gb) + "GB KV prefix (over the "
            + String(format: "%.1f", capGB)
            + "GB per-entry cap; raise KRILL_PREFIX_CACHE_MAX_ENTRY_GB or set it "
            + "to 0 to disable). Long full-attention contexts are not reused to "
            + "avoid memory pressure; windowed models (Gemma 4) are hit only at "
            + "much longer context.\n"
        ).utf8))
    }

    /// Size of a single cache file in bytes, or 0 if it does not exist.
    private func fileSizeOnDisk(_ url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
    }

    /// Sum of all `*.safetensors` cache files currently on disk.
    private func scanDiskBytes() -> Int64 {
        let fm = FileManager.default
        guard let urls = try? fm.contentsOfDirectory(
            at: cacheDir, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles])
        else { return 0 }
        var total: Int64 = 0
        for url in urls where url.pathExtension == "safetensors" {
            total += fileSizeOnDisk(url)
        }
        return total
    }

    /// Total bytes the on-disk tier currently occupies (live directory stat,
    /// not the cached estimate). Used by the budget gate in tests; not on any
    /// hot path.
    var diskBytes: Int64 { scanDiskBytes() }

    /// Block until all queued write-behind disk writes (and their budget
    /// eviction) have completed. Test-only synchronization point — the
    /// production paths are deliberately fire-and-forget.
    func waitForDiskWrites() {
        Self.diskQueue.sync {}
    }
}
