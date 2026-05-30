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
/// - On-disk safetensors at ~/.krillm/cache/ (persistent across restarts)
///
/// Cache key = SHA256 of the token prefix bytes + model identifier.
public final class PrefixCache: @unchecked Sendable {
    /// Background queue for write-behind disk persistence. Shared
    /// across PrefixCache instances; QoS `.utility` keeps it off
    /// the user-interactive band but still progresses in a
    /// reasonable timeframe.
    private static let diskQueue = DispatchQueue(
        label: "krillm.prefix-cache.disk", qos: .utility)

    private let cacheDir: URL
    private let maxMemoryEntries: Int
    private let minPrefixLength: Int

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

    public init(
        cacheDir: URL? = nil,
        maxMemoryEntries: Int = 8,
        minPrefixLength: Int = 8
    ) {
        let dir = cacheDir ?? PrefixCache.defaultCacheDir()
        self.cacheDir = dir
        self.maxMemoryEntries = maxMemoryEntries
        self.minPrefixLength = minPrefixLength
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    }

    static func defaultCacheDir() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".krillm")
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

        let key = cacheKey(tokens: tokens, modelId: modelId, mediaHash: mediaHash, dtype: .fp16)
        let entry = MemoryCacheEntry(storage: .fp16(keys: keys, values: values))

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

        let key = cacheKey(tokens: tokens, modelId: modelId, mediaHash: mediaHash, dtype: .int8)
        let entry = MemoryCacheEntry(storage: .int8(snapshots: snapshots))

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

    /// How many tokens this cache covers.
    public let prefixLength: Int
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

        try? save(arrays: arrays, url: fileURL)
    }
}
