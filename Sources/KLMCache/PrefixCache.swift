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
    private let cacheDir: URL
    private let maxMemoryEntries: Int
    private let minPrefixLength: Int

    /// In-memory LRU: key -> (kvState, accessTime)
    private var memoryCache: [String: MemoryCacheEntry] = [:]
    private var accessOrder: [String] = []

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
        // Try progressively shorter prefixes
        let maxLen = tokens.count
        guard maxLen >= minPrefixLength else { return nil }

        // Step down by chunks to avoid O(n) hash computations
        let step = max(1, minPrefixLength / 2)
        var checkLen = maxLen

        while checkLen >= minPrefixLength {
            let prefix = Array(tokens[0 ..< checkLen])
            let key = cacheKey(tokens: prefix, modelId: modelId, mediaHash: mediaHash)

            // Check memory first
            if let entry = memoryCache[key] {
                touchEntry(key)
                return PrefixCacheHit(
                    keys: entry.keys,
                    values: entry.values,
                    prefixLength: checkLen
                )
            }

            // Check disk
            if let entry = loadFromDisk(key: key) {
                // Promote to memory
                storeInMemory(key: key, entry: entry)
                return PrefixCacheHit(
                    keys: entry.keys,
                    values: entry.values,
                    prefixLength: checkLen
                )
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

        let key = cacheKey(tokens: tokens, modelId: modelId, mediaHash: mediaHash)
        let entry = MemoryCacheEntry(keys: keys, values: values)

        storeInMemory(key: key, entry: entry)

        // Write to disk in background (write-behind)
        nonisolated(unsafe) let selfRef = self
        nonisolated(unsafe) let capturedEntry = entry
        nonisolated(unsafe) let capturedKey = key
        Task.detached {
            selfRef.writeToDisk(key: capturedKey, entry: capturedEntry)
        }
    }

    /// Clear all cached entries (memory + disk).
    public func clear() {
        memoryCache.removeAll()
        accessOrder.removeAll()
        try? FileManager.default.removeItem(at: cacheDir)
        try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
    }

    /// Number of entries in memory cache.
    public var memoryCount: Int { memoryCache.count }

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

// MARK: - Internal

private struct MemoryCacheEntry {
    let keys: [[MLXArray]]
    let values: [[MLXArray]]
}

extension PrefixCache {
    /// Cache key schema version. Bump on backward-incompatible key changes
    /// so stale on-disk entries become unreachable instead of mis-served.
    /// v2: includes mediaHash to prevent multimodal cross-contamination.
    private static let keySchemaVersion: UInt8 = 2

    private func cacheKey(tokens: [Int], modelId: String, mediaHash: String?) -> String {
        // FNV-1a hash of: schema version || modelId || mediaHash || token bytes.
        var data = Data()
        data.append(PrefixCache.keySchemaVersion)
        data.append(modelId.data(using: .utf8) ?? Data())
        data.append(0xFF) // separator so modelId/mediaHash boundary is unambiguous
        if let mediaHash, !mediaHash.isEmpty {
            data.append(mediaHash.data(using: .utf8) ?? Data())
        }
        data.append(0xFF)
        data.append(Data(bytes: tokens, count: tokens.count * MemoryLayout<Int>.size))

        // Simple hash (not cryptographic - just for cache keying)
        var hash: UInt64 = 14695981039346656037 // FNV-1a offset basis
        for byte in data {
            hash ^= UInt64(byte)
            hash &*= 1099511628211 // FNV-1a prime
        }
        return String(format: "%016llx", hash)
    }

    private func touchEntry(_ key: String) {
        accessOrder.removeAll { $0 == key }
        accessOrder.append(key)
    }

    private func storeInMemory(key: String, entry: MemoryCacheEntry) {
        // Evict LRU if at capacity
        while memoryCache.count >= maxMemoryEntries, let oldest = accessOrder.first {
            memoryCache.removeValue(forKey: oldest)
            accessOrder.removeFirst()
        }

        memoryCache[key] = entry
        touchEntry(key)
    }

    private func loadFromDisk(key: String) -> MemoryCacheEntry? {
        let fileURL = cacheDir.appendingPathComponent("\(key).safetensors")
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }

        guard let arrays = try? loadArrays(url: fileURL) else { return nil }

        // Reconstruct per-layer keys/values from flat naming
        // Convention: "layer_0_k", "layer_0_v", "layer_1_k", etc.
        var layerKeys: [[MLXArray]] = []
        var layerValues: [[MLXArray]] = []

        var layerIdx = 0
        while true {
            guard let k = arrays["layer_\(layerIdx)_k"],
                  let v = arrays["layer_\(layerIdx)_v"] else { break }
            layerKeys.append([k])
            layerValues.append([v])
            layerIdx += 1
        }

        guard !layerKeys.isEmpty else { return nil }
        return MemoryCacheEntry(keys: layerKeys, values: layerValues)
    }

    private func writeToDisk(key: String, entry: MemoryCacheEntry) {
        let fileURL = cacheDir.appendingPathComponent("\(key).safetensors")

        // Flatten to named arrays
        var arrays: [String: MLXArray] = [:]
        for (i, layerK) in entry.keys.enumerated() {
            for k in layerK {
                arrays["layer_\(i)_k"] = k
            }
        }
        for (i, layerV) in entry.values.enumerated() {
            for v in layerV {
                arrays["layer_\(i)_v"] = v
            }
        }

        try? save(arrays: arrays, url: fileURL)
    }
}
