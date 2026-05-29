import Foundation
import os
import Logging
import KLMEngine
import KLMCache
import KLMRegistry

/// Thread-safe, synchronously-readable mirror of the registry's *active*
/// engine (the most-recently-activated resident model). Display endpoints
/// (`/healthz`, `/v1/status`, `/api/ps`, `/metrics`) read this without an
/// `await`; the ``EngineRegistry`` actor is its only writer.
public final class ActiveEngineRef: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock<InferenceEngine?>(initialState: nil)

    public init(_ initial: InferenceEngine? = nil) {
        if let initial { lock.withLock { $0 = initial } }
    }

    /// The currently-active engine, or nil if nothing is resident.
    public var current: InferenceEngine? { lock.withLock { $0 } }

    func set(_ engine: InferenceEngine?) { lock.withLock { $0 = engine } }
}

/// An LRU pool of resident ``InferenceEngine`` instances keyed by model
/// directory, lifting the historical `MAX_LOADED_MODELS == 1` ceiling
/// (follow-up #8, Stage A — routing first).
///
/// A request naming model *M* is **routed-or-loaded**: if *M* is already
/// resident it is returned immediately; otherwise it is loaded (evicting the
/// least-recently-used resident model when the pool is at `maxLoaded`).
/// Previously this was a single engine that *discarded* the old model on
/// every swap; now prior models stay resident up to the cap so switching
/// back is instant.
///
/// All engines share ONE ``PrefixCache`` (its keys already namespace by
/// model id, so models never read each other's prefixes); this keeps the
/// prefix-cache memory budget singular rather than multiplying by the
/// resident count. (Per-model pools are a deferred opt-in.)
///
/// Scope note (Stage A / "routing first"): per-model keep-alive,
/// in-flight-aware eviction, and the full+busy "meaningful 503" are the
/// next PR. Today eviction is LRU-on-overflow plus the existing global
/// idle evictor (which unloads the whole pool), and generation remains
/// serialized by `GenerationQueue`, so the LRU victim is never the
/// in-flight (active) model.
public actor EngineRegistry {
    public struct ModelNotFound: Error { public let name: String }

    private struct Entry {
        let engine: InferenceEngine
        var lastUsed: Date
    }

    /// Resident engines keyed by canonical model directory path.
    private var entries: [String: Entry] = [:]
    /// LRU order of keys, oldest first.
    private var order: [String] = []

    private let maxLoaded: Int
    private let registry: Registry
    private let prefixCache: PrefixCache
    private let kvCacheDtype: String?
    private let activeRef: ActiveEngineRef
    private let logger = Logger(label: "krillm.engine-registry")

    /// - Parameters:
    ///   - preloaded: an already-loaded engine (from `serve --model`) to
    ///     register as the initial resident/active model, if any.
    ///   - preloadedName: the name `--model` was given (used to derive the key).
    ///   - maxLoaded: `MAX_LOADED_MODELS` (clamped to >= 1).
    public init(preloaded: InferenceEngine? = nil,
                preloadedName: String? = nil,
                maxLoaded: Int,
                registry: Registry,
                prefixCache: PrefixCache,
                kvCacheDtype: String? = nil,
                activeRef: ActiveEngineRef) {
        self.maxLoaded = max(1, maxLoaded)
        self.registry = registry
        self.prefixCache = prefixCache
        self.kvCacheDtype = kvCacheDtype
        self.activeRef = activeRef
        if let preloaded {
            let key = Self.key(forName: preloadedName, registry: registry,
                               fallbackPath: preloaded.modelDirectoryPath)
            entries[key] = Entry(engine: preloaded, lastUsed: Date())
            order.append(key)
            if preloaded.isLoaded { activeRef.set(preloaded) }
        }
    }

    /// Canonical pool key for a model name: its registry directory path when
    /// the name is installed, else the engine's own directory path, else the
    /// raw name. Keying by directory dedupes aliases that resolve to one dir.
    private static func key(forName name: String?, registry: Registry,
                            fallbackPath: String?) -> String {
        if let name, registry.hasModel(name) { return registry.modelPath(name).path }
        return fallbackPath ?? (name ?? "")
    }

    /// Number of resident models (diagnostics/tests).
    public var residentCount: Int { entries.count }

    /// Resolve (route-or-load) the engine for `name`, mark it active, and
    /// return it. Loads on demand, evicting the LRU resident model when the
    /// pool is already at `maxLoaded`. Throws ``ModelNotFound`` if `name` is
    /// not an installed model.
    public func activate(name: String) async throws -> InferenceEngine {
        guard registry.hasModel(name) else { throw ModelNotFound(name: name) }
        let key = registry.modelPath(name).path

        if let existing = entries[key] {
            entries[key]?.lastUsed = Date()
            promote(key)
            activeRef.set(existing.engine)
            return existing.engine
        }

        // Evict LRU residents until there is room for one more. The active
        // (most-recently-used) model is at the tail of `order`, so the head
        // we evict is never the in-flight model under serialized generation.
        while entries.count >= maxLoaded, let victim = order.first {
            entries[victim]?.engine.unload()
            entries.removeValue(forKey: victim)
            order.removeFirst()
            logger.info("evicted LRU model to make room (cap=\(self.maxLoaded))")
        }

        let dir = registry.modelPath(name)
        let engine = InferenceEngine(modelDirectory: dir,
                                     prefixCache: prefixCache,
                                     kvCacheDtype: kvCacheDtype)
        try await engine.load()
        await engine.warmup()
        entries[key] = Entry(engine: engine, lastUsed: Date())
        order.append(key)
        activeRef.set(engine)
        logger.info("loaded model '\(name)' (resident=\(self.entries.count)/\(self.maxLoaded))")
        return engine
    }

    /// The active engine (mirror of ``ActiveEngineRef/current``).
    public func current() -> InferenceEngine? { activeRef.current }

    /// Unload every resident model and clear the active pointer. Used by the
    /// idle keep-alive evictor and `POST /v1/models/unload`.
    public func unloadAll() {
        for (_, entry) in entries { entry.engine.unload() }
        entries.removeAll()
        order.removeAll()
        activeRef.set(nil)
    }

    /// Move `key` to the most-recently-used (tail) position.
    private func promote(_ key: String) {
        order.removeAll { $0 == key }
        order.append(key)
    }
}
