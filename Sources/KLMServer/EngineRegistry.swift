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
/// Eviction is **in-flight-aware**: a resident engine with a live
/// generation (tracked via ``retain(_:)``/``release(_:)``, bracketed around
/// the `GenerationQueue` slot exactly like the keep-alive in-flight count)
/// is never evicted. When the pool is at `maxLoaded` and every resident
/// model is busy, ``activate(name:)`` throws ``PoolBusy`` rather than
/// tearing a model down mid-stream; the server turns that into a meaningful
/// 503. Without this guard, a concurrent request for a new model could
/// `unload()` a model that another request is still decoding against.
///
/// Scope note (Stage A / "routing first"): per-model keep-alive and the
/// full set of eviction policies are still the next PR; the idle evictor
/// here unloads the whole pool. (Known narrow boundary: a model is
/// protected from eviction only once its generation has entered the queue
/// slot and retained; the microsecond window between activation and that
/// retain is closed by an atomic reservation in the follow-up. Model loads
/// take far longer than that window, so it is not reachable in practice.)
public actor EngineRegistry {
    public struct ModelNotFound: Error { public let name: String }
    /// Thrown when a new model must be loaded but every resident model is at
    /// the `maxLoaded` cap AND currently in-flight, so none can be evicted.
    public struct PoolBusy: Error { public let maxLoaded: Int }

    private struct Entry {
        let engine: InferenceEngine
        var lastUsed: Date
    }

    /// Resident engines keyed by canonical model directory path.
    private var entries: [String: Entry] = [:]
    /// LRU order of keys, oldest first.
    private var order: [String] = []
    /// Live-generation refcount per key; a key with count > 0 is never
    /// evicted. Bracketed by ``retain(_:)``/``release(_:)`` around the
    /// generation-queue slot.
    private var inFlight: [String: Int] = [:]

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
    /// return it. Loads on demand, evicting the least-recently-used resident
    /// model that is NOT in-flight when the pool is at `maxLoaded`. Throws
    /// ``ModelNotFound`` if `name` is not installed, or ``PoolBusy`` if the
    /// pool is full and every resident model is currently generating.
    public func activate(name: String) async throws -> InferenceEngine {
        guard registry.hasModel(name) else { throw ModelNotFound(name: name) }
        let key = registry.modelPath(name).path

        if let existing = entries[key] {
            entries[key]?.lastUsed = Date()
            promote(key)
            activeRef.set(existing.engine)
            return existing.engine
        }

        // Make room: evict the LRU non-in-flight resident until under cap.
        // Never evict a model with a live generation (it would tear the
        // engine down mid-stream); if none is evictable, refuse with PoolBusy
        // so the server can return a meaningful 503 instead of crashing.
        while entries.count >= maxLoaded {
            guard let victim = Self.selectEvictionVictim(
                order: order, inFlight: inFlight) else {
                throw PoolBusy(maxLoaded: maxLoaded)
            }
            entries[victim]?.engine.unload()
            entries.removeValue(forKey: victim)
            order.removeAll { $0 == victim }
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

    /// Choose the eviction victim: the least-recently-used resident key whose
    /// in-flight count is zero, or nil if every resident model is in-flight.
    /// Pure (no actor state) so the invariant can be unit-tested directly.
    static func selectEvictionVictim(order: [String],
                                     inFlight: [String: Int]) -> String? {
        for key in order where (inFlight[key] ?? 0) == 0 { return key }
        return nil
    }

    /// Mark `engine`'s live generation as started (call once a generation
    /// queue slot is held). No-op for engines not in the pool (e.g. the
    /// unloaded display fallback).
    public func retain(_ engine: InferenceEngine?) {
        guard let engine, let key = key(for: engine) else { return }
        inFlight[key, default: 0] += 1
    }

    /// Release a live-generation hold taken by ``retain(_:)``.
    public func release(_ engine: InferenceEngine?) {
        guard let engine, let key = key(for: engine), let n = inFlight[key] else { return }
        if n <= 1 { inFlight.removeValue(forKey: key) } else { inFlight[key] = n - 1 }
    }

    /// The active engine (mirror of ``ActiveEngineRef/current``).
    public func current() -> InferenceEngine? { activeRef.current }

    /// Unload every resident model and clear the active pointer. Used by the
    /// idle keep-alive evictor and `POST /v1/models/unload`.
    public func unloadAll() {
        for (_, entry) in entries { entry.engine.unload() }
        entries.removeAll()
        order.removeAll()
        inFlight.removeAll()
        activeRef.set(nil)
    }

    /// Pool key for a resident engine, by object identity.
    private func key(for engine: InferenceEngine) -> String? {
        for (k, e) in entries where e.engine === engine { return k }
        return nil
    }

    /// Move `key` to the most-recently-used (tail) position.
    private func promote(_ key: String) {
        order.removeAll { $0 == key }
        order.append(key)
    }
}
