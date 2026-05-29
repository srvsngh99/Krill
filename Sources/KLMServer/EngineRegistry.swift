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
/// Keep-alive is **per model**: each resident `Entry` owns its own
/// ``KeepAliveController``, so a request's `keep_alive` (default / pin / 0)
/// sets that one model's idle deadline, and the background evictor unloads
/// each model independently when its own deadline passes and it is idle
/// (see ``evictExpired(now:)``). A model is never evicted while in-flight.
///
/// (Known narrow boundary: a model is protected from eviction only once its
/// generation has entered the queue slot and retained; the microsecond
/// window between activation and that retain is closed by an atomic
/// reservation in a follow-up. Model loads take far longer than that
/// window, so it is not reachable in practice.)
public actor EngineRegistry {
    public struct ModelNotFound: Error { public let name: String }
    /// Thrown when a new model must be loaded but every resident model is at
    /// the `maxLoaded` cap AND currently in-flight, so none can be evicted.
    public struct PoolBusy: Error { public let maxLoaded: Int }

    private struct Entry {
        let engine: InferenceEngine
        var lastUsed: Date
        /// Per-model idle deadline / keep-alive state.
        let keepAlive: KeepAliveController
        /// Per-model batch scheduler (follow-up #8, Stage B wiring): coalesces
        /// this model's concurrent requests into one batched forward.
        let scheduler: BatchScheduler
    }

    /// Resident engines keyed by canonical model directory path.
    private var entries: [String: Entry] = [:]
    /// LRU order of keys, oldest first.
    private var order: [String] = []
    /// Live-generation refcount per key; a key with count > 0 is never
    /// evicted. Bracketed by ``retain(_:)``/``release(_:)`` around the
    /// generation-queue slot. (Kept alongside each model's KeepAliveController
    /// in-flight count so the synchronous, pure ``selectEvictionVictim`` does
    /// not have to await each controller.)
    private var inFlight: [String: Int] = [:]

    private let maxLoaded: Int
    private let registry: Registry
    private let prefixCache: PrefixCache
    private let kvCacheDtype: String?
    private let activeRef: ActiveEngineRef
    /// Default idle keep-alive (seconds) for a newly-loaded model.
    private let defaultKeepAliveSeconds: Int
    /// `KRILL_NUM_PARALLEL`: max cohort size for each model's batch scheduler.
    private let numParallel: Int
    /// `KRILL_BATCH_WINDOW_MS`: coalescing window for each batch scheduler.
    private let batchWindowMs: Int
    private let logger = Logger(label: "krillm.engine-registry")

    /// - Parameters:
    ///   - preloaded: an already-loaded engine (from `serve --model`) to
    ///     register as the initial resident/active model, if any.
    ///   - preloadedName: the name `--model` was given (used to derive the key).
    ///   - maxLoaded: `MAX_LOADED_MODELS` (clamped to >= 1).
    ///   - defaultKeepAliveSeconds: idle TTL applied to each loaded model
    ///     until a request overrides it via `keep_alive`.
    public init(preloaded: InferenceEngine? = nil,
                preloadedName: String? = nil,
                maxLoaded: Int,
                registry: Registry,
                prefixCache: PrefixCache,
                kvCacheDtype: String? = nil,
                defaultKeepAliveSeconds: Int = 300,
                numParallel: Int = 1,
                activeRef: ActiveEngineRef) {
        self.maxLoaded = max(1, maxLoaded)
        self.registry = registry
        self.prefixCache = prefixCache
        self.kvCacheDtype = kvCacheDtype
        self.defaultKeepAliveSeconds = defaultKeepAliveSeconds
        self.numParallel = numParallel
        self.batchWindowMs = BatchScheduler.windowMsFromEnvironment()
        self.activeRef = activeRef
        if let preloaded {
            let key = Self.key(forName: preloadedName, registry: registry,
                               fallbackPath: preloaded.modelDirectoryPath)
            entries[key] = Entry(engine: preloaded, lastUsed: Date(),
                                 keepAlive: KeepAliveController(defaultSeconds: defaultKeepAliveSeconds),
                                 scheduler: BatchScheduler(engine: preloaded,
                                                           numParallel: numParallel,
                                                           windowMs: self.batchWindowMs))
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
        entries[key] = Entry(engine: engine, lastUsed: Date(),
                             keepAlive: KeepAliveController(defaultSeconds: defaultKeepAliveSeconds),
                             scheduler: BatchScheduler(engine: engine,
                                                       numParallel: numParallel,
                                                       windowMs: batchWindowMs))
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
    /// queue slot is held, or just before queueing - see the server). Bumps
    /// the eviction-safety refcount AND the model's own keep-alive in-flight
    /// count so neither LRU eviction nor idle eviction tears it down. No-op
    /// for engines not in the pool (e.g. the unloaded display fallback).
    public func retain(_ engine: InferenceEngine?) async {
        guard let engine, let key = key(for: engine) else { return }
        inFlight[key, default: 0] += 1
        await entries[key]?.keepAlive.beginRequest()
    }

    /// Release a live-generation hold taken by ``retain(_:)``.
    public func release(_ engine: InferenceEngine?) async {
        guard let engine, let key = key(for: engine) else { return }
        if let n = inFlight[key] {
            if n <= 1 { inFlight.removeValue(forKey: key) } else { inFlight[key] = n - 1 }
        }
        await entries[key]?.keepAlive.endRequest()
    }

    /// Apply a request's `keep_alive` (seconds; nil = default, < 0 = pin
    /// loaded, 0 = evict once this request drains) to the given model's
    /// per-model deadline. No-op for engines not in the pool.
    public func touch(_ engine: InferenceEngine?, keepAlive override: Int?) async {
        guard let engine, let key = key(for: engine) else { return }
        await entries[key]?.keepAlive.touch(override: override)
    }

    /// Unload every resident model whose per-model keep-alive deadline has
    /// passed and which is not in-flight. Returns the display names (last
    /// path component) of the models evicted this tick, for logging.
    public func evictExpired(now: Date = Date()) async -> [String] {
        var evicted: [String] = []
        for (key, entry) in entries {
            if await entry.keepAlive.shouldEvict(now: now) {
                entry.engine.unload()
                entries.removeValue(forKey: key)
                order.removeAll { $0 == key }
                inFlight.removeValue(forKey: key)
                evicted.append(entry.engine.modelName ?? (key as NSString).lastPathComponent)
            }
        }
        if let active = activeRef.current, key(for: active) == nil {
            // The active model was just evicted. Keep the active mirror in
            // step with residency: promote the most-recently-used survivor,
            // or clear it only when the pool is now empty. This preserves the
            // invariant `activeRef.current == nil` <=> no resident models,
            // which display endpoints rely on for a synchronous empty check.
            if let mruKey = order.last, let survivor = entries[mruKey] {
                activeRef.set(survivor.engine)
            } else {
                activeRef.set(nil)
            }
        }
        return evicted
    }

    /// Per-model status for `GET /api/ps`: every resident model with its
    /// display name and current keep-alive expiry (nil = pinned).
    public func residentInfo() async -> [(name: String, expiresAt: Date?)] {
        var out: [(name: String, expiresAt: Date?)] = []
        for key in order {
            guard let entry = entries[key] else { continue }
            let name = entry.engine.modelName ?? (key as NSString).lastPathComponent
            out.append((name: name, expiresAt: await entry.keepAlive.expiresAt()))
        }
        return out
    }

    /// The active engine (mirror of ``ActiveEngineRef/current``).
    public func current() -> InferenceEngine? { activeRef.current }

    /// The per-model ``BatchScheduler`` for a resident engine (by identity),
    /// or nil if the engine is not in the pool. Handlers route generation
    /// through this so concurrent same-model requests can be batched.
    /// Internal: `BatchScheduler` is a server-private type.
    func scheduler(for engine: InferenceEngine) -> BatchScheduler? {
        guard let key = key(for: engine) else { return nil }
        return entries[key]?.scheduler
    }

    /// Unload every resident model and clear the active pointer. Used by
    /// `POST /v1/models/unload` (a deliberate, immediate force-unload).
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

    #if DEBUG
    /// Test-only: register a resident entry without loading weights, so the
    /// per-model eviction policy (``evictExpired(now:)``) can be unit-tested
    /// against controllers in known states. Compiled out of release builds.
    func _insertForTesting(key: String, engine: InferenceEngine,
                           keepAlive: KeepAliveController) {
        entries[key] = Entry(engine: engine, lastUsed: Date(), keepAlive: keepAlive,
                             scheduler: BatchScheduler(engine: engine,
                                                       numParallel: numParallel,
                                                       windowMs: batchWindowMs))
        order.append(key)
        activeRef.set(engine)
    }
    #endif
}
