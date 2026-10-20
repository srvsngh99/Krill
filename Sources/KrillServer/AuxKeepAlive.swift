import Foundation
import KrillEngine

/// Idle-unload bookkeeping for the two auxiliary model slots, embedding and
/// rerank. They use the SAME ``KeepAliveController`` semantics, defaults
/// (`KRILL_KEEP_ALIVE` / `OLLAMA_KEEP_ALIVE`) and per-request `keep_alive`
/// field as chat models: a positive value sets the idle TTL, `0` unloads once
/// the request drains, a negative value pins. After expiry the weights AND
/// MLX's buffer cache are released, so a resting server falls back to the
/// bare-process footprint; the next request reloads transparently.
public final class AuxKeepAlive: Sendable {
    public let embed: KeepAliveController
    public let rerank: KeepAliveController

    public init(defaultSeconds: Int) {
        embed = KeepAliveController(defaultSeconds: defaultSeconds)
        rerank = KeepAliveController(defaultSeconds: defaultSeconds)
    }

    /// Unload whichever slot is resident, idle and past its deadline. Returns
    /// the names of what was unloaded (for logging).
    public func evictExpired(embedEngine: EmbeddingEngine, rerankEngine: RerankEngine,
                             now: Date = Date()) async -> [String] {
        var out: [String] = []
        if await embed.shouldEvict(now: now) {
            if let name = embedEngine.loadedModelName, embedEngine.unload() { out.append(name) }
            await embed.markEvicted()
        }
        if await rerank.shouldEvict(now: now) {
            if let name = rerankEngine.loadedModelName, rerankEngine.unload() { out.append(name) }
            await rerank.markEvicted()
        }
        return out
    }
}
