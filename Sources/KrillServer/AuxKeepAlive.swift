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
    /// EmbeddingGemma 2 modality towers load lazily and expire on their own
    /// clocks, armed ONLY by requests that carry that modality. A text-only
    /// server that once served an image drops back to the text-only footprint
    /// after the idle TTL even while text requests keep the text tower warm.
    public let vision: KeepAliveController
    public let audio: KeepAliveController

    public init(defaultSeconds: Int) {
        embed = KeepAliveController(defaultSeconds: defaultSeconds)
        rerank = KeepAliveController(defaultSeconds: defaultSeconds)
        vision = KeepAliveController(defaultSeconds: defaultSeconds)
        audio = KeepAliveController(defaultSeconds: defaultSeconds)
    }

    /// Unload whichever slot is resident, idle and past its deadline. Returns
    /// the names of what was unloaded (for logging).
    public func evictExpired(embedEngine: EmbeddingEngine, rerankEngine: RerankEngine,
                             now: Date = Date()) async -> [String] {
        var out: [String] = []
        // Check-and-unload is one actor step (see KeepAliveController.evictIfDue):
        // a request that begins in between can never lose its model.
        await embed.evictIfDue(now: now) {
            if let name = embedEngine.loadedModelName, embedEngine.unload() { out.append(name) }
        }
        await rerank.evictIfDue(now: now) {
            if let name = rerankEngine.loadedModelName, rerankEngine.unload() { out.append(name) }
        }
        await vision.evictIfDue(now: now) {
            if embedEngine.unloadVisionTower() { out.append("vision tower") }
        }
        await audio.evictIfDue(now: now) {
            if embedEngine.unloadAudioTower() { out.append("audio tower") }
        }
        return out
    }
}
