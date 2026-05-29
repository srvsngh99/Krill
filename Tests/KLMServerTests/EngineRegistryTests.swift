import XCTest
import Foundation
import KLMEngine
import KLMCache
import KLMRegistry
@testable import KLMServer

/// Unit tests for the `EngineRegistry` eviction invariant — the most
/// failure-prone part of the Stage A multi-model pool: an in-flight
/// (currently-generating) model must NEVER be chosen as an eviction victim,
/// or a concurrent load would tear an engine down mid-stream. The victim
/// selection is factored into a pure function so it can be tested without
/// loading real model weights.
final class EngineRegistryTests: XCTestCase {

    func testPicksLeastRecentlyUsedWhenNothingInFlight() {
        // order is oldest-first; the head (LRU) is the victim.
        let victim = EngineRegistry.selectEvictionVictim(
            order: ["a", "b", "c"], inFlight: [:])
        XCTAssertEqual(victim, "a")
    }

    func testSkipsInFlightAndPicksNextLRU() {
        // 'a' is the LRU but is generating; the next idle LRU ('b') is chosen.
        let victim = EngineRegistry.selectEvictionVictim(
            order: ["a", "b", "c"], inFlight: ["a": 1])
        XCTAssertEqual(victim, "b")
    }

    func testSkipsMultipleInFlight() {
        let victim = EngineRegistry.selectEvictionVictim(
            order: ["a", "b", "c"], inFlight: ["a": 2, "b": 1])
        XCTAssertEqual(victim, "c")
    }

    func testReturnsNilWhenEveryResidentIsInFlight() {
        // Pool full and all busy -> no victim -> caller must throw PoolBusy
        // (a meaningful 503) instead of evicting a generating model.
        let victim = EngineRegistry.selectEvictionVictim(
            order: ["a", "b"], inFlight: ["a": 1, "b": 3])
        XCTAssertNil(victim)
    }

    func testReturnsNilForEmptyPool() {
        XCTAssertNil(EngineRegistry.selectEvictionVictim(order: [], inFlight: [:]))
    }

    func testZeroCountEntryIsEvictable() {
        // A released hold decrements to 0; the registry removes the key, but
        // even a lingering explicit 0 must be treated as idle/evictable.
        let victim = EngineRegistry.selectEvictionVictim(
            order: ["a", "b"], inFlight: ["a": 0])
        XCTAssertEqual(victim, "a")
    }

    // MARK: - Per-model keep-alive eviction (Stage A-2)

    private func makeRegistry() -> EngineRegistry {
        EngineRegistry(maxLoaded: 4, registry: Registry(),
                       prefixCache: PrefixCache(), activeRef: ActiveEngineRef())
    }

    private func unloadedEngine(_ name: String) -> InferenceEngine {
        // The init stores the directory only; no weights are loaded.
        InferenceEngine(modelDirectory: URL(fileURLWithPath: "/tmp/krill-test-\(name)"))
    }

    func testEvictExpiredRemovesOnlyExpiredIdleModels() async {
        let reg = makeRegistry()
        // 'a' expires immediately (deadline = load time); 'b' has an hour left.
        await reg._insertForTesting(key: "a", engine: unloadedEngine("a"),
                                    keepAlive: KeepAliveController(defaultSeconds: 0))
        await reg._insertForTesting(key: "b", engine: unloadedEngine("b"),
                                    keepAlive: KeepAliveController(defaultSeconds: 3600))
        let evicted = await reg.evictExpired(now: Date().addingTimeInterval(1))
        XCTAssertEqual(evicted, ["a"])
        let remaining = await reg.residentCount
        XCTAssertEqual(remaining, 1, "the fresh model must stay resident")
    }

    func testEvictExpiredSkipsInFlightModelEvenIfDeadlinePassed() async {
        let reg = makeRegistry()
        let ka = KeepAliveController(defaultSeconds: 0)   // deadline already passed
        await ka.beginRequest()                            // ...but it is generating
        await reg._insertForTesting(key: "c", engine: unloadedEngine("c"), keepAlive: ka)
        let evicted = await reg.evictExpired(now: Date().addingTimeInterval(1))
        XCTAssertTrue(evicted.isEmpty, "an in-flight model is never auto-evicted")
        let remaining = await reg.residentCount
        XCTAssertEqual(remaining, 1)
    }

    func testPinnedModelIsNeverEvicted() async {
        let reg = makeRegistry()
        let ka = KeepAliveController(defaultSeconds: 0)
        await ka.touch(override: -1)                       // keep_alive: -1 -> pinned
        await reg._insertForTesting(key: "d", engine: unloadedEngine("d"), keepAlive: ka)
        let evicted = await reg.evictExpired(now: Date().addingTimeInterval(100_000))
        XCTAssertTrue(evicted.isEmpty, "a pinned model has no deadline")
        let remaining = await reg.residentCount
        XCTAssertEqual(remaining, 1)
    }
}
