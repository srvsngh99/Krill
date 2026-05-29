import XCTest
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
}
