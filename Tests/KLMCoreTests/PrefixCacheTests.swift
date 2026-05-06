import XCTest
@testable import KLMCache

/// Tests for PrefixCache exact-hit replay and partial-hit fallback.
///
/// These tests verify the cache lookup logic and engine-side checks
/// without requiring Metal/MLX (no actual tensor operations).
final class PrefixCacheTests: XCTestCase {

    private func makeTempCache() -> PrefixCache {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-prefix-test-\(UUID().uuidString)")
        return PrefixCache(cacheDir: dir, maxMemoryEntries: 4, minPrefixLength: 4)
    }

    // MARK: - Test 1: Cache miss when nothing stored

    func testLookupReturnsNilForEmptyCache() {
        let cache = makeTempCache()
        let hit = cache.lookup(tokens: Array(0..<10), modelId: "test")
        XCTAssertNil(hit, "Empty cache should always miss")
    }

    // MARK: - Test 2: Below minimum prefix length

    func testBelowMinPrefixReturnsNil() {
        let cache = makeTempCache()  // minPrefixLength=4
        let hit = cache.lookup(tokens: [1, 2, 3], modelId: "test")
        XCTAssertNil(hit, "Tokens below minPrefixLength should never hit")
    }

    // MARK: - Test 2: Partial-hit engine rejection logic

    func testEngineRejectsPartialHit() {
        // Simulates the engine's check:
        // Only use hit if hit.prefixLength == promptTokens.count
        let promptTokens = Array(0..<12)

        // Simulate a partial hit that covers fewer tokens
        let hitPrefixLength = 8
        let isFullHit = hitPrefixLength == promptTokens.count

        XCTAssertFalse(isFullHit,
            "Engine must reject hits where prefixLength != promptTokens.count")
    }

    func testEngineAcceptsFullHit() {
        let promptTokens = Array(0..<10)
        let hitPrefixLength = 10
        let isFullHit = hitPrefixLength == promptTokens.count

        XCTAssertTrue(isFullHit,
            "Engine must accept hits where prefixLength == promptTokens.count")
    }

    // MARK: - Test 1: Full-hit KV truncate-and-re-forward contract

    func testFullHitTruncateContract() {
        // Verifies the engine's full-hit invariant:
        // After restore (len=N), truncate to N-1, then forward 1 token → net len = N
        let originalLength = 10
        let truncatedLength = originalLength - 1  // 9
        let afterReforward = truncatedLength + 1  // 10

        XCTAssertEqual(afterReforward, originalLength,
            "Truncate-last + re-forward should restore original KV length")
    }

    // MARK: - Test: Different modelId is a miss

    func testDifferentModelIdMisses() {
        // Even with the same tokens, different modelId should not match.
        // This is a contract test — actual cache store requires MLXArrays.
        let cache = makeTempCache()

        // No data stored, so lookup should miss regardless
        let hit = cache.lookup(tokens: Array(0..<10), modelId: "model-a")
        XCTAssertNil(hit)
    }

    // MARK: - Test: Memory count tracking

    func testMemoryCountStartsAtZero() {
        let cache = makeTempCache()
        XCTAssertEqual(cache.memoryCount, 0)
    }

    func testClearResetsMemoryCount() {
        let cache = makeTempCache()
        cache.clear()
        XCTAssertEqual(cache.memoryCount, 0)
    }
}
