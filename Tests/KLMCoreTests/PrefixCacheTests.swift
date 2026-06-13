import XCTest
@testable import KLMCache
import KLMRuntime

#if canImport(MLX)
import MLX
#endif

/// Tests for PrefixCache exact-hit replay and partial-hit fallback.
final class PrefixCacheTests: XCTestCase {

    private func makeTempCache() -> PrefixCache {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-prefix-test-\(UUID().uuidString)")
        addTeardownBlock {
            try? FileManager.default.removeItem(at: dir)
        }
        return PrefixCache(cacheDir: dir, maxMemoryEntries: 4, minPrefixLength: 4)
    }

    // MARK: - Cache miss behavior

    func testLookupReturnsNilForEmptyCache() {
        let cache = makeTempCache()
        let hit = cache.lookup(tokens: Array(0..<10), modelId: "test")
        XCTAssertNil(hit, "Empty cache should always miss")
    }

    func testBelowMinPrefixReturnsNil() {
        let cache = makeTempCache()  // minPrefixLength=4
        let hit = cache.lookup(tokens: [1, 2, 3], modelId: "test")
        XCTAssertNil(hit, "Tokens below minPrefixLength should never hit")
    }

    // MARK: - Model isolation miss behavior

    func testDifferentModelIdMisses() {
        // Even with the same tokens, different modelId should not match.
        // This is a contract test — actual cache store requires MLXArrays.
        let cache = makeTempCache()

        // No data stored, so lookup should miss regardless
        let hit = cache.lookup(tokens: Array(0..<10), modelId: "model-a")
        XCTAssertNil(hit)
    }

    // MARK: - Memory count tracking

    func testMemoryCountStartsAtZero() {
        let cache = makeTempCache()
        XCTAssertEqual(cache.memoryCount, 0)
    }

    func testClearResetsMemoryCount() {
        let cache = makeTempCache()
        cache.clear()
        XCTAssertEqual(cache.memoryCount, 0)
    }

    // MARK: - MLX-backed prefix cache behavior

    func testStoreAndExactLookupReturnsKVArrays() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 6)
            let keys = [
                [makeKV(seqLen: tokens.count, start: 10)],
                [makeKV(seqLen: tokens.count, start: 100)],
            ]
            let values = [
                [makeKV(seqLen: tokens.count, start: 1_000)],
                [makeKV(seqLen: tokens.count, start: 2_000)],
            ]

            cache.store(tokens: tokens, modelId: "model-a", keys: keys, values: values)

            let hit = try XCTUnwrap(cache.lookup(tokens: tokens, modelId: "model-a"))
            XCTAssertEqual(hit.prefixLength, tokens.count)
            XCTAssertEqual(hit.keys.count, 2)
            XCTAssertEqual(hit.values.count, 2)
            XCTAssertEqual(hit.keys[0][0].shape, [1, 2, 6, 2])
            XCTAssertEqual(hit.values[1][0].shape, [1, 2, 6, 2])
            XCTAssertEqual(hit.keys[0][0].asArray(Int32.self), keys[0][0].asArray(Int32.self))
            XCTAssertEqual(hit.keys[1][0].asArray(Int32.self), keys[1][0].asArray(Int32.self))
            XCTAssertEqual(hit.values[0][0].asArray(Int32.self), values[0][0].asArray(Int32.self))
            XCTAssertEqual(hit.values[1][0].asArray(Int32.self), values[1][0].asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testFullHitTruncateAndReforwardKeepsOriginalLength() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 6)
            let storedKeys = makeKV(seqLen: tokens.count, start: 10)
            let storedValues = makeKV(seqLen: tokens.count, start: 100)
            cache.store(
                tokens: tokens,
                modelId: "model-a",
                keys: [[storedKeys]],
                values: [[storedValues]]
            )

            let hit = try XCTUnwrap(cache.lookup(tokens: tokens, modelId: "model-a"))
            XCTAssertEqual(hit.prefixLength, tokens.count)

            let kvCache = KVCache()
            kvCache.restore(keys: hit.keys[0][0], values: hit.values[0][0])
            kvCache.truncate(to: tokens.count - 1)
            _ = kvCache.update(
                keys: storedKeys[0..., 0..., (tokens.count - 1) ..< tokens.count, 0...],
                values: storedValues[0..., 0..., (tokens.count - 1) ..< tokens.count, 0...]
            )

            let snapshot = try XCTUnwrap(kvCache.snapshot())
            XCTAssertEqual(kvCache.sequenceLength, tokens.count)
            XCTAssertEqual(snapshot.keys.shape, [1, 2, 6, 2])
            XCTAssertEqual(snapshot.values.shape, [1, 2, 6, 2])
            XCTAssertEqual(snapshot.keys.asArray(Int32.self), storedKeys.asArray(Int32.self))
            XCTAssertEqual(snapshot.values.asArray(Int32.self), storedValues.asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testCrossModelIsolationWithStoredKVArrays() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 5)
            let modelAKeys = makeKV(seqLen: tokens.count, start: 10)
            let modelAValues = makeKV(seqLen: tokens.count, start: 100)
            let modelBKeys = makeKV(seqLen: tokens.count, start: 1_000)
            let modelBValues = makeKV(seqLen: tokens.count, start: 2_000)

            cache.store(
                tokens: tokens,
                modelId: "model-a",
                keys: [[modelAKeys]],
                values: [[modelAValues]]
            )

            XCTAssertNil(cache.lookup(tokens: tokens, modelId: "model-b"))

            cache.store(
                tokens: tokens,
                modelId: "model-b",
                keys: [[modelBKeys]],
                values: [[modelBValues]]
            )

            let hitA = try XCTUnwrap(cache.lookup(tokens: tokens, modelId: "model-a"))
            let hitB = try XCTUnwrap(cache.lookup(tokens: tokens, modelId: "model-b"))
            XCTAssertEqual(hitA.prefixLength, tokens.count)
            XCTAssertEqual(hitB.prefixLength, tokens.count)
            XCTAssertEqual(hitA.keys[0][0].asArray(Int32.self), modelAKeys.asArray(Int32.self))
            XCTAssertEqual(hitA.values[0][0].asArray(Int32.self), modelAValues.asArray(Int32.self))
            XCTAssertEqual(hitB.keys[0][0].asArray(Int32.self), modelBKeys.asArray(Int32.self))
            XCTAssertEqual(hitB.values[0][0].asArray(Int32.self), modelBValues.asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    // MARK: - Per-entry size cap

    /// A temp cache with an explicit per-entry GB cap (bypasses env/default).
    private func makeCappedCache(maxEntryGB: Double) -> PrefixCache {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-prefix-cap-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return PrefixCache(cacheDir: dir, maxMemoryEntries: 4, minPrefixLength: 4,
                           maxEntryGB: maxEntryGB)
    }

    /// An entry under the cap stores and replays normally (the cap is inert on
    /// the common small-prefix path).
    func testEntryWithinCapIsStored() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            // makeKV default is 96 bytes/array; a 1e-6 GB (1000-byte) cap leaves
            // the ~192-byte entry comfortably under.
            let cache = makeCappedCache(maxEntryGB: 1e-6)
            let tokens = Array(0 ..< 6)
            cache.store(tokens: tokens, modelId: "m",
                        keys: [[makeKV(seqLen: tokens.count, start: 10)]],
                        values: [[makeKV(seqLen: tokens.count, start: 100)]])
            XCTAssertEqual(cache.memoryCount, 1, "under-cap entry must be retained")
            XCTAssertNotNil(cache.lookup(tokens: tokens, modelId: "m"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    /// An entry over the cap is skipped wholesale: nothing in memory, lookup
    /// misses, and no disk file is written. This is the full-attention
    /// long-context guard (a real ~10GB KV would otherwise spike memory).
    func testOversizedEntryIsSkipped() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            // ~192-byte entry against a 1e-7 GB (100-byte) cap -> over budget.
            let cache = makeCappedCache(maxEntryGB: 1e-7)
            let tokens = Array(0 ..< 6)
            cache.store(tokens: tokens, modelId: "m",
                        keys: [[makeKV(seqLen: tokens.count, start: 10)]],
                        values: [[makeKV(seqLen: tokens.count, start: 100)]])
            cache.waitForDiskWrites()
            XCTAssertEqual(cache.memoryCount, 0, "over-cap entry must not be retained")
            XCTAssertNil(cache.lookup(tokens: tokens, modelId: "m"))
            XCTAssertEqual(cache.diskCount, 0, "over-cap entry must not hit disk")
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    /// `maxEntryGB <= 0` disables the cap: the same entry that the tiny cap
    /// rejected is now stored (legacy unbounded behavior, opt-in).
    func testCapDisabledStoresAnySize() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeCappedCache(maxEntryGB: 0)
            let tokens = Array(0 ..< 6)
            cache.store(tokens: tokens, modelId: "m",
                        keys: [[makeKV(seqLen: tokens.count, start: 10)]],
                        values: [[makeKV(seqLen: tokens.count, start: 100)]])
            XCTAssertEqual(cache.memoryCount, 1, "cap disabled -> entry stored regardless of size")
            XCTAssertNotNil(cache.lookup(tokens: tokens, modelId: "m"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    // MARK: - Longest-common-prefix (shared-prefix) lookup

    func testCommonPrefixLength() {
        XCTAssertEqual(PrefixCache.commonPrefixLength([1, 2, 3, 4], [1, 2, 3, 4]), 4)
        XCTAssertEqual(PrefixCache.commonPrefixLength([1, 2, 3, 9], [1, 2, 3, 4]), 3)
        XCTAssertEqual(PrefixCache.commonPrefixLength([1, 2], [1, 2, 3, 4]), 2, "bounded by shorter")
        XCTAssertEqual(PrefixCache.commonPrefixLength([9, 1], [1, 2]), 0)
        XCTAssertEqual(PrefixCache.commonPrefixLength([], [1]), 0)
    }

    /// Core of issue #0: a request that SHARES a long prefix with a recent
    /// prefill but diverges in the tail must find that prefix and report the
    /// shared length, carrying the stored entry's full KV for the caller to
    /// restore-then-truncate. This is the path that turns an agentic/RAG
    /// re-prefill into a suffix-only prefill.
    func testLookupLongestPrefixSharedPrefixHits() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()  // minPrefixLength = 4
            let shared = Array(0 ..< 10)
            let stored = shared + [100, 101]                 // prefix + tail A
            let storedKeys = makeKV(seqLen: stored.count, start: 10)
            let storedValues = makeKV(seqLen: stored.count, start: 100)
            cache.store(tokens: stored, modelId: "m",
                        keys: [[storedKeys]], values: [[storedValues]])

            // A different tail over the same shared prefix.
            let query = shared + [200, 201, 202]
            let hit = try XCTUnwrap(
                cache.lookupLongestPrefix(tokens: query, modelId: "m"),
                "shared-prefix request must find the recent prefill")
            XCTAssertEqual(hit.prefixLength, shared.count,
                           "reported length is the shared prefix, not the stored length")
            // Carries the entry's FULL stored KV (caller truncates to prefixLength).
            XCTAssertEqual(hit.keys[0][0].shape, [1, 2, stored.count, 2])
            XCTAssertEqual(hit.keys[0][0].asArray(Int32.self), storedKeys.asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testLookupLongestPrefixBelowMinMisses() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()  // minPrefixLength = 4
            let stored = Array(0 ..< 8)
            cache.store(tokens: stored, modelId: "m",
                        keys: [[makeKV(seqLen: stored.count, start: 1)]],
                        values: [[makeKV(seqLen: stored.count, start: 9)]])
            // Shares only 3 leading tokens -> below minPrefixLength -> miss.
            let query = [0, 1, 2, 77, 78, 79, 80, 81]
            XCTAssertNil(cache.lookupLongestPrefix(tokens: query, modelId: "m"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testLookupLongestPrefixModelIsolation() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let stored = Array(0 ..< 10) + [5, 6]
            cache.store(tokens: stored, modelId: "model-a",
                        keys: [[makeKV(seqLen: stored.count, start: 1)]],
                        values: [[makeKV(seqLen: stored.count, start: 9)]])
            // Same shared prefix, different model -> must not match.
            XCTAssertNil(cache.lookupLongestPrefix(
                tokens: Array(0 ..< 10) + [9, 9], modelId: "model-b"))
            // And a different mediaHash must not match a text (nil) entry.
            XCTAssertNil(cache.lookupLongestPrefix(
                tokens: Array(0 ..< 10) + [9, 9], modelId: "model-a", mediaHash: "img:X"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testLookupLongestPrefixPrefersLongestMatch() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()  // maxMemoryEntries = 4
            let short = Array(0 ..< 6) + [50]            // shares 6 with query
            let long = Array(0 ..< 9) + [60]             // shares 9 with query
            cache.store(tokens: short, modelId: "m",
                        keys: [[makeKV(seqLen: short.count, start: 1)]],
                        values: [[makeKV(seqLen: short.count, start: 5)]])
            let longKeys = makeKV(seqLen: long.count, start: 200)
            cache.store(tokens: long, modelId: "m",
                        keys: [[longKeys]],
                        values: [[makeKV(seqLen: long.count, start: 400)]])

            let query = Array(0 ..< 9) + [99, 99]
            let hit = try XCTUnwrap(cache.lookupLongestPrefix(tokens: query, modelId: "m"))
            XCTAssertEqual(hit.prefixLength, 9, "must pick the entry with the longer shared prefix")
            XCTAssertEqual(hit.keys[0][0].asArray(Int32.self), longKeys.asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    // MARK: - Concurrency

    /// Hammer `store` / `lookup` / `clear` from many threads at once. The
    /// serial `generate` path and the batched decode path now share one
    /// `PrefixCache` and each prefills on its own Task, so these calls genuinely
    /// race the `memoryCache` dictionary + `accessOrder` LRU array. Without the
    /// guarding lock this traps (concurrent dictionary/array mutation) or leaves
    /// the LRU bookkeeping inconsistent; with it the cache survives and never
    /// exceeds its capacity bound.
    func testConcurrentStoreLookupClearIsSafe() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()   // maxMemoryEntries = 4, minPrefixLength = 4
            let threads = 8
            let iterations = 200
            DispatchQueue.concurrentPerform(iterations: threads) { t in
                for i in 0 ..< iterations {
                    let tokens = Array((t * 1000 + i) ..< (t * 1000 + i + 6))
                    let modelId = "m\(i % 3)"
                    cache.store(
                        tokens: tokens, modelId: modelId,
                        keys: [[self.makeKV(seqLen: tokens.count, start: Int32(i))]],
                        values: [[self.makeKV(seqLen: tokens.count, start: Int32(i + 5000))]])
                    _ = cache.lookup(tokens: tokens, modelId: modelId)
                    _ = cache.lookup(tokens: Array(0 ..< 6), modelId: modelId)
                    if i % 50 == 0 { cache.clear() }
                }
            }
            // The capacity bound must still hold after the storm - proof the LRU
            // eviction never raced into an inconsistent state.
            XCTAssertLessThanOrEqual(cache.memoryCount, 4)
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    // MARK: - Disk budget enforcement (issue #177)

    private func tempDir() -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-prefix-budget-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    /// A `KRILL_PREFIX_CACHE_GB=0` budget disables the disk tier: nothing is
    /// persisted, but the in-memory LRU still serves hits.
    func testDiskBudgetZeroDisablesDiskWrites() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = PrefixCache(
                cacheDir: tempDir(), maxMemoryEntries: 4, minPrefixLength: 4, diskBudgetGB: 0)
            let tokens = Array(0 ..< 8)
            cache.store(
                tokens: tokens, modelId: "m",
                keys: [[makeKV(seqLen: tokens.count, start: 1)]],
                values: [[makeKV(seqLen: tokens.count, start: 9)]])
            cache.waitForDiskWrites()

            XCTAssertEqual(cache.diskBytes, 0, "budget=0 must write nothing to disk")
            XCTAssertNotNil(
                cache.lookup(tokens: tokens, modelId: "m"),
                "in-memory tier must still serve hits when the disk tier is disabled")
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    /// A run of distinct prefixes (none ever reused) must not grow the disk
    /// tier without bound: LRU eviction keeps it within the byte budget.
    func testDiskBudgetEvictsColdEntriesToStayBounded() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            // Measure one entry's on-disk footprint with an unbounded cache.
            let probe = PrefixCache(
                cacheDir: tempDir(), maxMemoryEntries: 16, minPrefixLength: 4, diskBudgetGB: -1)
            probe.store(
                tokens: [0, 1, 2, 3, 4, 5], modelId: "m",
                keys: [[makeKV(seqLen: 6, start: 1)]],
                values: [[makeKV(seqLen: 6, start: 9)]])
            probe.waitForDiskWrites()
            let oneSize = probe.diskBytes
            XCTAssertGreaterThan(oneSize, 0, "probe entry should have written to disk")

            // Budget sized to hold ~3 entries; store 12 distinct prefixes.
            let budgetGB = Double(oneSize * 3) / 1_000_000_000
            let budgetBytes = Int64(budgetGB * 1_000_000_000)
            let cache = PrefixCache(
                cacheDir: tempDir(), maxMemoryEntries: 4, minPrefixLength: 4, diskBudgetGB: budgetGB)
            for i in 0 ..< 12 {
                let tokens = (0 ..< 6).map { i * 100 + $0 }  // distinct prefix per i
                cache.store(
                    tokens: tokens, modelId: "m",
                    keys: [[makeKV(seqLen: 6, start: Int32(i))]],
                    values: [[makeKV(seqLen: 6, start: Int32(i + 1000))]])
            }
            cache.waitForDiskWrites()

            let bytes = cache.diskBytes
            XCTAssertLessThanOrEqual(bytes, budgetBytes, "disk tier must stay within its byte budget")
            XCTAssertGreaterThan(bytes, 0, "recent entries should survive eviction")
            XCTAssertLessThan(
                bytes, oneSize * 12, "eviction must have dropped cold entries (not unbounded growth)")
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    #if canImport(MLX) && os(macOS) && arch(arm64)
    private func withMLXCPU(_ body: () throws -> Void) throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is unavailable to this test process.")
        }

        try Device.withDefaultDevice(.cpu) {
            try body()
        }
    }

    private func makeKV(
        batch: Int = 1,
        heads: Int = 2,
        seqLen: Int,
        headDim: Int = 2,
        start: Int32
    ) -> MLXArray {
        let count = batch * heads * seqLen * headDim
        let values = (0 ..< count).map { start + Int32($0) }
        return MLXArray(values, [batch, heads, seqLen, headDim])
    }
    #endif
}
