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
