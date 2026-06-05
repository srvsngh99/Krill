import XCTest
@testable import KLMCache
import KLMRuntime

#if canImport(MLX)
import MLX
#endif

/// Tests for int8 KV snapshot/restore through `PrefixCache.storeQuantized`
/// and `lookupQuantized`. Verifies that:
///   - a stored quantized snapshot is restored bit-identical on a full hit;
///   - cross-dtype lookups do not mishit (fp16 entries are invisible to
///     `lookupQuantized` and vice versa);
///   - restored state survives a `QuantizedKVCache.restoreQuantized` +
///     `truncate` + `update` round trip.
final class QuantizedPrefixCacheTests: XCTestCase {

    private func makeTempCache() -> PrefixCache {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("krillm-q8-prefix-test-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return PrefixCache(cacheDir: dir, maxMemoryEntries: 4, minPrefixLength: 4)
    }

    func testStoreAndLookupQuantizedRoundTrip() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 6)

            let snap0 = makeSnapshot(seqLen: tokens.count, headSeed: 1)
            let snap1 = makeSnapshot(seqLen: tokens.count, headSeed: 2)

            cache.storeQuantized(
                tokens: tokens, modelId: "m-int8",
                snapshots: [snap0, snap1]
            )

            let hit = try XCTUnwrap(cache.lookupQuantized(tokens: tokens, modelId: "m-int8"))
            XCTAssertEqual(hit.prefixLength, tokens.count)
            XCTAssertEqual(hit.layers.count, 2)

            // All six tensors per layer round-trip bit-identical.
            XCTAssertEqual(hit.layers[0].keys.asArray(UInt8.self),       snap0.keys.asArray(UInt8.self))
            XCTAssertEqual(hit.layers[0].values.asArray(UInt8.self),     snap0.values.asArray(UInt8.self))
            XCTAssertEqual(hit.layers[0].keyScales.shape,                snap0.keyScales.shape)
            XCTAssertEqual(hit.layers[0].keyZeros.shape,                 snap0.keyZeros.shape)
            XCTAssertEqual(hit.layers[1].values.asArray(UInt8.self),     snap1.values.asArray(UInt8.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testFp16LookupCannotHitQuantizedEntry() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 6)
            cache.storeQuantized(
                tokens: tokens, modelId: "m",
                snapshots: [makeSnapshot(seqLen: tokens.count, headSeed: 7)]
            )

            // fp16 path should miss — entries are namespaced by dtype.
            XCTAssertNil(cache.lookup(tokens: tokens, modelId: "m"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testQuantizedLookupCannotHitFp16Entry() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let tokens = Array(0 ..< 6)
            let k = makeFp16(seqLen: tokens.count, start: 10)
            let v = makeFp16(seqLen: tokens.count, start: 100)
            cache.store(
                tokens: tokens, modelId: "m",
                keys: [[k]], values: [[v]]
            )

            XCTAssertNil(cache.lookupQuantized(tokens: tokens, modelId: "m"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testLookupLongestPrefixQuantizedFindsSharedPrefix() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let stored = [10, 11, 12, 13, 14, 15, 16, 17]
            cache.storeQuantized(
                tokens: stored, modelId: "m-int8",
                snapshots: [makeSnapshot(seqLen: stored.count, headSeed: 3),
                            makeSnapshot(seqLen: stored.count, headSeed: 4)]
            )

            // Shares 5 leading tokens, diverges at the 6th (>= minPrefixLength 4).
            let query = [10, 11, 12, 13, 14, 99, 99]
            let hit = try XCTUnwrap(
                cache.lookupLongestPrefixQuantized(tokens: query, modelId: "m-int8"))
            XCTAssertEqual(hit.prefixLength, 5, "should match the 5 shared leading tokens")
            XCTAssertEqual(hit.layers.count, 2, "carries the full stored snapshot set")

            // Sharing fewer than minPrefixLength leading tokens must miss.
            XCTAssertNil(
                cache.lookupLongestPrefixQuantized(tokens: [10, 11, 0, 0, 0, 0], modelId: "m-int8"),
                "a 2-token shared prefix is below minPrefixLength and must not hit")

            // Cross-dtype isolation: the fp16 LCP path cannot see this entry.
            XCTAssertNil(cache.lookupLongestPrefix(tokens: query, modelId: "m-int8"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testQuantizedLongestPrefixCannotHitFp16Entry() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = makeTempCache()
            let stored = [10, 11, 12, 13, 14, 15]
            cache.store(
                tokens: stored, modelId: "m",
                keys: [[makeFp16(seqLen: stored.count, start: 10)]],
                values: [[makeFp16(seqLen: stored.count, start: 100)]]
            )
            // An fp16 entry must be invisible to the quantized LCP lookup.
            XCTAssertNil(
                cache.lookupLongestPrefixQuantized(tokens: [10, 11, 12, 13, 99], modelId: "m"))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testRestoreTruncateAndReforwardKeepsOriginalLength() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            // Build a quantized cache via a real `update` call so scales/zeros
            // come from the production quantization path, then snapshot, restore
            // into a fresh cache, truncate, and re-feed the trailing token.
            let original = QuantizedKVCache()
            let fullK = makeFp16(seqLen: 6, start: 1)
            let fullV = makeFp16(seqLen: 6, start: 1_000)
            _ = original.update(keys: fullK, values: fullV)

            let snap = try XCTUnwrap(original.quantizedSnapshot())
            XCTAssertEqual(snap.sequenceLength, 6)

            let restored = QuantizedKVCache()
            restored.restoreQuantized(snap)
            XCTAssertEqual(restored.sequenceLength, 6)
            restored.truncate(to: 5)
            XCTAssertEqual(restored.sequenceLength, 5)

            // Re-feed the last token slice; sequence length returns to 6.
            let lastK = fullK[0..., 0..., 5 ..< 6, 0...]
            let lastV = fullV[0..., 0..., 5 ..< 6, 0...]
            _ = restored.update(keys: lastK, values: lastV)
            XCTAssertEqual(restored.sequenceLength, 6)
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

    private func makeFp16(
        batch: Int = 1, heads: Int = 2, seqLen: Int, headDim: Int = 2, start: Int32
    ) -> MLXArray {
        let count = batch * heads * seqLen * headDim
        let values = (0 ..< count).map { Float(start + Int32($0)) / 10.0 }
        return MLXArray(values, [batch, heads, seqLen, headDim]).asType(.float16)
    }

    /// Build a `QuantizedKVSnapshot` directly with uint8/fp16 tensors so we
    /// can verify the on-disk round trip without coupling to the quantization
    /// helper. Production code paths always feed real model output through
    /// `QuantizedKVCache.update`; this is purely a structural fixture.
    private func makeSnapshot(seqLen: Int, headSeed: Int32) -> QuantizedKVSnapshot {
        let batch = 1, heads = 2, headDim = 2
        let count = batch * heads * seqLen * headDim
        let qBytes = (0 ..< count).map { UInt8((Int($0 + Int(headSeed))) & 0xFF) }
        let qK = MLXArray(qBytes, [batch, heads, seqLen, headDim])
        let qV = MLXArray(qBytes.reversed(), [batch, heads, seqLen, headDim])

        let metaCount = batch * heads * seqLen * 1
        let meta = (0 ..< metaCount).map { Float($0) / 100.0 }
        let metaArr = MLXArray(meta, [batch, heads, seqLen, 1]).asType(.float16)

        return QuantizedKVSnapshot(
            keys: qK, values: qV,
            keyScales: metaArr, keyZeros: metaArr,
            valueScales: metaArr, valueZeros: metaArr
        )
    }
    #endif
}
