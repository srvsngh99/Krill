import XCTest
@testable import KrillCache
import KrillRuntime

#if canImport(MLX)
import MLX
#endif

/// Tests for KVCache snapshot/restore/truncate behavior.
final class KVCacheTests: XCTestCase {

    // MARK: - Basic KVCache contract (no MLX required)

    func testEmptyCacheHasZeroLength() {
        let cache = KVCache()
        XCTAssertEqual(cache.sequenceLength, 0)
        XCTAssertNil(cache.snapshot())
    }

    func testResetClearsState() {
        let cache = KVCache()
        // After reset, should still be empty
        cache.reset()
        XCTAssertEqual(cache.sequenceLength, 0)
        XCTAssertNil(cache.snapshot())
    }

    func testTruncateOnEmptyCacheIsNoop() {
        let cache = KVCache()
        cache.truncate(to: 5)
        XCTAssertEqual(cache.sequenceLength, 0, "Truncating empty cache should be no-op")
    }

    // MARK: - MLX-backed KV behavior

    func testUpdateReturnsExpectedShapesAndSequenceLength() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            let result = cache.update(
                keys: makeKV(seqLen: 2, headDim: 4, start: 10),
                values: makeKV(seqLen: 2, headDim: 4, start: 100)
            )

            XCTAssertEqual(result.0.shape, [1, 2, 2, 4])
            XCTAssertEqual(result.1.shape, [1, 2, 2, 4])
            XCTAssertEqual(cache.sequenceLength, 2)
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testUpdateConcatenatesAlongSequenceAxis() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            _ = cache.update(
                keys: makeKV(seqLen: 1, headDim: 3, start: 0),
                values: makeKV(seqLen: 1, headDim: 3, start: 50)
            )
            let result = cache.update(
                keys: makeKV(seqLen: 2, headDim: 3, start: 100),
                values: makeKV(seqLen: 2, headDim: 3, start: 150)
            )

            XCTAssertEqual(result.0.shape, [1, 2, 3, 3])
            XCTAssertEqual(result.0.asArray(Int32.self), [
                0, 1, 2,
                100, 101, 102,
                103, 104, 105,
                3, 4, 5,
                106, 107, 108,
                109, 110, 111,
            ])
            XCTAssertEqual(result.1.asArray(Int32.self), [
                50, 51, 52,
                150, 151, 152,
                153, 154, 155,
                53, 54, 55,
                156, 157, 158,
                159, 160, 161,
            ])
            XCTAssertEqual(cache.sequenceLength, 3)
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testSnapshotReturnsCurrentStateShape() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            _ = cache.update(
                keys: makeKV(seqLen: 3, headDim: 2, start: 10),
                values: makeKV(seqLen: 3, headDim: 2, start: 20)
            )

            let snapshot = try XCTUnwrap(cache.snapshot())
            XCTAssertEqual(snapshot.keys.shape, [1, 2, 3, 2])
            XCTAssertEqual(snapshot.values.shape, [1, 2, 3, 2])
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testRestoreOverwritesStateAndSetsSequenceLength() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            _ = cache.update(
                keys: makeKV(seqLen: 4, headDim: 2, start: 0),
                values: makeKV(seqLen: 4, headDim: 2, start: 20)
            )

            let restoredKeys = makeKV(seqLen: 2, headDim: 2, start: 200)
            let restoredValues = makeKV(seqLen: 2, headDim: 2, start: 300)
            cache.restore(keys: restoredKeys, values: restoredValues)

            let snapshot = try XCTUnwrap(cache.snapshot())
            XCTAssertEqual(cache.sequenceLength, 2)
            XCTAssertEqual(snapshot.keys.shape, [1, 2, 2, 2])
            XCTAssertEqual(snapshot.values.shape, [1, 2, 2, 2])
            XCTAssertEqual(snapshot.keys.asArray(Int32.self), restoredKeys.asArray(Int32.self))
            XCTAssertEqual(snapshot.values.asArray(Int32.self), restoredValues.asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testTruncateSlicesSequenceAxisAndBeyondLengthIsNoop() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            _ = cache.update(
                keys: makeKV(seqLen: 4, headDim: 2, start: 0),
                values: makeKV(seqLen: 4, headDim: 2, start: 100)
            )

            cache.truncate(to: 3)
            var snapshot = try XCTUnwrap(cache.snapshot())
            XCTAssertEqual(cache.sequenceLength, 3)
            XCTAssertEqual(snapshot.keys.shape, [1, 2, 3, 2])
            XCTAssertEqual(snapshot.values.shape, [1, 2, 3, 2])
            XCTAssertEqual(snapshot.keys.asArray(Int32.self), [
                0, 1,
                2, 3,
                4, 5,
                8, 9,
                10, 11,
                12, 13,
            ])
            XCTAssertEqual(snapshot.values.asArray(Int32.self), [
                100, 101,
                102, 103,
                104, 105,
                108, 109,
                110, 111,
                112, 113,
            ])

            let keysAfterTruncate = snapshot.keys.asArray(Int32.self)
            let valuesAfterTruncate = snapshot.values.asArray(Int32.self)
            cache.truncate(to: 6)

            snapshot = try XCTUnwrap(cache.snapshot())
            XCTAssertEqual(cache.sequenceLength, 3)
            XCTAssertEqual(snapshot.keys.shape, [1, 2, 3, 2])
            XCTAssertEqual(snapshot.values.shape, [1, 2, 3, 2])
            XCTAssertEqual(snapshot.keys.asArray(Int32.self), keysAfterTruncate)
            XCTAssertEqual(snapshot.values.asArray(Int32.self), valuesAfterTruncate)
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testResetAfterUpdateClearsState() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            _ = cache.update(
                keys: makeKV(seqLen: 2, start: 10),
                values: makeKV(seqLen: 2, start: 20)
            )

            cache.reset()

            XCTAssertEqual(cache.sequenceLength, 0)
            XCTAssertNil(cache.snapshot())
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testUpdateAfterRestoreConcatenates() throws {
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            cache.restore(
                keys: makeKV(seqLen: 2, headDim: 2, start: 200),
                values: makeKV(seqLen: 2, headDim: 2, start: 300)
            )

            let result = cache.update(
                keys: makeKV(seqLen: 1, headDim: 2, start: 400),
                values: makeKV(seqLen: 1, headDim: 2, start: 500)
            )

            XCTAssertEqual(cache.sequenceLength, 3)
            XCTAssertEqual(result.0.shape, [1, 2, 3, 2])
            XCTAssertEqual(result.1.shape, [1, 2, 3, 2])
            XCTAssertEqual(result.0.asArray(Int32.self), [
                200, 201,
                202, 203,
                400, 401,
                204, 205,
                206, 207,
                402, 403,
            ])
            XCTAssertEqual(result.1.asArray(Int32.self), [
                300, 301,
                302, 303,
                500, 501,
                304, 305,
                306, 307,
                502, 503,
            ])
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testTruncateThenUpdateOverwritesStaleRows() throws {
        // The spec-decode rollback pattern: append K tokens, reject some
        // (truncate), then append replacements. With the in-place buffer the
        // truncated rows are stale storage that the next update must overwrite.
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            _ = cache.update(
                keys: makeKV(seqLen: 10, headDim: 2, start: 0),
                values: makeKV(seqLen: 10, headDim: 2, start: 1000)
            )
            cache.truncate(to: 6)
            XCTAssertEqual(cache.sequenceLength, 6)

            let result = cache.update(
                keys: makeKV(seqLen: 2, headDim: 2, start: 500),
                values: makeKV(seqLen: 2, headDim: 2, start: 600)
            )
            XCTAssertEqual(cache.sequenceLength, 8)
            XCTAssertEqual(result.0.shape, [1, 2, 8, 2])
            // Reference: a fresh cache fed the same final sequence.
            let ref = KVCache()
            _ = ref.update(
                keys: makeKV(seqLen: 10, headDim: 2, start: 0)[0..., 0..., 0 ..< 6, 0...],
                values: makeKV(seqLen: 10, headDim: 2, start: 1000)[0..., 0..., 0 ..< 6, 0...]
            )
            let refResult = ref.update(
                keys: makeKV(seqLen: 2, headDim: 2, start: 500),
                values: makeKV(seqLen: 2, headDim: 2, start: 600)
            )
            XCTAssertEqual(result.0.asArray(Int32.self), refResult.0.asArray(Int32.self))
            XCTAssertEqual(result.1.asArray(Int32.self), refResult.1.asArray(Int32.self))
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testRestoreDoesNotAliasSourceArrays() throws {
        // restore() must copy: the restored source (e.g. a prefix-cache LRU
        // entry) must NOT change when the cache is subsequently truncated and
        // updated in place.
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let sourceKeys = makeKV(seqLen: 4, headDim: 2, start: 200)
            let sourceValues = makeKV(seqLen: 4, headDim: 2, start: 300)
            let sourceKeysBefore = sourceKeys.asArray(Int32.self)
            let sourceValuesBefore = sourceValues.asArray(Int32.self)

            let cache = KVCache()
            cache.restore(keys: sourceKeys, values: sourceValues)
            cache.truncate(to: 2)
            _ = cache.update(
                keys: makeKV(seqLen: 2, headDim: 2, start: 900),
                values: makeKV(seqLen: 2, headDim: 2, start: 950)
            )

            XCTAssertEqual(sourceKeys.asArray(Int32.self), sourceKeysBefore,
                "restore() aliased the source keys - in-place writes corrupted them")
            XCTAssertEqual(sourceValues.asArray(Int32.self), sourceValuesBefore,
                "restore() aliased the source values - in-place writes corrupted them")
        }
        #else
        throw XCTSkip("MLX tensor tests require MLX on macOS arm64.")
        #endif
    }

    func testGrowthAcrossCapacityBoundaryPreservesContents() throws {
        // Push the cache across at least one internal capacity-growth
        // reallocation and verify the whole sequence survives intact.
        #if canImport(MLX) && os(macOS) && arch(arm64)
        try withMLXCPU {
            let cache = KVCache()
            var expectedKeys: [Int32] = []
            let steps = 300  // > one 256-row growth step with seqLen 2 chunks
            for i in 0 ..< steps {
                let start = Int32(i * 10)
                _ = cache.update(
                    keys: makeKV(heads: 1, seqLen: 2, headDim: 1, start: start),
                    values: makeKV(heads: 1, seqLen: 2, headDim: 1, start: start)
                )
                expectedKeys.append(start)
                expectedKeys.append(start + 1)
            }
            XCTAssertEqual(cache.sequenceLength, steps * 2)
            let snapshot = try XCTUnwrap(cache.snapshot())
            XCTAssertEqual(snapshot.keys.shape, [1, 1, steps * 2, 1])
            XCTAssertEqual(snapshot.keys.asArray(Int32.self), expectedKeys)
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
