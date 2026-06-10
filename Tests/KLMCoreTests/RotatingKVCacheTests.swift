import XCTest
@testable import KLMCache
import KLMCore
import KLMRuntime

#if canImport(MLX)
import MLX
import MLXFast
import MLXRandom
#endif

/// Spec tests for `RotatingKVCache`: the retained window + NO mask must be
/// numerically identical to the full `KVCache` + sliding-window mask, at both
/// decode (L=1) and chunked-prefill (L > window) shapes. This is the gate for
/// the window-1 retention invariant (a query attends itself + the previous
/// `window - 1` keys).
final class RotatingKVCacheTests: XCTestCase {

    #if canImport(MLX) && os(macOS) && arch(arm64)
    private func requireMLX() throws {
        guard MLXMetalRuntime.canInitializeMLXForTests else {
            throw XCTSkip("MLX Metal runtime is unavailable to this test process.")
        }
    }

    private func randKV(L: Int, h: Int = 2, d: Int = 8) -> (MLXArray, MLXArray) {
        (MLXRandom.normal([1, h, L, d]).asType(.float32),
         MLXRandom.normal([1, h, L, d]).asType(.float32))
    }

    private func sdpa(
        _ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        let mode: MLXFast.ScaledDotProductAttentionMaskMode =
            mask != nil ? .array(mask!) : .none
        return MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: mode)
    }

    /// DECODE PARITY: T single-token steps. Reference = full KVCache + sliding
    /// mask over the whole history; rotating = retained window + no mask. The
    /// outputs must match (the reference's -10000 additive leaks ~exp(-10000)
    /// of masked keys; tolerance 1e-4 absorbs it).
    func testDecodeParityAgainstFullCachePlusMask() throws {
        try requireMLX()
        MLXRandom.seed(42)
        let window = 8
        let steps = 3 * window  // well past the window
        let full = KVCache()
        let rot = RotatingKVCache(window: window)

        for t in 0 ..< steps {
            let (nk, nv) = randKV(L: 1)
            let q = MLXRandom.normal([1, 2, 1, 8]).asType(.float32)

            let cacheLen = full.sequenceLength
            XCTAssertEqual(rot.sequenceLength, cacheLen,
                "sequenceLength must report TOTAL tokens seen")

            let (fk, fv) = full.update(keys: nk, values: nv)
            let mask = createSlidingWindowCausalMask(
                newLen: 1, cacheLen: cacheLen, window: window, dtype: .float32)
            let refOut = sdpa(q, fk, fv, mask: mask)

            let (rk, rv) = rot.update(keys: nk, values: nv)
            XCTAssertEqual(rk.dim(2), Swift.min(t + 1, window),
                "retained width after L=1 append must be min(totalSeen, window)")
            let rotOut = sdpa(q, rk, rv, mask: nil)

            let diff = MLX.max(MLX.abs(refOut - rotOut)).item(Float.self)
            XCTAssertLessThan(diff, 1e-4,
                "decode parity broke at step \(t): maxAbsDiff=\(diff)")
        }
    }

    /// CHUNK PARITY: multi-token appends with chunk sizes straddling the
    /// window (including chunk > window, the chunked-prefill shape).
    /// Reference = full cache + sliding mask (absolute positions); rotating =
    /// retained window + the SAME mask builder with `cacheLen =
    /// maskCacheLength` (relative distances are preserved).
    func testChunkParityAgainstFullCachePlusMask() throws {
        try requireMLX()
        MLXRandom.seed(7)
        let window = 8
        let full = KVCache()
        let rot = RotatingKVCache(window: window)

        for (i, L) in [5, 3, 12, 1, 9, 6].enumerated() {  // 12 and 9 exceed window
            let (nk, nv) = randKV(L: L)
            let q = MLXRandom.normal([1, 2, L, 8]).asType(.float32)

            let cacheLen = full.sequenceLength
            let (fk, fv) = full.update(keys: nk, values: nv)
            let refMask = createSlidingWindowCausalMask(
                newLen: L, cacheLen: cacheLen, window: window, dtype: .float32)
            let refOut = sdpa(q, fk, fv, mask: refMask)

            let rotCacheLen = rot.maskCacheLength
            let (rk, rv) = rot.update(keys: nk, values: nv)
            XCTAssertEqual(rk.dim(2), rotCacheLen + L,
                "retained width after append must be maskCacheLength + L")
            let rotMask = createSlidingWindowCausalMask(
                newLen: L, cacheLen: rotCacheLen, window: window, dtype: .float32)
            let rotOut = sdpa(q, rk, rv, mask: rotMask)

            let diff = MLX.max(MLX.abs(refOut - rotOut)).item(Float.self)
            XCTAssertLessThan(diff, 1e-4,
                "chunk parity broke at append \(i) (L=\(L)): maxAbsDiff=\(diff)")
        }
    }

    /// The window-1 invariant, explicitly: after a steady-state L=1 append the
    /// retained set is exactly `window` rows = the allowed key set (self +
    /// previous window-1). One off in either direction changes attention.
    func testRetentionIsExactlyWindowAtSteadyState() throws {
        try requireMLX()
        let window = 4
        let rot = RotatingKVCache(window: window)
        var expected: [Int32] = []
        for t in 0 ..< 10 {
            let row = MLXArray([Int32(t)], [1, 1, 1, 1]).asType(.float32)
            let (k, _) = rot.update(keys: row, values: row)
            expected.append(Int32(t))
            let want = Array(expected.suffix(window))
            XCTAssertEqual(
                k.asArray(Float.self).map { Int32($0) }, want,
                "retained rows at step \(t) must be the last min(t+1, window) tokens")
        }
        XCTAssertEqual(rot.retainedLength, window)
        XCTAssertEqual(rot.sequenceLength, 10)
    }

    /// truncate(): tail drops within the retained range work and update resumes
    /// correctly; canTruncate() refuses below the retained range.
    func testTruncateAndCanTruncateBounds() throws {
        try requireMLX()
        let window = 4
        let rot = RotatingKVCache(window: window)
        for t in 0 ..< 10 {
            let row = MLXArray([Int32(t)], [1, 1, 1, 1]).asType(.float32)
            _ = rot.update(keys: row, values: row)
        }
        // Retained (steady state): rows 6,7,8,9 = positions 6..<10.
        XCTAssertTrue(rot.canTruncate(to: 10))
        XCTAssertTrue(rot.canTruncate(to: 8))
        XCTAssertTrue(rot.canTruncate(to: 6))
        XCTAssertFalse(rot.canTruncate(to: 5), "position 5 was rotated out")

        rot.truncate(to: 8)  // drop rows 8, 9
        XCTAssertEqual(rot.sequenceLength, 8)
        XCTAssertEqual(rot.retainedLength, 2)

        let row = MLXArray([Int32(99)], [1, 1, 1, 1]).asType(.float32)
        let (k, _) = rot.update(keys: row, values: row)
        XCTAssertEqual(rot.sequenceLength, 9)
        XCTAssertEqual(k.asArray(Float.self).map { Int32($0) }, [6, 7, 99])
    }

    /// restore(): adopts a (possibly window-trimmed) span at an absolute
    /// position without aliasing the source, and decode resumes from there.
    func testRestoreAtAbsolutePositionAndNoAliasing() throws {
        try requireMLX()
        let window = 4
        let source = MLXArray((0 ..< 3).map { Float(100 + $0) }, [1, 1, 3, 1])
        let sourceBefore = source.asArray(Float.self)

        let rot = RotatingKVCache(window: window)
        rot.restore(keys: source, values: source, totalSeen: 20)
        XCTAssertEqual(rot.sequenceLength, 20)
        XCTAssertEqual(rot.retainedLength, 3)
        XCTAssertTrue(rot.canTruncate(to: 19))
        XCTAssertFalse(rot.canTruncate(to: 16))

        let row = MLXArray([Float(999)], [1, 1, 1, 1])
        let (k, _) = rot.update(keys: row, values: row)
        XCTAssertEqual(rot.sequenceLength, 21)
        XCTAssertEqual(k.asArray(Float.self), [100, 101, 102, 999])

        XCTAssertEqual(source.asArray(Float.self), sourceBefore,
            "restore() aliased the source - in-place writes corrupted it")
    }

    /// Growth/compaction across the internal capacity boundary preserves the
    /// retained window (start compacts back to the buffer front).
    func testCompactionAcrossCapacityBoundary() throws {
        try requireMLX()
        let window = 8
        let rot = RotatingKVCache(window: window)
        // Enough single-row appends to force at least one buffer compaction
        // (capacity grows in 256-row steps; start advances every append).
        let steps = 600
        for t in 0 ..< steps {
            let row = MLXArray([Float(t)], [1, 1, 1, 1])
            _ = rot.update(keys: row, values: row)
        }
        let snap = try XCTUnwrap(rot.snapshot())
        let want = ((steps - window) ..< steps).map { Float($0) }
        XCTAssertEqual(snap.keys.asArray(Float.self), want)
        XCTAssertEqual(rot.sequenceLength, steps)
    }
    #endif
}
