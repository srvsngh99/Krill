import XCTest
import MLX
@testable import KrillCache

/// Unit tests for the per-head min-max KV quantizer.
///
/// These tests do NOT depend on a live model — they exercise the quantizer math
/// on small synthetic tensors and check the round-trip error stays inside one
/// quantization step worth of the input range.
final class QuantizedKVCacheUnitTests: XCTestCase {

    /// Shape used for all tests: [B=1, H=4, S=8, D=16].
    private let shape = [1, 4, 8, 16]

    /// Build a deterministic fp16 tensor with the given range, varying along the
    /// last axis so per-head min/max actually compute non-trivial scales.
    private func deterministicTensor(min lo: Float, max hi: Float) -> MLXArray {
        let total = shape.reduce(1, *)
        let span = max(hi - lo, 1e-3)
        var values = [Float](repeating: 0, count: total)
        for i in 0 ..< total {
            // Pseudo-random but deterministic: Weyl sequence in [0, 1).
            let frac = Float(i) * 0.6180339887 - Float(Int(Float(i) * 0.6180339887))
            values[i] = lo + frac * span
        }
        return MLXArray(values).reshaped(shape).asType(.float16)
    }

    /// Maximum absolute error between two tensors (as a host Float).
    private func maxAbsError(_ a: MLXArray, _ b: MLXArray) -> Float {
        let diff = MLX.abs(a.asType(.float32) - b.asType(.float32))
        let m = MLX.max(diff)
        MLX.eval(m)
        return m.item(Float.self)
    }

    /// Round-trip with purely positive values.
    func testRoundTripPositiveValues() {
        let cache = QuantizedKVCache()
        let lo: Float = 0.0
        let hi: Float = 10.0
        let k = deterministicTensor(min: lo, max: hi)
        let v = deterministicTensor(min: lo, max: hi * 0.5)

        let (kOut, vOut) = cache.update(keys: k, values: v)

        XCTAssertEqual(kOut.shape, k.shape)
        XCTAssertEqual(vOut.shape, v.shape)

        // Per-head range divided by 255 levels — accept ~1.5 quant steps to absorb
        // fp16 rounding plus the per-head scale's own fp16 storage.
        let kRange = hi - lo
        let vRange = (hi * 0.5) - lo
        let kTol = (kRange / 255.0) * 1.5 + 1e-3
        let vTol = (vRange / 255.0) * 1.5 + 1e-3

        let kErr = maxAbsError(k, kOut)
        let vErr = maxAbsError(v, vOut)
        XCTAssertLessThan(kErr, kTol, "key round-trip error \(kErr) >= tol \(kTol)")
        XCTAssertLessThan(vErr, vTol, "value round-trip error \(vErr) >= tol \(vTol)")
    }

    /// Round-trip with mixed-sign values across the fp16 range. This is the
    /// regression test for the int8-cast wraparound bug: 200 cast to int8 wraps
    /// to -56, dequant produces garbage. uint8 storage avoids the wrap.
    func testRoundTripMixedSignValues() {
        let cache = QuantizedKVCache()
        let lo: Float = -64.0
        let hi: Float = 64.0
        let k = deterministicTensor(min: lo, max: hi)
        let v = deterministicTensor(min: lo * 0.5, max: hi)

        let (kOut, vOut) = cache.update(keys: k, values: v)

        let kRange = hi - lo
        let vRange = hi - (lo * 0.5)
        let kTol = (kRange / 255.0) * 1.5 + 1e-2
        let vTol = (vRange / 255.0) * 1.5 + 1e-2

        let kErr = maxAbsError(k, kOut)
        let vErr = maxAbsError(v, vOut)
        XCTAssertLessThan(kErr, kTol, "key round-trip error \(kErr) >= tol \(kTol)")
        XCTAssertLessThan(vErr, vTol, "value round-trip error \(vErr) >= tol \(vTol)")
    }

    /// Multiple updates accumulate along the sequence axis.
    func testSequentialUpdatesAccumulate() {
        let cache = QuantizedKVCache()
        let k1 = deterministicTensor(min: -2.0, max: 2.0)
        let v1 = deterministicTensor(min: -2.0, max: 2.0)
        _ = cache.update(keys: k1, values: v1)
        XCTAssertEqual(cache.sequenceLength, shape[2])

        let k2 = deterministicTensor(min: -2.0, max: 2.0)
        let v2 = deterministicTensor(min: -2.0, max: 2.0)
        let (kOut, vOut) = cache.update(keys: k2, values: v2)

        XCTAssertEqual(cache.sequenceLength, shape[2] * 2)
        XCTAssertEqual(kOut.dim(2), shape[2] * 2)
        XCTAssertEqual(vOut.dim(2), shape[2] * 2)
    }
}
