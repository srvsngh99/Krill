import XCTest
import MLX
import MLXRandom
@testable import KLMKernels

/// Tests for the fused affine-4bit dequant + GEMV decode probe
/// (`KLMKernels.fusedQ4Gemv`). The numerical test is the correctness gate (the
/// kernel must match MLX's built-in `quantizedMatmul` bit-for-bit-ish); the
/// benchmark is gated on `KLM_FUSED_Q4_BENCH` and only prints timings (it never
/// asserts a speedup -- the probe exists precisely to measure whether one
/// exists, see docs/FUSED_Q4_PROBE.md).
final class FusedQ4KernelTests: XCTestCase {

    /// Quantize a random weight `[O, I]` to affine 4-bit and return the packed
    /// representation MLX produces.
    private func quantize(O: Int, I: Int, groupSize: Int)
        -> (w: MLXArray, scales: MLXArray, biases: MLXArray)
    {
        let weight = MLXRandom.normal([O, I], key: MLXRandom.key(3))
        let (wq, scales, biasesOpt) = MLX.quantized(
            weight, groupSize: groupSize, bits: 4, mode: .affine)
        return (wq, scales, biasesOpt!)
    }

    func testMatchesMLXQuantizedMatmul() throws {
        let O = 512, I = 256, gs = 64
        let (w, scales, biases) = quantize(O: O, I: I, groupSize: gs)
        let x = MLXRandom.normal([1, I], key: MLXRandom.key(7))

        let ref = MLX.quantizedMatmul(
            x, w, scales: scales, biases: biases,
            transpose: true, groupSize: gs, bits: 4, mode: .affine)
        let got = KLMKernels.fusedQ4Gemv(
            x: x, w: w, scales: scales, biases: biases, groupSize: gs)

        XCTAssertEqual(got.shape, [1, O])
        eval(ref, got)
        let a = got.reshaped([O]).asArray(Float.self)
        let b = ref.reshaped([O]).asArray(Float.self)
        var dot: Double = 0, na: Double = 0, nb: Double = 0, maxAbs: Double = 0
        for i in 0 ..< O {
            dot += Double(a[i]) * Double(b[i])
            na += Double(a[i]) * Double(a[i])
            nb += Double(b[i]) * Double(b[i])
            maxAbs = max(maxAbs, abs(Double(a[i]) - Double(b[i])))
        }
        let cosine = dot / (na.squareRoot() * nb.squareRoot())
        XCTAssertGreaterThan(cosine, 0.9999,
            "fused Q4 GEMV must match MLX quantizedMatmul (cosine \(cosine))")
        // Loose abs tolerance: fp32 accumulation, same math, different reduction
        // order than MLX's tuned kernel.
        XCTAssertLessThan(maxAbs, 1e-2, "max abs diff \(maxAbs) too large")
    }

    func testMatchesAcrossGroupSizes() throws {
        for gs in [32, 128] {
            let O = 256, I = 256
            let (w, scales, biases) = quantize(O: O, I: I, groupSize: gs)
            let x = MLXRandom.normal([1, I], key: MLXRandom.key(11))
            let ref = MLX.quantizedMatmul(
                x, w, scales: scales, biases: biases,
                transpose: true, groupSize: gs, bits: 4, mode: .affine)
            let got = KLMKernels.fusedQ4Gemv(
                x: x, w: w, scales: scales, biases: biases, groupSize: gs)
            eval(ref, got)
            let a = got.reshaped([O]).asArray(Float.self)
            let b = ref.reshaped([O]).asArray(Float.self)
            var maxAbs: Double = 0
            for i in 0 ..< O { maxAbs = max(maxAbs, abs(Double(a[i]) - Double(b[i]))) }
            XCTAssertLessThan(maxAbs, 1e-2, "gs=\(gs) max abs diff \(maxAbs)")
        }
    }

    /// Benchmark vs MLX built-in. Gated (prints only, never asserts a speedup).
    func testBenchmarkVsBuiltin() throws {
        guard ProcessInfo.processInfo.environment["KLM_FUSED_Q4_BENCH"] == "1" else {
            throw XCTSkip("Set KLM_FUSED_Q4_BENCH=1 to run the fused-Q4 benchmark")
        }
        let O = 4096, I = 4096, gs = 64
        let (w, scales, biases) = quantize(O: O, I: I, groupSize: gs)
        let x = MLXRandom.normal([1, I], key: MLXRandom.key(1))
        let iters = 200

        func time(_ body: () -> MLXArray) -> Double {
            // Warmup.
            for _ in 0 ..< 10 { eval(body()) }
            let t0 = CFAbsoluteTimeGetCurrent()
            for _ in 0 ..< iters { eval(body()) }
            return (CFAbsoluteTimeGetCurrent() - t0) / Double(iters) * 1e6  // us
        }

        let builtinUs = time {
            MLX.quantizedMatmul(
                x, w, scales: scales, biases: biases,
                transpose: true, groupSize: gs, bits: 4, mode: .affine)
        }
        let fusedUs = time {
            KLMKernels.fusedQ4Gemv(x: x, w: w, scales: scales, biases: biases, groupSize: gs)
        }
        print("[fused-Q4 bench] O=\(O) I=\(I) gs=\(gs) iters=\(iters)")
        print(String(format: "  MLX quantizedMatmul: %.1f us/call", builtinUs))
        print(String(format: "  fused_q4_gemv:       %.1f us/call", fusedUs))
        print(String(format: "  ratio (fused/builtin): %.2fx", fusedUs / builtinUs))
    }
}
