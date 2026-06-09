import XCTest
import MLX
import MLXRandom
@testable import KLMKernels

/// Tests for the fused SwiGLU kernel (`KLMKernels.fusedSwiGLU`), which computes
/// `silu(gate) * up = gate * sigmoid(gate) * up` in one pass.
///
/// The bf16 case is a CRASH REGRESSION: the kernel previously stored the fp32
/// result through Metal's implicit conversion (`out[elem] = g * sig * u`), which
/// the Metal compiler accepts for `half`/`float` but REJECTS for `bfloat`
/// ("assigning to 'bfloat16_t' from incompatible type 'float'"). The first
/// dispatch with a bf16 output therefore failed to JIT-compile and aborted the
/// whole process. The fix casts explicitly (`static_cast<OUT_T>(...)`), so this
/// test must (a) run at all on bf16 (the kernel compiles) and (b) match a
/// reference within the dtype's rounding tolerance.
final class FusedSwiGLUKernelTests: XCTestCase {

    private func runCase(_ dtype: DType, tol: Float) throws {
        let n = 2048
        let gate = MLXRandom.normal([n], key: MLXRandom.key(11)).asType(dtype)
        let up = MLXRandom.normal([n], key: MLXRandom.key(13)).asType(dtype)

        // The kernel: computes in fp32, casts the result to `dtype`.
        let got = KLMKernels.fusedSwiGLU(gate: gate, up: up)
        XCTAssertEqual(got.dtype, dtype, "output dtype must match the input dtype")
        XCTAssertEqual(got.shape, [n])

        // Reference: same fp32 math, rounded to `dtype` the same way, so we test
        // the kernel's arithmetic — not the final rounding step.
        let gf = gate.asType(.float32)
        let uf = up.asType(.float32)
        let sig = 1.0 / (1.0 + MLX.exp(-gf))
        let ref = (gf * sig * uf).asType(dtype)

        eval(got, ref)
        let a = got.asType(.float32).asArray(Float.self)
        let b = ref.asType(.float32).asArray(Float.self)
        var maxAbs: Float = 0
        for i in 0 ..< n { maxAbs = max(maxAbs, abs(a[i] - b[i])) }
        XCTAssertLessThan(maxAbs, tol, "\(dtype) fusedSwiGLU max abs error \(maxAbs)")
    }

    func testFusedSwiGLUFloat32() throws { try runCase(.float32, tol: 1e-4) }

    func testFusedSwiGLUFloat16() throws { try runCase(.float16, tol: 1e-2) }

    /// Regression: a bf16 output must not crash the JIT compiler and must be
    /// numerically correct.
    func testFusedSwiGLUBFloat16CompilesAndMatches() throws {
        try runCase(.bfloat16, tol: 5e-2)
    }
}
