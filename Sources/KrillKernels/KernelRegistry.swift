import Foundation
import MLX
import MLXNN
import MLXFast

/// Registry for custom fused Metal kernels.
///
/// Uses MLX's JIT metalKernel API to dispatch fused operations that avoid
/// materializing intermediate tensors, saving memory bandwidth.
///
/// Available kernels:
/// - `fusedSwiGLU`: silu(gate) * up in one pass (5-12% FFN speedup)
public enum KrillKernels {

    /// JIT-compiled fused SwiGLU kernel.
    /// Computes output[i] = silu(gate[i]) * up[i] without materializing silu(gate).
    ///
    /// The sigmoid/multiply intermediates run in fp32; the result is then
    /// **explicitly** cast to the output element type (`OUT_T`, bound to the
    /// output dtype at dispatch) before the store. Metal allows implicit
    /// `float -> half`/`float -> float` but REJECTS implicit `float -> bfloat`
    /// ("assigning to 'bfloat16_t' from incompatible type 'float'"), which made
    /// the prior implicit-conversion version fail to compile and crash the
    /// process whenever a bf16 model (e.g. Gemma 4 12B) reached this kernel.
    /// The explicit `static_cast<OUT_T>` is correct for fp16, bf16, and fp32 and
    /// keeps the bf16 output (the prior `half(...)` would fp16-truncate bf16).
    private static let _fusedSwiGLUKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "fused_swiglu",
            inputNames: ["gate", "up"],
            outputNames: ["out"],
            source: """
                uint elem = thread_position_in_grid.x;
                float g = float(gate[elem]);
                float u = float(up[elem]);
                float sig = 1.0f / (1.0f + exp(-g));
                out[elem] = static_cast<OUT_T>(g * sig * u);
            """,
            ensureRowContiguous: true
        )
    }()

    /// Apply fused SwiGLU: output = silu(gate) * up
    ///
    /// Dispatches a custom Metal kernel that computes silu(gate) * up in a single
    /// pass, avoiding the intermediate tensor allocation for silu(gate).
    /// This saves one full read+write of the intermediate_size tensor per layer.
    ///
    /// - Parameters:
    ///   - gate: Output of gate_proj, shape [B*L, intermediate_size]
    ///   - up: Output of up_proj, shape [B*L, intermediate_size]
    /// - Returns: silu(gate) * up, same shape
    public static func fusedSwiGLU(gate: MLXArray, up: MLXArray) -> MLXArray {
        let totalElements = gate.size
        // Threadgroup: use 256 threads (standard for elementwise)
        let threadGroup = min(256, totalElements)
        let grid = (totalElements + threadGroup - 1) / threadGroup * threadGroup

        let results = _fusedSwiGLUKernel(
            [gate, up],
            template: [("OUT_T", gate.dtype)],
            grid: (grid, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [gate.shape],
            outputDTypes: [gate.dtype]
        )
        return results[0]
    }

    // MARK: - Fused Q4-affine dequant + GEMV (decode probe)

    /// JIT-compiled fused affine-4bit dequant + GEMV for the decode (single
    /// query row) shape. One thread per output row `o` reads the packed 4-bit
    /// weights, dequantizes on the fly (`w = q * scale + bias`, MLX's affine
    /// convention), and accumulates `sum_i x[i] * w[o, i]` in fp32 -- no
    /// dequantized weight matrix is ever materialized.
    ///
    /// This is a PROBE to test whether a hand-fused kernel can beat MLX's
    /// built-in `quantizedMatmul` on M-series decode. Layout matches MLX's
    /// affine pack: `w` is `[O, I/8]` uint32 (8 nibbles per word), `scales` /
    /// `biases` are `[O, I/groupSize]`. `params` is int32 `[O, I, groupSize]`.
    private static let _fusedQ4GemvKernel: MLXFast.MLXFastKernel = {
        MLXFast.metalKernel(
            name: "fused_q4_gemv",
            inputNames: ["x", "w", "scales", "biases", "params"],
            outputNames: ["out"],
            source: """
                uint o = thread_position_in_grid.x;
                int O = params[0];
                int I = params[1];
                int gs = params[2];
                if (o >= (uint)O) { return; }
                uint wordsPerRow = (uint)(I / 8);
                uint groupsPerRow = (uint)(I / gs);
                uint wBase = o * wordsPerRow;
                uint gBase = o * groupsPerRow;
                float acc = 0.0f;
                for (uint word = 0; word < wordsPerRow; word++) {
                    uint packed = w[wBase + word];
                    uint i0 = word * 8u;
                    for (uint k = 0; k < 8u; k++) {
                        uint i = i0 + k;
                        uint q = (packed >> (4u * k)) & 0xFu;
                        uint g = i / (uint)gs;
                        float s = float(scales[gBase + g]);
                        float b = float(biases[gBase + g]);
                        acc += float(x[i]) * (float(q) * s + b);
                    }
                }
                out[o] = acc;
            """,
            ensureRowContiguous: true
        )
    }()

    /// Fused affine-4bit dequant + GEMV: `out = x @ dequant(w)^T` for a single
    /// query row, bit-compatible with
    /// `MLX.quantizedMatmul(x, w, scales:, biases:, transpose: true, bits: 4)`.
    ///
    /// - Parameters:
    ///   - x: `[1, I]` (or `[I]`) activation row.
    ///   - w: `[O, I/8]` uint32 packed 4-bit weights (MLX affine pack).
    ///   - scales / biases: `[O, I/groupSize]`.
    ///   - groupSize: affine group size (32 / 64 / 128).
    /// - Returns: `[1, O]` in `x`'s dtype.
    ///
    /// PROBE ONLY: not wired into the decode hot path (see
    /// docs/FUSED_Q4_PROBE.md). MLX's built-in quantizedMatmul is the shipped
    /// path.
    public static func fusedQ4Gemv(
        x: MLXArray, w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int
    ) -> MLXArray {
        let O = w.dim(0)
        let I = x.size
        let params = MLXArray([Int32(O), Int32(I), Int32(groupSize)])
        let threadGroup = min(256, O)
        let grid = (O + threadGroup - 1) / threadGroup * threadGroup
        let results = _fusedQ4GemvKernel(
            [x.reshaped([I]), w, scales, biases, params],
            grid: (grid, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [[1, O]],
            outputDTypes: [x.dtype]
        )
        return results[0]
    }

    /// Whether the fused-Q4 decode probe is opted in (`KRILL_FUSED_Q4=1`). Off
    /// by default: the probe is not wired into the decode path regardless (the
    /// flag exists for benchmark harnesses and a possible future wiring).
    public static var fusedQ4Enabled: Bool {
        ProcessInfo.processInfo.environment["KRILL_FUSED_Q4"] == "1"
    }

    /// Custom kernels are always available (JIT compiled via MLX metalKernel API).
    public static var isAvailable: Bool { true }

    /// Kernel status for diagnostics.
    public static var status: String {
        "Custom Metal kernels: active (fused_swiglu via JIT)"
    }
}
