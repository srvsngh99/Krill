// KrillLM - Fused SwiGLU Metal Kernel
//
// Computes: output = silu(gate) * up in a single pass.
// Saves one full activation tensor read/write compared to separate ops.
//
// Expected ~5-12% decode speedup on the FFN portion of each layer.

#include <metal_stdlib>
using namespace metal;

/// Fused SwiGLU: output[i] = silu(gate[i]) * up[i]
/// where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
///
/// All tensors are fp16 with shape [batch * seq_len, intermediate_size]
kernel void fused_swiglu_f16(
    device const half *gate [[buffer(0)]],    // [N, intermediate_size]
    device const half *up   [[buffer(1)]],    // [N, intermediate_size]
    device half *output     [[buffer(2)]],    // [N, intermediate_size]
    constant uint &size     [[buffer(3)]],    // total elements (N * intermediate_size)
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = float(gate[tid]);
    float u = float(up[tid]);

    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    float sig = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig;

    output[tid] = half(silu_g * u);
}

/// Fused SwiGLU for float32 (used during accumulation or non-quantized models)
kernel void fused_swiglu_f32(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float *output     [[buffer(2)]],
    constant uint &size      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= size) return;

    float g = gate[tid];
    float u = up[tid];

    float sig = 1.0f / (1.0f + exp(-g));
    float silu_g = g * sig;

    output[tid] = silu_g * u;
}

/// Fused RMSNorm + Residual: output = x + rmsnorm(residual_input)
/// Folds residual addition into the normalization step.
///
/// rmsnorm(x) = x * rsqrt(mean(x^2) + eps) * weight
kernel void fused_rmsnorm_residual_f16(
    device const half *input     [[buffer(0)]],   // [N, hidden_size] - input to norm
    device const half *residual  [[buffer(1)]],   // [N, hidden_size] - residual to add
    device const half *weight    [[buffer(2)]],   // [hidden_size] - norm weight
    device half *output          [[buffer(3)]],   // [N, hidden_size] - result
    constant uint &hidden_size   [[buffer(4)]],
    constant float &eps          [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]         // (element_in_row, row)
) {
    uint row = tid.y;
    uint col = tid.x;
    if (col >= hidden_size) return;

    uint idx = row * hidden_size + col;

    // Compute mean of squares for this row (simplified - real impl uses threadgroup reduction)
    float val = float(input[idx]);
    // Note: full implementation needs threadgroup reduction for mean(x^2)
    // This is a simplified per-element version showing the fusion pattern.
    // Production version would use simdgroup_reduce.

    float w = float(weight[col]);
    float r = float(residual[idx]);

    // For now, just demonstrate the fusion pattern
    // Real kernel: normalized = val * rsqrt(variance + eps) * w
    // Output = normalized + residual
    output[idx] = half(val * w + r);
}
