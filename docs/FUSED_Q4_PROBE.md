# Fused Q4-affine matmul probe (closed lever)

## What this is

A probe testing whether a hand-written fused affine-4bit dequant + GEMV Metal
kernel can beat MLX's built-in `quantizedMatmul` on the M-series decode shape
(single query row, `x @ dequant(w)^T`). The kernel lives in
`Sources/KrillKernels/KernelRegistry.swift` as `KrillKernels.fusedQ4Gemv` (JIT via
`MLXFast.metalKernel`), one thread per output row, dequantizing on the fly
(`w = q * scale + bias`, MLX's affine convention) and accumulating in fp32 with
no materialized weight matrix.

## Result: it loses, so it is NOT wired in

The kernel is **numerically correct** (matches MLX `quantizedMatmul` bit-for-bit
within fp tolerance: cosine > 0.9999, max abs diff < 1e-2, across group sizes
32 / 64 / 128 -- see `FusedQ4KernelTests`), but it is **slower** than the
built-in:

```
O=4096 I=4096 gs=64, 200 iters, M-series, release:
  MLX quantizedMatmul: 270.3 us/call
  fused_q4_gemv:       791.8 us/call
  ratio (fused/builtin): 2.93x   (i.e. ~3x SLOWER)
```

MLX's `quantizedMatmul` is a heavily tuned kernel (SIMD-group reductions,
vectorized packed-weight loads, coalesced memory access). A naive
one-thread-per-output-row GEMV is memory-bound and cannot compete. Beating it
would need matching those techniques -- a multi-day kernel-engineering effort
with no guarantee of a win, since MLX is already near the memory-bandwidth roof
for this op.

So this is a **closed lever**, in the same category as the compiled-decode probe
(`docs/COMPILED_DECODE_PROBE.md`, PR #128) and the >=1.5x speculative-decode
gate: the probe is landed (kernel + correctness gate + benchmark) but **not
wired into the decode hot path**. `MLX.quantizedMatmul` / `gatherQuantizedMM`
remain the shipped quantized-matmul paths.

## Reproducing

```
make metallib CONFIGURATION=release
KRILL_FUSED_Q4_BENCH=1 swift test -c release \
    --filter 'FusedQ4KernelTests/testBenchmarkVsBuiltin'
```

`KrillKernels.fusedQ4Enabled` reads `KRILL_FUSED_Q4=1`; it is off by default and,
because the kernel is not wired into any forward path, the flag currently only
documents intent for a benchmark harness or a future re-attempt.

## If revisited

A competitive kernel would need: vectorized `uint4` packed-weight loads, a
SIMD-group reduction over the contraction axis (not a per-thread serial loop),
and threadgroup tiling of the activation row. Even then, the upside over MLX's
built-in is likely marginal on memory-bandwidth-bound decode. Re-open only with
a concrete profiling hypothesis for where MLX's kernel leaves bandwidth on the
table.
