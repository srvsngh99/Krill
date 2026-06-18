# Compiled text-decode block: probe and verdict

**Status: investigated and closed (no net win on M-series).** The production
decode path stays on the uncompiled growing `KVCache`.

## The lever

The Qwen 2.5-VL *vision* blocks are `MLX.compile`'d (they hold no KV cache, so
their forward is shape-stable and compiling fuses the residual / norm / SwiGLU
elementwise ops and removes per-call graph-build overhead). The open question
was whether the *text* decoder block could get the same treatment to speed up
decode.

The blocker named in the backlog was that `KVCache` was not MLX `Updatable`, so
it could not be passed to `MLX.compile(inputs:outputs:)`. That part is now
fixed (`KVCache: Updatable` in `Sources/KrillCache/KVCache.swift`).

## Why it is hard

`MLX.compile` re-traces whenever an input's shape changes. The default
`KVCache` returns a slice that grows one row per decode step, so a block forward
built on it changes shape every step and the compiled graph is never reused.

The only way to keep the shape constant is a fixed-size buffer
(`FixedBufferKVCache`): a pre-allocated `[B, H, capacity, D]` buffer written in
place at a **runtime** offset (via scatter, so the write position is not a
static slice bound that would itself force a re-trace), with an additive
position mask hiding the unwritten tail. The cost: attention runs over the full
padded `capacity` every step instead of the actual length.

## The probe

`Tests/KrillCoreTests/CompiledDecodeProbeTests.swift` (gated on
`KRILL_DECODE_PROBE`) builds Llama-3.2-3B-dim transformer blocks (GQA attention +
SwiGLU MLP + two RMSNorms, random weights) and:

1. **Correctness** -- asserts the compiled fixed-buffer path is numerically
   equivalent to the uncompiled growing-cache path at a matched offset (the
   position mask must make the padded buffer behave like a length-`offset`
   cache). Passes (max abs diff < 5e-3 in fp16).
2. **Timing** -- 16 stacked blocks (so the per-step lazy graph-build overhead
   `MLX.compile` eliminates, which scales with layer count, is realistically
   represented), 128 decode steps, growing vs compiled-fixed.

## Result (M4 Pro, fp16)

```
[16 blocks cap=256]  growing 57.3 tok/s  |  compiled-fixed 56.9 tok/s  |  0.99x
[16 blocks cap=1024] growing 59.5 tok/s  |  compiled-fixed 55.3 tok/s  |  0.93x
```

(A single block is worse for the compiled path -- 0.76x at cap=256 -- because
graph-build overhead is negligible relative to one block's compute; stacking 16
blocks is where compile's overhead-elimination actually shows, lifting it to
~break-even.)

## Verdict

**Break-even at best, and it loses as capacity grows.** The graph-build overhead
`MLX.compile` removes is real and grows with layer count, but the fixed buffer's
padded-attention tax (computing attention over the whole `capacity` every step)
cancels it out. There is no regime where compiled fixed-buffer decode is a clear
win for typical decode, so wiring it into the engine would add complexity and a
larger memory footprint for no speedup -- and would *regress* at large capacity.

This mirrors the WS2 speculative-decode finding: a structural M-series limit,
not a code defect. The lever is closed.

## What landed regardless

- `KVCache: Updatable` -- clean infrastructure, matches mlx-swift conventions,
  lets a cache be passed to `eval` / `MLX.compile` for any future experiment.
- `FixedBufferKVCache` -- the shape-stable mechanism, correctness-tested by the
  probe. Kept as a building block, not wired into the production decode path.
- The probe itself -- a reproducible benchmark + correctness regression guard.
