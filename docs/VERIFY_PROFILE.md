# Verify-forward `beta` profile (n-gram spec-decode go/no-go)

Probe: `Tests/KrillCoreTests/VerifyProfileTests.swift`, gated on `KRILL_VERIFY_PROFILE`.

## Why

WS2 (`docs/workstreams/WS2_SPECULATIVE_DECODING.md`) capped speculative decoding
using a **fitted** per-round overhead `beta ~ 0.78`, never a measured one. The
n-gram / prompt-lookup variant removes the draft model (`alpha -> 0`), so its
single-stream speedup ceiling collapses to `1 / beta`. Whether the lever is worth
shipping depends entirely on the **measured** `beta` — so we measured it directly
instead of trusting the fit.

## Method

A stack of 28 transformer decode blocks at Llama-3.2-3B dims (hidden 3072, 24/8
GQA heads, head 128, inter 8192, vocab 128256) with **production-faithful 4-bit
affine-quantized weights** (`MLX.quantizedMatmul`). Against a fixed length-`L` KV
history, time a width-`1` forward (`t1`, one plain decode step) vs a width-`K`
verify forward (`tK`, the spec verify shape), **including the K-wide lm_head**
(the verify path needs an argmax at every position). New positions are not
persisted, so the history stays length `L` across iterations — a clean,
repeatable measurement of the marginal forward cost. 60 iters, release build,
M-series.

Derived per K: `r = tK/t1`, `beta = (r+1)/(K+1)`, `ideal = (K+1)/(r+1) = 1/beta`
(speedup at 100% acceptance — an **upper bound**; real workloads accept < 1).

## Result (measured, release)

```
L=512:
  K    t1(us)    tK(us)   r=tK/t1   beta    ideal   lm_head(us)
  1    11773.6   11867.4    1.008   1.004   0.996    1218.8
  2    11773.6   13053.5    1.109   0.703   1.423    1360.2
  4    11773.6   23145.1    1.966   0.593   1.686    2650.5
  8    11773.6   42998.4    3.652   0.517   1.935    4915.3
  16   11773.6   51240.4    4.352   0.315   3.176    5649.3
  32   11773.6   52448.6    4.455   0.165   6.050    5776.1

L=4096:
  K    t1(us)    tK(us)   r=tK/t1   beta    ideal   lm_head(us)
  1    27905.1   27829.5    0.997   0.999   1.001    1220.4
  2    27905.1   30261.7    1.084   0.695   1.439    1352.2
  4    27905.1   44346.7    1.589   0.518   1.931    2539.2
  8    27905.1   71896.1    2.576   0.397   2.516    4883.9
  16   27905.1   82966.5    2.973   0.234   4.279    5561.7
  32   27905.1   84502.1    3.028   0.122   8.192    5996.2
```

## Reading

- **Measured `beta` is far below the fitted 0.78.** At K=16 it is 0.23–0.31
  (ceiling 3.2–4.3×); at K=8, 0.40–0.52 (ceiling 1.9–2.5×). The fit was
  pessimistic because it folded in draft-model forward cost, eval-sync, rollback
  and adaptive-K churn — none of which a clean draft-free verify pays.
- **`tK` saturates in K.** K=16→32 barely moves `tK` (51.2→52.4ms at L=512). Once
  the one weight stream is paid, extra verify positions ride along nearly free —
  this is exactly the regime n-gram exploits, since a confident long match
  proposes many tokens at almost the cost of one forward.
- **lm_head is the main K-scaling term** (1.2ms→5.8ms over K=1→32) but plateaus
  alongside `tK`; it does not erode the ceiling at the K values that matter.
- These are 100%-acceptance upper bounds and exclude per-round host overhead
  (propose scan, accept loop, cache truncate, eval syncs). Real speedup will be
  lower, but the margin over 1.5× at K=8–16 is large.

## Decision

**GO.** `ideal >= 1.5×` clears at K≥4 (L=512) / K≥8 (L=4096), with 3–4× headroom
at K=16. Ship n-gram single-stream speculative decode. Proposer `maxDraft` default
**16** (still-climbing ceiling, but a fully-rejected round costs ~4.4·t1, so cap
the wasted-forward downside; rely on longest-match precision + the k=0 plain-decode
fallback to hold the ≥1.0× floor). Re-validate end-to-end with the real release
gate (`bench-ngram-spec`) on repetitive vs non-repetitive suites.
