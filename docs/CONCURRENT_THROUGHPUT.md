# Concurrent throughput: KrillLM vs Ollama

Tool: `tools/krillm_concurrent_benchmark.py` (`make bench-concurrent`). Drives N
simultaneous `/api/generate` streams against each engine and reports AGGREGATE
decode tok/s — the axis where the continuous batcher wins. Single-stream decode
is memory-bandwidth bound, so neither engine out-decodes the other on one
request; concurrency is the lever (one weight read serves many decode rows).

## Result (qwen2.5-3b 4-bit, M-series, max_tokens=96, distinct prompts per stream)

**KrillLM batched (`KRILL_NUM_PARALLEL=16`, `KRILL_NGRAM_SPEC=1`) vs Ollama 0.24:**

| N | KrillLM agg tok/s | Ollama agg tok/s | ratio | KrillLM p99 TTFT | Ollama p99 TTFT |
|---|------------------:|-----------------:|------:|-----------------:|----------------:|
| 1 | 85.2  | 36.4 | 2.34x | 160 ms | 1545 ms |
| 4 | 158.0 | 85.2 | 1.86x | 324 ms | 3449 ms |
| 8 | 172.6 | 85.7 | 2.01x | 541 ms | 7903 ms |

**KrillLM serial arm (`KRILL_NUM_PARALLEL=1`, baseline) — same sweep:**

| N | KrillLM agg tok/s | p99 TTFT |
|---|------------------:|---------:|
| 1 | 97.3  | 86 ms |
| 4 | 101.3 | 2899 ms |
| 8 | 101.1 | 6707 ms |

## Reading

- **Batching scales; serial does not.** Batched aggregate climbs 85 → 158 → 173
  tok/s; serial is flat ~100 (requests serialize through the generation queue,
  and p99 TTFT balloons). The **crossover is N\* = 2** — batched beats
  serial-aggregate as soon as two requests overlap. This justifies the
  load-adaptive default `KRILL_SPEC_CONCURRENCY_MAX = 1` (n-gram spec only when
  solo; batch at N >= 2).
- **KrillLM beats Ollama on every concurrency level** (1.9–2.3x aggregate) and
  dramatically on tail latency: at N=8 KrillLM p99 TTFT is 541 ms vs Ollama's
  7.9 s (~14x). Ollama's aggregate throughput is flat (~85 tok/s) — it does not
  amortize weights across concurrent generates the way the continuous batcher
  does.
- **N-gram interaction:** the batched-arm N=1 used n-gram (85 tok/s) and is a
  touch below the serial-arm N=1 without n-gram (97) — these are generic,
  *non-echo* prompts, so n-gram sits at its ~floor here (the multi-token win
  needs echo-heavy work: RAG verbatim quoting, code, structured output, where it
  measured ~1.85x single-stream, see `docs/VERIFY_PROFILE.md`). N-gram is
  therefore **opt-in** (`KRILL_NGRAM_SPEC`), workload-gated by the operator; the
  default server is unaffected.

## Scaling shape (averaged, 3 runs, 24 distinct prompts)

Averaged wall-based aggregate (qwen2.5-3b, max_tokens=128): 97 -> 139 -> 162 -> 174
-> 255 tok/s at N=1,2,4,8,16. The aggregate is non-monotonic in efficiency (the
N=4->8 step adds little, N=8->16 jumps), which initially looks like a batcher
scheduling artifact. It is not. Two things explain it:

1. **Prefix-cache contamination (a benchmark bug, now fixed).** With only 8 distinct
   prompts, an N=16 sweep reused prompts -> prefix-cache hits shrink wall time and
   inflate the wall-based aggregate. The harness now ships >=24 distinct prompts and
   warns when the sweep exceeds the prompt-set size. (Removing it dropped N=16 from
   270 to 255 - a real but small effect.)

2. **GPU occupancy (structural, the real driver).** Using the per-request decode
   tok/s (server `eval_count/eval_duration`), the batched step time per token is
   9.6 / 13.6 / 23.2 / 43.1 / 55.6 ms at R=1/2/4/8/16 — its marginal cost grows
   ~5 ms/row up to R=8, then the slope drops to +1.6 ms/row at R=16 (the step time
   is still rising, just much more slowly). That marginal flattening is the opposite
   of compute-bound: MLX's batched matmul does not fill the GPU until ~R=16, so each
   added row is costly at R=4-8 and much cheaper at R=16. This is an MLX/hardware
   occupancy curve, **not** a KrillLM batcher scheduling bug — admission + epochs are
   correct and rolling. Aggregate grows monotonically (174 -> 255 from R=8 -> 16, a
   1.47x step, not linear) and beats Ollama at every N.

   Two caveats on the metric, so this is not over-read:
   - It is *server-measured*, which is the point: it sidesteps any client-side
     `ThreadPoolExecutor`/GIL confound in the wall-based aggregate. But it is not
     pure GEMM — `eval_count/eval_duration` folds in the per-epoch `scatterBack`
     `MLX.eval` (`ContinuousBatcher.swift`). It reads as ~GEMM here only because the
     benchmark's equal-budget, simultaneously-arriving rows keep the epoch intact
     (one scatter at the end), so per-step overhead is negligible. A ragged workload
     (rows finishing at different times -> frequent epoch breaks) would add real
     scatter/re-stack cost that this run does not show.
   - The steady-state GEMM step time has no KrillLM-side lever (it is MLX's kernel
     occupancy). `KRILL_BATCH_WINDOW_MS` and `KRILL_NUM_PARALLEL` are real levers for
     *coalescing* and the *row cap*, but they do not change the per-step GEMM cost.

The benchmark averages `--runs` passes per level because a single pass at moderate N
is noisy.

## Reproduce

Enable n-gram with the `--ngram-spec` serve flag (equivalent to `KRILL_NGRAM_SPEC=1`):

```bash
# batched arm
KRILL_NUM_PARALLEL=16 krillm serve --ngram-spec --model qwen2.5-3b --port 11500 &
make bench-concurrent KRILLM_URL=http://127.0.0.1:11500 KRILL_MODEL=qwen2.5-3b \
     OLLAMA_HOST=http://127.0.0.1:11434 OLLAMA_MODEL=qwen2.5:3b \
     CONCURRENCY_SWEEP=1,2,4,8,16 SERVER_ARM=batched BENCH_RUNS=3 BENCH_WARMUP=1

# serial baseline (find the crossover)
KRILL_NUM_PARALLEL=1 krillm serve --model qwen2.5-3b --port 11500 &
make bench-concurrent KRILLM_URL=http://127.0.0.1:11500 KRILL_MODEL=qwen2.5-3b \
     CONCURRENCY_SWEEP=1,2,4,8,16 SERVER_ARM=serial BENCH_RUNS=3
```
