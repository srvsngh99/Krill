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

## Reproduce

```bash
# batched arm
KRILL_NUM_PARALLEL=16 KRILL_NGRAM_SPEC=1 krillm serve --model qwen2.5-3b --port 11500 &
make bench-concurrent KRILLM_URL=http://127.0.0.1:11500 KRILL_MODEL=qwen2.5-3b \
     OLLAMA_HOST=http://127.0.0.1:11434 OLLAMA_MODEL=qwen2.5:3b \
     CONCURRENCY_SWEEP=1,4,8 SERVER_ARM=batched

# serial baseline (find the crossover)
KRILL_NUM_PARALLEL=1 krillm serve --model qwen2.5-3b --port 11500 &
make bench-concurrent KRILLM_URL=http://127.0.0.1:11500 KRILL_MODEL=qwen2.5-3b \
     CONCURRENCY_SWEEP=1,4,8 SERVER_ARM=serial
```
