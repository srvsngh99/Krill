# KrillLM vs Ollama — Benchmarks

Reproducible head-to-head benchmarks across the four axes KrillLM serves
natively: **text, vision, voice, tool calling**, in both **hot** (warm,
model-resident) and **cold** (fresh model load) states, plus the **concurrency**
sweep (the axis where KrillLM's continuous batcher wins architecturally).

The numbers below are **fresh** (regenerated, not transcribed from old runs).
Re-run them with the scripts in §How to run before quoting — absolute tok/s
drifts with thermal state and OS load; the *ratios* are the durable signal.

## Environment (last refresh: 2026-06-04)

| | |
|---|---|
| Machine | Apple M4 Pro, macOS (Darwin 25.5.0, arm64) |
| KrillLM | v0.4.0, commit `9a21941` (release build, `make release`) |
| Ollama | 0.24.0 |
| Common model | Gemma 4 E2B — KrillLM `gemma-4-e2b` / Ollama `gemma4:e2b` (the one model that does text+vision+voice+tools) |
| Popular text | Llama 3.2 3B — `llama-3.2-3b` / `llama3.2:3b` |
| Tool model | Qwen 2.5 3B — `qwen2.5-3b` / `qwen2.5:3b` (strong tool caller) |

**Fairness controls.** Same quant class (MLX affine 4-bit vs GGUF Q4_K_M — both
4-bit families, not bit-identical quantizers). Greedy (`temperature 0`, `seed 0`).
Each engine is hit **sequentially, one at a time** — never concurrently — so the
GPU/RAM is never contended. Each engine's OWN Ollama-compatible timing block is
read (`prompt_eval_*`, `eval_*`, `load_duration`), so prefill/decode rates are
self-reported by each engine, not wall-clock-inferred.

---

## Results

### Text (single stream)

| Model | State | Metric | KrillLM | Ollama | KrillLM |
|---|---|---|--:|--:|:--|
| Gemma 4 E2B | hot | decode tok/s | **114.3** | 93.4 | **1.22x** |
| Gemma 4 E2B | hot | total ms (64 tok) | **571** | 995 | **1.74x faster** |
| Gemma 4 E2B | cold | model load ms | **1700** | 2137 | **1.26x faster** |
| Gemma 4 E2B | cold | total ms | **1830** | 2900 | **1.58x faster** |
| Llama 3.2 3B | hot | decode tok/s | **107.2** | 95.5 | **1.12x** |
| Llama 3.2 3B | hot | total ms (64 tok) | **694** | 794 | **1.14x faster** |
| Llama 3.2 3B | cold | model load ms | 700 | 690 | ~parity |

**Read:** single-stream decode is memory-bandwidth bound — both engines hit the
same Metal/GGUF RAM-bandwidth roof, so the **12–22% decode edge is the real
single-stream headroom, not "by miles."** KrillLM's clearer wins are total
latency (last-token-only prefill slice) and cold model load. (Short-prompt
prefill tok/s is noisy and is intentionally not headlined.)

### Vision (Gemma 4 E2B + image)

| State | Metric | KrillLM | Ollama |
|---|---|--:|--:|
| hot | total ms | **545** | 809 (**1.48x faster**) |
| cold | total ms | **2470** | 3941 (**1.60x faster**) |
| hot | image prefill tok/s | **24,080** | 5,149 |
| — | answer | ✅ "solid red rectangular block…" | ❌ **empty output** |

**Caveat:** Ollama 0.24's `gemma4:e2b` *processes* the image (prefill rises to
~289 tokens) but emits **empty content** — a Gemma-4n-vision quirk in Ollama,
NOT a general Ollama-vision failure (its qwen2.5-vl / llava paths answer fine).
The timing comparison is valid (both ran the full prefill+decode); the answer
column is a same-model quality gap.

### Voice (Gemma 4 E2B + real speech)

| Metric | KrillLM | Ollama |
|---|--:|--:|
| transcription | ✅ exact ("…Tokyo today is sunny, high of 25 degrees") | ❌ empty output |
| decode tok/s / TTFT | 86.8 / 447 ms | — (no text) |

Ollama ingests the audio (≈120 prefill tokens, same as KrillLM's 117) but emits
no text. **KrillLM transcribes correctly.** NOTE: KrillLM voice currently works
via the native CLI (`krillm run --audio`), **not** the HTTP API — see
`docs/BENCHMARK_ISSUES.md` #1.

### Tool calling (Qwen 2.5 3B, single-shot, scored)

| Metric | KrillLM | Ollama |
|---|--:|--:|
| valid tool call | **4/4** | 4/4 |
| exact args | **4/4** | 4/4 |
| median latency | **377–558 ms** | 528–578 ms |

**Read:** tool-call **correctness is at parity** on a capable model; KrillLM's
edge is **latency per call** (it inherits the decode/total wins). Multi-step
*agentic* prompts are flaky on **both** engines for small 3B models — the call
*decision* is the model's weakness, prompt-sensitive on both (not an engine bug;
KrillLM emits correct `tool_calls` when prompted directly). See issues #4/#6.

### Concurrency (Gemma 4 E2B, N simultaneous streams) — the "scales under load" axis

| N streams | KrillLM agg tok/s | Ollama agg tok/s | Ratio |
|--:|--:|--:|--:|
| 1 | 107.8 | 72.8 | 1.48x |
| 2 | 126.5 | 78.0 | 1.62x |
| 4 | 123.9 | 81.8 | 1.52x |
| 8 | **153.1** | 83.5 | **1.83x** |

**This is the architectural win:** KrillLM aggregate throughput *climbs* with load
(108 → 153 tok/s) because the continuous batcher serves many decode rows from one
weight read; Ollama is flat (~73-84) because it serializes. (The harness now also
captures Ollama p99 TTFT, resolved in issue #3.)

### Agentic / RAG (shared-prefix KV reuse) - the closed-gap axis

A realistic agent/RAG request reuses a long shared scaffold (system prompt + tool
schemas + retrieved docs) across many calls with a short varying tail. KrillLM now
reuses that shared prefix - serial (PRs #148) and concurrent batched (#151) for
the standard per-layer families, plus **Gemma 4 on every path** (#156 serial fp16,
#157 int8 serial, #158 honest gate, #159 concurrent batched) - instead of
re-prefilling it. Measured on **qwen2.5-14b** vs Ollama `qwen2.5:14b` (shared
~1300-token RAG context, varied question, greedy; run sequentially so only one
14B is resident):

| | cold prefill (first req) | repeated-context prefill (reuse) |
|--:|--:|--:|
| **KrillLM** | 4556 ms | **180 ms** |
| **Ollama**  | 4781 ms | 193 ms |

KrillLM's repeated-context prefill is **180 ms, at parity with Ollama's 193 ms** -
where before #148 it re-prefilled the whole context every request (~4556 ms, i.e.
~24x slower than Ollama's cached path). The critical agentic/RAG gap is closed.
Concurrent rows share the scaffold too: on qwen2.5-3b at `KRILL_NUM_PARALLEL=4`, a
~440-token scaffold prefills cold once (~441 ms) and the 4 concurrent shared-prefix
requests then prefill in 13 / 43 / 44 / 111 ms each.

**Gemma 4 now reuses too (closed 2026-06-06).** Shared-prefix reuse forwards a
suffix span over a restored prefix. This is byte-exact for standard per-layer
caches (Llama, Qwen, Mistral, Phi, dense MoE). Gemma 4 needed one fix - its
cross-layer KV-shared layers must rotate the suffix query at its true positions
`[LCP, count)` rather than offset 0 - and now reuses on all paths. Gemma 4
computes in bf16, so the reused result is numerically correct but NOT strictly
byte-identical: the shorter suffix GEMM rounds a few percent differently and can
flip a downstream greedy near-tie into an equally-valid different continuation
(dense families stay byte-exact; Gemma 4 is gated on the reused cache matching
cold within bf16 noise + first-token match - see `docs/BACKLOG.md`). KrillLM
self-measured on gemma-4-e2b over HTTP (562-token shared prefix): cold prefill
**1001 ms -> partial reuse 158 ms** (full-match 17 ms).

> **Refresh pending:** a clean Gemma-4 head-to-head vs Ollama `gemma4:e2b`
> agentic table is not yet captured here; re-run `tools/agentic_benchmark.py`
> with `--krill-model gemma-4-e2b --ollama-model gemma4:e2b` and replace this note.

---

## How to run

```bash
# 1. Build the release binary (needed for the KrillLM cold/CLI path).
make release

# 2. Bring up both engines (Ollama on 11434 already; KrillLM on its default 57455).
#    For the concurrency sweep, enable the batcher + n-gram spec.
KRILL_NUM_PARALLEL=16 KRILL_NGRAM_SPEC=1 \
  .build/release/krillm serve --model gemma-4-e2b &

# 3. Text / vision / voice / tools head-to-head (hot + cold):
python3 tools/bench_suite.py --axis all \
  --krill-model gemma-4-e2b --ollama-model gemma4:e2b --repo .
#   (text axis on the popular model:)
python3 tools/bench_suite.py --axis text \
  --krill-model llama-3.2-3b --ollama-model llama3.2:3b --repo .
#   (tools on a strong caller:)
python3 tools/bench_suite.py --axis tools \
  --krill-model qwen2.5-3b --ollama-model qwen2.5:3b --repo .

# 4. Concurrency sweep:
python3 tools/krillm_concurrent_benchmark.py \
  --krillm-url http://127.0.0.1:57455 --krill-model gemma-4-e2b \
  --ollama-host http://127.0.0.1:11434 --ollama-model gemma4:e2b \
  --concurrency-sweep "1,2,4,8" --max-tokens 96 --runs 2 --warmup 1 --server-arm batched

# 5. Agentic / RAG shared-prefix reuse (any family now, incl. Gemma 4; run the
#    engines sequentially on a 24GB box so only one large model is resident):
python3 tools/agentic_benchmark.py \
  --krill-model qwen2.5-14b --ollama-model qwen2.5:14b --concurrency 1,4 --max-tokens 48
#   (Gemma 4 agentic head-to-head - the path closed in #156-#159:)
python3 tools/agentic_benchmark.py \
  --krill-model gemma-4-e2b --ollama-model gemma4:e2b --concurrency 1,4 --max-tokens 48
```

`tools/bench_suite.py` auto-uses `/tmp/klmbench/red.png` and
`/tmp/klmbench/speech.wav`; generate a speech asset with
`say -o a.aiff "…"; afconvert -f WAVE -d LEI16@16000 -c 1 a.aiff /tmp/klmbench/speech.wav`.

## Companion harnesses
- `tools/krillm_concurrent_benchmark.py` — averaged N-stream sweep (serial vs batched arms, p99 TTFT).
- `tools/gemma4_multimodal_benchmark.py` — the release-gate multimodal harness (text/image/audio, memory sampling).
- `tools/tool_calling_benchmark.py` — scored tool-call parity gate.

## Known gaps
Open issues found during benchmarking (to revisit/fix) live in
`docs/BENCHMARK_ISSUES.md`.
