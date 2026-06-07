# KrillLM vs Ollama — Benchmarks

Reproducible head-to-head benchmarks across the four axes KrillLM serves
natively: **text, vision, voice, tool calling**, in both **hot** (warm,
model-resident) and **cold** (fresh model load) states, plus the **concurrency**
sweep (the axis where KrillLM's continuous batcher wins architecturally).

The numbers below are **fresh** (regenerated, not transcribed from old runs).
Re-run them with the scripts in §How to run before quoting — absolute tok/s
drifts with thermal state and OS load; the *ratios* are the durable signal.

## Environment (last refresh: 2026-06-07)

| | |
|---|---|
| Machine | Apple M4 Pro, 24 GB, macOS 26.5.1 (Darwin 25.5.0, arm64) |
| KrillLM | commit `c0b4a63` (release build, `swift build -c release`) |
| Ollama | 0.30.6 |
| Common model | Gemma 4 E2B — KrillLM `gemma-4-e2b` / Ollama `gemma4:e2b` (the one model that does text+vision+voice+tools) |
| Popular text | Llama 3.2 3B — `llama-3.2-3b` / `llama3.2:3b` |
| Tool model | Qwen 2.5 3B — `qwen2.5-3b` / `qwen2.5:3b` (strong tool caller) |
| Vision (fair) | Qwen 2.5 VL 3B — `Qwen2.5-VL-3B-Instruct-4bit` / `qwen2.5vl:3b` (both engines render it) |
| Large model | Qwen 2.5 14B — `qwen2.5-14b` / `qwen2.5:14b` (at this box's RAM ceiling) |

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
| Gemma 4 E2B | hot | decode tok/s | **96.0** | 88.9 | **1.08x** |
| Gemma 4 E2B | hot | total ms (64 tok) | **678** | 1023 | **1.51x faster** |
| Gemma 4 E2B | cold | model load ms | **1800** | 2697 | **1.50x faster** |
| Gemma 4 E2B | cold | total ms | **2640** | 3525 | **1.34x faster** |
| Llama 3.2 3B | hot | decode tok/s | **92.8** | 86.1 | **1.08x** |
| Llama 3.2 3B | hot | total ms (64 tok) | **780** | 876 | **1.12x faster** |
| Llama 3.2 3B | cold | total ms | **1130** | 1814 | **1.61x faster** |
| Qwen 2.5 3B | hot | decode tok/s | **91.4** | 87.4 | **1.05x** |
| Qwen 2.5 3B | hot | total ms (64 tok) | **752** | 839 | **1.12x faster** |
| Qwen 2.5 3B | cold | total ms | **1130** | 1507 | **1.33x faster** |
| Qwen 2.5 **14B** | hot | decode tok/s | 19.6 | **23.1** | **0.85x** (see note) |

**Read:** single-stream decode is memory-bandwidth bound — both engines hit the
same Metal/GGUF RAM-bandwidth roof, so the **5–8% decode edge on small/mid models
is the real single-stream headroom, not "by miles."** KrillLM's clearer wins are
total latency (last-token-only prefill slice) and cold total. (Short-prompt
prefill tok/s is noisy and is intentionally not headlined.)

**14B note (RAM ceiling, not a defect):** a 14B 4-bit is ~9 GB; on this 24 GB box
KrillLM's single-stream decode runs ~15% behind Ollama (0.85x) and the concurrent
agentic path collapses under memory pressure (free RAM drops to ~21% with both
engines holding a 14B). This is the documented ~14B stability ceiling for 24 GB
(`docs/BENCHMARK_ISSUES.md` #0b), not a code regression — on a box with headroom
the 14B tracks the small/mid models. KrillLM leads on every model that fits
comfortably.

### Vision

**Gemma 4 E2B + image** (the all-modality model):

| State | Metric | KrillLM | Ollama |
|---|---|--:|--:|
| hot | total ms | **643** | 985 (**1.53x faster**) |
| — | answer | ✅ describes the image | ❌ **empty output** |

Ollama 0.30's `gemma4:e2b` still *processes* the image but emits **empty
content** — a Gemma-4-vision quirk in Ollama, NOT a general Ollama-vision failure.
For a fair vision latency comparison, use a model Ollama renders:

**Qwen 2.5 VL 3B + image** (both engines answer):

| State | Metric | KrillLM | Ollama |
|---|---|--:|--:|
| hot | total ms | **577** | 699 (**1.21x faster**) |
| — | answer | ✅ renders | ✅ renders |

### Voice (Gemma 4 E2B + real speech)

| Metric | KrillLM | Ollama |
|---|--:|--:|
| transcription | ✅ exact ("…The quick brown fox jumps over the lazy dog.") | ❌ empty output |
| decode tok/s / TTFT | 72.8 / 2254 ms | — (no text) |

Ollama ingests the audio but emits no text. **KrillLM transcribes correctly.**
NOTE: KrillLM voice currently works via the native CLI (`krillm run --audio`),
**not** the HTTP API — see `docs/BENCHMARK_ISSUES.md` #1.

### Tool calling (single-shot, scored)

| Model | Metric | KrillLM | Ollama |
|---|---|--:|--:|
| Qwen 2.5 3B | valid tool call | **4/4** | 4/4 |
| Qwen 2.5 3B | median latency | **498 ms** | 560 ms |
| Gemma 4 E2B | valid tool call | **4/4** | **0/4** |
| Gemma 4 E2B | median latency | **311 ms** | 1731 ms |

**Read:** on a strong caller (Qwen 2.5 3B) tool-call **correctness is at parity**
and KrillLM's edge is **latency per call** (it inherits the decode/total wins). On
**Gemma 4 E2B** KrillLM emits valid tool calls **4/4** where Ollama's `gemma4:e2b`
returns **0/4** — a same-model correctness win (KrillLM's native per-family tool
adapter). Multi-step *agentic* tool prompts remain prompt-sensitive on small
models (a model-decision weakness, not an engine bug). See issues #4/#6.

### Concurrency (Gemma 4 E2B, N simultaneous streams) — the "scales under load" axis

| N streams | KrillLM agg tok/s | Ollama agg tok/s | Ratio |
|--:|--:|--:|--:|
| 1 | 97.8 | 67.9 | 1.44x |
| 2 | 120.0 | 74.0 | 1.62x |
| 4 | 117.1 | 75.7 | 1.55x |
| 8 | **155.2** | 76.7 | **2.02x** |

**This is the architectural win:** KrillLM aggregate throughput *climbs* with load
(98 → 155 tok/s) because the continuous batcher serves many decode rows from one
weight read; Ollama is flat (~68-77) because it serializes. At N=8 KrillLM is
**2.02x** Ollama's aggregate throughput.

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

**Head-to-head agentic throughput (2026-06-07).** Shared ~710-word RAG context +
JSON output, concurrency sweep (`tools/agentic_benchmark.py`), aggregate decode
tok/s and tasks/s, all responses valid JSON on both engines:

| Model | N | KrillLM tok/s | Ollama tok/s | KrillLM tasks/s | Ollama tasks/s |
|---|--:|--:|--:|--:|--:|
| Gemma 4 E2B | 1 | **33.8** | 15.5 | **4.84** | 2.21 |
| Gemma 4 E2B | 4 | **63.8** | 34.1 | **6.23** | 3.25 |
| Gemma 4 E2B | 8 | **69.5** | 34.1 | **7.83** | 3.79 |
| Llama 3.2 3B | 1 | **38.6** | 27.1 | 2.57 | 3.87 |
| Llama 3.2 3B | 4 | **56.0** | 46.4 | **4.98** | 4.88 |
| Llama 3.2 3B | 8 | **62.4** | 46.7 | **6.65** | 5.50 |

On **Gemma 4 E2B** KrillLM runs the agentic/RAG workload at **~2x** Ollama's
throughput and tasks/s across the sweep (and this is exactly the long-context
decode path repaired in #168 — before that fix it returned 0/N valid JSON on long
varied context). On **Llama 3.2 3B** KrillLM leads **1.2–1.4x**. JSON validity is
8/8 on both engines.

**14B agentic is RAM-bound on this box.** On `qwen2.5-14b` the concurrent agentic
path does NOT lead (KrillLM ~4.7 vs Ollama ~11 tok/s, and N=4 slower than N=1) —
memory pressure at the 24 GB ~14B ceiling, the same constraint flagged in the Text
section, not a code defect. The shared-prefix *reuse* win above (180 ms cached
prefill) still holds; it is the *concurrent decode* under a near-full RAM budget
that regresses on this hardware.

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
