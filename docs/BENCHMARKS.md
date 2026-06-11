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
| Ollama | 0.30.6 (GGUF default; MLX engine available for Gemma 4) |
| Common model | Gemma 4 E2B — KrillLM `gemma-4-e2b` / Ollama `gemma4:e2b` (GGUF) (the one model that does text+vision+voice+tools) |
| Gemma 4 MLX | Ollama `gemma4:e2b-mlx` (nvfp4, text-only) — the MLX-vs-MLX comparison target |
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

### Text (single stream, vs Ollama's default GGUF engine)

These rows compare KrillLM (MLX) against Ollama's **GGUF / llama.cpp** path (the
default for these models). For Gemma 4 — where Ollama also has an MLX engine — see
the MLX-vs-MLX section right below, which is the apples-to-apples decode number.
Llama 3.2 and Qwen 2.5 have **no MLX tag in Ollama** (its MLX backend only covers
newer architectures), so GGUF is the only available Ollama comparison there.

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

### Text — vs Ollama's MLX engine (Gemma 4 E2B) — the apples-to-apples decode number

Ollama added an MLX backend (v0.21.0, Apr 2026) with a text-only Gemma 4 runtime,
tagged `gemma4:e2b-mlx` (nvfp4 4-bit). This is the fair MLX-vs-MLX comparison:
both engines run the same model family on MLX, so the GGUF quant-class confound is
gone. KrillLM uses `mlx-community/gemma-4-e2b-it-4bit` (affine 4-bit) — not a
bit-identical quant to nvfp4, but both are MLX 4-bit.

| Metric | KrillLM (MLX affine-4bit) | Ollama `gemma4:e2b-mlx` (MLX nvfp4) | KrillLM |
|---|--:|--:|:--|
| hot decode tok/s (N=1) | 109.1 | **114.9** | 0.95x |
| hot total ms (64 tok) | 621 | 624 | ~parity |
| **cold total ms** | **1080** | 3124 | **2.9x faster** |
| **concurrent agg tok/s, N=4** | **206** | 110 | **1.88x** |
| **concurrent agg tok/s, N=8** | **219** | 110 | **1.99x** |
| vision / voice | ✅ native | ❌ MLX tag is text-only | KrillLM only |

> This is a SEPARATE, later measurement run from the GGUF tables above (different
> session, thermal state, and build), so KrillLM's absolute self-numbers here
> (decode 109 vs 96, N=8 219 vs 155) do not line up with that section - only the
> within-section RATIO vs the engine measured alongside it is meaningful, per the
> "ratios are the durable signal" note at the top.

**Read (honest):** on **single-stream** text decode, KrillLM is at **parity with
Ollama's MLX engine — marginally behind (~5%)**; Ollama's nvfp4 + fused top-P/top-K
sampling edge out the shared bandwidth roof. The 1.5x text-total figure in the
GGUF table above is partly a GGUF-vs-MLX artifact and should NOT be read as a
single-stream MLX win. KrillLM's durable wins over Ollama's MLX engine are
**concurrency (~2x, the continuous batcher vs Ollama's per-slot serialization,
which holds even on MLX), cold start (2.9x), and multimodal** (Ollama's MLX Gemma
tag is text-only; KrillLM serves vision + voice natively). Numbers drift ~±5% with
thermal state, so treat single-stream as parity-within-noise.

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

### Long context (Gemma 4 12B, 16k-99k) - the survivability axis (2026-06-11)

Same prompt on both engines (repeated structured sections, needle planted at the
start, question at the end - `tools/longctx_head2head.py`), both read via their
own `/api/generate` timing fields, engines run strictly sequentially. KrillLM
`gemma-4-12b` vs Ollama `gemma4:12b-mlx` (their MLX backend; v0.30.6).

**Decode (128-token answers - see the measurement note below):**

| ctx tokens | KrillLM tok/s | Ollama tok/s | Ratio |
|--:|--:|--:|:--|
| ~16k | 21.9 | **24.4** | 0.90x |
| ~53k | **20.5** | 8.0-13.0 | **1.6-2.6x** |
| ~99k | **17.2** | 7.5-11.6 | **1.5-2.3x** |

**Prefill (same runs):**

| ctx tokens | KrillLM s | Ollama s | KrillLM |
|--:|--:|--:|:--|
| ~16k | **90.8** | 120.3 | **1.3x faster** |
| ~53k | **345** | 644-686 | **1.9-2.0x faster** |
| ~99k | **746** | 2701-3057 | **3.6-4.1x faster** |

**Memory at ~99k:** KrillLM peaks ~18.5GB inside the box's budget (RotatingKVCache
caps the 40 sliding-window layers; only the 8 full-attention layers grow). Ollama's
MLX runner preallocates full-`num_ctx` KV on ALL layers - at 99k it reports **41GB**
on the 24GB box and drives system swap to ~36GB. It survives, but by thrashing the
whole machine; its long-ctx decode varies run-to-run with accumulated swap state
(the ranges above), KrillLM's does not. Needle retrieval was correct in every cell
on both engines. KrillLM's usable ceiling is ~123k at 15.6 tok/s
(`docs/CEILINGS_AND_REATTEMPTS.md` #7); Ollama was not probed past 99k - the box
was already 36GB into swap there.

**Prefix-cache bonus (agentic axis at long ctx):** repeating a 16k prompt against
KrillLM hits the shared-prefix KV cache - prefill drops 88.5s -> **0.1s**. Ollama
re-prefills unless its own slot cache happens to hold the exact context.

**Measurement note (the 8-token trap).** With a direct needle question the model
answers in ~8 tokens, and "decode tok/s" over 8 tokens is dominated by fixed
post-prefill overhead (on KrillLM's side the O(ctx) prefix-cache store), NOT the
per-token rate - it made the serve path read 5.4 tok/s at 99k when its true decode
rate is 17.2 (proven by a cold/warm pair on an identical prompt: 11.0 vs 24.6 on
the same 8-token answer). Quote long-ctx decode only from runs with >=128 decoded
tokens. At <=16k, where their runner is not yet swap-bound, Ollama's nvfp4 decode
is marginally ahead (same parity-by-physics as the short-ctx section; this run used
KrillLM's mixed-weights blob, not `gemma-4-12b-nvfp4`). Re-measured on the nvfp4
blob same-day: 22.7 tok/s at 16k (0.93x) - tighter, inside the parity-within-noise
band, still not a win. Short-ctx single-stream decode stays parity; the long-ctx
cells above are where the engines diverge.

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
#   (apples-to-apples MLX-vs-MLX: Ollama's MLX Gemma 4 tag - pull it first:)
ollama pull gemma4:e2b-mlx
python3 tools/bench_suite.py --axis text \
  --krill-model gemma-4-e2b --ollama-model gemma4:e2b-mlx --repo .

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

# 6. Long-context head-to-head (12B; one engine loaded at a time, ~2h total -
#    the summary question + 128 tokens is what makes decode tok/s meaningful):
python3 tools/longctx_head2head.py --engine krillm --port 57455 \
  --model gemma-4-12b --ctx 14300,47400,88600 --question summary --max-tokens 128
python3 tools/longctx_head2head.py --engine ollama --port 11434 \
  --model gemma4:12b-mlx --ctx 14300,47400,88600 --question summary --max-tokens 128
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
