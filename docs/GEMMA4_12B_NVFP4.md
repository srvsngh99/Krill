# Gemma 4 12B: nvfp4 on KrillLM vs Ollama (MLX-vs-MLX)

KrillLM runs Gemma 4 12B as a native encoder-free "unified" runtime. This note
records the work to make its 4-bit-class checkpoint match-or-beat Ollama's
`gemma4:12b-mlx` (nvfp4) across quality, speed, concurrency, and capability on a
24 GB Apple-Silicon box, measured one 12B model resident at a time so no number
is a memory-pressure artifact.

## TL;DR

| Checkpoint (same 12B weights) | MMLU-500 | single-stream decode | source |
| --- | --- | --- | --- |
| **KrillLM `gemma-4-12b` (nvfp4, o_proj 8-bit)** | **77.6%** (388/500) | **27.7 tok/s** | bf16 -> nvfp4, attn o_proj kept 8-bit |
| KrillLM nvfp4 (uniform, from bf16) | 75.6% (378/500) | 25.9 tok/s | bf16 -> nvfp4 |
| Ollama `gemma4:12b-mlx` (uniform nvfp4) | 75.8% (379/500) | 27.4 tok/s | reference |
| KrillLM mixed (4-bit attn + 8-bit MLP) | 73.6% (368/500) | 19.2 tok/s | prior default |
| KrillLM nvfp4 (from the **4-bit** ckpt) | 71.6% (358/500) | 26.3 tok/s | the old defect |

The shipped `gemma-4-12b` is numerically best on quality and at parity-or-faster
on single-stream speed, while winning decisively on concurrency, cold start, and
capability (below). All five rows are measured on the **same 500-question set**,
no-CoT, greedy (temp 0), last-letter extraction, on the same mlx-swift 0.31.4
binary, one 12B resident at a time.

## What changed

### 1. Requantize nvfp4 from bf16, not from the 4-bit checkpoint

The nvfp4 checkpoint first validated for native nvfp4 support was requantized
from the already-4-bit mixed checkpoint
(`mlx-community/gemma-4-12B-it-4bit`):

```
dequant(4-bit-affine attn / 8-bit-affine MLP) -> fp -> quantize(nvfp4)
```

so attention inherited affine-int4 damage *before* nvfp4 ever applied. On the
500-question set this scores **71.6%** (358/500) vs Ollama-nvfp4's 75.8%.
Requantizing from the **original bf16** weights
(`mlx-community/gemma-4-12B-it-bf16`, ungated) so attention is single-quantized
lifts this to **75.6%** (378/500) - a +20-question gain that is statistically
significant (paired McNemar p=0.02) and reaches parity with Ollama (within
0.2 pt). This closes issue #174. (An earlier 68.8% figure for the defect was
measured on a 50-question subset; the 71.6% above is the full 500-set number.)

### 2. Mixed-precision nvfp4 to push past parity

KrillLM's loader threads a per-module quantization `mode` (PR #173,
`ModuleQuant.mode`), so a checkpoint can keep the bulk at nvfp4 (speed) while
protecting a small, precision-sensitive set at 8-bit. A broad sweep over which
modules to protect found the attention output projection (`o_proj`, 48 modules)
is the high-yield, low-cost lever:

| protected @ 8-bit | MMLU-500 | decode tok/s | size |
| --- | --- | --- | --- |
| **o_proj only (shipped)** | **77.6%** | **27.7** | 6.7 G |
| o_proj + down_proj | 77.6% | 23.5 | 8.0 G |
| embed_tokens only | 76.0% | 27.6 | 6.8 G |
| down_proj only | 75.8% | 24.7 | 7.6 G |
| none (uniform nvfp4) | 75.6% | 25.9 | 6.7 G |

Protecting `o_proj` alone captures the full quality lift at no speed cost; adding
more 8-bit modules only slows decode.

**Honesty on significance.** What *is* statistically significant: both
bf16-sourced checkpoints beat the from-4-bit defect on the paired 500-set -
uniform nvfp4 +20 net (p=0.02), o_proj-nvfp4 +30 net (p=0.0004). What is *not*:
the quality edge of o_proj-nvfp4 over Ollama-nvfp4 or over our own uniform
baseline is suggestive only (paired McNemar p=0.31 vs Ollama, p=0.10 vs
uniform). At ~4-bit both engines sit near the bf16 quality ceiling, so the
defensible single-stream claim is parity-or-slightly-better, not a blowout. The
decisive wins are elsewhere.

## Where KrillLM wins decisively

### Concurrency (the batcher)

Aggregate decode throughput under N simultaneous `/api/generate` streams
(`tools/krillm_concurrent_benchmark.py`, max_tokens 128):

| N | KrillLM `gemma-4-12b` | Ollama `gemma4:12b-mlx` |
| --- | --- | --- |
| 1 | 18.4* | 27.7 |
| 2 | 26.0 | 27.6 |
| 4 | 31.1 | 27.2 |
| 8 | **37.8** | 27.4 |

KrillLM's continuous batcher scales ~2x from N=1 to N=8 (one weight read serves
many decode rows); Ollama is flat (~27.4 at every N - it serializes). At N=8
KrillLM is **1.38x** Ollama. (*The N=1 aggregate figure is wall-clock including
batch-formation TTFT, and this arm ran with a small co-resident e2b daemon; the
clean steady-state single-stream decode is ~27.7 tok/s, so the concurrency
numbers are conservative for KrillLM.)

### Cold start

KrillLM cold-loads the 12B in ~1.6 s.

### Capability Ollama's MLX gemma tag cannot do

- **Structured / grammar-constrained output** (JSON, JSON-schema, regex, CFG) now
  works on Gemma 4 (PR #175). It was silently disabled by a padded-vocab mask
  mismatch (262144 logits vs 261707 tokenizer pieces); the mask now emits at the
  logits width with padding slots blocked. Ollama's MLX gemma tag has no
  constrained decoding.
- **Multimodal** (native vision + audio) via the unified runtime. Ollama's
  `gemma4:12b-mlx` tag is text-only.

## Registration

The winning checkpoint is installed as **`gemma-4-12b-nvfp4`**, and the canonical
**`gemma-4-12b`** name is promoted to it (`krillm cp`, weights referenced):

```
krillm run gemma-4-12b "..."          # nvfp4, o_proj 8-bit
krillm run gemma-4-12b-nvfp4 "..."    # same checkpoint
```

The `AliasMap` `repo` for `gemma-4-12b` still points at the 4-bit HF repo, so a
fresh `krillm pull gemma-4-12b` on another machine fetches 4-bit until the nvfp4
checkpoint is published to the Hub - a separate, opt-in follow-up.

## Reproduce

```
# 1. bf16 source (ungated)
hf download mlx-community/gemma-4-12B-it-bf16

# 2. requant nvfp4 from bf16, o_proj kept 8-bit (the shipped checkpoint)
python tools/requant_gemma4_nvfp4.py \
    --out ~/.cache/huggingface/krillm-requant/gemma-4-12B-it-nvfp4-oproj8 \
    --protect o_proj
# omit --protect for the uniform nvfp4 baseline

# 3. serve / eval (one 12B at a time; wipe ~/.krillm/cache between runs)
krillm serve --model <checkpoint-dir> --port 57461
```

`tools/requant_gemma4_nvfp4.py` learns which modules to quantize from the 4-bit
checkpoint's index (the proven coverage), pulls each weight from bf16, and
emits a top-level nvfp4 block plus per-module 8-bit overrides for the protected
set.

## Iteration 2: where "miles ahead" is real vs physics-capped

A second pass pushed on beating Ollama across speed, accuracy, tool-calls, and
agentic. The honest result: **on a 24 GB Mac the 12B is raw-throughput-capped at
parity-to-1.5x** (same MLX, same memory-bandwidth roof, and a 12B nearly fills
the box so the GPU saturates under batching). The genuine, shipped "miles ahead"
is **capability** - things Ollama's MLX gemma tag cannot do at all.

### Performance (raw throughput) - parity-to-modest, by physics

- **Single-stream decode:** ~28 tok/s, parity with Ollama (bandwidth roof).
- **Concurrency (NUM_PARALLEL=16):** KrillLM aggregate 18.7 / 25.9 / 31.5 / 37.3 /
  41.9 tok/s at N=1/2/4/8/16; Ollama flat ~28 (it serializes). KrillLM wins from
  N>=4, **1.5x at N=16** - real but not 2x (the ~2x in `CONCURRENT_THROUGHPUT.md`
  is a 3B-model result; a 12B saturates compute sooner).
- **Fused GEGLU (closed):** no decode win - decode is weight-bandwidth bound, not
  activation bound (see `CEILINGS_AND_REATTEMPTS.md` #4).

### Capability - genuinely miles ahead (Ollama MLX gemma tag does NONE of these)

- **Grammar-constrained tool calls** (PR #179): `tool_choice` forces a call
  decoded under the tool's JSON schema -> valid, schema-matched `{name,arguments}`
  (best-effort with a fail-open net). Verified e2e: forced `get_weather` ->
  `{"days":3,"city":"Paris"}`.
- **Structured / grammar output** (PR #175): JSON / schema / regex / CFG.
- **Multimodal:** native vision + audio (the Ollama gemma MLX tag is text-only).

### Accuracy

- nvfp4-from-bf16 + o_proj-8bit: **77.6% MMLU-500**, at/above Ollama-nvfp4 (75.8%);
  both near the bf16 ceiling. AWQ calibration was scoped but is blocked on the
  unified model (mlx_lm cannot load the multimodal weights to calibrate), and
  affine-int4 trails nvfp4 on quality regardless - not pursued.

**Takeaway:** lead with capability + concurrency-on-smaller-models; do not claim a
single-stream raw-speed lead over Ollama on the 12B - that is physics, not effort.
