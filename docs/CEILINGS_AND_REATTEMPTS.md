# Ceilings and re-attempts

A running index of levers KrillLM has **tried and hit a ceiling on** -- where the
work landed as a probe / gated path / negative result rather than a shipped win.
The point of this doc is twofold:

1. **Stop blind re-tries.** Each entry records what was tried and the measured
   result, so a future session does not waste days re-discovering the same wall.
2. **Make re-attempts deliberate.** Each entry names a concrete **re-attempt
   trigger** -- the specific change in the world (new MLX/Metal capability, new
   hardware, or a fundamentally different algorithm) that would make another
   attempt worthwhile. Re-open an item ONLY when its trigger fires.

Two kinds of ceiling:

- **Structural (M-series perf):** MLX already sits near the memory-bandwidth roof
  for decode on Apple silicon, so several raw-throughput levers cannot win with
  the current approach. These need a *different algorithm* or a *new MLX/Metal
  capability*, not more tuning.
- **Resource (RAM-blocked):** the code is believed correct but cannot be
  verified / run on the 24GB dev box. These need *bigger hardware*, not code.

When you re-attempt one and it changes status, update its entry here and the
linked detail doc.

---

## Structural perf ceilings (M-series)

### 1. Speculative decode >= 1.5x

- **Goal:** draft-model speculative decoding hitting the >= 1.5x decode speedup
  the workstream targeted.
- **Status (draft-model):** CLOSED as a strict gate. Batched verification landed
  (PR #41); the >= 1.5x gate was demoted to advisory (PR #42/#43). A hard
  >= 1.0x floor (never slower than non-spec) holds and is enforced.
- **Why the draft path capped:** on M-series the verification forward is
  bandwidth-bound and the draft overhead (`alpha ~ 1/8`) eats most of the
  acceptance win. WS2 fitted a per-round overhead `beta ~ 0.78`, ceiling
  `1/(alpha+beta) ~ 1.10x`.
- **RE-OPENED and partially solved via a different algorithm (n-gram /
  prompt-lookup).** The WS2 `beta` was *fitted*, never profiled. A direct probe
  (`Tests/KLMCoreTests/VerifyProfileTests.swift`, `docs/VERIFY_PROFILE.md`)
  measured the clean width-K verify cost: `beta ~ 0.23` at K=16 (ceiling
  3.2-4.3x), because `tK` *saturates* in K (extra positions ride along on one
  weight stream). Removing the draft model (n-gram drafts from context,
  `alpha -> 0`) lands a real win: **echo-heavy single-stream 1.85x** on
  llama-3.2-3b, **~floor (>=0.95x via an acceptance-adaptive draft cap)** on
  non-echo. Shipped opt-in (`KRILL_NGRAM_SPEC`), greedy-only, prefix-consistent
  with standard decode (differs only at fp16 verify-vs-decode near-ties, same as
  the batched decoder). The single-stream lever's marginal value shrinks under
  concurrency, so a load-adaptive gate (`KRILL_SPEC_CONCURRENCY_MAX`, default 1)
  uses n-gram solo and **continuous batching** above the crossover (N*=2), where
  KrillLM already beats Ollama 1.9-2.3x on aggregate throughput
  (`docs/CONCURRENT_THROUGHPUT.md`).
- **Still open:** the strict >= 1.5x gate for the *general* (non-echo) case, and
  tree-attention / Medusa heads (multiple candidate continuations verified in one
  forward) to lift acceptance on non-repetitive workloads. The n-gram verify path
  + parity harness are the template for that next attempt.
- **Detail:** `docs/workstreams/WS2_SPECULATIVE_DECODING.md`, `docs/VERIFY_PROFILE.md`,
  `docs/CONCURRENT_THROUGHPUT.md`.

### 2. Compiled text-decode block (`MLX.compile`)

- **Goal:** wrap the per-token text-decode step in `MLX.compile` to fuse the
  graph and cut dispatch overhead.
- **Status:** CLOSED. The `KVCache: Updatable` conformance + `FixedBufferKVCache`
  + a gated compiled-decode probe landed (PR #128, `a7bc2d4`), but the compiled
  block is **not wired in**: it benches **0.93-0.99x** (break-even to slightly
  slower) on M-series.
- **Why it's capped:** the decode step is already a small, bandwidth-bound graph;
  compilation's fusion/dispatch savings are smaller than the fixed-buffer copy +
  capture overhead it introduces.
- **Re-attempt trigger:** a new MLX release whose `compile` meaningfully lowers
  capture/dispatch cost for tiny graphs, OR an Updatable-cache path that avoids
  the fixed-buffer copy. Re-run the gated probe; wire in only if it clears a
  real margin (say > 1.05x).
- **Detail:** `docs/COMPILED_DECODE_PROBE.md`.

### 3. Fused Q4-affine matmul kernel

- **Goal:** a hand-written fused dequant + GEMV Metal kernel beating MLX's
  built-in `quantizedMatmul` on the 4-bit decode shape (a universal,
  every-token, every-model win if it worked).
- **Status:** CLOSED. `KLMKernels.fusedQ4Gemv` is numerically correct (matches
  MLX, cosine > 0.9999) but **~2.9x slower** (PR #134, `b6cd07a`). Landed as a
  probe + benchmark + doc; not wired in.
- **Why it's capped:** decode matmul is memory-bandwidth bound (you must stream
  the whole 4-bit weight matrix once per token); MLX's kernel already reads at
  near the bandwidth roof. A plain GEMV cannot beat it -- at best match it.
- **Re-attempt trigger:** do NOT re-attempt the plain GEMV. The only custom-kernel
  win here is **fusion** -- a kernel that removes traffic MLX is forced to pay by
  materializing intermediates between ops (e.g. fused quantized-SwiGLU:
  dequant+gate-matmul + dequant+up-matmul + SiLU*mul in one pass; or fused QKV).
  That is a different lever; track it as new work, using this kernel's
  parity-gate + benchmark harness as the template (cf. the unquantized
  `fusedSwiGLU`, which already nets 5-12%).
- **Detail:** `docs/FUSED_Q4_PROBE.md`.

**Cross-cutting read:** three M-series decode-speed levers have now closed the
same way (spec-decode, compiled-decode, fused-Q4). The consistent signal is that
MLX puts KrillLM near the hardware bandwidth limit for decode, so raw-throughput
tuning has low ROI. Direct future effort at **coverage/capability** (more models,
VLM serving) or at **op fusion** (the one kernel direction with demonstrated
wins), not at out-tuning MLX's core ops.

---

## Resource ceilings (RAM-blocked on the 24GB dev box)

These are verified-by-design but un-run here; they need a larger-RAM machine, not
code changes. Re-attempt trigger for all three: **access to a box with enough
unified memory** (rule of thumb: >= 2x the fp16 footprint for headroom).

### 4. bge-multilingual-gemma2 embedder

- 9B Gemma-2 decoder embedder; ~37GB fp32 exceeds the host's ~25.8GB RAM.
- The Gemma `DecoderEmbedder` + the `embed_tokens` fp32-upcast path (both on
  `main`, #115) should serve it. Re-attempt: pull `BAAI/bge-multilingual-gemma2`,
  confirm cosine vs transformers + a retrieval check, add the alias (family
  `.gemma`). Detail: `~/.claude/plans/krillm-embeddings-handover.md`.

### 5. DeepSeek-V3 671B real-checkpoint run

- Native V3 absorbed-MLA runtime landed and is synthetic-parity verified
  (PR #127); the 671B real checkpoint is RAM-blocked. Re-attempt: run the real
  checkpoint on a large box and confirm decode coherence + mlx-lm parity.

### 6. Llama-3.2-11B-Vision (mllama) real-checkpoint run

- Native mllama runtime landed and is mlx-vlm synthetic-parity verified
  (PR #133, image + text-only). The 11B real checkpoint is RAM-blocked here, and
  the image-serving wiring (tile/aspect-ratio preprocessing + a cross-KV decode
  driver) is a separate open follow-up. Re-attempt the real-checkpoint run on a
  larger box alongside the serving wiring. Detail: `docs/MAC_NATIVE_MODEL_COVERAGE_ROADMAP.md`.

---

## How to use this doc

- Before starting any "make decode faster" / "custom matmul kernel" /
  "speculative decoding" task, read the relevant entry. If its re-attempt
  trigger has not fired, don't.
- When a trigger fires and you re-attempt, keep the same discipline as the
  original probe: a correctness/parity gate first, then a benchmark that decides
  wire-in vs stay-closed, then update this entry + the detail doc with the new
  result.
