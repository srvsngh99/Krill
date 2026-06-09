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

### 4. Activation fusion (fused GEGLU) for Gemma decode

- **Goal:** wire Gemma 4's GEGLU FFN to a fused `gelu_tanh(gate) * up` Metal
  kernel (the GEGLU analogue of `fusedSwiGLU`), expecting the same FFN win on the
  12B decode path.
- **Status:** CLOSED (no measured win). Implemented `fusedGEGLU`, parity-gated it,
  and A/B'd it on `gemma-4-12b` (nvfp4): **single-stream decode 28.0 (fused) vs
  28.0 (unfused) tok/s; concurrent aggregate 31.7/37.2 (fused) vs 32.6/37.1
  (unfused) at N=4/8** - all within noise. Reverted (not shipped).
- **Why it's capped:** the prior `fusedSwiGLU` "5-12%" is a *prefill / large-batch*
  (big `[B*L, inter]`) win. At **decode** the FFN activation is `[rows, inter]`
  with rows = 1 (or the small batch N), tiny next to streaming the 12B weights
  each step - so decode (single AND batched) stays weight-bandwidth bound and
  fusing the activation saves nothing. Same root cause as #1-#3.
- **Re-attempt trigger:** a *prefill-bound* workload (very long prompts dominating
  wall time) where the `[B*L, inter]` activation is large - then a fused GEGLU
  could help prefill (not decode). Use the fused-Q4 parity-gate + bench template.
  Impl note for a reimplementation: cast the fp32 result to the output element
  type before the write
  (`static_cast<metal::remove_reference<decltype(out[elem])>::type>(...)`) - a
  bare `out[elem] = <float>` failed to compile for bf16 outputs under mlx-swift
  0.31.4 in this attempt (Gemma runs bf16), since Metal's bfloat16 has no
  implicit float ctor.

### 5. int8 KV cache as a default for (Gemma 4) agent sessions

- **Goal:** make int8 quantized KV cache the default for agent sessions (the
  `gemma-4-12b` unified model), expecting the halved KV footprint to relieve
  memory pressure on the 24GB box and keep long agent contexts off the swap
  cliff (the handoff noted 12B decode collapses 23-27 -> 3-7 tok/s under
  pressure).
- **Status:** CLOSED (negative result; not shipped, not even as an opt-in). The
  int8 path is *correct* for the unified family - the unified text decoder reuses
  the same `Gemma4Attention` (accepts `KVCacheProtocol`) and the loader already
  wires `batchedDecodeForwardQuantized` identically to plain Gemma 4, so flipping
  `ModelAdapter.kvCacheQuantization` for `.gemma4Unified` to `.supportsInt8` is a
  one-liner. A verification harness (int8 vs fp16 greedy, both prompts) confirmed
  **accuracy is fine** (greedy matches fp16 byte-for-byte at every context
  length - int8 KV is near-lossless, unlike int4 *weights*). **But speed loses,
  and the gap WIDENS with context** (gemma-4-12b nvfp4, clean box):

  | context | fp16 tok/s | int8 tok/s | int8 ratio |
  |---|---|---|---|
  | short (~tens) | 24.8 | 20.1 | 0.81x |
  | ~2.2k | 16.1 | 10.6 | 0.66x |
  | ~6.6k | 12.5 | 5.4 | 0.43x |

- **Why it's capped (two independent reasons):**
  1. **Decode dequant cost scales with KV length.** Every decode step
     dequantizes the *entire* past KV before the attention matmul; there is no
     fused dequant+attention kernel on the KV path (same wall as #3 fused-Q4 -
     plain dequant-then-GEMM is slower than the bandwidth it saves). On a box
     that is not swapping, this is pure overhead that grows linearly with context.
  2. **The memory win is structurally muted for Gemma 4.** Sliding-window
     attention already caps KV growth for most layers (window 512), so halving
     the KV barely moves total memory. And the real long-context OOM is the
     **prefill attention matrix** (O(L^2), bf16: a single 21GB buffer at ~16k ctx
     exceeded MLX's 14.3GB max-buffer limit) - which int8 KV does nothing for.
- **Re-attempt trigger:** (a) a **full-attention** family (no sliding window) on
  this box where KV genuinely dominates memory and fp16 OOMs where int8 fits AND
  the decode is bandwidth-bound enough that smaller KV nets out faster; or (b) a
  fused int8-dequant+SDPA Metal kernel (or MLX native quantized-KV attention) that
  removes the per-step dequant pass; or (c) int4-KV *with* such a fused kernel for
  a 4x footprint cut, if the accuracy holds. Reuse the greedy-match + short/long
  tok/s harness from this attempt. NOTE: plain `.gemma4` stays `.supportsInt8`
  (opt-in via `KRILL_KV_CACHE_DTYPE=int8`) and is unchanged by this finding.

**Cross-cutting read:** five M-series decode levers have now closed the same way
(spec-decode, compiled-decode, fused-Q4, fused-GEGLU, int8-KV). The consistent
signal is that MLX puts KrillLM near the hardware bandwidth limit for decode, so
raw-throughput tuning has low ROI. Direct future effort at **coverage/capability**
(more models, VLM serving, structured output, tool/agentic) - NOT at out-tuning
MLX's core decode ops. (Op fusion's demonstrated wins are prefill/large-batch
shapes, not decode; and lossless KV compression only pays off when memory, not
compute, is the actual binding constraint.)

### 6. No flash prefill kernel in MLX (long-prompt OOM) - MITIGATED

- **Limit:** MLX (0.31.4) `scaledDotProductAttention` has no flash/streaming
  prefill kernel - for any query length > 1 it materializes the full per-head
  `[heads, L, L]` bf16 score matrix (verified: peak grows quadratically; a single
  buffer crosses Metal's 14.3GB max around L ~ 21k tokens and HARD-OOMs - e.g. a
  ~35k-token prompt asks for a 39.86GB buffer and aborts the process). Mask mode
  is irrelevant: `.causal` and `.array(additiveMask)` both materialize (probed in
  `Gemma4PrefillSDPAProbeTests`). This is a hard cap on single-prompt context.
- **Status: MITIGATED (not closed) via chunked prefill** - the engine forwards
  the prompt in query-chunks of `KRILL_PREFILL_CHUNK` (default 2048), `eval`'ing
  each so its scores free before the next, while the shared KV cache accumulates
  exactly as one pass would (the verified partial-prefix-reuse mechanism). Result
  is numerically the single-pass prefill (greedy byte-identical at 6.6k); ~35k
  contexts that previously crashed now run and answer a planted needle; short
  prompts are untouched (single forward). Gate: `Gemma4ChunkedPrefillTests`.
- **Residual / re-attempt trigger:** chunking bounds the SCORE matrix, but the KV
  cache itself still grows with context (Gemma sliding-window caps most layers;
  full-attention layers do not), so very long contexts remain RAM-bound on 24GB.
  A real win needs (a) an MLX flash prefill kernel (then drop chunking), or (b)
  chunking the KEY dimension too (true flash, custom kernel - weigh against the
  fused-Q4 #3 lesson that hand-rolled attention kernels lose to MLX). The
  multimodal prefill path is not yet chunked (text-only today) - a follow-up.

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
