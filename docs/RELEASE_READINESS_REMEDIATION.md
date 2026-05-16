# Release Readiness Remediation Plan

Originally drafted: 2026-05-10.
Baseline commit: `4a2b6e6` (`Release readiness baseline and Gemma4 fixes`).
PR: #9 (`feat/release-readiness-remediation`).
Last updated: 2026-05-11 (post PR #12).

This document is structured as four sections:

1. **Historical Baseline** — the pre-PR state that motivated the plan.
   Frozen for reference.
2. **PR #9 State** — what was true at the head of the original remediation
   PR. Frozen for reference.
3. **Remaining Release Blockers** — what is still required before this can
   be tagged a production release. Updated incrementally.
4. **Post-baseline progress** — what has shipped on top of the PR #9 baseline.
   Append-only.

It is not yet a production release. Sections 3 and 4 describe what remains
and what has retired.

---

## 1. Historical Baseline (frozen)

The state at commit `4a2b6e6`, before any work in this PR. Listed for
historical reference only; do not treat any item below as "current".

Behavior:

- Native Gemma 4 text and image worked end-to-end on the CLI.
- Gemma 4 audio routed through the `mlx-vlm` Python bridge (per-call
  subprocess).
- The HTTP server accepted text only; image/audio payloads were
  explicitly rejected.
- Multimodal benchmark exercised the `mlx-vlm` bridge for image, not the
  native Swift path.
- Audio fixture was ambiguous (`mlx-vlm` heard "click", Ollama heard
  "dog barking").
- Benchmark reports did not label `cache_mode`.
- 81 tests passed.

Baseline release-gate measurements (text/server, 5 runs / 2 warmup):

| Metric | KrillLM | Ollama | Ratio |
| --- | ---: | ---: | ---: |
| Wall time | 0.3384 s | 0.5380 s | 0.6289x |
| Decode throughput | 98.57 tok/s | 87.65 tok/s | 1.1246x |
| Prefill throughput | 1389.85 tok/s | 1938.30 tok/s | 0.7170x |

Baseline multimodal gate (2 runs / 1 warmup, bridge mode):

| Metric | Ratio |
| --- | ---: |
| Geometric mean speedup | 0.429x |
| text_decode_ratio | 1.4413x |
| text_wall_ratio | 0.6043x |
| image_wall_ratio | 2.2921x |
| image_prefill_ratio | 0.0237x |
| audio_wall_ratio | 1.2525x |
| audio_prefill_ratio | 0.0592x |

The original plan listed five release blockers: release gates fail, server
multimodal not implemented, documentation does not match behavior, audio
fixture ambiguous, cache-affected results need explicit labels.

---

## 2. PR #9 State (HEAD of `feat/release-readiness-remediation`)

What is true after the commits in this PR. Code verification was performed
on `206791d`; the head also includes docs-only commits on top
(`2c6066b`, `b4dbda1`, and any subsequent docs-only fixups).

### What now works

- **Server multimodal is implemented.** `/api/generate`, `/api/chat`, and
  `/v1/chat/completions` accept image and audio payloads (Ollama and OpenAI
  shapes). Image routes through the native Swift path; audio (and combined
  image + audio) routes through the `mlx-vlm` bridge. Reject paths cover
  `>1` image, non-Gemma 4 model, and oversized payloads (`413` returns
  before the model-loaded check).
- **Chat path actually conditions on the image.** A prior bug had the chat
  overload accept `imageData` but never insert the `<|image|>` placeholder
  run, so vision embeddings had no positions to land in and chat image
  requests were silently text-only. Fixed via a shared
  `injectMediaPlaceholders` helper. Live regression
  (`testTwoDifferentImagesProduceDifferentOutputs`) verifies two visually
  different images with the same prompt produce different outputs.
- **`supportsNativeImage` and `supportsAudio`** require `family ==
  "gemma4"` AND `loadedModel?.multimodalForward != nil`. A text-only
  Gemma 4 checkpoint (no `vision_config`) correctly reports no media
  capability and the server rejects with `400`.
- **Persistent `mlx-vlm` sidecar.** Replaces per-call Python subprocess
  with a long-running helper. One model load instead of N. Requests
  serialize through the helper and correlate by id; helper crash surfaces
  a clear error and respawns lazily.
- **SHA-256-keyed SigLIP2 vision encoder cache.** LRU, capacity 4, lives
  on the loaded model instance so unload invalidates automatically.
  Bypassed for audio and multi-image batches.
- **Decode loop pipelining + on-GPU sampler chain.** GPU forward overlaps
  CPU tokenizer decode + yield; sampled token stays as a lazy MLXArray
  fed directly into the next forward.
- **`QuantizedKVCache` wired in (opt-in).** `kv_cache_dtype = "int8"` (or
  `KRILL_KV_CACHE_DTYPE=int8`) selects the int8 path. Default stays
  `fp16`. int8 is gated to Gemma 4 — non-Gemma families warn to stderr
  and fall back to fp16 (the relevant loaders downcast caches to
  `[KVCache]` concrete type and would silently drop int8 state). int8
  also disables prefix cache and speculative decoding because the
  snapshot path dequantizes on every call and the SpeculativeDecoder API
  takes `[KVCache]` concretely.
- **Multimodal prefix-cache key includes media hash (schema v2).** Prior
  key was `FNV1a(modelId || tokenIds)` with no image/audio bytes, so two
  requests with the same prompt but different images collided and served
  each other's KV state — silent wrong answers. New key shape:
  `FNV1a(v2 || modelId || 0xFF || mediaHash || 0xFF || tokenBytes)` where
  `mediaHash = "img:<sha256>|aud:<sha256>"` (empty for pure text). Old
  v1 entries on disk become unreachable.
- **OpenAI bridge streaming returns SSE.** When `stream: true` and the
  request hits the audio bridge, the server now emits an SSE head, one
  `chat.completion.chunk` content delta, a `finish_reason=stop` chunk,
  and `data: [DONE]\n\n`. Prior behavior was a single non-SSE JSON.
- **Per-item media size limit aligned with HTTP body limit.**
  `ServerMultimodal.maxPayloadBytes` is now equal to
  `ServerLimits.maxBodySize`. `validatePayloadSizes` runs in
  `validateMediaShape` (i.e. before the model-loaded check) so oversized
  payloads return `413` regardless of server state. Test helper's
  `maxBodySizeOverride` is now plumbed through `HTTPHandler.init`.
- **Documentation updated.** `README.md`, `docs/ARCHITECTURE.md`,
  `docs/BENCHMARKING.md`, `docs/GEMMA4_INTERNALS.md`, and
  `docs/SERVER_API.md` reflect the support matrix above.
- **Benchmark harness hardening.** Both benchmark scripts emit
  `cache_mode` per run and per group (`cold` / `warm` / `cache_hit` /
  `mixed`), `output_preview`, `output_sha256`, and `input_parity` fields.
  `release_gate.py` adds `--scope {release, multimodal_release}`.
  `gemma4_multimodal_benchmark.py` adds `--krillm-image-mode
  {bridge, native_cli, native_server}` so the native Swift image path can
  be benchmarked head-to-head against an Ollama daemon.
- **Audio fixture and quality rubric.** `tools/generate_audio_fixture.py`
  emits deterministic 1 kHz sine + silence WAVs (verified bit-identical
  across runs). `tools/quality_rubric.json` captures expected and
  forbidden terms per fixture for text, image, and audio.

### Verification on the target M4 Pro 24 GB machine

Performed on `206791d` (the head of code changes; docs-only commits since
do not affect these results):

| Check | Result |
| --- | --- |
| `make test` | Passed: 123 tests, 8 skipped (env-gated), 0 failures |
| `make release` | Passed; `.build/release/krillm` 37 MB |
| Live int8 KV parity test (`KLM_GEMMA4_MODEL_PATH=…`) | Passed |
| Live chat image-conditioning test (`KLM_GEMMA4_MODEL_PATH=…`) | Passed |
| Native CLI text smoke | Coherent output |
| Native CLI image smoke | Identifies red-box fixture |
| CLI audio smoke (`mlx-vlm` bridge) | Runs end to end |
| Audio fixture determinism | sha256 stable across runs |

The two live tests previously named in the review (the int8 KV "no
tokens" failure and the chat-multimodal-not-conditioning question) both
pass against the local Gemma 4 e2b checkpoint at
`/Users/sourav/.krillm/models/blobs/gemma-4-e2b`.

### Latest benchmark measurements

Text/server benchmark (`v3-text.json`, 5 runs / 2 warmup, `native_server`
vs Ollama `gemma4:e2b`):

| Metric | KrillLM | Ollama | Ratio | vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 0.305 s | 0.539 s | 0.5655x | 0.6289x |
| Decode throughput | 108 tok/s | 89 tok/s | 1.2119x | 1.1246x |
| Prefill throughput | 1701 tok/s | 1932 tok/s | 0.8807x | 0.7170x |

Multimodal `--krillm-image-mode native_server` gate (`v5-mm-gate.json`,
4 runs / 2 warmup, peak-memory sampling on):

| Metric | Ratio | Threshold | Status | Baseline |
| --- | ---: | ---: | --- | ---: |
| text_decode_ratio | 1.1738x | >=1.5 | FAIL (hard) | 1.4413x |
| text_prefill_ratio | 1.2231x | >=1.5 | WARN (advisory) | 0.0237x |
| text_ttft_ratio | 0.2126x | <=0.67 | OK | n/a |
| text_wall_ratio | 0.6351x | <=0.67 | OK | 0.6043x |
| image_prefill_ratio | 0.8899x | >=1.5 | WARN (advisory) | 0.0237x |
| image_wall_ratio | 0.6656x | <=0.67 | OK (margin <1%) | 2.2921x |
| memory_ratio | 1.1447x | <=1.0 | FAIL (hard) | n/a |
| audio_wall_ratio | 3.7868x | <=0.67 | SKIP (out_of_scope) | 1.2525x |

Geometric mean speedup: **1.157x** (memory excluded from the headline).
Text and image wall-time metrics still pass under `release_candidate`, but
the gate exits `1` because `text_decode_ratio` and `memory_ratio` are hard
misses. The text decode miss is driven by Ollama daemon variance between
the v4 and v5 runs; KrillLM's absolute decode rate is essentially unchanged.

---

## 3. Remaining Release Blockers

What is still required before this can be tagged a production release.
Items that have shipped since PR #9 are crossed out; see Section 4 for the
landing details.

> **macOS Ollama parity gate (new track, 2026-05-17).** A production tag now
> requires *both* the speedup `release_candidate` gate **and** the
> `mac_parity` gate (`make parity-gate`) green, per
> [`OLLAMA_MAC_PARITY_PLAN.md`](OLLAMA_MAC_PARITY_PLAN.md) §6. Phase 1 +
> Phase 2 tool calling are complete (`--compat`,
> `/api/version|ps|show|pull|delete|copy`, `/api/blobs`,
> `/v1/models/{id}`, **WS-B embeddings — dedicated BERT encoder**,
> **WS-D D1 tools/function calling on `/v1/chat/completions` + `/api/chat`,
> verified live**). `make parity-gate` is **GREEN — 10/10** on both
> `mac_parity` and `strict_parity`. Remaining plan workstreams (WS-C
> Modelfile, WS-D D2/D3/D4, WS-E keep-alive/concurrency, WS-F Anthropic,
> WS-G CORS/env) are Phase 2–4, not yet in the gated check-set, and must
> be added before the DoD `11435→11434` port flip.

### 3.1 Release benchmark gate

Against the accepted multimodal report (`.build/benchmarks/v6-mm.json`):

- **`release_candidate` exits `0` (GATE: PASS)** under the owner-accepted
  decode gate semantics (PR #16 + the 2026-05-16 gate proposal). All
  user-visible-latency and class-equal-memory metrics hard-pass, plus the
  hard `text_decode_ratio_floor >= 1.0x` (KrillLM never decodes slower
  than Ollama). `text_decode_ratio`'s `>= 1.5x` target is **advisory** and
  still printed as a WARN — the gate does not claim KrillLM hit 1.5x
  decode.
- **`strict` exits `1`** — unchanged; the uncompromised reference still
  fails `text_decode_ratio`, prefill TPS, and audio.

See `docs/RELEASE_GATE_DECODE_PROPOSAL.md` for the full rationale,
anti-relaxation safeguards, and the objective re-promotion contract, and
`docs/BENCHMARKING.md` for the per-metric kind table.

- **`text_decode_ratio` ~1.15x** (v6 run: 1.1937x; 1.13–1.19x across 5
  fresh runs) — **advisory** at the `>= 1.5x` target under
  `release_candidate`, with a **hard `>= 1.0x` non-regression floor**.
  This is a *structural*, not variance, gap: KrillLM decodes ~103–106
  tok/s vs Ollama's ~88–95 tok/s on the tiny 5B 4-bit Gemma 4 e2b, where
  llama.cpp's hand-tuned Metal decode kernels are genuinely competitive,
  and per-token weight-read bandwidth bounds dense decode. User-visible
  latency still wins decisively (text TTFT ~5x, text wall ~1.57x, image
  wall ~1.77x faster) — those are the metrics that substantiate the
  "1.5x–3x faster" product claim and they hard-pass. `text_decode_ratio`
  re-promotes to hard `>= 1.5x` when **either** Gemma 4 speculative
  decoding (Workstream 2) sustains `>= 1.5x` with greedy parity **or** the
  matrix adds a long-output decode task where decode dominates wall time.
  `strict` keeps it hard `>= 1.5x` regardless. This build remains a
  release-readiness baseline, not a production tag.
- ~~**`memory_ratio` 1.1447x**~~ — **CLOSED in PR #16.** Two compounding
  causes, both now fixed (see Section 4.4):
  1. **Measurement.** The v5 reading of ~9.6 GB KrillLM phys_footprint was
     measured without the clean `--krillm-server-pid` override the plan's
     own "Benchmark Rules" prescribe, so MLX's *unbounded* Metal
     buffer-recycling pool (it had no cap) plus process-tree contamination
     dominated the figure. With the prescribed clean per-process sampling
     KrillLM's text/image phys_footprint is ~2.85–3.0 GB.
  2. **Unbounded pool.** MLX never had a cache cap, so the recycling pool
     could grow into the multi-GB range under sustained load. PR #16 adds
     `MLXMemoryConfig` (default 256 MB cap, `KRILL_MLX_CACHE_LIMIT_MB`
     override) wired into every native model load.
  Across 5 fresh `native_server` runs on the M4 Pro 24 GB target,
  `memory_ratio` is 0.32–0.84 (always `<= 1.0`, the canonical run 0.322),
  with no decode regression. The class-equal 4-bit-vs-4-bit comparison is
  unchanged; memory is genuinely hard-gateable and now passes with margin.
- **`text_prefill_ratio` 1.2231x** — below the 1.5x threshold.
  Currently advisory under `release_candidate`. Re-promote to hard once a
  drafter, fused kernel, or short-prompt eval-cadence change pushes it
  consistently over 1.5x.
- **`image_prefill_ratio` 0.8899x** — structurally below the 1.5x
  threshold because the vision-encoder cache moves SigLIP2 forward and
  projector cost out of the prefill window. Currently advisory under
  `release_candidate`. Re-promote either by counting vision-encoder time
  inside the prefill bucket or by switching the multimodal gate to a
  wall/TTFT-based metric.
- **`audio_wall_ratio` 3.7868x** — out_of_scope under
  `release_candidate` until native Swift audio lands. The persistent
  `mlx-vlm` sidecar removed subprocess startup; the remaining gap is the
  bridge's generate cost vs Ollama's native path. Closing it requires
  porting Gemma 4's Conformer audio encoder and audio token expansion
  into native Swift+MLX. Multi-week effort.

Required outcome before release tag:

- The active gate (strict, or release_candidate after every advisory has
  been reviewed and every out_of_scope has been documented) exits `0`
  against the accepted multimodal benchmark report.
- The accepted report is committed or attached to release notes with
  machine, model, quantization, warmup, run count, `cache_mode`, and
  `kv_cache_dtype`.

### 3.2 Engineering work explicitly deferred to a follow-up PR

These were originally out-of-scope for PR #9. Items 4 and 5 have shipped;
see Section 4.

1. **Native audio in Swift.** Required to close `audio_wall_ratio`.
   Status: pending.
2. **Vocab-compatible Gemma 4 drafter or self-speculative decoding.**
   Required to push `text_decode_ratio` and `text_prefill_ratio`
   meaningfully past 1.5x. Status: pending.
3. **Custom Metal attention/MLP kernels.** Fused attention+RMSNorm or
   fused MLP gates would cut launch overhead and help short-prompt
   prefill. Status: pending.
4. ~~**Restructure `prefill_tps` measurement for multimodal.**~~ Replaced
   by a softer fix in PR #12: image and text prefill TPS are advisory
   under `release_candidate` with documented rationale. The harder
   restructure — counting vision-encoder time in prefill, or switching
   to a wall/TTFT-based multimodal metric — is still desirable and can
   re-promote both metrics to hard.
5. ~~**Quantized KV state save/restore in the prefix cache.**~~ Shipped
   in PR #11. int8 KV and prefix cache compose end-to-end with no
   dequant→requant round trip. See Section 4.

---

## Acceptance Criteria For Release Tag

A follow-up PR (or this one, if the gate is revised) can be considered
release-ready only when all of the following are true:

- `make test` passes.
- `make release` passes.
- Direct CLI smoke checks pass for Gemma 4 text, image, and audio
  (audio via the bridge unless native audio is implemented).
- `tools/release_gate.py` exits `0` for the accepted multimodal benchmark
  on the target machine, against the active gate definition.
- Documentation accurately states native versus bridge, CLI versus
  server, known unsupported paths, and benchmark caveats. (Done in this
  PR for the public docs; verify on each follow-up.)
- Release notes include benchmark report paths or attached artifacts
  with full reproducibility metadata.
- No benchmark compares non-equivalent KrillLM/Ollama inputs (e.g. text
  placeholder vs real media).

## Recommended Release Language

> This build implements Gemma 4 native text and image, audio via the
> persistent `mlx-vlm` bridge, server multimodal end-to-end (Ollama and
> OpenAI shapes), benchmark harness hardening, a corrected multimodal
> prefix cache, and an opt-in int8 KV cache that composes with the
> prefix cache. A bounded MLX Metal buffer cache
> (`KRILL_MLX_CACHE_LIMIT_MB`, default 256 MB) keeps peak `phys_footprint`
> in check (KrillLM ~2.85–3.0 GB vs Ollama ~8.2–8.4 GB on Gemma 4 e2b).
> On the v6 multimodal snapshot the **`release_candidate` gate passes**:
> user-visible latency wins decisively (text TTFT ~5x, text wall ~1.57x,
> image wall ~1.77x faster than Ollama) and class-equal peak memory passes
> hard, with a hard floor guaranteeing KrillLM never decodes slower than
> Ollama. KrillLM is competitive but **not** 1.5x ahead on raw
> decode-token/s against llama.cpp's Metal kernels on this tiny 4-bit
> model; that `>= 1.5x` decode target is a tracked advisory pending
> speculative decoding, and no release language should claim faster raw
> decode. The `strict` gate still exits `1` (decode, prefill, audio). This
> is a release-readiness baseline plus a documented follow-up roadmap; a
> production tag still requires the `strict` gate and native audio.

---

## 4. Post-baseline progress

Append-only log of what has shipped on top of PR #9. Each entry references
the PR for full detail.

### 4.1 PR #11 — `perf: compose int8 KV cache with prefix cache`

Merged onto `main` at commit `d6ed1ea`. Closes Workstream 5 of
[`OLLAMA_SPEEDUP_EXECUTION_PLAN.md`](../OLLAMA_SPEEDUP_EXECUTION_PLAN.md).

What landed:

- `QuantizedKVCache.quantizedSnapshot()`, `restoreQuantized(_:)`, and
  `truncate(to:)` mirror the fp16 contract while keeping K/V in uint8
  with their fp16 scales/zeros.
- `PrefixCache.storeQuantized` / `lookupQuantized` persist int8 entries
  to a separate `<key>.q8.safetensors` filename; the cache key schema
  bumps to v3 with a dtype tag so fp16 and int8 entries cannot collide.
- `InferenceEngine` removes the previous `&& !useInt8KV` guard on
  prefix cache; int8 lookup/restore/truncate mirrors fp16, including
  the Gemma 4 KV-sharing case where the shared-layer suffix leaves
  caches empty (only the non-shared prefix is persisted).

Coverage:

- `Tests/KLMCoreTests/QuantizedPrefixCacheTests.swift` — 4 unit tests
  (round trip, dtype isolation in both directions, restore+truncate+update
  length invariants).
- `Tests/KLMEngineTests/QuantizedPrefixCacheLiveTests.swift` (gated by
  `KLM_GEMMA4_MODEL_PATH`) — cold and warm runs through `InferenceEngine`
  with `kvCacheDtype: "int8"` and a shared `PrefixCache` produce
  identical greedy tokens.

Net effect on the test count: `123 / 8` → `128 / 9`.

### 4.2 PR #12 — `feat: release_candidate gate profile + kv_cache_dtype in reports`

Merged onto `main` at commit `2f3386a`. Closes Workstream 4 of the
execution plan.

What landed:

- `tools/release_gate.py --profile <name>` (default `strict`). The
  `release_candidate` profile classifies each metric as hard, advisory,
  or out_of_scope and only fails the gate on hard misses. **Missing
  hard metrics also fail the gate** — a release claim cannot rest on
  unmeasured numbers.
- Per-metric kind under `release_candidate`:
  - hard: `text_decode`, `text_wall`, `text_ttft`, `image_wall`
  - advisory: `text_prefill`, `image_prefill`, `memory`
  - out_of_scope: `audio_wall`, `audio_prefill`
- `memory_ratio` is advisory until
  `gemma4_multimodal_benchmark.py` records `peak_memory_gb_median`. The
  re-promotion contract is documented in `docs/BENCHMARKING.md`.
- `tools/gemma4_multimodal_benchmark.py` records
  `benchmark.kv_cache_dtype` (sourced from `KRILL_KV_CACHE_DTYPE`); the
  gate echoes it in the gate report and on the terminal header.
- `docs/BENCHMARKING.md` documents both profiles, the per-metric kind
  table, and the rationale for each downgrade.

Coverage:

- `tools/test_release_gate.py` — 7 unit tests covering strict vs
  release_candidate dispatch, audio scope skipping, hard-metric
  regression detection, missing-hard-metric → fail, and
  `kv_cache_dtype` surfacing.

Verified gate behavior on `.build/benchmarks/v4-mm.json`:

```text
strict                                  -> exit 1 (unchanged)
release_candidate --allow-dtype-mismatch -> exit 0
   hard pass: text_decode, text_wall, text_ttft, image_wall
   advisory: text_prefill WARN, image_prefill WARN, memory N/A
   skip:     audio_wall, audio_prefill
```

### 4.3 PR #14 — `feat: peak-memory sampling + release_candidate memory hard-gate`

Branch `feat/peak-memory-sampling`. Closes the PR #12 contract
"`memory_ratio` is advisory until the benchmark records peak memory"; see
the [`OLLAMA_SPEEDUP_EXECUTION_PLAN.md`](../OLLAMA_SPEEDUP_EXECUTION_PLAN.md)
"Goal For The Next PR" item 1.

What landed:

- `gemma4_multimodal_benchmark.py` samples each engine's process-tree
  memory from a daemon thread (default 50 ms poll) and records the peak
  alongside every measured run as `peak_memory_gb` plus
  `peak_memory_basis`. On macOS the per-PID number is `phys_footprint`
  from `proc_pid_rusage(RUSAGE_INFO_V2)` — the same figure Activity
  Monitor's "Memory" column reports, which counts resident mmap'd pages
  (KrillLM's safetensors weights). Non-Darwin platforms fall back to
  RSS from `ps`. Ollama PIDs auto-resolve via `pgrep ollama`; the
  KrillLM server via `pgrep -f 'krillm.*serve'`; the krillm CLI
  subprocess by its own PID. Operators can override with `--ollama-pids`
  / `--krillm-server-pid`, or skip sampling entirely with
  `--sample-memory off`. The KrillLM bridge path keeps using mlx-vlm's
  `GenerationResult.peak_memory` (MLX Metal allocator peak); all three
  bases are documented under `memory_sampling.basis`.
- `release_gate.py` promotes `memory_ratio` to `hard` under
  `release_candidate`. A new `resolve_metric_kinds()` helper
  auto-downgrades it to advisory whenever the report's
  `quantization.comparison.class_equal` is false (e.g. one engine bf16
  and the other Q4_K_M), with the downgrade recorded in
  `scope.memory_ratio` and added as a caveat. `strict` keeps it hard
  regardless. `memory_ratio` is excluded from the geometric-mean
  speedup headline because footprint is not a speed dimension.
- Docs (`docs/BENCHMARKING.md`, this file, the execution plan) updated.

Coverage:

- `tools/test_memory_sampling.py` — 17 unit tests covering the
  cross-platform sampler, the `phys_footprint` / RSS basis switch,
  process-tree topology walking, `pgrep` parsing, override handling,
  thread join on `__exit__`, the `_MemoryProbe` report block, and the
  `RSSSampler → MemorySampler` back-compat alias.
- `tools/test_release_gate.py` — 6 new tests covering memory present →
  recorded; quant-class-mismatch → advisory + caveat; quant-class-equal
  + over budget → hard fail; quant-class-equal + under budget → hard
  pass; strict keeps memory hard regardless of class; memory excluded
  from the geomean headline.

Net effect on the test count: `tools/` Python suite goes from 7 → 30
(13 gate + 17 memory). Swift `make test` is untouched.

Two findings the fresh benchmark surfaced:

1. **The canonical comparison is 4-bit-vs-4-bit, not bf16-vs-Q4.** The
   v4-mm.json snapshot recorded `quantization_class: "unknown"` for
   KrillLM because the bench was invoked with `--krill-model gemma-4-e2b`
   (registry name), and `krill_quantization()` couldn't find a local
   config and got a HuggingFace 401 falling back to "unknown". With the
   full path (`/Users/sourav/.krillm/models/blobs/gemma-4-e2b`), the
   on-disk config correctly identifies as 4-bit affine MLX and the
   comparison is class-equal with Ollama's Q4_K_M GGUF. The auto-downgrade
   for `memory_ratio` therefore does *not* apply on the canonical
   snapshot — memory is genuinely hard-gated, and currently failing.
2. **Ollama's text decode varies meaningfully run-to-run.** On the same
   machine and same Ollama version (0.21.0), the daemon ran ~73 tok/s in
   v4-mm and ~94 tok/s in v5-mm. KrillLM's absolute text decode is
   essentially unchanged (110.30 → 110.36 tok/s). The drop in
   `text_decode_ratio` from 1.50x → 1.17x is entirely Ollama variance.

Verified gate behavior on the refreshed `.build/benchmarks/v5-mm.json`
(produced fresh on the M4 Pro 24 GB target with `--krillm-image-mode
native_server`, `KRILL_KV_CACHE_DTYPE=fp16`, peak-memory sampling on,
class-equal 4-bit-vs-4-bit comparison):

```text
strict                                    -> exit 1
release_candidate --allow-dtype-mismatch  -> exit 1

  HARD pass:  text_wall_ratio   0.6351x   (target <= 0.67)
              text_ttft_ratio   0.2126x   (target <= 0.67)
              image_wall_ratio  0.6656x   (target <= 0.67)
  HARD fail:  text_decode_ratio 1.1738x   (target >= 1.5)
              memory_ratio      1.1447x   (target <= 1.0)
  ADV  warn:  text_prefill_ratio  1.2231x  (target >= 1.5)
              image_prefill_ratio 0.8899x  (target >= 1.5)
  SKIP:       audio_wall_ratio  3.7868x   out_of_scope
              audio_prefill_ratio  N/A    out_of_scope

  geomean speedup: 1.157x   (memory_ratio excluded from headline)

  KrillLM phys_footprint:  text 9.611 GB / image 9.611 GB / audio 10.481 GB
  Ollama  phys_footprint:  text 8.396 GB / image 8.495 GB / audio 8.511 GB
```

The release-candidate gate is now red on a hard-gated `text_decode_ratio`
and `memory_ratio`. That is the honest current state, surfaced for the
first time by the corrected quant identification + the new memory
sampling. Closing either one is follow-up work outside this PR.

### 4.4 PR #16 — `feat: cap MLX Metal buffer cache; close memory_ratio`

Branch `feat/mlx-cache-cap-memory-gate`. Closes the PR #14 hard
`memory_ratio` miss and the "Goal For The Next PR" item 1 (memory footprint
narrow slice).

Root-causing the v5 ~9.6 GB KrillLM phys_footprint reading found **two
compounding causes**:

1. **Unbounded MLX Metal buffer pool.** mlx-swift's buffer-recycling pool is
   sized from Metal's `recommendedMaxWorkingSetSize` (≈16 GB on a 24 GB M4
   Pro) and KrillLM never capped it. Freed intermediate buffers stay
   resident and are counted by `phys_footprint` / `RSIZE` (the exact figure
   the benchmark samples), so the pool could grow into the multi-GB range
   under sustained load even though MLX considers it "free".
2. **Contaminated measurement.** The v5 number was taken without the clean
   `--krillm-server-pid` override the plan's own "Benchmark Rules"
   prescribe; with clean per-process `native_server` sampling KrillLM's
   text/image footprint is ~2.85–3.0 GB.

What landed:

- `Sources/KLMCore/MLXMemoryConfig.swift` — `resolveCacheLimitMB`
  (pure, env-driven) + `apply()` which sets `MLX.Memory.cacheLimit`.
  Default 256 MB; `KRILL_MLX_CACHE_LIMIT_MB` overrides (`0` = legacy
  unbounded). 256 MB comfortably covers Gemma 4 e2b's fixed-size
  decode-step buffers so the hot loop still recycles — no decode
  regression.
- `loadModel(from:)` calls `MLXMemoryConfig.apply()` right after
  `MLXMetalRuntime.validateForNativeInference()` — one chokepoint, every
  native load, idempotent.

Coverage:

- `Tests/KLMCoreTests/MLXMemoryConfigTests.swift` — 5 unit tests (default,
  explicit value, `0`→disabled, whitespace trim, invalid→default).
- Net Swift test count: `128 / 9` → `133 / 9`, 0 failures.

Verified on the M4 Pro 24 GB target (`native_server`,
`--krillm-server-pid`, `KRILL_KV_CACHE_DTYPE=fp16`, peak-memory sampling
on, class-equal 4-bit-vs-4-bit), 5 fresh runs:

```text
release_candidate --allow-dtype-mismatch  -> exit 0   GATE: PASS
strict                                    -> exit 1   (unchanged)

  HARD pass:  memory_ratio              0.3221  (target <= 1.0)
              text_wall_ratio           0.6373  (target <= 0.67)
              text_ttft_ratio           0.2102  (target <= 0.67)
              image_wall_ratio          0.5645  (target <= 0.67)
              text_decode_ratio_floor   1.1937  (target >= 1.0)  ← floor
  ADV  warn:  text_decode_ratio         1.1937  (target >= 1.5)  ← demoted
              text_prefill_ratio / image_prefill_ratio
  SKIP:       audio_*                   out_of_scope

  KrillLM phys_footprint:  text/image ~2.85–3.0 GB  (was a contaminated
                           9.611 GB in v5)
  Ollama  phys_footprint:  text ~8.2–8.4 GB
```

Accepted report: `.build/benchmarks/v6-mm.json` (+ `v6-mm-gate.json`,
`v6-mm-strict-gate.json`). `make test` 133/9/0, `make release` passed,
CLI Gemma 4 text smoke coherent, `python3 -m unittest
tools.test_release_gate` 17/17.

**Gate semantics (owner-accepted 2026-05-16, `release_candidate` only;
`strict` unchanged):** `text_decode_ratio` is demoted from hard to
**advisory** at the `>= 1.5x` target, with a new synthetic **HARD
`text_decode_ratio_floor >= 1.0x`** so a decode regression vs Ollama (or
an unmeasured decode) still breaks the gate. Rationale, anti-relaxation
safeguards, and the objective re-promotion contract:
`docs/RELEASE_GATE_DECODE_PROPOSAL.md`. The gate report records the
demotion in `scope.text_decode_ratio` and a caveat; the summary still
prints `text_decode_ratio` as an advisory WARN at 1.19x — no claim that
KrillLM hit 1.5x decode.

Net effect: `memory_ratio` hard-passes; `release_candidate` exits `0`
honestly on the metrics that substantiate the product claim plus a hard
non-regression floor; `strict` still exits `1`. The `>= 1.5x` decode
aspiration remains tracked and re-promotable (Workstream 2 — speculative
decoding). This is the agreed release-candidate gate; `strict`-green and
a production tag still require Workstreams 1–2.
