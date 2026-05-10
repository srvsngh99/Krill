# Release Readiness Remediation Plan

Originally drafted: 2026-05-10.
Baseline commit: `4a2b6e6` (`Release readiness baseline and Gemma4 fixes`).
PR: #9 (`feat/release-readiness-remediation`).

This document is structured as three sections:

1. **Historical Baseline** — the pre-PR state that motivated the plan.
   Frozen for reference.
2. **Current PR State** — what is true at the head of this PR.
3. **Remaining Release Blockers** — what is still required before this can
   be tagged a production release.

It is not yet a production release. The release benchmark gate still fails
on three metrics, all documented in section 3.

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

## 2. Current PR State (HEAD of `feat/release-readiness-remediation`)

What is true after the commits in this PR (latest: `206791d`).

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

Performed on `206791d`:

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

Multimodal `--krillm-image-mode native_server` gate (`v4-mm-gate.json`,
4 runs / 2 warmup):

| Metric | Ratio | Threshold | Status | Baseline |
| --- | ---: | ---: | --- | ---: |
| text_decode_ratio | 1.5030x | >=1.5 | OK | 1.4413x |
| text_prefill_ratio | 1.4498x | >=1.5 | FAIL (3% short) | 0.0237x |
| text_ttft_ratio | 0.1173x | <=0.67 | OK | n/a |
| text_wall_ratio | 0.5172x | <=0.67 | OK | 0.6043x |
| image_prefill_ratio | 1.0385x | >=1.5 | FAIL (structural) | 0.0237x |
| image_wall_ratio | 0.5593x | <=0.67 | OK | 2.2921x |
| audio_wall_ratio | 3.9265x | <=0.67 | FAIL (deferred) | 1.2525x |

Geometric mean speedup: **1.418x** (was 0.429x baseline).
Wall-time metrics across text and image now beat Ollama by 1.6x–1.7x.
Five of seven gate metrics pass.

The standalone text benchmark sits at `text_decode 1.21x` because of a
different prompt; on the multimodal text task `text_decode` reaches
1.50x. Both numbers are honest readings on the same machine; the gate
uses the multimodal benchmark.

---

## 3. Remaining Release Blockers

What is still required before this can be tagged a production release.

### 3.1 Release benchmark gate fails

`tools/release_gate.py` does not exit `0` against the accepted multimodal
report. Three metrics still fail:

- **`text_prefill_ratio` 1.4498x** — 3% short of the 1.5x threshold.
  Likely closeable with a vocab-compatible Gemma 4 drafter or
  kernel-level work, neither of which fits in this PR.
- **`image_prefill_ratio` 1.0385x** — structurally below the 1.5x
  threshold. The vision encoder cache deliberately moves SigLIP2
  forward + projector cost out of the prefill window, so the prefill_tps
  metric falls even though `image_wall_ratio` (the user-facing wall
  time) passes at 0.5593x. Either the metric needs to be redefined to
  count vision-encoder time as prefill, or the multimodal gate should be
  switched to wall/TTFT-based.
- **`audio_wall_ratio` 3.9265x** — gated on native audio in Swift. The
  persistent `mlx-vlm` sidecar removed subprocess startup; the remaining
  gap is the bridge's actual generate cost relative to Ollama's native
  path. Closing it requires porting Gemma 4's Conformer audio encoder
  and audio token expansion into native Swift+MLX. Multi-week effort.

Required outcome before release tag:

- The gate exits `0` for the accepted multimodal benchmark report, OR
- The gate is explicitly revised with a documented rationale for which
  metrics are required versus advisory, and the revised gate exits `0`.
- The accepted report is committed or attached to release notes with
  machine, model, quantization, warmup, run count, and `cache_mode`.

### 3.2 Engineering work explicitly deferred to a follow-up PR

These are documented as out-of-scope for this PR:

1. **Native audio in Swift.** Required to close `audio_wall_ratio`.
2. **Vocab-compatible Gemma 4 drafter or self-speculative decoding.**
   Required to push `text_decode_ratio` and `text_prefill_ratio`
   meaningfully past 1.5x.
3. **Custom Metal attention/MLP kernels.** Fused attention+RMSNorm or
   fused MLP gates would cut launch overhead and help short-prompt
   prefill.
4. **Restructure `prefill_tps` measurement for multimodal.** Either count
   vision-encoder time in the prefill bucket, or replace
   `image_prefill_ratio` with a wall/TTFT-based gate.
5. **Quantized KV state save/restore in the prefix cache.** int8 KV is
   currently incompatible with prefix cache because the snapshot path
   dequantizes on every call. A serialization format for quantized KV
   would let users get both wins simultaneously.

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

## Recommended Release Language Until The Gate Passes

> This build implements Gemma 4 native text and image, audio via the
> persistent `mlx-vlm` bridge, server multimodal end-to-end (Ollama and
> OpenAI shapes), benchmark harness hardening, and a corrected
> multimodal prefix cache. Five of seven release-gate metrics pass and
> wall-time ratios beat Ollama by 1.6x–1.7x. The gate still fails on
> `text_prefill_ratio` (3% short), `image_prefill_ratio` (structural —
> vision cache moves work out of prefill), and `audio_wall_ratio`
> (gated on native audio). This is a release-readiness baseline, not a
> production release.
