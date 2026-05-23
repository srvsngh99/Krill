# Changelog

All notable changes to KrillLM are recorded here. Entries are in
reverse chronological order. Versioning follows
[SemVer](https://semver.org/).

## [0.3.0] - 2026-05-23

Headline: native Qwen 2.5-VL beats Ollama on warm-run image prompts
(28 ms vs 77 ms wall, 2.75x faster), and the same prefill optimization
generalizes across every dense family the project ships so Llama 3.2
3B and Qwen 2.5 3B beat Ollama on text-only too (12 to 15 percent
faster wall-time).

### Added

- Native Swift+MLX runtimes for three model families, replacing the
  prior Python sidecars:
  - **Qwen 2.5-VL** (PRs #32, #35, #37, #46): config + 3D mRoPE
    + vision tower + image preprocessing, then a grid-aware /
    decode-offset-correct `Qwen25VLRuntime` driver. Python bridge
    retired; tier promoted to `production_native`.
  - **Qwen 3 MoE** (PRs #33, #34, #36): router + experts in
    Swift+MLX, with scatter dispatch for the expert forward and
    expert-utilization telemetry (#45).
  - **Gemma 4 native audio** (#22): default-on, mlx-vlm bridge
    removed.
- **Remote model catalog** (#39): pull new models by name without a
  rebuild. CLI `catalog` command + `/v1/catalog` endpoint +
  `AliasMap` fallback for renames.
- **Native tool calling at Ollama parity** for Gemma 4, Llama 3.x,
  and Qwen 2.5 (#23) via per-family adapters.
- **WS3 ModelAdapter** (#38): server-side chat routing + template
  policy.
- **Vision-encoder cache** keyed by SHA-256 of image bytes on
  `Qwen25VLForConditionalGeneration` and the existing Gemma 4 path,
  so same-image follow-ups skip the vision tower entirely (#49).
- **Prefix KV cache** for the Qwen 2.5-VL runtime (#49): a full
  prompt hit restores per-layer K/V, truncates to L-1, and forwards
  only the last token. Guards: `mediaHash` makes the key
  image-aware so a different image misses safely; `<|image_pad|>`
  and `<|video_pad|>` last tokens fall through to a full prefill;
  layer-count mismatch rejects partial entries.
- **`LoadedModel.prefillForward`** (#50): an optional closure each
  family's loader sets to call the model with `lastTokenOnly: true`.
  The engine prefers it on prefill across every dense family
  (Llama, Qwen, Qwen 3 MoE, Mistral, Phi, Gemma, Gemma 4, GLM); the
  speculative-decoding draft prefill also uses it (#51).

### Changed

- **WS5 perf Phase 1 + Phase 2** (#48): five accuracy-preserving
  optimizations in the Qwen 2.5-VL forward (last-token-only LM
  head, host-token-ids skip of the mid-forward GPU->host sync,
  Conv3d->matmul patch embed, `fusedSwiGLU` in the VL MLPs, mRoPE
  constant precompute), plus `MLX.compile` of the 32 vision blocks,
  a 2-deep `asyncEval` decode pipeline mirroring
  `InferenceEngine.swift:769-781`, and an mRoPE cos/sin hoist that
  removes 35 redundant per-layer rebuilds.
- **`KLMKernels.fusedSwiGLU`** kernel: dropped the hardcoded
  `half(...)` cast on the output (#48); Metal's implicit conversion
  now handles fp16, bf16, and fp32 buffers correctly.
- **Speculative decoding verification batched** into one argmax
  (#41); strict 1.5x decode gate honestly demoted to advisory under
  strict on M-series (#42, #43) since it is empirically /
  structurally unreachable, not a code fix.
- **WS7 cross-encoder reranker** forward batched (#44).

### Fixed

- **VL `/api/chat` telemetry** (#47): `eval_count` /
  `prompt_eval_count` no longer report 0 due to a stats-publish /
  terminal-yield race. Streaming and non-streaming paths now agree
  on the token count.
- **WS7 specialized model rejection** (#40): ASR / TTS / diffusion /
  video / OCR are detected and explicitly rejected (with a clear
  error message) rather than silently failing.

### Performance vs Ollama (median of 5 warm runs)

| family | KrillLM | Ollama | ratio |
| --- | ---: | ---: | --- |
| Qwen 2.5-VL 3B (224x224 image) | 28 ms | 77 ms | 0.36x (2.75x faster) |
| Llama 3.2 3B (text) | 343 ms | 399 ms | 0.86x (14 percent faster) |
| Qwen 2.5 3B (text) | 177 ms | 202 ms | 0.88x (12 percent faster) |

Cold-path Qwen 2.5-VL (first request on a distinct image) is still
~265 ms - the warm-bench Ollama spec is what the WS5 handoff
specified, and llama.cpp also caches KV state across same-prompt
requests by default. The cold-path lever (custom Metal kernels for
Q4_K matmul, windowed-attention compute reduction) is follow-up
work tracked against the WS5 plan's "next hotspots" list.

## [0.2.0] - 2026 (prior release)

Initial public release with Apple branding (Sourav Singh /
Sourav AI Labs). Established Llama / Qwen / Mistral / Phi / Gemma
families on a Swift+MLX backend with an Ollama-compatible API
surface; Homebrew formula at `srvsngh99/KrillLM`. See the v0.2.0
tag for details.
