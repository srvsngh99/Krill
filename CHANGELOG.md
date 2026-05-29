# Changelog

All notable changes to KrillLM are recorded here. Entries are in
reverse chronological order. Versioning follows
[SemVer](https://semver.org/).

## [Unreleased]

### Added

- **Live batched serving (`BatchScheduler`)** (follow-up #8, Stage B —
  wiring): concurrent same-model requests are now coalesced into ONE batched
  forward, turning the verified Stage B engine into a live throughput feature.
  A per-model `BatchScheduler` gathers eligible requests into a static cohort
  (up to `KRILL_NUM_PARALLEL`, within a small `KRILL_BATCH_WINDOW_MS` window,
  default 8 ms) and drives a new streaming `InferenceEngine.generateBatched`
  entry that demuxes each row to its own token stream — with per-row sampling
  (temperature / top-p / top-k / penalties), per-row stop/maxTokens, and
  ragged prefill. `KRILL_NUM_PARALLEL < 2`, ineligible families, multimodal
  requests, seeded non-greedy sampling, and explicit speculative opt-ins fall
  through to today's serial path **byte-identically**. fp16 KV only; the
  prefix cache and speculative decode stay bypassed on the batched path;
  finished rows remain in the batch until the cohort completes (mid-flight
  shrinking and continuous admission are Stage C).
- **Batched concurrent decode engine** (follow-up #8, Stage B - core): the
  inference engine can now decode several ragged-length prompts in ONE
  batched forward for plain-causal families (Llama 3.x, Qwen 2.5/3 dense).
  Each row carries its own RoPE position (threaded into the attention
  forward) and an additive mask hides its left-padded prefix in the stacked
  KV cache, so a batched row reproduces that prompt's solo decode. Verified
  on real checkpoints: batched per-row logits match the single-prompt logits
  within fp16 rounding (~1 ULP), with no cross-row attention bleed. fp16 KV
  only; greedy/per-row sampling; speculative decode and the prefix cache are
  bypassed on the batched path. This lands the verified engine; wiring it to
  concurrent server requests (`KRILL_NUM_PARALLEL >= 2`) is the next PR.
- **Multi-model resident pool (`MAX_LOADED_MODELS > 1`)** (follow-up #8,
  Stage A — routing first): a new `EngineRegistry` keeps more than one
  model resident at once (`KRILL_MAX_LOADED_MODELS` /
  `OLLAMA_MAX_LOADED_MODELS`, default 1). Generate requests are now
  **routed-or-loaded** by model name: a request for an installed model
  loads it on demand and routes to it, keeping previously-loaded models
  resident up to the cap instead of discarding the prior model on every
  swap. Eviction is **in-flight-aware**: the least-recently-used resident
  that is NOT currently generating is evicted, and when the pool is full
  and every model is busy a new-model request gets a meaningful 503
  (naming `KRILL_MAX_LOADED_MODELS`) rather than tearing a model down
  mid-stream. All resident engines share one prefix cache (keys already
  namespace by model). At the default `MAX_LOADED_MODELS=1` behavior is
  unchanged. (Batched concurrent decode is Stage B.)
- **Per-model keep-alive** (follow-up #8, Stage A-2): each resident model
  carries its own idle deadline, so a request's `keep_alive` (default /
  `-1` to pin / `0` to evict-after-drain) applies to that model alone. The
  background evictor now unloads each model independently when its own
  deadline passes and it is idle, leaving other resident models loaded
  (previously a single global deadline unloaded the whole pool).
  `GET /api/ps` lists every resident model with its own `expires_at`,
  not just the active one. At `MAX_LOADED_MODELS=1` this matches the prior
  single-model behavior.

## [0.4.0] - 2026-05-28

Headline: Gemma 4 26B-A4B native MoE serves on Apple Silicon and beats
Ollama on every published Gemma 4 SKU (decode, prefill, and total wall
time). Plus Qwen3-MoE coherence, a vision_config-driven Gemma 4
encoder, daemon-mode CLI routing, the KLMAgent skeleton, and the
default-port flip to `11434` that makes KrillLM a zero-config Ollama
drop-in.

### Added

- **Gemma 4 26B-A4B native text MoE** (#81): first native inference of
  the sparse 26B-A4B variant on Apple Silicon. Router + top-K expert
  dispatch in Swift+MLX. Closes #80.
- **KLMAgent skeleton** (#65): `OperatorLoop`, `OperatorTool`,
  `OperatorEvent`, `HardwareInfo`, and `Recommender` land the structural
  foundation for agent mode (slice 3 sub-PR A). Tool wiring + CLI follow
  in later sub-PRs.
- **Daemon-mode CLI routing for `krillm run`** (#63): when a `krillm
  serve` daemon is already running, `krillm run` detects it (probes
  `/v1/status`), routes the request through `/v1/chat/completions`, and
  skips the per-call model load. TTFT drops from seconds to tens of
  milliseconds (~5x warm-daemon speedup). Text-only single-shot
  requests are routed; multimodal, draft-model, Modelfile-override, and
  REPL paths stay in-process. `KRILL_NO_AUTO_DAEMON=1` forces
  in-process.
- **Modelfile `TEMPLATE` override applied at decode**: created models
  carrying a `TEMPLATE` directive now render their prompt with it
  instead of the model's built-in chat template. Ollama `TEMPLATE`s are
  Go `text/template`, so this ships a from-scratch Go-template engine
  (`GoTemplate`: actions, pipelines, `if`/`range`/`with`, `{{- -}}`
  trimming, and the `eq`/`and`/`len`/`index`/`slice`/`printf`/... builtin
  set) plus the `OllamaTemplateContext` bridge from chat messages to the
  `.System`/`.Messages`/`.Prompt` render context. The override was
  already parsed and round-tripped through `/api/show`; the renderer was
  the missing piece. A template that fails to parse/evaluate falls back
  to the built-in chat template rather than failing the request.
- **SDK usage docs** (#64): verified end-to-end snippets for the OpenAI
  Python SDK, LangChain, LlamaIndex, and the Anthropic SDK pointing at
  the local server.

### Changed

- **SwitchGLU MoE dispatch replaces scatter** (#82): the per-layer
  scatter dispatch (a Swift loop driven by a host read of per-expert
  token counts) is gone. A new `Gemma4SwitchGLU` /
  `Gemma4QuantizedSwitchedLinear` runs one `gatherQuantizedMM` kernel
  per (gate, up, down) projection across all top-K experts with zero
  host syncs in the layer loop. Decode (N=1) pays no Swift loop. This
  flipped 26B-A4B from 9% behind Ollama to 43% ahead on total wall time.
- **Gemma 4 vision encoder reads `vision_config` from the checkpoint**
  (#79): the SigLIP2 tower shapes are parsed from the model's own
  config instead of hardcoded, so checkpoints with different vision
  dimensions load correctly.
- **Default server port flipped `11435` -> `11434`** (#83): KrillLM now
  listens on the same port stock Ollama uses, so existing Ollama
  clients connect with no configuration. The previous default `11435`
  still works for one release when set explicitly (`--port 11435` or
  `KRILL_PORT=11435`). This activates the T0-1 flip that
  `docs/OLLAMA_MAC_PARITY_PLAN.md` deferred until the `mac_parity` gate
  went green (18/18, 2026-05-28).
- **Quantization config requires explicit `group_size` and `bits`**
  (#74): the decode path no longer falls back to silent defaults when a
  checkpoint omits quant metadata; it now requires the values be
  present, surfacing malformed quant configs instead of guessing.
- **Qwen3-MoE SwitchGLU dispatch** (opt-in via `KRILL_NATIVE_MOE=1`):
  the native Qwen3-MoE runtime now dispatches the top-K experts with a
  single `gatherQuantizedMM` per projection (`Qwen3SwitchGLU`), the same
  pattern PR #82 applied to Gemma 4. The stacked
  `mlp.switch_mlp.{proj}.*` checkpoint tensors bind directly (no
  per-expert unpacking), and the per-layer host sync that drove the old
  scatter dispatch is gone. Decode on Qwen3-Coder-30B-A3B benches **2.7x
  faster (24 -> 66 tok/s)**. (At the time this landed the unsorted gather
  still regressed long-prompt prefill, so the path was opt-in; the #87
  sort path fixed prefill and #88 then made native the default - see
  those entries below.)
- **SwitchGLU sort path recovers prefill parity** (#87): the unsorted
  `gatherQuantizedMM` dispatch (#82, #85) does an `M=1` matmul per
  `(token, expert)` with experts gathered in router-score order, which
  regresses long-prompt prefill. Mirroring mlx-lm's `switch_layers` sort
  step, the SwitchGLU now sorts the flattened `(token, slot)`
  assignments by expert id once `indices.size >= 64` (prefill) so each
  expert's gather slice is contiguous and `gather_qmm`'s sorted-indices
  fast path applies, then unsorts the output back to `(token, slot)`
  order. Decode (`N=1`, below the threshold) stays on the unsorted fast
  path, so the #85 decode win is untouched. Measured on a 256-token
  prompt: Qwen3-Coder-30B-A3B prefill **229 -> 536 tok/s (+134%)** with
  decode held (65 tok/s); gemma-4-26b-a4b prefill **~230 -> 494 tok/s**
  with decode held (~59 tok/s). Applied to both `Gemma4SwitchGLU` and
  `Qwen3SwitchGLU`; shared helpers in `MoESortPath.swift`. This unblocks
  promoting native Qwen3-MoE to the default.
- **Native Qwen3-MoE runtime is now the DEFAULT** (#88): with #85
  decode (2.7x) and #87 prefill parity both landed, the native Swift+MLX
  Qwen3-MoE runtime no longer waits behind the `KRILL_NATIVE_MOE=1`
  opt-in. Qwen3-MoE checkpoints now load, serve, and tool-call on the
  native path with no env var; `KRILL_NATIVE_MOE=0` is the opt-out that
  forces the legacy mlx-lm bridge for one release. The model loader,
  `nativeMoEDispatchSupported`, and the server MoE routing all default to
  native; the still-unmigrated MoE families (Mixtral / Qwen2-MoE / OLMoE
  / DeepSeek-V3) continue to route to the bridge. `/api/show` reports a
  checkpoint-aware `support_tier`: a served Qwen3-MoE checkpoint is now
  `production_native` (the new `supportTier(for:at:)` resolves it from the
  installed config), while the bridge-only members and the family-level
  floor stay `compatible_fallback`. Verified end-to-end on
  Qwen3-Coder-30B-A3B: coherent generation and OpenAI tool calling
  (`finish_reason: tool_calls`) on the native default.

### Fixed

- **Qwen3-MoE coherence** (#78): mlx-community ships stacked
  `switch_mlp` expert weights; KrillLM now unpacks them into per-expert
  keys, so Qwen3-Coder-30B-A3B serves coherent text and tool calls
  instead of garbage.
- **Gemma 4 e4b / 26B-A4B crash on load** (#72): `layer_types` is now
  parsed from the checkpoint config; the previous hardcoded assumption
  crashed these variants.
- **Mixed-precision quant support** (#73): per-module `bits` /
  `group_size` overrides let checkpoints that quantize different modules
  at different precisions load correctly.
- **External `chat_template.jinja` loading** (#77): the tokenizer loads
  an external chat template file when present and bypasses a lossy
  round-trip that corrupted some templates.
- **Model puller HF file allowlist** (#71): extended to cover newer
  tokenizer file conventions so recent HF repos pull completely.
- **Removed fake `gemma-4-12b` / `gemma-4-27b` aliases** (#70): these
  SKUs do not exist; the aliases pointed at nothing and are gone.

### Performance vs Ollama

Median across 5 runs, warmed servers, 128-token generation, on the M4
target (full report + raw JSONs archived with the release). KrillLM
wins decode, prefill, and total wall time on every published Gemma 4
SKU:

| Variant | KrillLM decode | Ollama decode | KrillLM total | Ollama total | Total delta |
|---|---:|---:|---:|---:|---:|
| e2b (dense, ~2B, 4-bit) | 110.1 tok/s | 88.4 tok/s | 1.18s | 1.65s | +40% |
| e4b (dense, ~4B, 4-bit) | 62.6 tok/s | 55.2 tok/s | 2.07s | 2.53s | +22% |
| 26B-A4B (sparse MoE, 4-bit) | 61.6 tok/s | 49.0 tok/s | 2.11s | 3.02s | +43% |

The 26B-A4B SwitchGLU rewrite (#82) drove the headline gain: decode
41.2 -> 61.6 tok/s (+50%), prefill 3516 -> 5193 tok/s (+48%), total
3.17s -> 2.11s, flipping it from 9% behind Ollama to 43% ahead.

## [0.3.1] - 2026-05-24

Headline: cold-path multimodal prefill on Gemma 4 drops the per-position
vocab matmul over a 262144-vocab head to a single position, and a
family-aware engine warmup pass eliminates the first-request MLX
compile / Metal JIT spike. Plus a Homebrew install fix that was
already broken on v0.2.0 (binary alone, no metallib), and the
project's first Swift CI workflow.

### Added

- **`LoadedModel.multimodalPrefillForward`** (#53): an optional
  last-token-only variant of `multimodalForward`. Gemma 4 wires it.
  The engine prefers it on multimodal prefill and falls back when
  the family does not. Bit exact for the sampled token. Closes the
  cold-path gap on Gemma 4's `262144 x L -> 262144 x 1` matmul that
  PR #50 had only addressed for the text path.
- **Family-aware engine warmup** (#54): `InferenceEngine.warmup()`
  runs a tiny dry forward after `load()` and `swap()`. Vision-capable
  models include a 224x224 synthetic gray PNG so the 32 vision-block
  MLX.compile slots (from PR #48) get populated. Behind
  `KRILL_SKIP_WARMUP=1` for CI / cold-start-sensitive use. Best
  effort: warmup errors never block accepting real requests.
- **Swift CI workflow** (#60): `.github/workflows/swift-tests.yml`
  builds the Swift package and runs `swift test` on every PR and on
  `main`. Single `macos-15` job, SwiftPM cache keyed on
  `Package.resolved`, metallib step gated by `REQUIRE_METALLIB=1`.
  The repo previously only ran the Python tools-tests workflow; the
  entire Swift core had no automated coverage.
- **`KrillLMVersion` constant** in `Sources/KLMRegistry/KrillLMVersion.swift`
  (#56) replaces four hardcoded "0.3.0" string literals across CLI,
  server, and Ollama-compat. `KrillLMVersionMatchesVersionFile` test
  asserts agreement with the repo-root `VERSION` file at build time.
- **Per-family `lastTokenOnly` slice-equivalence tests** (#57): pin
  the bit-exact property of PRs #50 and #53 against a future
  refactor that moves a non-elementwise op after the slice.
- **Both-paths vision-stage profile** (#59): the Qwen 2.5-VL profile
  test now times the pre-PR-58 additive-mask windowed-attention
  path and the batched-per-window path side by side, keeping the
  delta auditable in CI.

### Changed

- **Qwen 2.5-VL vision tower** (#58): replaces the additive `-1e4`
  inter-window mask with per-window batched SDPA for the 28
  windowed vision blocks. SDPA cost drops from `O(L^2)` to
  `O(L * windowSize)` per block (16x on the canonical
  224x224 / 8x8 LLM grid). Wall-time impact is small (~1 percent)
  because MLP + Linears dominate the vision-block cost; the SDPA
  reduction is the right code shape regardless.
- **Homebrew formula** (#55): bumped to v0.3.0 (URL + sha256) and
  fixed the install layout to put `mlx.metallib` and the
  mlx-swift `Cmlx` bundle next to the binary via libexec/+symlink.

### Fixed

- **Homebrew install was broken on v0.2.0** (#55): the formula
  installed only the bare `krillm` binary, but the MLX runtime
  needs `mlx.metallib` adjacent to the executable to initialize
  Metal. No user hit it because source builds dominate on a small
  project; now correct from v0.3.0 onward.

### Performance vs Ollama

- **First request after engine load** is ~20 ms faster on
  Qwen 2.5-VL 3B (317 ms -> 287 ms median across 5 cold-cold-cold
  back-to-back distinct-image requests) thanks to the warmup pass
  populating MLX.compile slots and the Metal kernel JIT before the
  first user request lands (#54).
- **Cold-path Gemma 4 image prefill** drops the `262144 x L`
  vocab matmul to `262144 x 1`. Wall-time delta scales with the
  number of prompt tokens; bit exact for the sampled token (#53).

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
