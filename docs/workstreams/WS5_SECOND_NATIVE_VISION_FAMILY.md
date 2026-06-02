# WS5: Second Native Vision Family

Status: COMPLETE. Qwen 2.5-VL runs end-to-end on a native
Swift+MLX runtime - vision tower, 3D mRoPE text tower, image
preprocessing, and a grid- and decode-offset-correct generation
loop. The native path is the default and only runtime: the Python
bridge (`Qwen25VLEngine` + `qwen25vl_bridge.py`) is retired, the
support tier is promoted `compatible_fallback` -> `production_native`,
and a request flows HTTP -> image decode -> native prefill -> native
decode -> response with no Python sidecar. Validated on a real
`Qwen2.5-VL-3B-Instruct-4bit` checkpoint against a recorded mlx-vlm
oracle baseline.

## What landed in the runtime PR (this PR)

- `Qwen25VLRuntime` (`Sources/KLMEngine/Qwen25VLRuntime.swift`):
  the native decode driver - prefill then an incremental KV-cached
  decode loop. It threads the per-step mRoPE offset
  (`Qwen25VLForConditionalGeneration.mropePositionOffset`): the
  driver captures the prefill's mRoPE frontier
  (`Qwen25VLPositions.Coords.nextPos`, newly exposed) and forwards
  decode step `k` at `frontier + k`. After an image prompt the
  KV-cache length is NOT the next mRoPE position - the image span
  compresses `gridH * gridW` placeholder tokens to
  `max(gridH, gridW)` positions - so this offset is load-bearing.
- Real `(gridH, gridW)` grid threading. `Qwen25VLImagePreprocessor.preprocess`
  decodes a PNG/JPEG (CoreGraphics), `smartResize`s it to a multiple
  of `patch_size * spatial_merge_size`, and returns the per-patch
  batch together with the actual - generally non-square - post-merge
  grid. No more square-grid assumption.
- Window attention. `Qwen25VLVisionTower.windowedAttentionPlan` groups
  patches into per-window mini-sequences and runs them as a batched
  `[numWindows, windowSize, hidden]` attention, dropping the windowed-block
  SDPA cost from O(L^2) to O(L * windowSize). It now covers BOTH uniform and
  ragged grids: a ragged edge window (image dims not a multiple of the window
  edge) is padded up to the uniform window size with an additive padding mask
  that zeroes the padded keys. The original `windowAttentionMask` (a
  block-diagonal additive mask, equivalent to the HF permutation +
  `cu_window_seqlens` scheme without reordering) is retained as the numerical
  reference the equivalence test checks against, and now only serves the
  single-window degenerate case at runtime.
- Engine + tokenizer. `InferenceEngine.generate` detects a VL model
  and routes through `generateQwen25VL`, which preprocesses the
  image and renders the ChatML prompt with the `<|image_pad|>` run
  via `KLMTokenizer.formatQwen25VLTokenIds`.
- Server. `ModelAdapter` routes `.qwen25vl` to `.denseEngine`; the
  standard chat path handles a VL request exactly like Gemma 4
  vision (`decodeMediaForRequest` -> `engine.generate(... imageData:)`).
  `handleVLMChat` / `decodeMediaForRequestVLM` / the `vlmEngine`
  field / the `.visionBridge` routing are removed.
- Bridge retired. `Sources/KLMEngine/Qwen25VLEngine.swift` and
  `tools/qwen25vl_bridge.py` are deleted; the shared sidecar
  plumbing (`LineReader`, `VLMError`, the venv interpreter path)
  moved to `Sources/KLMEngine/PythonSidecar.swift` for `MoEEngine`.
  Tier promoted to `production_native` in `ModelCapabilities`.
- Tests. `Qwen25VLRuntimeTests` is the decode-correctness gold
  standard: an incremental prefill+decode loop must match a single
  full forward over `prompt + generated`, per-position, including a
  non-square image span and a decode run long enough to cross the
  KV-cache compaction threshold. `Qwen25VLSmokeTests` (gated on
  `KLM_QWEN25VL_MODEL_PATH`) loads the real checkpoint, asserts the
  image conditions the answer, and checks rubric-equivalence to the
  recorded mlx-vlm oracle (`Fixtures/ws5_oracle_baseline.json`).
  Window-mask and preprocessing tests cover the foundation pieces.

## What landed in the native multimodal-forward PR

- `Qwen25VLMRoPE` is now consumed by a native text tower:
  - `Qwen25VLTextAttention` applies 3D mRoPE to Q/K with explicit
    per-axis (t, h, w) position arrays, instead of the standard
    1D `RoPE` module. QKV-bias, GQA, no q_norm/k_norm - dense
    Qwen 2.5 attention shape otherwise.
  - `Qwen25VLTextBlock` / `Qwen25VLTextModel` build the language
    transformer stack; the text model accepts pre-computed input
    embeddings so vision embeddings can be injected first.
- `Qwen25VLPositions.compute` computes 3D mRoPE position ids on
  the host for a prompt with at most one image span: text tokens
  advance all three axes together; `<|image_pad|>` tokens form a
  `gridHMerged x gridWMerged` 2D grid at a fixed temporal index;
  text resumes at `startPos + max(gridH, gridW)` - matching the
  HF `get_rope_index` reference.
- `Qwen25VLForConditionalGeneration` is the full multimodal
  model. Module keys match the mlx-vlm checkpoint layout:
  `vision_tower.*`, `language_model.model.*`,
  `language_model.lm_head.*`. The forward embeds tokens, runs the
  vision tower, splices vision embeddings into the `<|image_pad|>`
  span (`injectVisionEmbeds`), computes mRoPE positions, runs the
  text stack, and projects to logits.
- `Sources/KLMCore/ModelLoader.swift`: the `KRILL_NATIVE_QWEN25VL=1`
  arm now calls `loadQwen25VL`, which builds the native model,
  loads weights, and returns a `LoadedModel` with both a text-only
  `forward` and a `multimodalForward` closure.
- Tests: `Qwen25VLNativeTests` adds mRoPE position-id tests
  (text-only, square + non-square image grids), `injectVisionEmbeds`
  splice tests, module-key-layout tests, text-only + multimodal
  forward finite-logit tests, the WS5 acceptance-bar tests (image
  changes output vs text-only; two images differ), and a
  multimodal micro-benchmark.

## Runtime follow-up: complete

Every item the multimodal-forward PR deferred shipped in the
runtime PR above: server image-preprocessing wiring, real
`(gridH, gridW)` grid threading, the decode-step mRoPE offset,
real-checkpoint validation + bridge retirement + tier promotion,
and the window-attention mask builder. There is no remaining WS5
follow-up.

## What landed in the native foundation PR (prior)

- `Sources/KLMCore/Qwen25VLModel.swift` (new): native Swift+MLX
  modules for Qwen 2.5-VL.
  - `Qwen25VLConfig` parses the full `config.json`: language-side
    fields (compatible with the existing dense Qwen 2.5 attention
    + MLP modules via `qwenTextConfig`), the multimodal token
    ids (`<|image_pad|>`, `<|vision_start|>`, `<|vision_end|>`,
    `<|video_pad|>`), the `rope_scaling.mrope_section` split,
    and the full `vision_config` (depth, hidden_size, num_heads,
    patch_size, spatial_merge_size, fullatt_block_indexes,
    window_size, etc.).
  - `Qwen25VLMRoPE` implements 3D mRoPE: splits per-head dim
    into three sub-vectors along the `mrope_section` lengths and
    rotates each with its own (t, h, w) position axis. Tolerates
    both conventions for the section sum (`head_dim` or
    `head_dim/2`).
  - `Qwen25VLPatchMerger` collapses each `spatial_merge_size^2`
    block of patches into one language-side token via RMSNorm +
    Linear -> GELU -> Linear. Matches the `merger.{ln_q, mlp.0,
    mlp.2}` weight keys (GELU at index 1 has no parameters; its
    array slot keeps the second Linear at index 2 to match the
    shipped checkpoint exactly).
  - `Qwen25VLPatchEmbed`, `Qwen25VLVisionAttention`,
    `Qwen25VLVisionMLP`, `Qwen25VLVisionBlock` build the
    32-layer SigLIP-style vision tower. The block index check
    against `fullAttnBlockIndexes` is wired in so the runtime
    PR only needs to attach the real window mask.
  - `Qwen25VLVisionTower` composes patch_embed + N blocks +
    merger. Forward runs the tower over packed-patch input and
    returns language-aligned vision embeddings.
  - `Qwen25VLImagePreprocessor`: CLIP-mean normalize +
    pack-patches helpers (resize to multiple of 28 stays on the
    caller side so platform-native resize backends can plug in).
- `Sources/KLMCore/ModelLoader.swift`: the qwen2_5_vl arm now
  reserves the `KRILL_NATIVE_QWEN25VL=1` env-gate. With the gate
  set the loader emits a "foundation modules landed, runtime is
  the follow-up" rejection so users see the env-gate is intentional;
  without the gate the existing bridge redirect stands.

## Acceptance status (native runtime - final)

- "Selected family handles image-only prompts natively." -
  **DONE.** A VL chat request runs end-to-end on the native
  Swift+MLX engine: `Qwen25VLSmokeTests` loads the real
  `Qwen2.5-VL-3B-Instruct-4bit` checkpoint and generates from a
  PNG with no Python sidecar.
- "Image fixture changes output versus text-only prompt." -
  **DONE.** `testImageInputChangesOutputVsTextOnly` (synthetic)
  and `testImageConditionsTheAnswer` (real checkpoint: red vs
  green).
- "Two different image fixtures produce different outputs." -
  **DONE.** `testTwoImagesProduceDifferentOutputs` (synthetic);
  the real-checkpoint smoke distinguishes red / green / blue.
- "Server rejects unsupported media/family combinations clearly." -
  **DONE.** A non-vision model rejects an image payload with
  HTTP 400 on the standard chat path's capability gate.
- "Benchmark shows production-native performance or marks the
  path experimental/fallback." - **DONE.** The path is
  `production_native`; correctness is gated by the
  decode-equivalence test and rubric-equivalence to the recorded
  mlx-vlm oracle.

## What landed in the bridge PR (prior)

## What landed in the runtime PR

- `Sources/KLMEngine/Qwen25VLEngine.swift`: long-lived Python
  sidecar wrapping `mlx_vlm.generate`. JSON-over-stdin/stdout
  protocol matches the shape Gemma 4 audio used before WS1
  retired its bridge. One sidecar per server instance, lazy
  load on first VL request.
- `tools/qwen25vl_bridge.py`: the Python side. Sidesteps an
  mlx-vlm 0.5.0 chat-template bug by constructing the Qwen
  2.5-VL chat sequence manually (system + user + vision_start /
  image_pad / vision_end + user text + assistant marker).
- `Sources/KLMServer/Server.swift`: chat-completion routes
  (`/v1/chat/completions` and `/api/chat`) dispatch VL-family
  requests through `handleVLMChatOpenAI` before the
  InferenceEngine model-loaded gate, so a VL request does not
  get refused for "no model loaded". One image per request; the
  bridge ignores per-request sampling parameters.
- `Sources/KLMRegistry/ModelCapabilities.swift`: `.qwen25vl`
  tier moved from `experimental` to `compatible_fallback` (the
  workstream's tier definition for bridge-backed runtimes).

## Benchmark vs Ollama native

M-series, Qwen 2.5-VL-3B-Instruct-4bit, 3 warm runs of
"What color do you see?" + 64x64 red PNG, max 32 tokens:

| Engine                              | Median warm latency |
| ----------------------------------- | ------------------- |
| KrillLM `/api/chat` (bridge)        | 300 ms              |
| Ollama `/api/chat` (qwen2.5vl:3b)   | 277 ms              |

KrillLM is 8% slower than Ollama's native C++ + mlx-vlm path,
despite going through a Python sidecar. The cold-start cost
(first request after server boot) is ~2.9 s on both (model load
into mlx-vlm + first prefill); subsequent requests reuse the
loaded model.

Quality is at parity by construction: KrillLM's bridge invokes
the same `mlx_vlm.generate` Ollama wraps internally on Mac.

## What landed in this PR

Selected family: **Qwen 2.5-VL** (`Qwen2_5_VLForConditionalGeneration`,
`model_type: qwen2_5_vl`). Selection criteria from the workstream:

- High user demand (Qwen 2.5-VL is the current best open VLM).
- Stable MLX checkpoint availability
  (`mlx-community/Qwen2.5-VL-{3B,7B,32B}-Instruct-4bit`).
- Clear reference implementation in mlx-vlm and the upstream
  Transformers code.
- Inherits the Qwen 2.5 text-side architecture this build already
  ships natively, so the multimodal port is incremental rather than
  a full new family.
- Ollama 0.21+ has Qwen 2.5-VL support for benchmark comparison
  once the native runtime lands.

What this PR ships:

- `ModelFamily.qwen25vl` (rawValue: `"qwen2_5_vl"` to match Ollama's
  snake_case identifier convention) with detection from both
  `architectures: ["Qwen2_5_VLForConditionalGeneration"]` and
  `model_type: "qwen2_5_vl"` (also `qwen2_vl` for the older
  generation). Detection order matches VL BEFORE the generic `qwen`
  arm so a VL checkpoint never routes to the text loader.
- Capability declaration: `textGeneration`, `visionInput`, `tools`.
  Audio is intentionally NOT declared (WS5 scope is image-only).
- Support tier: `experimental`. Promotion to `production_native`
  is gated on the items below shipping in follow-up PRs.
- Alias entries for `qwen2.5-vl-3b`, `qwen2.5-vl-7b`,
  `qwen2.5-vl-32b`.
- Explicit `ModelLoadError.unsupportedArchitecture` from
  `ModelLoader.swift` BEFORE any weight load runs. The message names
  the family, points at this workstream doc, and suggests the
  qwen2.5 text-only path for text workflows so users are not stuck.
- Tests pin both halves of the contract: detection routes to
  `.qwen25vl`, and the loader rejects with the documented error
  text.

## What is NOT in this PR (follow-ups for the native path)

These items are intentionally deferred. None are needed to use
Qwen 2.5-VL today; promotion from `compatible_fallback` to
`production_native` is gated on them:

- 3D mRoPE (`rope_scaling.type == "mrope"` with `mrope_section:
  [16, 24, 24]` split across temporal/height/width axes). Reuses
  the existing Qwen RoPE module with per-section bases.
- Window-attention vision tower (32 layers, hidden 1280, patch 14)
  with periodic full-attention at `fullatt_block_indexes: [7, 15,
  23, 31]`.
- Spatial patch merger (`spatial_merge_size: 2`).
- Image preprocessing in Swift (resize to a multiple of 28,
  normalization, packing into the model's expected layout).
- Multimodal forward in Swift with `<|vision_start|>` (151652) /
  `<|image_pad|>` (151655) / `<|vision_end|>` (151653) placeholder
  injection on the text side, identical pattern to Gemma 4
  vision but with Qwen-shaped tokens.
- Removing the Python sidecar dependency once the native path is
  validated.

## Acceptance status

From the workstream's acceptance bar:

- "Server rejects unsupported media/family combinations clearly." -
  **DONE** (the family declares only `visionInput`, not
  `audioInput`, so audio requests fail the WS3 media gate at the
  server layer; non-VL chat callers that ask for VL models still
  get the loader's explicit redirect to `/api/chat`).
- "Benchmark shows production-native performance OR marks the
  path experimental/fallback." - **DONE** (path marked
  `compatible_fallback`; benchmark vs Ollama qwen2.5vl:3b shows
  300 vs 277 ms warm median, 8% gap).
- "Selected family handles image-only prompts natively." -
  **DONE for the bridge path.** The bridge runs natively on Mac
  (no x86 emulation, no remote inference), just through a Python
  sidecar process. The strict reading ("Swift+MLX in-process")
  is a non-goal for this PR and is the follow-up.
- "Image fixture changes output versus text-only prompt." -
  **DONE** (`testImageInputChangesOutputVsTextOnly` in
  `Tests/KLMEngineTests/Qwen25VLBridgeTests.swift`).
- "Two different image fixtures produce different outputs." -
  **DONE** (`testTwoFixturesProduceDifferentOutputs`; red
  fixture -> "Red", green fixture -> "Green").

## Goal

Add one additional native vision-language family beyond Gemma 4 image.

Candidate families:

```text
Qwen-VL / Qwen2.5-VL / Qwen3-VL
Llama vision variants
LLaVA-style models
Mistral vision variants
Gemma 3 vision if chosen as a compatibility target
```

Pick one family first. Do not attempt generic vision support across all
families in one PR.

## Selection Criteria

- High user demand.
- Stable MLX checkpoint availability.
- Clear reference implementation.
- Reasonable image preprocessing complexity.
- Ollama/reference model available for benchmark comparison.
- Compatible memory footprint on target Macs.

## Required Components

```text
vision preprocessing
vision encoder
projector into language hidden size
media token/chat-template handling
masked embedding injection or model-specific equivalent
server request validation
benchmark fixture and gate
```

## Key Files

```text
Sources/KLMCore/VisionEncoder.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMServer/ServerMultimodal.swift
tools/gemma4_multimodal_benchmark.py
docs/GEMMA4_INTERNALS.md
docs/SERVER_API.md
```

New family-specific files are expected. Do not overload Gemma 4's
`VisionEncoder.swift` with unrelated architecture logic.

## Acceptance

- Selected family handles image-only prompts natively.
- Image fixture changes output versus text-only prompt.
- Two different image fixtures produce different outputs.
- Server rejects unsupported media/family combinations clearly.
- Benchmark shows production-native performance or marks the path
  experimental/fallback.

## Non-Goals

- Do not add audio for the selected family in this workstream.
- Do not implement multiple vision families at once.
- Do not accept image payloads for unsupported families.
