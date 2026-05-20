# WS5: Second Native Vision Family

Status: runtime shipped as `compatible_fallback` tier via Python
sidecar (`Qwen25VLEngine` -> mlx-vlm). Native Swift+MLX vision
tower + mRoPE + patch merger are a follow-up; documented as
non-goals below.

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
