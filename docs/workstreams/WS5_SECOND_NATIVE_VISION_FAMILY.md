# WS5: Second Native Vision Family

Status: foundation only (family detection + capability metadata +
explicit loader rejection). Native vision tower lands in follow-up
PRs.

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

## What is NOT in this PR

The native runtime work. Each item below is its own follow-up PR:

- 3D mRoPE (`rope_scaling.type == "mrope"` with `mrope_section:
  [16, 24, 24]` split across temporal/height/width axes). Reuses
  the existing Qwen RoPE module with per-section bases.
- Window-attention vision tower (32 layers, hidden 1280, patch 14)
  with periodic full-attention at `fullatt_block_indexes: [7, 15,
  23, 31]`.
- Spatial patch merger (`spatial_merge_size: 2`).
- Image preprocessing (resize to a multiple of `patch_size * merge`
  = 28, normalization, packing into the model's expected layout).
- Multimodal forward: `<|vision_start|>` (151652) /
  `<|image_pad|>` (151655) / `<|vision_end|>` (151653) placeholder
  injection on the text side, identical pattern to Gemma 4 vision
  but with Qwen-shaped tokens.
- Server pre-generation media gating (image accepted for
  `.qwen25vl`, audio rejected with a clear error).
- Vision-fixture benchmark vs Ollama qwen2.5-vl.

## Acceptance status

From the workstream's acceptance bar:

- "Server rejects unsupported media/family combinations clearly." -
  **DONE** (this PR ships the explicit loader rejection and the
  family declares only `visionInput`, not `audioInput`, so audio
  requests fail the WS3 media gate at the server layer).
- "Benchmark shows production-native performance OR marks the
  path experimental/fallback." - **DONE** (this PR marks the path
  experimental in the registry; once the native runtime lands the
  benchmark replaces the tier).
- "Selected family handles image-only prompts natively." -
  **PENDING** (follow-up PR).
- "Image fixture changes output versus text-only prompt." -
  **PENDING** (follow-up PR).
- "Two different image fixtures produce different outputs." -
  **PENDING** (follow-up PR).

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
