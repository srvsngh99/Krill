# WS5: Second Native Vision Family

Status: planned

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
