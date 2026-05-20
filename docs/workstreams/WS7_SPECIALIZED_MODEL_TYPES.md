# WS7: Specialized Model Types

Status: planned

## Goal

Decide which non-chat model types KrillLM should support, and add them as
separate product tracks instead of forcing them through the LLM generation
loop.

Candidate types:

```text
rerankers / cross-encoders
ASR / Whisper-style speech recognition
TTS / speech generation
diffusion image generation
video-language models
OCR/document models
code-specific fill-in-middle models
```

## Principle

Each model type needs its own API shape, runtime path, tests, and benchmark.
Do not hide specialized models behind chat-completions if that produces
incorrect semantics or bad performance.

## Subtracks

### Rerankers

Likely closest to existing embedding support, but still needs a scoring head
and ranking API.

Status: foundation only (family detection + capability metadata +
explicit loader rejection). Cross-encoder scoring runtime + `/v1/rerank`
endpoint land in follow-up PRs.

What landed:

- `ModelFamily.reranker` (rawValue `"reranker"`). Detection from
  architectures matching `*ForSequenceClassification` or
  `*CrossEncoder*` (covers BGE Reranker `XLMRobertaForSequenceClassification`,
  Cohere-style cross-encoders, BERT classification heads). Matched
  BEFORE the generic bert/roberta arm so a reranker checkpoint
  never silently routes to the embedding loader (which has no
  scoring head and would either crash on the classifier weights
  or silently run with no head).
- Capability: ONLY `reranker`. No `textGeneration`, no `embeddings`.
  The server's existing capability checks therefore refuse
  `/api/generate`, `/v1/chat`, `/v1/embeddings`, `/api/embed` on
  these models BEFORE the engine is touched.
- Tier: `experimental`.
- Aliases: `bge-reranker-base`, `bge-reranker-large`, `bge-reranker-v2-m3`.
- `ModelLoader.swift` rejects with an explicit
  `unsupportedArchitecture` pointing at this workstream doc and
  suggesting the `bge-small-en` embedding-plus-dot-product stand-in.

Acceptance:

- Pair/list scoring works. **PENDING** (follow-up PR).
- `/v1/rerank` or documented local API exists. **PENDING**.
- Scores match a reference model within tolerance. **PENDING**.

### ASR

Requires audio frontend plus encoder-decoder or CTC-style decoding depending
on target model.

Acceptance:

- Deterministic WAV transcription works.
- Word/error quality smoke exists.
- Streaming behavior is defined before server production support.

### TTS

Separate generation pipeline. Usually not an autoregressive text-token loop
compatible with current `InferenceEngine`.

Acceptance:

- Text-to-audio file generation works.
- Output format and sample rate are documented.
- Latency and memory benchmarks exist.

### Diffusion

Not in current runtime scope.

Acceptance:

- Only start after a separate image-generation runtime proposal is accepted.

### Video-Language

Requires frame sampling, video preprocessing, visual temporal encoder or
multi-image strategy, and larger memory planning.

Acceptance:

- Fixed short-video fixture changes model output versus text-only.
- Frame sampling and max payload limits are documented.

## Key Files

```text
Sources/KLMCore/EmbeddingModel.swift
Sources/KLMEngine/EmbeddingEngine.swift
Sources/KLMServer/Server.swift
Sources/KLMServer/ServerParsing.swift
docs/SERVER_API.md
docs/BENCHMARKING.md
```

New runtime files are expected for ASR/TTS/diffusion/video. Do not overload
the text `InferenceEngine` with incompatible generation semantics.

## Acceptance For Any Specialized Type

- Has an explicit capability in the model registry.
- Has a server/API contract or is CLI-only by design.
- Has unit tests and at least one live fixture.
- Has benchmark output and memory data.
- Has a clear support tier.

## Non-Goals

- Do not promise all specialized model types as one release.
- Do not add aliases for unsupported runtime types.
- Do not weaken text or multimodal gates to include unrelated models.
