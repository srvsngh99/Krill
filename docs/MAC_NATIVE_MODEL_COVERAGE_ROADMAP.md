# Mac-Native Model Coverage Roadmap

Created: 2026-05-17
Status: planning handoff

## Objective

Grow KrillLM from a fast, focused Mac-native runtime into a broader
Ollama-compatible runtime without losing the core product constraint:

```text
KrillLM must remain faster, lighter, and more explicit than Ollama on Mac
for every workflow it claims as production-supported.
```

The path is not "support every model by accepting every repo." The path is
capability-gated native support, measured fallback support, and clear
unsupported errors.

## Support Tiers

Every model or model type must be labeled with exactly one support tier.

| Tier | Meaning | Release claim |
| --- | --- | --- |
| Production native | Swift + MLX/Metal path, tests, docs, benchmark gate | Supported and recommended |
| Compatible fallback | Works through bridge/reference path or slower compatibility path | Works, not a performance claim |
| Experimental | Runs on known fixtures but lacks full gates | Developer preview |
| Unsupported | Explicit error before execution | Not supported |

Do not silently promote a model from one tier to another. Promotion requires
tests, benchmarks, and docs.

## Current Baseline

Production-native or close:

```text
Dense text LLMs: selected Llama, Qwen, Mistral, Gemma, Phi, GLM,
                 DeepSeek distills via Llama/Qwen architecture
Embeddings:      selected BERT/MiniLM/BGE style encoders
Vision:          Gemma 4 image only
Server APIs:     Ollama + OpenAI-compatible core text/generation surface
```

Fallback / scoped:

```text
Gemma 4 audio:   mlx-vlm bridge
Image+audio:     bridge path
```

Not broadly supported:

```text
MoE base models
Qwen-VL / Llama vision / LLaVA-style models
Whisper/ASR
TTS
Rerankers/cross-encoders
Diffusion/image generation
Video-language models
Arbitrary new Hugging Face architectures
```

## Workstreams

Each workstream has its own handoff doc.

| ID | Workstream | Doc | Purpose |
| --- | --- | --- | --- |
| WS1 | Native Gemma 4 audio | [WS1_NATIVE_GEMMA4_AUDIO.md](workstreams/WS1_NATIVE_GEMMA4_AUDIO.md) | Move voice/audio from bridge to native Metal |
| WS2 | Speculative decoding | [WS2_SPECULATIVE_DECODING.md](workstreams/WS2_SPECULATIVE_DECODING.md) | Restore strict decode speed margin |
| WS3 | Model adapter and capability registry | [WS3_MODEL_ADAPTER_REGISTRY.md](workstreams/WS3_MODEL_ADAPTER_REGISTRY.md) | Make support explicit and scalable |
| WS4 | New dense text families | [WS4_NEW_DENSE_TEXT_FAMILIES.md](workstreams/WS4_NEW_DENSE_TEXT_FAMILIES.md) | Add Qwen3/new Llama/Gemma/Mistral/Phi safely |
| WS5 | Second native vision family | [WS5_SECOND_NATIVE_VISION_FAMILY.md](workstreams/WS5_SECOND_NATIVE_VISION_FAMILY.md) | Expand beyond Gemma 4 image |
| WS6 | MoE runtime support | [WS6_MOE_RUNTIME_SUPPORT.md](workstreams/WS6_MOE_RUNTIME_SUPPORT.md) | Support router/expert models efficiently |
| WS7 | Specialized model types | [WS7_SPECIALIZED_MODEL_TYPES.md](workstreams/WS7_SPECIALIZED_MODEL_TYPES.md) | Rerankers, ASR, TTS, diffusion, video |

## Priority

1. **WS3: Model adapter and capability registry.**
   This prevents the next model additions from becoming scattered
   conditionals. It is the foundation for honest support tiers.
2. **WS1: Native Gemma 4 audio.**
   Closes the biggest multimodal gap and moves voice toward production.
3. **WS2: Speculative decoding.**
   Needed for strict text speed margin versus Ollama on long decode.
4. **WS4: New dense text families.**
   Adds breadth where the architecture delta is manageable.
5. **WS5: Second native vision family.**
   Adds real multimodal breadth after Gemma 4.
6. **WS6: MoE runtime support.**
   High value, but needs careful memory and dispatch design.
7. **WS7: Specialized model types.**
   Treat these as separate products, not incidental LLM features.

## Gates

Every production-native workstream must add or extend gates:

```text
api_parity_gate
text_gate
vision_gate
audio_gate
embedding_gate
moe_gate
memory_gate
quality_smoke_gate
```

Promotion to production-native requires:

```text
unit tests pass
live tests pass when model path is present
Ollama/reference parity checks pass
release_candidate gate passes
strict gate either passes or has an accepted scope proposal
docs and support matrix updated
```

## Performance Principles

- Prefer server-mode benchmarks. CLI process startup is not the production
  path and will distort wall time.
- Keep fast paths model-specific where performance matters.
- Use bridges as compatibility fallbacks, not production performance paths.
- Measure peak memory with process-tree sampling and record quantization
  class; do not compare unlike formats without caveats.
- Never mark unsupported media as supported just because the endpoint accepts
  a JSON field.
- Any new model family must document tokenizer, chat template, weight map,
  cache behavior, and supported capabilities.

## Product Positioning

Near-term honest claim:

```text
KrillLM is a Mac-native, lightweight, Ollama-compatible runtime optimized
for selected high-value local models, with native text and Gemma 4 vision
today, and native audio / broader multimodal support in progress.
```

Do not claim:

```text
KrillLM supports every Ollama model.
```

That claim would force breadth-first compatibility and would likely make the
runtime slower and heavier than Ollama.
