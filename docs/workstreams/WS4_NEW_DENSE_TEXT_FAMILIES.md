# WS4: New Dense Text Families

Status: Qwen 3 dense landed (qwen3-0.6b through qwen3-14b aliases).

## What landed in this PR

Native MLX/Metal support for the Qwen 3 dense family inside the
existing Qwen loader. The architectural deltas vs Qwen 2.5 are switched
on by flags carried through `QwenConfig`, so a single
`QwenForCausalLM` covers both variants:

- `attention_bias: false` (Qwen 3) vs the historical Qwen 2.5
  `bias: true` on QKV projections.
- Per-head RMSNorm on Q and K before RoPE (Qwen 3's `q_norm` /
  `k_norm`). Optional modules so Qwen 2.5 checkpoints continue to
  load with zero unused parameters.
- Tied embeddings (`tie_word_embeddings: true`): `lm_head` is now an
  Optional module; when tied, the LM projection reuses
  `model.embed_tokens.asLinear(hidden)`, the same path Gemma 4 uses.
  `WeightLoader.loadWeights` learned a `tieWordEmbeddings: Bool`
  parameter so the embed_tokens -> lm_head key duplication step does
  not run on models without an `lm_head` property.
- Explicit `head_dim` in config (Qwen 3 sets it, Qwen 2.5 derives it).
- Mask dtype is now sourced from the embedding output dtype, so the
  Qwen 3 bf16 inference path no longer crashes with
  `[scaled_dot_product_attention] Mask type must promote to output
  type bfloat16`.

Aliases:

```text
qwen3-0.6b  mlx-community/Qwen3-0.6B-4bit
qwen3-1.7b  mlx-community/Qwen3-1.7B-4bit
qwen3-4b    mlx-community/Qwen3-4B-4bit
qwen3-8b    mlx-community/Qwen3-8B-4bit
qwen3-14b   mlx-community/Qwen3-14B-4bit
```

All inherit the qwen family's WS3 capability set
(`textGeneration`, `tools`) and `production_native` support tier.

## Benchmark vs Ollama

Qwen 3 1.7B, max 64 tokens, 3 runs / 1 warmup, M-series:

- Krill: 144.0 tok/s
- Ollama:  135.1 tok/s

Krill is 1.07x faster than Ollama on this pair. Plain decode; no
speculative decoding.

## Acceptance status

- `krill pull <alias>` resolves to the new entries.
- `krill run qwen3-1.7b "..."` loads and produces coherent output
  including Qwen 3's `<think>` chain-of-thought trace.
- Server `/api/generate`, `/api/chat`, OpenAI chat paths inherit the
  Qwen family handling without further changes.
- Tool calling behavior tracked by the existing qwen tool template
  (PR #23 native parity); structured output is family-agnostic via
  the structured sampler.
- Bench report against Ollama attached above.

## Non-goals

- Qwen 3 MoE variants (defer to WS6).
- Qwen 3 VL (multimodal) variants (defer to WS5).

## Goal

Keep Krill current with popular dense text model families while preserving
Mac-native speed, memory discipline, and honest support claims.

Priority examples:

```text
Qwen3 and newer Qwen text models
new Llama dense text variants
new Gemma dense text variants
new Mistral dense text variants
new Phi dense text variants
DeepSeek distills that map to existing dense architectures
```

## Scope

This workstream is for dense decoder-only text models. It is not for MoE,
vision-language, ASR, TTS, or diffusion models.

## Key Files

```text
Sources/KLMCore/LlamaModel.swift
Sources/KLMCore/QwenModel.swift
Sources/KLMCore/MistralModel.swift
Sources/KLMCore/GemmaModel.swift
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCore/PhiModel.swift
Sources/KLMCore/GLMModel.swift
Sources/KLMCore/ModelConfig.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMRegistry/AliasMap.swift
Sources/KLMTokenizer/TokenizerWrapper.swift
```

## Implementation Checklist For A New Dense Model

1. Confirm architecture and `model_type`.
2. Confirm tokenizer and special token behavior.
3. Confirm chat template.
4. Confirm RoPE/scaling policy.
5. Confirm attention head layout and KV shape.
6. Confirm MLP activation and norm type.
7. Confirm safetensors key names and quantization format.
8. Add alias only after native load/run works.
9. Add deterministic smoke tests.
10. Run server-mode benchmark against Ollama/reference.

## Acceptance

- `krill pull <alias>` resolves to a known native adapter.
- `krill run <alias>` produces coherent deterministic output.
- Server `/api/generate`, `/api/chat`, and OpenAI chat paths work.
- Tool/structured-output behavior is documented as model-quality dependent.
- Benchmark report is attached for production-native promotion.

## Non-Goals

- Do not add aliases for models that only partially load.
- Do not treat a new multimodal variant as a dense text model just because
  text generation works.
- Do not claim Ollama parity until endpoint and benchmark gates run.
