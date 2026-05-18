# WS4: New Dense Text Families

Status: planned

## Goal

Keep KrillLM current with popular dense text model families while preserving
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

- `krillm pull <alias>` resolves to a known native adapter.
- `krillm run <alias>` produces coherent deterministic output.
- Server `/api/generate`, `/api/chat`, and OpenAI chat paths work.
- Tool/structured-output behavior is documented as model-quality dependent.
- Benchmark report is attached for production-native promotion.

## Non-Goals

- Do not add aliases for models that only partially load.
- Do not treat a new multimodal variant as a dense text model just because
  text generation works.
- Do not claim Ollama parity until endpoint and benchmark gates run.
