# KrillLM Architecture

## Overview

KrillLM is a Mac-native LLM inference engine for Apple Silicon built on MLX. It ships as a single CLI binary (`krillm`) with an HTTP server, supporting 7 model families with prefix caching, speculative decoding, and native Gemma 4 multimodal.

## Module Dependency Graph

```
KLMCLI (executable)
  |-- KLMEngine        (inference orchestration)
  |     |-- KLMCore    (model architectures)
  |     |-- KLMCache   (KV cache, prefix cache)
  |     |-- KLMTokenizer (tokenizer wrapper)
  |     |-- KLMSampler (greedy, top-k, top-p)
  |-- KLMServer        (HTTP API)
  |     |-- KLMEngine
  |-- KLMRegistry      (model store, HF puller)
  |-- KLMRuntime       (Metal GPU validation)
```

## Source Layout

```
Sources/
  KLMCLI/             CLI commands (run, serve, pull, bench, etc.)
  KLMEngine/          Inference engine, speculative decoder, Python fallback
  KLMCore/            Model architectures (Llama, Gemma4, Qwen, etc.) + model loader
  KLMCache/           KV cache (batched concat) + prefix cache (LRU + disk)
  KLMServer/          HTTP server (OpenAI + Ollama APIs)
  KLMTokenizer/       HuggingFace tokenizer wrapper
  KLMSampler/         Token sampling (greedy, temperature, top-k, top-p)
  KLMRegistry/        Model registry, HF puller, manifests
  KLMRuntime/         Metal runtime validation
  KLMKernels/         Custom Metal shaders (planned)
```

## Generation Pipeline

```
User prompt
  |
  v
Tokenizer (chat template -> token IDs)
  |  Gemma4 uses direct token ID path to preserve special tokens
  v
Prefix Cache Lookup (full-hit only)
  |  Hit -> restore KV, truncate to last-1, re-forward last token
  |  Miss -> forward entire prompt
  v
Prefill (forward all prompt tokens)
  |  Multimodal: preprocessImage() -> VisionEncoder -> embedVision -> inject at <|image|> positions
  |  Graph compaction: MLX.eval() every 5 layers
  v
Prefix Cache Store (write-behind, async disk)
  |
  v
Decode Loop
  |  Standard: one token per forward pass
  |  Speculative: draft K tokens, verify in single target pass
  v
Token Stream (AsyncStream<TokenEvent>)
  |
  v
Output (CLI print / SSE stream / JSON response)
```

## Model Family Support

| Family | File | Attention | MLP | RMSNorm | Notes |
|--------|------|-----------|-----|---------|-------|
| Llama | LlamaModel.swift | GQA + RoPE | SwiGLU | Standard | Base architecture |
| Qwen | QwenModel.swift | GQA + RoPE (with bias) | SwiGLU | Standard | rope_theta=1M |
| Mistral | MistralModel.swift | GQA + RoPE | SwiGLU | Standard | Aliases LlamaConfig |
| Gemma | GemmaModel.swift | GQA + RoPE | GeGLU | +1 offset | Embedding scaling |
| Phi | PhiModel.swift | GQA + RoPE | SwiGLU (fused gate+up) | Standard | Fused QKV |
| GLM-4 | GLMModel.swift | GQA (fused QKV) + RoPE | SwiGLU (fused) | Standard | Post-norm |
| Gemma 4 | Gemma4Model.swift | Sliding+Full, KV sharing | GeGLU | 4-norm blocks | PLE, softcap, multimodal |

## HTTP API

Default port: 11435

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI chat (SSE streaming) |
| `/v1/completions` | POST | OpenAI text completion |
| `/v1/models` | GET | List models |
| `/v1/models/load` | POST | Load model by name |
| `/v1/models/unload` | POST | Unload model |
| `/v1/status` | GET | Server status + timing |
| `/api/chat` | POST | Ollama chat |
| `/api/generate` | POST | Ollama generate (with timing fields) |
| `/api/tags` | GET | Ollama model list |
| `/healthz` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mlx-swift | >= 0.21.0 | MLX arrays, NN modules, Metal kernels |
| swift-transformers | >= 0.1.12 | HuggingFace tokenizer loading |
| swift-argument-parser | >= 1.5.0 | CLI argument parsing |
| swift-nio | >= 2.70.0 | HTTP server |
| swift-crypto | >= 3.8.0 | SHA256 hashing |
| swift-log | >= 1.6.0 | Structured logging |
