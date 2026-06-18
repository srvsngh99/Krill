# Krill Architecture

## Overview

Krill is a Mac-native LLM inference engine for Apple Silicon built on MLX. It ships as a single CLI binary (`krill`) with an HTTP server, supporting 7 model families with prefix caching and speculative decoding.

### Release Status

This is not a production release because the release benchmark gate still fails on three metrics. Server multimodal is implemented for Gemma 4 — native image and bridge-backed audio — as shown in the support matrix below. See the [README support matrix](../README.md#support-matrix) and [`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md) for the authoritative status and the remaining gate gaps.

### Gemma 4 Multimodal Support Matrix

| Path | Text | Image | Audio |
|------|------|-------|-------|
| CLI native | Supported | Supported (SigLIP2) | — |
| CLI bridge | — | — | Supported (mlx-vlm) |
| Server | Supported | Supported (Gemma 4 only, native) | Supported (Gemma 4 only, mlx-vlm bridge) |

Native Swift covers Gemma 4 text and image (text model + SigLIP2 vision encoder). When `--audio` is present, RunCommand routes the entire request (including any `--image`) through the mlx-vlm Python bridge because native audio is not implemented. Image-only requests use the native Swift vision path. The HTTP server mirrors this routing: image-only requests run through the native engine; audio (with or without an image) goes through the bridge.

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
  KLMCLI/             CLI commands (run, serve, launch, pull, bench, etc.)
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

Default port: 57455

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/chat/completions` | POST | OpenAI chat (SSE streaming) |
| `/v1/completions` | POST | OpenAI text completion |
| `/v1/responses` | POST | OpenAI Responses API (Codex `wire_api="responses"`) |
| `/v1/messages` | POST | Anthropic Messages API (Claude Code / Anthropic SDK) |
| `/v1/models` | GET | List models |
| `/v1/models/load` | POST | Load model by name |
| `/v1/models/unload` | POST | Unload model |
| `/v1/status` | GET | Server status + timing |
| `/api/chat` | POST | Ollama chat |
| `/api/generate` | POST | Ollama generate (with timing fields) |
| `/api/tags` | GET | Ollama model list |
| `/healthz` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

The three generation wire protocols share one internal generate path. The
Anthropic (`/v1/messages`) and Responses (`/v1/responses`) surfaces are thin,
pure translation layers (`AnthropicCompat.swift`, `ResponsesCompat.swift`) that
map a foreign request shape onto the internal messages/tools/sampling params and
format the result back; tool calls on every surface flow through one
model-agnostic `<tool_call>`/`<tool_response>` sentinel extractor
(`ToolCalling.swift`). This is what lets a single server back agents that speak
three different protocols.

## Coding Agent Backend (`krill launch`)

`krill launch <agent>` boots a terminal coding agent (Claude Code, Codex,
OpenCode, Hermes, Pi, Copilot CLI, Droid) pre-wired to the local server, the way
`ollama launch` does. The design principle: **the adapter is an endpoint inside
the server, not a per-agent external proxy.** Each agent speaks one of the three
generation wire protocols above, and `launch` only points it at the matching
endpoint.

| Agent's protocol | Endpoint | Agents |
|---|---|---|
| Anthropic Messages | `/v1/messages` | Claude Code |
| OpenAI Chat Completions | `/v1/chat/completions` | OpenCode, Hermes, Pi, Copilot, Droid |
| OpenAI Responses | `/v1/responses` | Codex (it dropped `wire_api="chat"`) |

**Layout.** Agent knowledge is a declarative table in
`Sources/KLMCLI/AgentProfiles.swift` (one `AgentProfile` literal per agent:
wire protocol, env to export, config files to `write`/`mergeJSON`, setup
`preExec` commands, binary, install hint). `Sources/KLMCLI/LaunchCommand.swift`
stays generic over the table.

**Flow.** Resolve the profile and model (`--model` or first installed) -> ensure
a server is up with that model loaded (auto-start a detached `krill serve` and
poll `/healthz`, or fail loud; `--no-serve` opts out) -> apply the agent's
config files + env + setup commands -> `execvp` the agent so it inherits the
real TTY/stdin/signals. The auto-started server survives the exec (it is a
separate process) and keeps running after the agent exits; the keep-alive
controller unloads its idle *model* to free memory (and `krill stop` unloads
it on demand), but the server *process* itself stays resident until killed.

**Config safety.** `write` targets are krill-owned paths only (e.g. Codex gets
an isolated `config.toml` under a krill-owned `CODEX_HOME`, so the user's real
`~/.codex` is never touched). `mergeJSON` deep-merges only our keys into the
user's config, keeps a `.bak`, and concatenates+dedups arrays (so e.g. Droid's
`custom_models` never clobbers the user's existing entries).

Adding an agent is a one-literal edit. Two roster members are documented for
manual setup rather than auto-wired: `codex-app` (the desktop app reads the real
`~/.codex` and would change the user's default provider) and `openclaw`
(config surface unverified). See
[`CONNECT_CODING_AGENTS.md`](CONNECT_CODING_AGENTS.md) for usage and
[`SERVER_API.md`](SERVER_API.md) for the raw endpoint shapes.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mlx-swift | >= 0.21.0 | MLX arrays, NN modules, Metal kernels |
| swift-transformers | >= 0.1.12 | HuggingFace tokenizer loading |
| swift-argument-parser | >= 1.5.0 | CLI argument parsing |
| swift-nio | >= 2.70.0 | HTTP server |
| swift-crypto | >= 3.8.0 | SHA256 hashing |
| swift-log | >= 1.6.0 | Structured logging |
