# KrillLM

A Mac-native LLM inference CLI for Apple Silicon.

Built on Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. Ships as a single CLI binary.

## Benchmarks

Benchmark results depend on local hardware, OS, model quantization, installed binaries, and daemon state. KrillLM does not publish fixed KrillLM-vs-Ollama numbers in this README without an attached reproducibility report.

Use the local harness to compare an installed KrillLM model with an installed Ollama model:

```bash
make bench-compare \
  KRILL_MODEL=llama-3.2-1b \
  OLLAMA_MODEL=llama3.2:1b \
  BENCH_MAX_TOKENS=32 \
  BENCH_RUNS=5 \
  BENCH_WARMUP=2
```

The harness writes `.build/benchmarks/krillm-vs-ollama.json` by default and records:

- KrillLM and Ollama model names
- Prompt text and SHA256
- requested max tokens, actual prompt/generated token counts, runs, warmups, seed, temperature, and top-p
- per-run throughput/timing plus median/min/max summaries
- host environment, Swift version, KrillLM version, Ollama version, git commit, and git status

Prerequisites:

- Build KrillLM first with `make release`, or pass `--krillm-bin` when running the Python harness directly.
- Install the KrillLM model with `krillm pull <model>`.
- Install Ollama, start its daemon with `ollama serve`, and install the comparison model with `ollama pull <model>`.

The harness exits `77` and writes an actionable skip report when `ollama`, the Ollama daemon, or either model is missing. `krillm bench <model>` remains available for the native synthetic-token benchmark.

## Install

```bash
# From source (requires Xcode + Metal Toolchain)
git clone https://github.com/srvsngh99/KrillLM.git
cd KrillLM
make release
make install  # installs to /usr/local/bin/krillm

# Or just build and run from repo
swift build -c release
.build/release/krillm version
```

### Prerequisites

- macOS 14+ on Apple Silicon (M1 or newer)
- Xcode with Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`
- Swift 6.2+

## Quick Start

```bash
# Pull a model
krillm pull llama-3.2-3b

# Chat
krillm run llama-3.2-3b "What is the meaning of life?"

# Interactive REPL
krillm run llama-3.2-3b

# Start API server (OpenAI + Ollama compatible)
krillm serve --model llama-3.2-3b
# Then: curl http://localhost:11435/v1/chat/completions ...

# Native synthetic-token benchmark
krillm bench llama-3.2-3b
```

## Available Models

```bash
krillm pull llama-3.2-1b      # Llama 3.2 1B (fast, small)
krillm pull llama-3.2-3b      # Llama 3.2 3B
krillm pull llama-3.1-8b      # Llama 3.1 8B
krillm pull qwen2.5-7b        # Qwen 2.5 7B
krillm pull mistral-7b        # Mistral 7B v0.3
krillm pull gemma-2-9b        # Gemma 2 9B
krillm pull gemma-4-e2b       # Gemma 4 E2B (text via native experimental or mlx-vlm; media via mlx-vlm)
krillm pull phi-4-mini         # Phi-4 Mini
```

Or pull any mlx-community model directly:
```bash
krillm pull mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

## Commands

| Command | Description |
|---------|-------------|
| `krillm run <model> [prompt]` | Chat (interactive or single-shot) |
| `krillm pull <model>` | Download model from HuggingFace |
| `krillm serve` | Start OpenAI/Ollama-compatible HTTP server |
| `krillm bench <model>` | Run performance benchmarks |
| `krillm list` | Show installed models |
| `krillm rm <model>` | Remove a model |
| `krillm quantize <hf-path>` | Convert HF model to MLX format |
| `krillm version` | Print version and system info |

## Gemma 4 Multimodal Support

Gemma 4 image/audio support is routed through the Python `mlx-vlm` bridge. The native Swift Gemma 4 path is text-only and experimental; native image and audio preprocessing intentionally fails instead of producing placeholder tensors or ignoring media.

| Path | Text | Image | Audio | Tools |
|------|------|-------|-------|-------|
| Gemma 4 via `mlx-vlm` | Supported | Supported with `--image` | Supported with `--audio` | Not supported by `krillm run` |
| Gemma 4 native Swift | Experimental | Not supported | Not supported | Not supported |
| Other native models | Supported | Not supported | Not supported | Not supported |

Install the bridge dependency:

```bash
make setup-mlx-vlm

# Optional: force a specific interpreter instead of ~/.krillm/venv/bin/python3
export KRILLM_PYTHON=/path/to/venv/bin/python3
```

Smoke checks:

```bash
# Dependency detection used by krillm
python3 -c "from mlx_vlm import load; print('ok')"

# Image/audio route through mlx-vlm for Gemma 4
krillm run gemma-4-e2b "Describe this image" --image ./sample.jpg --max-tokens 64
krillm run gemma-4-e2b "Transcribe or summarize this audio" --audio ./sample.wav --max-tokens 64

# Test image and audio separately. Gemma 4 E2B is large enough that combined
# image+audio prompts should be treated as a separate load/performance test.

# Without mlx-vlm, Gemma 4 media fails loudly instead of falling back to native text-only inference.
KRILLM_PYTHON=/usr/bin/python3 krillm run gemma-4-e2b "Describe this image" --image ./sample.jpg
```

## API Compatibility

KrillLM serves on port 11435 (configurable) with both:

- **OpenAI API**: `POST /v1/chat/completions` (SSE streaming), `POST /v1/completions`, `GET /v1/models`
- **Ollama API**: `POST /api/chat`, `POST /api/generate`, `GET /api/tags`

Drop-in replacement - just change the port in your client config.

## Architecture

```
KLMCLI           CLI entry point (swift-argument-parser)
KLMEngine        Inference orchestration + speculative decoding
KLMCore          Model architectures (Llama, Qwen, Mistral, Gemma, Phi)
KLMCache         KV cache (fp16, int8 quantized, prefix cache)
KLMKernels       Custom fused Metal shaders (SwiGLU)
KLMRegistry      Model store, HF Hub puller, config
KLMServer        OpenAI + Ollama HTTP server (swift-nio)
KLMSampler       Greedy, temperature, top-k, top-p
KLMTokenizer     swift-transformers tokenizer wrapper
```

## Configuration

```bash
# Config file
~/.krillm/config.toml

# Environment variables
KRILL_DEFAULT_MODEL=llama-3.2-3b
KRILL_PORT=11435
KRILL_KV_CACHE_DTYPE=fp16
```

## Why KrillLM?

1. **Mac-native backend** - Uses MLX on Apple Silicon.
2. **Normal VM memory** - Does not wire model pages through `mlock`.
3. **Prefix cache** - Reuses repeated prompt prefixes.
4. **Speculative decoding** - Supports draft-model verification.
5. **Single binary** - No required background daemon for local CLI inference.

## Author

**Sourav Singh** / [Sourav AI Labs](https://github.com/srvsngh99)

## License

MIT
