# KrillLM

A faster, Mac-native LLM inference CLI for Apple Silicon.

Built on Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. Ships as a single CLI binary.

**1.57x faster than Ollama** on decode, **58% less memory**, zero CoreAudio conflicts.

## Performance (llama3.2:1b, M4 Pro)

| Metric | KrillLM | Ollama | Delta |
|--------|---------|--------|-------|
| Decode (32 tok) | 252 tok/s | 161 tok/s | **1.57x** |
| TTFT | 17ms | 136ms | **8x faster** |
| Memory | 704 MB | 1,685 MB | **58% less** |
| mlock usage | None | 1.8 GB wired | No system conflicts |

<details>
<summary>Benchmark methodology</summary>

- **Hardware**: Apple M4 Pro
- **Model**: `llama3.2:1b` (Hugging Face canonical weights via `mlx-community/Llama-3.2-1B-Instruct-4bit`)
- **Prompt**: "Explain quantum computing in simple terms"
- **Settings**: temperature=0, max_tokens=32
- **Measurement**: Median of 5 runs after 2 warmup runs
- **Ollama version**: v0.5.x (specify exact version when reproducing)
- **KrillLM version**: v0.2.0
- **macOS**: 15.x, Xcode 16.x, Swift 6.2

To reproduce:
```
krillm run llama3.2:1b --prompt "Explain quantum computing in simple terms" --max-tokens 32
```
</details>

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

# Benchmark
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

## Why KrillLM over Ollama?

1. **Faster** - MLX on Metal is 1.5x+ faster than Ollama's llama.cpp backend on Apple Silicon
2. **Less memory** - No mlock. Models use normal VM pages, not wired memory
3. **No system conflicts** - Ollama's mlock exhausts the system mlock budget, breaking CoreAudio (browser audio). KrillLM doesn't.
4. **Prefix cache** - Repeated system prompts (agent loops) get sub-20ms TTFT
5. **Speculative decoding** - Draft model verification for 1.5-3x sustained decode
6. **Single binary** - No daemon, no client/server split, no background process

## Author

**Sourav Singh** / [Sourav AI Labs](https://github.com/srvsngh99)

## License

MIT
