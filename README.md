# KrillLM

A Mac-native LLM inference CLI for Apple Silicon.

Built on Apple's [MLX](https://github.com/ml-explore/mlx-swift) framework. Ships as a single CLI binary.

## Release Status

This is not yet a production release. Server multimodal is implemented for Gemma 4 — native image, bridge-backed audio — and shipped with end-to-end tests. The int8 KV cache composes with the prefix cache on Gemma 4 (PR #11), and the release gate now distinguishes hard, advisory, and out_of_scope metrics via `--profile release_candidate` (PR #12). Peak-memory sampling is wired into the benchmark and `memory_ratio` is hard-gated for class-equal comparisons (PR #14). PR #16 capped the MLX Metal buffer pool (`KRILL_MLX_CACHE_LIMIT_MB`), closed the `memory_ratio` hard miss (now 0.32–0.84, passing), and — per an owner-accepted gate proposal (`docs/RELEASE_GATE_DECODE_PROPOSAL.md`) — demoted `text_decode_ratio` to advisory under `release_candidate` with a new **hard `text_decode_ratio_floor ≥1.0x`** (KrillLM must never decode slower than Ollama). **`release_candidate` now exits `0` (GATE: PASS)** on `.build/benchmarks/v6-mm.json`; `strict` still exits `1` (text decode hard ≥1.5x, prefill, audio). The gate makes **no claim** that KrillLM hit 1.5x raw decode — that target is a tracked advisory pending speculative decoding. See [`docs/RELEASE_READINESS_REMEDIATION.md`](docs/RELEASE_READINESS_REMEDIATION.md) and [`OLLAMA_SPEEDUP_EXECUTION_PLAN.md`](OLLAMA_SPEEDUP_EXECUTION_PLAN.md) for the full status, the per-metric promotion contract, and acceptance criteria.

### Support Matrix

The matrix below separates CLI vs Server, native Swift vs Python bridge, and per-modality support.

Gemma 4 (`gemma-4-e2b`):

| Path | Text | Image | Audio |
|------|------|-------|-------|
| CLI native | Supported | Supported | — |
| CLI bridge | — | — | Supported (mlx-vlm) |
| Server | Supported | Supported (Gemma 4 only) | Supported (Gemma 4 only) [^audio-bridge] |

[^audio-bridge]: Server-side audio uses the same `mlx-vlm` Python bridge as the CLI; install with `make setup-mlx-vlm`. Image input on the server runs through the native Swift SigLIP2 path.

All other model families (Llama, Qwen, Mistral, Gemma 2, Phi, GLM-4) are text-only on both CLI and server.

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

For Gemma 4 text/image/audio comparison, install the Python bridge and run:

```bash
make setup-mlx-vlm
make bench-gemma4-multimodal
```

This writes `.build/benchmarks/gemma4-e2b-multimodal-4bit.json`. The harness benchmarks text, image, and audio separately and records the exact quantization metadata. KrillLM's local Gemma 4 E2B checkpoint uses MLX affine 4-bit; Ollama `gemma4:e2b` reports `Q4_K_M`, so the default report labels this as a 4-bit-class comparison, not bit-identical quantization.

### Server-mode benchmarking

For fair warm-server-vs-warm-server comparison (no CLI process startup overhead):

```bash
# Start KrillLM server
krillm serve --model llama-3.2-1b --port 11435

# In another terminal
make bench-compare KRILLM_URL=http://127.0.0.1:11435
```

### Release benchmark gate

Evaluate benchmark reports against release thresholds (1.5x decode, 0.67x wall time):

```bash
# Run against existing benchmark report (strict profile, default)
make bench-release-gate

# With custom report
make bench-release-gate GATE_INPUT=.build/benchmarks/krillm-vs-ollama.json

# Sequential comparison (disk-constrained)
make bench-release-gate GATE_KRILLM=krillm.json GATE_OLLAMA=ollama.json

# release_candidate profile — hard-gates user-visible latency and class-equal
# memory, marks prefill TPS advisory, scopes audio out until native Swift audio
python3 tools/release_gate.py .build/benchmarks/v6-mm.json \
  --profile release_candidate --allow-dtype-mismatch
```

The gate writes `.build/benchmarks/release-gate.json` with per-metric pass/fail, geometric mean speedup, worst metric, bottleneck classification, the active profile, and the KV cache dtype the run used. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the per-metric kind table and rationale.

### Performance claims

KrillLM beats Ollama decisively on user-visible latency for Gemma4 E2B on Apple Silicon — text TTFT ~5x, text wall-time ~1.57x, image wall-time ~1.77x faster on the v6 multimodal snapshot — and now also passes the hard class-equal peak-memory gate (PR #16 capped the MLX Metal buffer pool: KrillLM ~2.85–3.0 GB vs Ollama ~8.2–8.4 GB). It remains competitive but not 1.5x ahead on raw decode-token/s against llama.cpp's hand-tuned Metal kernels on this tiny 4-bit model. Ollama is currently stronger on multimodal prefill TPS and on audio (where KrillLM still routes through `mlx-vlm`). KrillLM's remaining milestones are a sustained `text_decode_ratio >= 1.5x` (Workstream 2 — Gemma 4 speculative decoding; re-promotes the metric to hard) and a fully native multimodal path with the `strict`-profile gate exiting `0`.

The `strict` benchmark gate (the uncompromised reference) still fails on text decode, prefill TPS, and audio wall time. The `release_candidate` profile keeps prefill TPS advisory, audio out_of_scope, and `text_decode_ratio` advisory-with-a-hard-≥1.0x-floor; on `.build/benchmarks/v6-mm.json` it **exits `0` (GATE: PASS)** — user-visible latency and class-equal memory hard-pass and KrillLM is never slower than Ollama at decode. This is a release-readiness baseline plus a documented follow-up roadmap, not a production release (which needs `strict`-green and native audio). Performance claims in this README are not updated unless `make bench-release-gate` passes under the agreed profile and the report is committed or linked; no claim states KrillLM decodes faster than Ollama.

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

# Ollama drop-in (opt-in): match Ollama's port + full /api surface
krillm serve --model llama-3.2-3b --port 11434 --compat both
# Default port stays 11435 until the mac_parity gate is green
# (deferred per docs/OLLAMA_MAC_PARITY_PLAN.md). Track with: make parity-gate

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
krillm pull gemma-4-e2b       # Gemma 4 E2B (text+image native, audio via mlx-vlm)
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

Gemma 4 supports text, image, and audio inputs. Text and image-only use the native Swift path (text model + SigLIP2 vision encoder). Audio requires the Python `mlx-vlm` bridge. When both `--image` and `--audio` are used together, the entire request routes through mlx-vlm because native audio is not implemented.

See the [Support Matrix](#support-matrix) above for the full CLI-native / CLI-bridge / Server breakdown. The HTTP server accepts image and audio payloads on Ollama and OpenAI endpoints when a Gemma 4 model is loaded; see [`docs/SERVER_API.md`](docs/SERVER_API.md) for request shapes.

### Image (native, no Python needed)

```bash
krillm run gemma-4-e2b "Describe this image" --image ./photo.png --max-tokens 64
```

### Audio (requires mlx-vlm)

```bash
# Install the bridge dependency
make setup-mlx-vlm

krillm run gemma-4-e2b "What sound is this?" --audio ./clip.wav --max-tokens 64
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
