<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/srvsngh99/Krill/main/assets/krill-lockup-paper.png">
    <img src="https://raw.githubusercontent.com/srvsngh99/Krill/main/assets/krill-lockup-ink.png" alt="Krill — a Sourav AI Labs project" width="340">
  </picture>
</p>

<p align="center">
  A Mac-native LLM inference CLI for Apple Silicon, built on Apple's <a href="https://github.com/ml-explore/mlx-swift">MLX</a>. Ships as a single CLI binary.
</p>

## Release Status

**Current release: v0.10.0 - GLM-4, faster decode, and a fully native quantizer** (`brew tap srvsngh99/krill && brew install krill`).

The headline of this release is the **full-screen chat TUI and on-device voice**:
`krill run` now opens an opencode-style alternate-screen chat interface (branded
masthead, scrollable conversation, slash-command autosuggest, streaming markdown,
resize-aware; `--classic` for the line REPL, auto-fallback when stdout is not a TTY),
with interactive image/audio/mic attach. Voice is fully on-device, both ways:
push-to-dictate via Apple Speech plus a native MLX Whisper runtime with automatic
language detection, and an opt-in `/speak` that reads replies back aloud
(text-to-speech) to complete the hands-free loop. It builds on **Gemma 4 12B
running natively** - an encoder-free
"unified" multimodal runtime (text + vision + audio on one dense backbone, no separate
vision encoder) in 4-bit-float nvfp4, near-flat long-context decode (~17-23 tok/s to
~99k on a 24GB Mac) - plus **one-command coding-agent launch**: `krill launch <agent>`
wires Claude Code, Codex, OpenCode, GitHub Copilot, Droid, Hermes, or Pi to your local
server with no manual config (see
[`docs/CONNECT_CODING_AGENTS.md`](docs/CONNECT_CODING_AGENTS.md)).

Everything runs on the native Swift+MLX engine with **no Python dependency**. The
server is a drop-in for both the OpenAI and Ollama HTTP APIs and listens on the
default port `57455`, so Krill coexists with Ollama on `11434` (run `--port
11434` for a literal drop-in replacement).

**Where Krill leads Ollama on Mac** (all reproducible with the bundled harness -
see Benchmarks below, and [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) /
[`docs/GEMMA4_12B_NVFP4.md`](docs/GEMMA4_12B_NVFP4.md)):

- **Capability**: native multimodal (vision + audio), native voice,
  grammar-constrained structured output, and `tool_choice` / schema-constrained
  tool calls - none of which Ollama's MLX Gemma tag does.
- **Concurrency**: ~2x aggregate throughput at load via the continuous batcher
  (Ollama serializes even on its MLX backend).
- **Cold start** and **agentic/RAG latency**: faster model load, and shared-prefix
  KV reuse that turns repeat-prefix requests from full re-prefill into a cache hit.
- **Quality at speed**: the shipped `gemma-4-12b` nvfp4 checkpoint (mixed-precision,
  attention `o_proj` at 8-bit) scores **77.6% MMLU-500 at ~27.7 tok/s** single
  stream - Ollama-parity quality at nvfp4 speed.

Single-stream text decode is **at parity** with Ollama's MLX backend, bounded by
the MLX memory-bandwidth roof on a 24GB box; raw-throughput chasing is an
explicitly closed lever (see [`docs/CEILINGS_AND_REATTEMPTS.md`](docs/CEILINGS_AND_REATTEMPTS.md)).
Krill does not claim a single-stream decode lead over Ollama-MLX.

### Support Matrix

The matrix separates CLI vs Server and per-modality support. Gemma 4 runs text,
image, and audio on the native Swift+MLX path (no Python bridge).

Gemma 4 (`gemma-4-12b` flagship, `gemma-4-e2b` / `gemma-4-e4b` smaller):

| Path | Text | Image | Audio |
|------|------|-------|-------|
| CLI    | Supported | Supported | Supported (native) [^native-audio] |
| Server | Supported | Supported | Supported (native) [^native-audio] |

Additional native vision runtimes (server image serving): LLaVA-1.5,
Llama-3.2-Vision (mllama, multi-image), and Qwen2.5-VL.

[^native-audio]: The native Swift+MLX Gemma 4 USM Conformer audio path (`Sources/KLMCore/AudioPreprocessor.swift` + `AudioEncoder.swift`) is the **only** audio path. It was numerically validated against the (now-removed) `mlx-vlm` oracle on a real Gemma 4 checkpoint on the M4 target (verbatim-equivalent transcription on a deterministic speech fixture) and benchmarked **faster than Ollama** (audio prefill ~2.4×, audio wall ~0.53×). The oracle outputs were pinned to `Tests/KLMEngineTests/Fixtures/ws6_oracle_baseline.json` before bridge removal so the parity contract stays testable without the Python dependency. See [`docs/NATIVE_GEMMA4_AUDIO_PLAN.md`](docs/NATIVE_GEMMA4_AUDIO_PLAN.md).

Text generation also runs natively for Llama 3.x, Qwen 2.5 / Qwen 3 (incl. MoE),
Mistral, Gemma 2, Phi, GLM-4, Mixtral, OLMoE, and DeepSeek-V2/V3, plus a ~15-family
embedding/reranker stack.

## Benchmarks

Benchmark results depend on local hardware, OS, model quantization, installed binaries, and daemon state. Krill does not publish fixed Krill-vs-Ollama numbers in this README without an attached reproducibility report.

Use the local harness to compare an installed Krill model with an installed Ollama model:

```bash
make bench-compare \
  KRILL_MODEL=llama-3.2-1b \
  OLLAMA_MODEL=llama3.2:1b \
  BENCH_MAX_TOKENS=32 \
  BENCH_RUNS=5 \
  BENCH_WARMUP=2
```

The harness writes `.build/benchmarks/krill-vs-ollama.json` by default and records:

- Krill and Ollama model names
- Prompt text and SHA256
- requested max tokens, actual prompt/generated token counts, runs, warmups, seed, temperature, and top-p
- per-run throughput/timing plus median/min/max summaries
- host environment, Swift version, Krill version, Ollama version, git commit, and git status

Prerequisites:

- Build Krill first with `make release`, or pass `--krill-bin` when running the Python harness directly.
- Install the Krill model with `krill pull <model>`.
- Install Ollama, start its daemon with `ollama serve`, and install the comparison model with `ollama pull <model>`.

The harness exits `77` and writes an actionable skip report when `ollama`, the Ollama daemon, or either model is missing. `krill bench <model>` remains available for the native synthetic-token benchmark.

For Gemma 4 text/image/audio comparison (all native — no Python bridge), run:

```bash
make bench-gemma4-multimodal
```

This writes `.build/benchmarks/gemma4-e2b-multimodal-4bit.json`. The harness benchmarks text, image, and audio separately and records the exact quantization metadata. Krill's local Gemma 4 E2B checkpoint uses MLX affine 4-bit; Ollama `gemma4:e2b` reports `Q4_K_M`, so the default report labels this as a 4-bit-class comparison, not bit-identical quantization.

### Server-mode benchmarking

For fair warm-server-vs-warm-server comparison (no CLI process startup overhead):

```bash
# Start Krill server
krill serve --model llama-3.2-1b --port 57455

# In another terminal
make bench-compare KRILL_URL=http://127.0.0.1:57455
```

### Release benchmark gate

Evaluate benchmark reports against release thresholds (1.5x decode, 0.67x wall time):

```bash
# Run against existing benchmark report (strict profile, default)
make bench-release-gate

# With custom report
make bench-release-gate GATE_INPUT=.build/benchmarks/krill-vs-ollama.json

# Sequential comparison (disk-constrained)
make bench-release-gate GATE_KRILL=krill.json GATE_OLLAMA=ollama.json

# release_candidate profile — hard-gates user-visible latency, class-equal
# memory, and native audio (WS6); marks prefill TPS advisory
python3 tools/release_gate.py .build/benchmarks/v6-mm.json \
  --profile release_candidate --allow-dtype-mismatch
```

The gate writes `.build/benchmarks/release-gate.json` with per-metric pass/fail, geometric mean speedup, worst metric, bottleneck classification, the active profile, and the KV cache dtype the run used. See [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md) for the per-metric kind table and rationale.

### Performance claims

Krill beats Ollama decisively on user-visible latency for Gemma4 E2B on
Apple Silicon in the accepted `v6-mm` multimodal report: text TTFT ~5x,
text wall-time ~1.57x faster, and native vision/image wall-time ~1.77x
faster. The same report passes the hard class-equal peak-memory gate
(Krill ~2.85-3.0 GB for text/image vs Ollama ~8.8 GB; PR #16 capped
the MLX Metal buffer pool). As of WS6, **native Swift+MLX audio is
default-on and part of the gated metrics**: on the M4 target it is
numerically validated against the `mlx-vlm` oracle (verbatim-equivalent
transcription on a deterministic speech fixture) and benchmarks **faster
than Ollama** (audio prefill ~2.4×, audio wall ~0.53×), so `audio_*` is
promoted from `out_of_scope` to **hard** in both profiles and the
`release_candidate` gate now enforces — and passes — it.

The `strict` benchmark gate (the uncompromised reference) still fails,
but only on the **pre-existing, owner-accepted** image-prefill item.
Audio now **passes strict**. Both gate profiles treat `text_decode_ratio`
as advisory (the ≥1.5× target is structurally unreachable on M-series;
owner-accepted for `strict` 2026-05-22,
`docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md`) with a hard
`text_decode_ratio_floor >= 1.0x`. The `release_candidate` profile keeps
prefill TPS advisory and, on the native-audio multimodal report,
**exits `0` (GATE: PASS)** with audio enforced as hard. No claim states
Krill decodes 1.5x faster than Ollama; the single-stream decode posture
is **parity** with Ollama's MLX backend by the bandwidth roof, and the
release leans on capability, concurrency, cold start, and agentic/RAG
latency (see Release Status above and `docs/BENCHMARKS.md`).

## Install

```bash
# From source (requires Xcode + Metal Toolchain)
git clone https://github.com/srvsngh99/Krill.git
cd Krill
make release
make install  # installs to /usr/local/bin/krill

# Or just build and run from repo
swift build -c release
.build/release/krill version
```

### Prerequisites

- macOS 14+ on Apple Silicon (M1 or newer)
- Xcode with Metal Toolchain: `xcodebuild -downloadComponent MetalToolchain`
- Swift 6.2+

## Quick Start

```bash
# Pull a model
krill pull llama-3.2-3b

# Chat
krill run llama-3.2-3b "What is the meaning of life?"

# Interactive REPL
krill run llama-3.2-3b

# Start API server (OpenAI + Ollama compatible)
krill serve --model llama-3.2-3b
# Then: curl http://localhost:57455/v1/chat/completions ...
# Default port is 57455 (unique; coexists with Ollama on 11434). For a
# drop-in Ollama replacement, run with --port 11434.

# Boot a coding agent wired to your local Krill server (no manual config)
krill launch              # list agents
krill launch claude       # also: codex, opencode, hermes, pi, copilot, droid

# Native synthetic-token benchmark
krill bench llama-3.2-3b
```

## Available Models

```bash
krill pull llama-3.2-1b      # Llama 3.2 1B (fast, small)
krill pull llama-3.2-3b      # Llama 3.2 3B
krill pull llama-3.1-8b      # Llama 3.1 8B
krill pull qwen2.5-7b        # Qwen 2.5 7B
krill pull mistral-7b        # Mistral 7B v0.3
krill pull gemma-2-9b        # Gemma 2 9B
krill pull gemma-4-e2b       # Gemma 4 E2B (text+image+audio all native)
krill pull phi-4-mini         # Phi-4 Mini
```

Or pull any mlx-community model directly:
```bash
krill pull mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

## Commands

| Command | Description |
|---------|-------------|
| `krill run <model> [prompt]` | Chat (interactive or single-shot) |
| `krill pull <model>` | Download model from HuggingFace |
| `krill serve` | Start OpenAI/Ollama-compatible HTTP server |
| `krill launch <agent>` | Boot a coding agent (Claude Code, Codex, OpenCode, ...) wired to Krill |
| `krill bench <model>` | Run performance benchmarks |
| `krill list` | Show installed models |
| `krill rm <model>` | Remove a model |
| `krill quantize <hf-path>` | Convert HF model to MLX format |
| `krill version` | Print version and system info |

### Speed up CLI with a background daemon

`krill run "<prompt>"` reloads the model on every invocation. Run `krill serve` in the background once and subsequent `krill run` calls detect it (probes `/v1/status` on `$KRILL_PORT` or 57455), route through `/v1/chat/completions`, and skip the per-call model load entirely. TTFT drops from seconds to tens of milliseconds.

```bash
KRILL_KEEP_ALIVE=24h krill serve --model qwen2.5-3b &
krill run qwen2.5-3b "hi"   # auto-routed; prints "(via daemon @ :57455)"
```

Text-only single-shot requests are routed; `--image`, `--audio`, `--draft-model`, models with Modelfile overrides, and the interactive REPL still run in-process. Set `KRILL_NO_AUTO_DAEMON=1` to force in-process behavior.

## Gemma 4 Multimodal Support

Gemma 4 supports text, image, and audio inputs, **all on the native Swift+MLX path** (text model + SigLIP2 vision encoder + USM Conformer audio). The legacy `mlx-vlm` Python bridge was removed in WS6 Step 4; combined `--image` + `--audio` requests run natively as well.

See the [Support Matrix](#support-matrix) above for the full CLI / Server breakdown. The HTTP server accepts image and audio payloads on Ollama and OpenAI endpoints when a Gemma 4 model is loaded; see [`docs/SERVER_API.md`](docs/SERVER_API.md) for request shapes.

### Image (native, no Python needed)

```bash
krill run gemma-4-e2b "Describe this image" --image ./photo.png --max-tokens 64
```

### Audio (native, no Python needed)

```bash
krill run gemma-4-e2b "Transcribe this audio." --audio ./clip.wav --max-tokens 64
```

`--image` works for **any** vision-capable family (Gemma 4, Qwen2.5-VL, LLaVA,
mllama), not just Gemma 4 - the flag is gated on the loaded model's real
capability, and fails loudly on a text-only model rather than silently dropping
the image.

### Interactive chat (full-screen TUI)

`krill run <model>` with no prompt opens a full-screen chat in the Sourav AI
Labs identity: a branded masthead, a scrollable conversation pane, a bottom
input box, and a status footer. It is a multi-turn conversation that remembers
context. Type `/` and a **slash-command autosuggest popup** appears; cycle it
with Up/Down and run with Enter (or Tab to fill it and add arguments). Replies
stream with light markdown styling; PgUp/PgDn or the mouse wheel scroll the pane,
Ctrl-C cancels a reply, Ctrl-D quits. Resize-aware.

Shades **adapt to a light or dark terminal** automatically (override with
`--theme light|dark` or `KRILL_TUI_THEME`). Run `krill run <model> --classic`
for the lighter libedit line REPL instead (history, in-line editing, Tab
completion). Krill auto-uses the line REPL when output is not a TTY
(piped/redirected), and disables color under `NO_COLOR`.

See **[docs/TUI.md](docs/TUI.md)** for the full reference (themes, every command,
custom slash commands, voice).

Attach images and audio without leaving the session, three ways:

```text
> /image ~/Pictures/cat.png        # explicit command (/audio, /img too)
> /Users/me/My Photos/cat.png      # drag a file into the terminal (its path is pasted)
> what breed is @~/Pictures/cat.png?   # inline @path inside your message
```

Attachments apply to your **next** message, then clear. `/attach` lists them
with index, dimensions, and size; `/remove <n>` drops one; `/drop` drops all.
Images accumulate for multi-image models (mllama); single-image models use the
first. `--image` / `--audio` passed on the command line pre-attach to the first
turn.

Session commands: `/system <text>` sets the system prompt, `/model [name]` opens
the model picker or switches in place (keeping the conversation), `/history`
prints the turns so far, `/compact` summarizes and shrinks the conversation to
free context, `/save [file]` writes the transcript, `/clear` clears the chat, and
`/help` lists everything. **Custom slash commands**: drop a prompt template at
`~/.krill/commands/<name>.md` (with `$ARGUMENTS` / `$1`..`$9` placeholders) and
it becomes `/<name>` - see [docs/TUI.md](docs/TUI.md).

### Live microphone voice input

In interactive chat with an audio-capable Gemma 4 model, hold **Space** on an
empty composer to talk (push-to-talk), or use `/mic` to record until Enter. By
default the clip is sent as an audio turn the model answers; `/voice dictate`
switches to best-effort transcription into the composer for review. (Gemma 4's
audio model tends to *answer* speech rather than transcribe it, so dictation is a
stopgap until a dedicated speech-to-text model is wired in.) `/mic` attaches a
clip for an explicit send:

```text
> /mic
Recording... press Enter to stop.
> what did I just say?
```

macOS attributes microphone access to the running app, so `/mic` needs Krill
to run from a code-signed bundle that declares the mic-usage string. Build it
with:

```bash
make app-bundle                 # produces dist/krill.app (ad-hoc signed)
dist/krill.app/Contents/MacOS/krill run gemma-4-e2b
```

The first `/mic` triggers the system microphone permission prompt under
Krill's own identity. Run the bare `.build/release/krill` binary instead and
the prompt (and permission) attach to your terminal app.

## API Compatibility

Krill serves on port 57455 (configurable) with all of:

- **OpenAI API**: `POST /v1/chat/completions` (SSE streaming), `POST /v1/completions`, `POST /v1/responses` (Responses API, for Codex), `GET /v1/models`
- **Ollama API**: `POST /api/chat`, `POST /api/generate`, `GET /api/tags`
- **Anthropic API**: `POST /v1/messages` (Claude SDK drop-in)

Drop-in replacement -- just change the port in your client config.

**Coding agents:** `krill launch <agent>` boots Claude Code, Codex, OpenCode,
Hermes, Pi, Copilot CLI, or Droid pre-wired to your local server. See
[`docs/CONNECT_CODING_AGENTS.md`](docs/CONNECT_CODING_AGENTS.md).

### Use with existing SDKs

Start `krill serve --model <name>` once. Then point your SDK at the right base URL: OpenAI-family clients (openai, langchain-openai, llama-index) use `http://localhost:57455/v1`, since their SDKs append paths like `/chat/completions` directly. The Anthropic SDK is the exception -- it appends its own `/v1/messages`, so it takes `http://localhost:57455` without the trailing `/v1`. All four snippets below are verified end-to-end against the running daemon.

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:57455/v1", api_key="not-used")
resp = client.chat.completions.create(
    model="llama-3.2-1b",
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.choices[0].message.content)
```

**LangChain (`langchain-openai`):**

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:57455/v1",
    api_key="not-used",
    model="llama-3.2-1b",
)
print(llm.invoke("hi").content)
```

**LlamaIndex (`llama-index-llms-openai-like`):**

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="llama-3.2-1b",
    api_base="http://localhost:57455/v1",
    api_key="not-used",
    is_chat_model=True,
)
print(llm.complete("hi").text)
```

**Anthropic Python SDK** (Krill exposes `/v1/messages`):

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:57455", api_key="not-used")
resp = client.messages.create(
    model="llama-3.2-1b",
    max_tokens=64,
    messages=[{"role": "user", "content": "hi"}],
)
print(resp.content[0].text)
```

For coding agents (Aider, gptme, OpenHands) configure them with the OpenAI-compatible base URL above and any model name from `krill list`.

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
~/.krill/config.toml

# Environment variables
KRILL_DEFAULT_MODEL=llama-3.2-3b
KRILL_PORT=57455
KRILL_KV_CACHE_DTYPE=fp16
KRILL_PREFILL_CHUNK=2048   # query tokens per prefill forward (0 disables)
KRILL_ROTATING_KV=1        # windowed KV for Gemma sliding layers (0 disables)
```

`KRILL_ROTATING_KV` (default on) caps each Gemma 4 sliding-window layer's KV
cache at the layer's trained window instead of the full context, so long-context
decode reads O(window) KV on 40 of 48 layers instead of O(context). Numerically
identical to the full-cache + sliding-mask path (the window is what the model
attends either way); set `0` to fall back to full-history caches.

`KRILL_PREFILL_CHUNK` bounds how many prompt tokens are forwarded per prefill
pass. MLX's attention has no flash prefill kernel, so a single forward over a
very long prompt materializes a quadratic `[heads, L, L]` score matrix and OOMs
past ~21k tokens on a 24GB box. Chunking processes the prompt in query-blocks
(default 2048) while the KV cache accumulates exactly as one pass would, so long
contexts (32k+) run without crashing and shorter prompts stay untouched. Lower
it for even longer contexts; set `0` to disable.

## Why Krill?

1. **Mac-native backend** - Uses MLX on Apple Silicon.
2. **Normal VM memory** - Does not wire model pages through `mlock`.
3. **Prefix cache** - Reuses repeated prompt prefixes.
4. **Speculative decoding** - Supports draft-model verification.
5. **Single binary** - No required background daemon for local CLI inference.

## Author

**Sourav Singh** / [Sourav AI Labs](https://github.com/srvsngh99)

## License

MIT
