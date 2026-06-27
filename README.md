<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/srvsngh99/Krill/main/assets/krill-lockup-paper.png">
    <img src="https://raw.githubusercontent.com/srvsngh99/Krill/main/assets/krill-lockup-ink.png" alt="Krill — a Sourav AI Labs project" width="340">
  </picture>
</p>

<p align="center">
  A Mac-native LLM runtime that's also a coding agent.<br>
  One Swift + MLX binary — text, vision, and voice. No Python.
</p>

---

## What makes Krill different

Most local-LLM stacks are one half of a pair:

- an **engine** (Ollama, llama.cpp) that runs a model but can't *do* anything, and
- a **harness** (Claude Code, Codex) that drives tools and edits files but borrows someone else's model.

**Krill is both, in one binary.** The same native Swift + MLX engine that serves tokens also runs a full agent loop — tools, file edits, web fetch, permissions — against the model already sitting in your RAM. No second process, no Python bridge, nothing leaves the machine.

## One engine, three modes

| Mode | Command | What you get |
|------|---------|--------------|
| **Chat** | `krill run <model>` | Full-screen TUI (or one-shot) — multimodal, streaming, on-device voice |
| **Serve** | `krill serve` | Drop-in **OpenAI · Ollama · Anthropic** API on `:57455` |
| **Agent** | `krill code <task>` | Coding agent — bash, edits, glob/grep, **web**, **deep research** — on your local model |

The agent isn't boxed into your filesystem. **`web_search`** ranks the open web through a local-first SearXNG backend you point it at, and **`web_fetch`** reads any page as clean text — both SSRF-guarded and untrusted-framed against prompt injection — while **`/research <question>`** runs a multi-source deep-research pass (plan queries → fetch → summarize each source → synthesize a cited answer). A local model that browses, and cites its work.

And the inverse: `krill launch claude` (or `codex`, `opencode`, `copilot`, `droid`, `hermes`, `pi`) points an **external** harness at Krill's engine. Krill is the model *for* other agents, or the agent *on* its own model.

Underneath: a continuous batcher (~2× throughput under load), shared-prefix KV reuse (repeat prompts hit cache instead of re-prefilling — the agentic/RAG fast path), speculative decoding, and native vision (SigLIP2) + audio (USM Conformer). All Swift + MLX.

> **⚠️ Early release.** Krill is young and still getting its polish — expect some rough edges, and pin a version if you need stability. Bug reports, ideas, and feedback are genuinely welcome → [open an issue](https://github.com/srvsngh99/Krill/issues).

## Install

```sh
# Homebrew
brew tap srvsngh99/krill && brew install krill

# …or the one-line installer (Apple Silicon, no Homebrew needed)
curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh | sh
```

**Updating:** Homebrew installs update with `brew upgrade krill`; installer
builds update in place with `krill update` (add `--check` to only see if a
newer release is available).

<details>
<summary>Build from source</summary>

```bash
# Requires macOS 14+ (Apple Silicon, M1+), Swift 6.2+, and the Metal Toolchain
#   xcodebuild -downloadComponent MetalToolchain
git clone https://github.com/srvsngh99/Krill.git && cd Krill
make release && make install      # → /usr/local/bin/krill
```
</details>

## Quick start

```bash
krill pull gemma-4-e2b                       # text + image + audio, all native

krill run  gemma-4-e2b                        # chat: full-screen TUI
krill run  gemma-4-e2b "explain MLX in one line"   # chat: one-shot
krill code "add a docstring to the top fn in main.swift"   # agent: tools + edits
krill serve --model gemma-4-e2b               # API server on :57455

krill run gemma-4-e2b "what's here?" --image ./photo.png   # vision
krill run gemma-4-e2b "transcribe this"      --audio ./clip.wav   # audio
```

The default port `57455` is unique, so Krill coexists with Ollama on `11434`; run `krill serve --port 11434` for a literal drop-in.

## On Mac, vs Ollama

Single-stream decode is **at parity** — both hit the MLX memory-bandwidth roof, and Krill makes no raw-decode-speed claim. Krill leads where real workloads live:

- **Capability** — native vision + audio + voice, grammar-constrained output, schema / `tool_choice` tool calls. Ollama's MLX Gemma tag has none.
- **Concurrency** — ~2× aggregate throughput under load (continuous batcher vs serialized).
- **Latency** — faster cold start; shared-prefix KV turns repeat-prefix / agentic / RAG calls into cache hits. Gemma-4-E2B: TTFT ~5×, wall ~1.57× faster.
- **Memory** — ~3 GB vs ~8.8 GB peak (class-equal gate; capped Metal buffer pool).

Numbers track *your* hardware — reproduce them, don't trust a banner:

```bash
make bench-compare KRILL_MODEL=llama-3.2-1b OLLAMA_MODEL=llama3.2:1b
```

Full methodology and gates: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md), [`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Models

**36 chat & multimodal models** ship as one-word `krill pull` shortcuts (plus ~19 embedding / reranker models), spanning ~15 architecture families. Switch between installed models **live in a chat** with `/model` — the conversation carries over — or import any `mlx-community` repo and it joins the picker.

```bash
krill pull gemma-4-e2b       # Gemma 4 — text + image + audio, all native (also: -e4b, -12b flagship)
krill pull qwen3-14b         # Qwen 3 (incl. MoE: qwen3-30b) — also Qwen 2.5: qwen2.5-7b
krill pull llama-3.2-3b      # Llama 3.2 / 3.1 (also: llama-3.2-1b, llama-3.1-8b)
krill pull mistral-7b        # Mistral 7B v0.3
krill pull gemma-2-9b        # Gemma 2 9B
krill pull phi-4-mini        # Phi-4 Mini

krill pull mlx-community/Meta-Llama-3.1-8B-Instruct-4bit   # …or any mlx-community repo
```

Native text also runs Phi, GLM-4, Mixtral, OLMoE, and DeepSeek-V2/V3, plus a ~15-family embedding/reranker stack. Vision serving adds LLaVA-1.5, Llama-3.2-Vision (mllama, multi-image), and Qwen2.5-VL.

**Formats:** not anything-goes — Krill is MLX-native. It runs **MLX-format** checkpoints (safetensors) in 4-bit, 8-bit, `nvfp4` (mixed-precision 4-bit-float), or bf16/fp16 — **GGUF is not supported**. Any `mlx-community` model of a supported architecture loads as-is; convert other Hugging Face checkpoints with `krill quantize <hf-path>`.

## Commands

| Command | Description |
|---------|-------------|
| `krill run <model> [prompt]` | Chat — interactive TUI or one-shot (`/agent` toggles agent mode) |
| `krill code [task]` | Open the chat TUI in agent mode (tools, file edits, web) |
| `krill serve` | Start the OpenAI / Ollama / Anthropic HTTP server |
| `krill launch <agent>` | Wire an external coding agent (Claude Code, Codex, …) to Krill |
| `krill pull / list / rm <model>` | Manage models (download from HuggingFace) |
| `krill quantize <hf-path>` | Convert an HF model to MLX |
| `krill bench <model>` · `krill version` | Benchmark · version + system info |

**Faster CLI:** `krill run` reloads the model each call. Start a daemon once and subsequent calls auto-route to it — TTFT drops from seconds to milliseconds:

```bash
KRILL_KEEP_ALIVE=24h krill serve --model qwen2.5-3b &
krill run qwen2.5-3b "hi"     # auto-routed → "(via daemon @ :57455)"
```

## API compatibility

One server speaks three protocols — change the port in your client, nothing else:

- **OpenAI** — `/v1/chat/completions` (SSE), `/v1/completions`, `/v1/responses`, `/v1/models`
- **Ollama** — `/api/chat`, `/api/generate`, `/api/tags`
- **Anthropic** — `/v1/messages` (Claude SDK drop-in)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:57455/v1", api_key="not-used")
print(client.chat.completions.create(
    model="llama-3.2-1b", messages=[{"role": "user", "content": "hi"}]
).choices[0].message.content)
```

OpenAI-family SDKs (openai, langchain-openai, llama-index) use `…/v1`; the Anthropic SDK takes the bare host (it appends its own `/v1/messages`). Request shapes and more SDKs: [`docs/SERVER_API.md`](docs/SERVER_API.md), [`docs/CONNECT_CODING_AGENTS.md`](docs/CONNECT_CODING_AGENTS.md).

## Architecture

One Swift package, no Python. The engine and the agent harness share a process and the *same loaded model*.

**Engine**
- `KrillEngine` — orchestration, continuous batcher, speculative decoding
- `KrillCore` — architectures (Llama, Qwen/MoE, Mistral, Gemma 4, Phi, GLM, DeepSeek …) + SigLIP2 vision + USM Conformer audio
- `KrillCache` — KV cache (fp16 / int8, prefix reuse) · `KrillKernels` — fused Metal shaders · `KrillSampler` · `KrillGrammar` — grammar-constrained decode

**Harness**
- `KrillHarness` — agent loop, permissions, and tools (bash, read/write/edit, glob/grep, web fetch/search, dispatch, deep research)
- `KrillAgent` — hardware-aware operator / recommender

**Surfaces**
- `KrillServer` — OpenAI/Ollama/Anthropic HTTP (swift-nio) · `KrillCLI` + `KrillTUI` — CLI and the full-screen chat/agent TUI · `KrillRegistry` — model store + HF puller · `KrillTokenizer`

The full-screen TUI (themes, slash commands, attachments, push-to-talk voice) has its own reference: [`docs/TUI.md`](docs/TUI.md).

## Configuration

`~/.krill/config.toml`, or environment variables:

| Var | Default | Purpose |
|-----|---------|---------|
| `KRILL_DEFAULT_MODEL` | — | Model used when none is named |
| `KRILL_PORT` | `57455` | Server port |
| `KRILL_KV_CACHE_DTYPE` | `fp16` | KV cache precision (`fp16` / `int8`) |
| `KRILL_PREFILL_CHUNK` | `2048` | Prompt tokens per prefill pass — lets 32k+ contexts run without OOM (`0` disables) |
| `KRILL_ROTATING_KV` | `1` | Windowed KV for Gemma sliding layers — O(window) long-context decode (`0` disables) |

## Author

**Sourav Singh** / [Sourav AI Labs](https://github.com/srvsngh99) · [souravailabs.ai](https://souravailabs.ai)

## License

MIT
