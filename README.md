<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/srvsngh99/Krill/main/assets/krill-lockup-paper.png">
    <img src="https://raw.githubusercontent.com/srvsngh99/Krill/main/assets/krill-lockup-ink.png" alt="Krill — a Sourav AI Labs project" width="340">
  </picture>
</p>

<p align="center">
  A Mac-native LLM runtime that's also a coding agent.<br>
  Chat, serve compatible APIs, or run agent workflows against the same on-device model.
</p>

---

## What makes Krill different

Local inference runtimes and agent harnesses are usually separate: one runs and
serves the model; the other adds tools and actions around it.

**Krill brings both together in one binary.** In chat and serve modes, its engine
runs the model for interactive or API inference. In agent mode, the harness adds
tools, file edits, web access, and permissions while using the same resident model.
Inference stays on your Mac; web tools make outbound requests only when used.

## One engine, three modes

| Mode | Command | What you get |
|------|---------|--------------|
| **Chat** | `krill run <model>` | Full-screen TUI (or one-shot) — multimodal, streaming, on-device voice |
| **Serve** | `krill serve` | Compatible **OpenAI · Ollama · Anthropic** API on `:57455` |
| **Agent** | `krill code <task>` | Coding agent — bash, edits, glob/grep, **web**, **deep research** — on your local model |

The agent isn't boxed into your filesystem. **`web_search`** ranks the open web — keyless out of the box (DuckDuckGo), or point it at Brave/Tavily (free-tier API key) or your own SearXNG for reliable results — and **`web_fetch`** reads public HTTP(S) pages as text, with SSRF defenses and untrusted-content framing — while **`/research <question>`** runs a multi-source deep-research pass (plan queries → fetch → summarize each source → synthesize a cited answer). A local model that browses, and cites its work.

And the inverse: `krill launch claude` (or `codex`, `opencode`, `copilot`, `droid`, `hermes`, `pi`) points an **external** harness at Krill's engine. Krill is the model *for* other agents, or the agent *on* its own model.

Underneath: a continuous batcher for concurrent requests, shared-prefix KV reuse
(repeat prompts hit cache instead of re-prefilling — the agentic/RAG fast path),
speculative decoding, and native vision (SigLIP2) + audio (USM Conformer).

> **⚠️ Early release.** Krill is young and still getting its polish — expect some rough edges, and pin a version if you need stability. Bug reports, ideas, and feedback are genuinely welcome → [open an issue](https://github.com/srvsngh99/Krill/issues).

## Install

```sh
# Homebrew
brew tap srvsngh99/krill && brew install krill

# …or the one-line installer (Apple Silicon, no Homebrew needed)
curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh | sh
```

The installer verifies the release archive against the SHA-256 digest
published by GitHub before extracting it. Set `KRILL_VERSION` to pin a release.

**Updating:** Homebrew installs update with `brew upgrade krill`; installer
builds update in place with `krill update`, which fetches the installer from
the target release tag (add `--check` to only see if a newer release is
available).

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
krill pull unlimited-ocr && krill run unlimited-ocr --image page.png "document parsing."   # OCR
```

The default port `57455` is unique, so Krill coexists with Ollama on `11434`.
Clients that expect port `11434` can use `krill serve --port 11434` with Krill's
supported Ollama-compatible endpoints.

> 📖 **New to Krill?** The **[User Guide](docs/GUIDE.md)** is a single indexed,
> example-driven walkthrough of every feature — chat & TUI, multimodal (image /
> audio / OCR), agentic coding, the HTTP server, structured output, embeddings,
> web search, model management, and configuration.

## On Mac, alongside Ollama

Krill does not make a blanket "faster than Ollama" claim. Single-stream decode
approaches the same Apple Silicon memory-bandwidth ceiling. Its architectural
focus is on workloads where the surrounding runtime matters:

- **Runtime + harness** — serve a model, use the built-in agent, or point an
  external agent at the same inference engine.
- **Concurrent inference** — continuous batching serves multiple active requests.
- **Repeated context** — shared-prefix KV reuse avoids reprocessing the same
  system prompts, tool schemas, and retrieved context.
- **Long context** — chunked prefill and sliding-window-aware KV management keep
  large Gemma contexts within the machine's memory budget.
- **Capability** — text, vision, audio, grammar-constrained output, and
  schema-guided tool calling from one runtime.

Benchmark methodology and engineering notes: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md),
[`docs/BENCHMARKING.md`](docs/BENCHMARKING.md).

## Models

**36 chat & multimodal models** ship as one-word `krill pull` shortcuts (plus ~19 embedding / reranker models), spanning ~15 architecture families. Switch between installed models **live in a chat** with `/model` — the conversation carries over — or import any `mlx-community` repo and it joins the picker.

```bash
krill pull gemma-4-e2b       # Gemma 4 — text + image + audio, all native (also: -e4b, -12b flagship)
krill pull qwen3-14b         # Qwen 3 (incl. MoE: qwen3-30b) — also Qwen 2.5: qwen2.5-7b
krill pull llama-3.2-3b      # Llama 3.2 / 3.1 (also: llama-3.2-1b, llama-3.1-8b)
krill pull mistral-7b        # Mistral 7B v0.3
krill pull gemma-2-9b        # Gemma 2 9B
krill pull phi-4-mini        # Phi-4 Mini
krill pull unlimited-ocr     # native document/image OCR (DeepSeek-OCR)

krill pull mlx-community/Meta-Llama-3.1-8B-Instruct-4bit   # …or any mlx-community repo
```

Native text also runs Phi, GLM-4, Mixtral, OLMoE, and DeepSeek-V2/V3, plus a ~15-family embedding/reranker stack. Vision serving adds LLaVA-1.5, Llama-3.2-Vision (mllama, multi-image), and Qwen2.5-VL.

**Formats:** not anything-goes — Krill is MLX-native. It runs **MLX-format** checkpoints (safetensors) in 4-bit, 8-bit, `nvfp4` (mixed-precision 4-bit-float), or bf16/fp16 — **GGUF is not supported**. Any `mlx-community` model of a supported architecture loads as-is; convert other Hugging Face checkpoints with `krill quantize <hf-path>`.

## Commands

| Command | Description |
|---------|-------------|
| `krill run <model> [prompt]` | Chat — interactive TUI or one-shot (`/agent` toggles agent mode) |
| `krill code [task]` | Open the chat TUI in agent mode (tools, file edits, web) |
| `krill serve` | Start the OpenAI / Ollama / Anthropic-compatible HTTP server |
| `krill launch <agent>` | Wire an external coding agent (Claude Code, Codex, …) to Krill |
| `krill pull / list / rm <model>` | Manage models (download from HuggingFace) |
| `krill quantize <hf-path>` | Convert an HF model to MLX |
| `krill bench <model>` · `krill version` | Benchmark · version + system info |

**Daemon-backed CLI:** `krill run` reloads the model each call. Start a daemon once
and subsequent calls auto-route to it, avoiding a model reload on every invocation:

```bash
KRILL_KEEP_ALIVE=24h krill serve --model qwen2.5-3b &
krill run qwen2.5-3b "hi"     # auto-routed → "(via daemon @ :57455)"
```

## API compatibility

One server exposes three compatible protocol surfaces:

- **OpenAI** — `/v1/chat/completions` (SSE), `/v1/completions`, `/v1/responses`, `/v1/models`
- **Ollama** — `/api/chat`, `/api/generate`, `/api/tags`
- **Anthropic** — `/v1/messages` (Claude SDK-compatible)

```bash
curl http://localhost:57455/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama-3.2-1b","messages":[{"role":"user","content":"hi"}]}'
```

OpenAI-family SDKs (openai, langchain-openai, llama-index) use `…/v1`; the Anthropic SDK takes the bare host (it appends its own `/v1/messages`). Request shapes and more SDKs: [`docs/SERVER_API.md`](docs/SERVER_API.md), [`docs/CONNECT_CODING_AGENTS.md`](docs/CONNECT_CODING_AGENTS.md).

## Architecture

One Swift package. The engine and the agent harness share a process and the
*same loaded model*.

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
| `KRILL_API_KEY` | — | Require bearer authentication; needed for safe non-loopback serving |
| `KRILL_KV_CACHE_DTYPE` | `fp16` | KV cache precision (`fp16` / `int8`) |
| `KRILL_PREFILL_CHUNK` | `2048` | Prompt tokens per prefill pass — lets 32k+ contexts run without OOM (`0` disables) |
| `KRILL_ROTATING_KV` | `1` | Windowed KV for Gemma sliding layers — O(window) long-context decode (`0` disables) |
| `KRILL_SEARCH_BACKEND` | `auto` | Web-search backend: `auto` (keyless DuckDuckGo), `brave`, `tavily`, or `searxng` |
| `KRILL_BRAVE_API_KEY` | — | API key for `search_backend=brave` (free tier available) |
| `KRILL_TAVILY_API_KEY` | — | API key for `search_backend=tavily` (free tier available) |

### Web search

`web_search` works with no setup via DuckDuckGo. For reliable, rate-limit-free
results, add a free-tier API key:

```sh
krill --config search_backend=brave         # or: tavily
krill --config brave_api_key=YOUR_KEY        # or export KRILL_BRAVE_API_KEY
```

Or point at a self-hosted SearXNG with `search_backend=searxng` + `searxng_url`.
See [docs/decisions/0002-web-search-backends.md](docs/decisions/0002-web-search-backends.md).

## Author

**Sourav Singh** / [Sourav AI Labs](https://github.com/srvsngh99) · [souravailabs.ai](https://souravailabs.ai)

## License

MIT
