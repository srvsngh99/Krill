# Krill User Guide

The complete, example-driven guide to using Krill — a Mac-native LLM runtime on
Apple's MLX framework. Run open models locally with Metal acceleration: chat, a
full-screen TUI with voice, agentic coding, native multimodal (image / audio /
OCR), and an Ollama / OpenAI / Anthropic–compatible server.

> New here? Skim **[Quick start](#quick-start)**, then jump to the feature you
> need from the index below.

---

## Index

1. [Install](#1-install)
2. [Quick start](#2-quick-start)
3. [Models: find, pull, switch](#3-models-find-pull-switch)
4. [Chat](#4-chat)
   - [Interactive TUI](#41-interactive-tui)
   - [Slash commands](#42-slash-commands)
   - [Single-shot](#43-single-shot)
5. [Multimodal: images, audio, OCR](#5-multimodal-images-audio-ocr)
   - [Vision (image input)](#51-vision-image-input)
   - [Document OCR (Unlimited-OCR)](#52-document-ocr-unlimited-ocr)
   - [Audio input & voice](#53-audio-input--voice)
6. [Agentic coding (`krill code`)](#6-agentic-coding-krill-code)
7. [Connect external coding agents (`krill launch`)](#7-connect-external-coding-agents-krill-launch)
8. [The HTTP server (`krill serve`)](#8-the-http-server-krill-serve)
   - [Endpoints](#81-endpoints)
   - [Examples](#82-examples)
9. [Structured output (JSON / regex / grammar)](#9-structured-output-json--regex--grammar)
10. [Embeddings & reranking](#10-embeddings--reranking)
11. [Web search & deep research](#11-web-search--deep-research)
12. [Model management (create, quantize, copy, remove)](#12-model-management)
13. [Configuration](#13-configuration)
14. [Performance: daemon, keep-alive, speculative decode](#14-performance)
15. [Troubleshooting & support](#15-troubleshooting--support)
16. [Reference docs](#16-reference-docs)

---

## 1. Install

**Homebrew** (recommended):
```bash
brew tap srvsngh99/krill
brew install krill
```

**One-line installer** (Apple Silicon, no Homebrew):
```sh
curl -fsSL https://raw.githubusercontent.com/srvsngh99/Krill/main/install.sh | sh
```
The installer verifies the downloaded archive against GitHub's published
SHA-256 digest before extraction. Set `KRILL_VERSION` to pin a release.

**From source** — requires macOS 14+ (Apple Silicon, M1+), Swift 6.2+, and the
Metal Toolchain (`xcodebuild -downloadComponent MetalToolchain`):
```bash
git clone https://github.com/srvsngh99/Krill.git && cd Krill
make release && make install      # → /usr/local/bin/krill
```

Update later with `krill update`; it fetches the installer from the exact target
release tag (`--check` to only check; Homebrew installs use `brew upgrade krill`).

---

## 2. Quick start

```bash
krill pull gemma-4-e2b               # download a small, fast multimodal model
krill run  gemma-4-e2b               # open the full-screen chat TUI
krill run  gemma-4-e2b "explain MLX in one line"   # single-shot

krill serve --model gemma-4-e2b      # start the HTTP API server
krill list                           # what's installed
```

Set a default so you can just type `krill`:
```bash
krill --config default_model=gemma-4-e2b
krill                                # opens chat with the default model
```

---

## 3. Models: find, pull, switch

Dozens of chat & multimodal models ship as one-word shortcuts; any
`mlx-community` repo of a supported architecture also loads as-is.

```bash
krill pull gemma-4-e2b       # Gemma 4 — text + image + audio (also -e4b, -12b)
krill pull qwen3-14b         # Qwen 3 (incl. MoE: qwen3-30b)
krill pull llama-3.2-3b      # Llama 3.2 / 3.1
krill pull mistral-7b        # Mistral 7B v0.3
krill pull unlimited-ocr     # native document OCR (see §5.2)

krill pull mlx-community/Meta-Llama-3.1-8B-Instruct-4bit   # any mlx-community repo
```

Browse and manage:
```bash
krill catalog            # built-in aliases + catalog models you can pull
krill list               # installed models (size, family, params, quant)
krill show <name>        # a model's metadata, template, system prompt
krill rm <name>          # remove
```

**Switch models live in a chat** with `/model` — the conversation carries over.

**Formats:** Krill is **MLX-native**. It runs MLX-format safetensors in 4-bit,
8-bit, `nvfp4` (mixed 4-bit-float), or bf16/fp16. **GGUF is not supported.**
Convert other Hugging Face checkpoints with [`krill quantize`](#12-model-management).

---

## 4. Chat

### 4.1 Interactive TUI

```bash
krill run <model>          # full-screen TUI (themes, slash commands, voice)
krill run <model> --classic   # plain line REPL instead of the TUI
krill run <model> --theme dark   # light | dark | auto (default)
```

Useful run flags: `--temp <0..>`, `--top-p <0..1>`, `--max-tokens <n>`,
`--seed <n>`, `--system "<prompt>"`. Press **`/help`** inside the TUI for keys.
Full TUI reference: [`docs/TUI.md`](TUI.md).

### 4.2 Slash commands

Type these inside the chat TUI:

| Command | What it does |
|---|---|
| `/help` | Keys + command list |
| `/model [name]` · `/model info [name]` | Switch model (picker if no name) · model deep-dive |
| `/system <text>` | Set the system prompt |
| `/clear` · `/reset` | Clear the conversation |
| `/compact` | Summarize history to free context |
| `/context` · `/status` | Context-window usage · session info |
| `/history` · `/save [file]` · `/copy` | History · write transcript · copy last reply |
| `/agent` | Toggle agent mode (tools + file edits) |
| `/bg <task>` · `/agents` · `/switch <n>` · `/main` | Background agents |
| `/research <question>` | Multi-source deep research with citations |
| `/image <path>` (`/img`) · `/audio <path>` · `/mic` | Attach image · audio · record |
| `/attach` · `/remove <n>` · `/drop` | List · drop one · drop all attachments |
| `/voice-mode <type\|dictate\|handsfree\|send>` · `/voice` · `/speak` | Voice posture · state · TTS |
| `/think` | Toggle the model's reasoning channel (also `Ctrl-T`) |
| `/cd <path>` · `/add-dir <path>` · `/diff` | Working dir · extra dir · git diff |
| `/config [key=value]` · `/init` | Show/set config · generate a `Krill.md` for the repo |
| `/quit` (`/exit`, `/q`) | Exit |

**Custom commands:** drop a markdown file at `~/.krill/commands/<name>.md` and it
becomes `/<name>`. Placeholders `$ARGUMENTS`, `$INPUT`, `$1`..`$9` are
substituted; optional `--- description: … ---` frontmatter shows in `/help`.

### 4.3 Single-shot

Pass a prompt to skip the REPL — great for scripts and pipes:
```bash
krill run qwen2.5-3b "summarize this in 5 bullets" --max-tokens 300
echo "translate to French: good morning" | krill run llama-3.2-3b
```

---

## 5. Multimodal: images, audio, OCR

### 5.1 Vision (image input)

Vision-capable models: **Gemma 4**, **Qwen2.5-VL**, **LLaVA-1.5**,
**Llama-3.2-Vision** (multi-image).

```bash
krill run gemma-4-e2b --image photo.jpg "what's in this image?"
```
In the TUI, attach with `/image <path>`, a dragged-in path, or `@path`, then ask
your question.

### 5.2 Document OCR (Unlimited-OCR)

`unlimited-ocr` is a dedicated, single-purpose **document/image OCR** model
(native DeepSeek-OCR runtime — no Python). It reads a page into **grounded
text**: each region with its bounding box and type. **Use the instruction
`document parsing.`** — that's the prompt it's trained on.

```bash
krill pull unlimited-ocr
krill run unlimited-ocr --image invoice.png "document parsing."
```
Output:
```
<|det|>title [48, 74, 402, 130]<|/det|>Invoice 2026
<|det|>text  [33, 229, 370, 290]<|/det|>Bill to: Acme Corporation
<|det|>text  [34, 526, 382, 588]<|/det|>Widget A 3 $12.00
```

Via the TUI: `krill run unlimited-ocr`, then `/image page.png`, then
`document parsing.`. Via the server, see [§8.2](#82-examples).

> It's OCR-only — not a chat or coding model. To use OCR inside a larger app or
> agent, run it behind [`krill serve`](#8-the-http-server-krill-serve) and have
> your other model call that endpoint to read a document, then act on the text.
> It serves the **base view** (full pages including wide layouts); very dense
> large scans (gundam tiling) are a follow-up.

### 5.3 Audio input & voice

Audio input (speech understanding) runs on **Gemma 4** (native USM):
```bash
krill run gemma-4-e2b --audio clip.wav "transcribe and summarize"   # wav/mp3/flac/ogg/m4a
```

In the TUI, **push-to-talk** voice is available on audio-capable models:

| Voice mode | Hold `Space` to… |
|---|---|
| `dictate` | transcribe into the composer, review, then Enter |
| `handsfree` | transcribe and auto-send |
| `send` | send the clip as an audio turn; the model answers the audio |

Cycle modes with `Ctrl-V` or `/voice-mode <mode>`. Choose the dictation engine
with `/voice engine apple|whisper` (Whisper SKUs download on first use, with
consent). Toggle read-aloud replies (TTS) with `/speak`.

> **Mic permission (macOS):** attributing mic access to Krill needs a
> code-signed bundle. Build one with `make app-bundle` and run
> `dist/krill.app/Contents/MacOS/krill run gemma-4-e2b`. See
> [`docs/TUI.md`](TUI.md#voice).

---

## 6. Agentic coding (`krill code`)

`krill code` opens the chat TUI in **agent mode**: the model can read/write/edit
files, run shell commands, search the web, and dispatch sub-agents to complete a
task.

```bash
krill code "add a --verbose flag to the CLI and update the README"
krill code --plan "investigate why the build is slow"     # read-only plan first
```

**Permission postures** (cycle with `Shift+Tab` in the TUI, or set with
`--permission-mode`):

| Posture | Behaviour |
|---|---|
| `plan` | Read-only: inspect files, propose a plan; no edits/commands. |
| `ask` | Confirm every file edit and shell command. |
| `accept-edits` | Auto-apply edits; still ask before commands. |
| `auto` | Run everything without asking. |

With no CLI override, `krill code` uses `default_agent_posture` (which defaults
to read-only `plan`). Unrestricted execution therefore requires an explicit
`--permission-mode auto` or a deliberate `default_agent_posture = "auto"`
configuration.

Key flags: `--max-iterations <n>`, `--no-bash`, `--allow-tool <name>` /
`--deny-tool <name>` (repeatable), `--system "<prompt>"`.

**Agent tools:** `read_file`, `list_dir`, `glob`, `grep`, `bash`, `write_file`,
`edit_file`, `multi_edit`, `web_search`, `web_fetch`, `dispatch_agent`.

You can also enter agent mode from a normal chat with `/agent`, or run a
background agent with `/bg <task>`.

---

## 7. Connect external coding agents (`krill launch`)

Wire a third-party coding agent to a local Krill model — it auto-starts the
server and points the agent at it.

```bash
krill launch claude          # Claude Code on a local model
krill launch codex --model qwen3-14b
krill launch opencode -- <extra args forwarded to the agent>
krill launch                 # no agent → lists what's available
```
Verified/tested agents: **Claude Code** (Anthropic `/v1/messages`), **Codex**
(OpenAI `/v1/responses`), **OpenCode**, **Hermes**, **Pi**, **Copilot CLI**,
**Droid**. Flags: `--model`, `--port`, `--host`, `--no-serve` (use an
already-running server), `--keep-alive <dur>`. Details:
[`docs/CONNECT_CODING_AGENTS.md`](CONNECT_CODING_AGENTS.md).

---

## 8. The HTTP server (`krill serve`)

One server speaks **OpenAI, Ollama, and Anthropic** protocols. Default port
**57455** ("KRILL" on a keypad); for a drop-in Ollama replacement use
`--port 11434`.

```bash
krill serve --model gemma-4-e2b                 # both compat surfaces
krill serve --model qwen2.5-3b --port 11434     # Ollama drop-in
KRILL_API_KEY='choose-a-secret' krill serve --compat openai --host 0.0.0.0
```

Loopback (`127.0.0.1`, `::1`, or `localhost`) needs no authentication by
default. A non-loopback bind is refused unless bearer authentication is enabled
with `KRILL_API_KEY`, `--api-key`, or `server_api_key`. To knowingly expose an
unauthenticated server, pass `--allow-remote-unauthenticated`. Prefer the
environment variable over `--api-key` so the secret does not enter shell
history or the process list. Authenticated clients send `Authorization: Bearer <key>`; CORS
preflight requests remain available without the header.

### 8.1 Endpoints

**OpenAI** (`/v1/*`): `POST /chat/completions` (SSE streaming), `POST /completions`,
`POST /responses`, `POST /embeddings`, `POST /rerank`, `GET /models`,
`GET /models/{id}`, `POST /models/load` · `POST /models/unload`, `GET /status`,
`GET /catalog`.

**Anthropic:** `POST /v1/messages` (Claude SDK drop-in).

**Ollama** (`/api/*`): `POST /chat`, `POST /generate`, `GET /tags`,
`GET /version`, `GET /ps`, `POST /show`, `POST /embed` · `POST /embeddings`,
`POST /pull`, `POST /create`, `DELETE /delete`, `POST /copy`.

**Krill / universal:** `POST /research` (deep research), `GET /healthz`,
`GET /metrics` (Prometheus).

Full request/response shapes: [`docs/SERVER_API.md`](SERVER_API.md).

### 8.2 Examples

**OpenAI SDK** (point `base_url` at `…/v1`):
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:57455/v1", api_key="not-used")
print(client.chat.completions.create(
    model="gemma-4-e2b", messages=[{"role": "user", "content": "hi"}]
).choices[0].message.content)
```

**Image via OpenAI content blocks** (data-URL only):
```bash
curl http://localhost:57455/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "gemma-4-e2b",
  "messages": [{"role":"user","content":[
    {"type":"text","text":"describe this"},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -i photo.png)"'"}}
  ]}]
}'
```

**OCR via the server** (OpenAI or Ollama shape):
```bash
# OpenAI
curl http://localhost:57455/v1/chat/completions -d '{"model":"unlimited-ocr",
  "messages":[{"role":"user","content":[
    {"type":"text","text":"document parsing."},
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'"$(base64 -i page.png)"'"}}]}]}'

# Ollama
curl http://localhost:57455/api/generate -d '{"model":"unlimited-ocr",
  "prompt":"document parsing.","images":["'"$(base64 -i page.png)"'"]}'
```

**Media limits:** 1 image + 1 audio clip per request, ≤25 MB per item, ≤10 MB
total body.

---

## 9. Structured output (JSON / regex / grammar)

Constrain generation token-by-token so the output is always valid. Works on both
compat surfaces.

**JSON** — OpenAI:
```jsonc
"response_format": {"type": "json_object"}
"response_format": {"type": "json_schema", "json_schema": {"schema": { /* … */ }}}
```
Ollama: `"format": "json"` or a JSON-schema object as `"format"`.

**Regex** (constrain to a pattern): OpenAI
`"response_format": {"type":"regex","regex":"\\d{4}-\\d{2}-\\d{2}"}`; Ollama
`"format": {"regex":"…"}`.

**Grammar (CFG / Lark)** for recursive structures: OpenAI
`"response_format": {"type":"lark","grammar":"start: item*\nitem: \"(\" item* \")\""}`;
Ollama `"format": {"lark":"…"}`. Reference: [`docs/SERVER_API.md`](SERVER_API.md).

`krill code` can grammar-constrain tool-call arguments to a schema
(`--constrain-args`, on by default) so small models emit valid calls.

---

## 10. Embeddings & reranking

Pull a dedicated encoder first (independent of any loaded chat model):
```bash
krill pull all-minilm        # also: bge-small-en, bge-base-en, E5 / BERT families
krill serve
```

**Embeddings:**
```bash
curl http://localhost:57455/v1/embeddings -d '{"model":"all-minilm","input":["hello","world"]}'   # OpenAI
curl http://localhost:57455/api/embed     -d '{"model":"all-minilm","input":["hello","world"]}'   # Ollama
```
Vectors are mean-pooled (override `KRILL_EMBED_POOLING=cls`) and L2-normalized.

**Reranking** (cross-encoder relevance scoring):
```bash
curl http://localhost:57455/v1/rerank -d '{"model":"<reranker>","query":"…","documents":["…","…"]}'
```

---

## 11. Web search & deep research

`web_search` (the agent tool) and `/research` work out of the box via a keyless
**DuckDuckGo** backend. For reliable, rate-limit-free results, add a free-tier
key:

```bash
krill --config search_backend=brave        # or: tavily
krill --config brave_api_key=YOUR_KEY      # or export KRILL_BRAVE_API_KEY
# self-hosted:
krill --config search_backend=searxng
krill --config searxng_url=http://localhost:8888
```

**Deep research** — plan → search → read sources → synthesize a cited answer:
```bash
# in the TUI:
/research what changed in the latest MLX release?
# via the server:
curl http://localhost:57455/research -d '{"question":"…"}'
```
Background and the public-vs-private backend split:
[`docs/decisions/0002-web-search-backends.md`](decisions/0002-web-search-backends.md).

---

## 12. Model management

```bash
krill pull <alias|org/repo> [--force]      # download (alias or any HF repo)
krill list                                 # installed
krill show <name> [--template|--system|--parameters|--modelfile]
krill cp <src> <dst>                       # copy (weights referenced, not duplicated)
krill rm <name>                            # remove
krill catalog [list]                       # browse pullable models
```

**Customize a model with a Modelfile** (Ollama-style):
```modelfile
FROM llama-3.2-1b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a terse, accurate assistant."
```
```bash
krill create my-assistant -f Modelfile
krill run my-assistant
```
Directives: `FROM`, `PARAMETER`, `SYSTEM`, `TEMPLATE`, `LICENSE`, `MESSAGE`.

**Quantize** any HF checkpoint to MLX natively (no Python / mlx-lm):
```bash
krill quantize mlx-community/GLM-4-9B-0414-bf16 --bits 4 --name glm4-9b
# mixed-precision nvfp4 (protect quality-critical modules at 8-bit):
krill quantize <src> --mode nvfp4 --protect o_proj --protect down_proj
```
For MoE / vision / Gemma checkpoints, pass `--reference <a-4bit-build>` so the
quantized module set matches that build exactly. See `krill quantize --help` for
all flags (`--group-size`, `--dtype`, `--protect-bits`, `--protect-vision`, …).

---

## 13. Configuration

Precedence: **CLI flags → `KRILL_*` env vars → `~/.krill/config.toml` →
defaults.** Set keys from the shell (`krill --config key=value`) or in-chat
(`/config key=value`); both persist to `config.toml`.

Common keys (each has a `KRILL_…` env equivalent):

| Key | Default | Purpose |
|---|---|---|
| `default_model` | — | Model when none is named |
| `default_mode` | `chat` | Launch surface: `chat` or `agent` |
| `default_agent_posture` | `plan` | `plan` \| `ask` \| `accept-edits` \| `auto` |
| `server_port` / `server_host` | `57455` / `127.0.0.1` | HTTP server bind |
| `server_api_key` | — | Bearer token for all HTTP routes (redacted in config output; prefer `KRILL_API_KEY`) |
| `keep_alive` | `5m` | Keep a model resident (`30m`, `0`, negative=pin) |
| `num_parallel` | `1` | Concurrent in-flight requests per model |
| `max_loaded_models` | `1` | Models resident at once |
| `kv_cache_dtype` | `fp16` | KV cache precision (`fp16` / `int8`) |
| `context_length` | model max | Prompt token limit |
| `thinking` | `true` | Reasoning channel on/off |
| `search_backend` | `auto` | `auto` \| `brave` \| `tavily` \| `searxng` |
| `brave_api_key` / `tavily_api_key` | — | BYOK search keys (redacted in output) |
| `voice_mode` / `speak_replies` | `off` / `false` | Voice posture · TTS |
| `models_dir` | `~/.krill/models` | Where models live |

Server knobs also read Ollama's env vars (`OLLAMA_HOST`, `OLLAMA_KEEP_ALIVE`,
`OLLAMA_NUM_PARALLEL`, `OLLAMA_MODELS`, `OLLAMA_ORIGINS`) for drop-in
compatibility. CORS: `KRILL_ORIGINS` (comma-separated allowlist). Server auth:
`KRILL_API_KEY`.

---

## 14. Performance

**Run via a daemon for instant TTFT.** `krill run` reloads the model each call;
start a server once and subsequent `krill run` calls auto-route to it
(milliseconds instead of seconds):
```bash
KRILL_KEEP_ALIVE=24h krill serve --model qwen2.5-3b &
krill run qwen2.5-3b "hi"      # "(via daemon @ :57455)"
```

**Speculative decoding** speeds up generation. N-gram (prompt-lookup) spec is on
by default and wins on repetitive workloads (RAG, code, structured output);
disable with `KRILL_NGRAM_SPEC=0`. For draft-model spec, pass
`--draft-model auto` (or an alias/path) to `run` / `serve`. See
[`docs/SPECULATIVE_DECODING.md`](SPECULATIVE_DECODING.md).

**Long context without OOM:** prompts are prefilled in chunks
(`KRILL_PREFILL_CHUNK`, default 2048); Gemma sliding layers use windowed KV
(`KRILL_ROTATING_KV`). `int8` KV cache (`kv_cache_dtype=int8`, Gemma 4) halves
cache memory.

---

## 15. Troubleshooting & support

- **`model 'X' is not loaded`** (server) — load it first (`POST /v1/models/load`
  or pass `--model X` to `serve`), or use the exact name the server reports
  (`GET /v1/models` / `GET /api/tags`). When you `serve` a local *path*, the
  model name is the directory name.
- **OCR returns nothing / stops immediately** — use the exact instruction
  `document parsing.`; other phrasings are less reliable for `unlimited-ocr`.
- **GGUF won't load** — Krill is MLX-only. Use an `mlx-community` build or
  `krill quantize` an HF checkpoint.
- **"Requires the Metal Toolchain"** (from source) —
  `xcodebuild -downloadComponent MetalToolchain`.
- **Mic does nothing** — build the signed app bundle (`make app-bundle`); see §5.3.
- **`krill debug <model>`** diagnoses model loading and inference.
- **Inspect a model:** `krill show <name>`; **server health:** `GET /healthz`,
  `GET /v1/status`, `GET /metrics`.

Issues & questions: <https://github.com/srvsngh99/Krill/issues>.

---

## 16. Reference docs

- [`docs/SERVER_API.md`](SERVER_API.md) — every endpoint, request/response shapes
- [`docs/TUI.md`](TUI.md) — full TUI: keys, slash commands, voice, background agents, custom commands
- [`docs/CONNECT_CODING_AGENTS.md`](CONNECT_CODING_AGENTS.md) — wiring Claude Code / Codex / OpenCode / …
- [`docs/ADDING_MODELS.md`](ADDING_MODELS.md) — add a model to the registry
- [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) — engine + harness internals
- [`docs/BENCHMARKS.md`](BENCHMARKS.md) — performance vs Ollama
- [`docs/decisions/`](decisions/) — architecture decision records
- [`RELEASES.md`](../RELEASES.md) · [`CHANGELOG.md`](../CHANGELOG.md) — what each version shipped
</content>
