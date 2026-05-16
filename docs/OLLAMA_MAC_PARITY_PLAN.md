# KrillLM ↔ Ollama macOS Parity Plan

Local handoff for the next agent/session.

Last updated: 2026-05-17
Base branch: `main`
Base commit: `c17356d` (merged PR #16)
Machine target: Apple Silicon (M-series), macOS 14+

## 0. Status (2026-05-17)

**Phase 1 COMPLETE — wire compatibility + embeddings (branch
`feat/ollama-parity-phase1`).**

Shipped: `--compat ollama|openai|both`; `GET /api/version`, `GET /api/ps`,
`POST /api/show`; `POST /api/pull` (NDJSON), `DELETE /api/delete`,
`POST /api/copy`, `HEAD|POST /api/blobs/:digest`; `GET /v1/models/{id}`;
**WS-B embeddings — dedicated BERT/MiniLM/BGE encoder (`EmbeddingModel.swift`
+ `EmbeddingEngine.swift`), `POST /api/embed`, `POST /api/embeddings`,
`POST /v1/embeddings`, `bert` family + embed aliases**;
`tools/parity_gate.py` + `make parity-gate`. Default port stays `11435`
(T0-1 deferral intact).

`make parity-gate` verdict: **9/10 hard checks PASS** (embeddings verified
live: `all-minilm` 384-d, L2-normalized, semantically correct —
cos(dog,puppy)=0.72 vs cos(dog,stocks)=0.04). The single remaining `H`
row is **T0-4 tools/function calling (WS-D D1, Phase 2)**. `mac_parity`
is correctly NOT yet green — one blocker left.
Next: WS-D D1 tool/function calling.

## 1. Goal

Make KrillLM a **drop-in replacement for Ollama on macOS / Apple Silicon**:
any tool, GUI, SDK, or agent that today points at a local Ollama server
should work unchanged when pointed at `krillm serve`, while KrillLM keeps its
existing wins (native MLX/Metal, lower resident memory, faster TTFT,
persistent prefix cache, speculative decoding).

"Parity" here is scoped to **macOS feature/configuration surface**, not
Ollama's Linux/Windows/multi-GPU/cloud surface. Cloud catalog, `ollama.com`
push/registry, Vulkan, and multi-GPU scheduling are explicit non-goals
(see §7).

This plan is the companion to
[`OLLAMA_SPEEDUP_EXECUTION_PLAN.md`](../OLLAMA_SPEEDUP_EXECUTION_PLAN.md):
that plan tracks *speed* parity gates; this plan tracks *feature &
configuration* parity. They share the same release-gate philosophy
(hard / advisory / out-of-scope).

## 2. Current Position (parity baseline, 2026-05-16)

KrillLM today exposes `/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/v1/models/load`, `/v1/models/unload`, `/v1/status`,
`/api/chat`, `/api/generate`, `/api/tags`, `/healthz`, `/metrics`; CLI
`run` / `pull` / `list` / `rm` / `serve` / `bench` / `quantize` / `version`;
config via `~/.krillm/config.toml` + `KRILL_*` env vars; MLX-safetensors
models pulled from HuggingFace.

Where KrillLM already **leads** Ollama (do not regress these):

- Persistent on-disk prefix cache (Ollama does not expose this).
- Native speculative decoding (adaptive K 2–6).
- int8 KV cache composable with prefix cache (PR #11).
- Single binary, no background daemon required.
- ~3× lower resident memory and ~5× faster TTFT on Gemma 4 E2B.

## 3. Parity Gap Matrix

Tiers reflect "how badly this breaks a drop-in Ollama replacement on Mac."
`H` = hard parity gate, `A` = advisory, `OOS` = out of scope for parity.

### Tier 0 — Breaks drop-in compatibility immediately

| ID | Gap | Ollama | KrillLM today | Gate |
|----|-----|--------|---------------|------|
| T0-1 | Default port | `11434` | `11435` (flip deferred — see §4 WS-A1) | H |
| T0-2 | Embeddings | `/api/embed`, `/api/embeddings`, `/v1/embeddings` | none | H |
| T0-3 | Discovery endpoints | `/api/version`, `/api/ps`, `/api/show` | none | H |
| T0-4 | Tool / function calling | native + OpenAI/Anthropic compat | `--tools` rejected, parser stubbed | H |

### Tier 1 — Major feature gaps

| ID | Gap | Ollama | KrillLM today | Gate |
|----|-----|--------|---------------|------|
| T1-1 | Structured output | `format:"json"` / JSON-schema | none | H |
| T1-2 | Modelfile + `create` | full directive set + `ollama create` | global `config.toml` only | H |
| T1-3 | Context-length override | `num_ctx` per req + `OLLAMA_CONTEXT_LENGTH` | fixed per-model max | H |
| T1-4 | `keep_alive` + auto-unload | per-request + 5m default eviction | global `idle_timeout`, manual unload | H |
| T1-5 | Concurrency / queue | `OLLAMA_NUM_PARALLEL` / `MAX_LOADED_MODELS` / `MAX_QUEUE` | single model, single request, no queue | H |
| T1-6 | Thinking/reasoning | `think` / `reasoning_effort` | none | A |
| T1-7 | Multimodal breadth | vision across many families | Gemma 4 only; audio via Python bridge | A |
| T1-8 | GGUF / model library | GGUF+ST+LoRA import, curated registry | MLX-ST only, HF pull only | OOS |

### Tier 2 — CLI / API surface gaps

| ID | Gap | Ollama | KrillLM today | Gate |
|----|-----|--------|---------------|------|
| T2-1 | CLI: `show` | yes | none | H |
| T2-2 | CLI: `ps` | yes | none | H |
| T2-3 | CLI: `stop` | yes | none | H |
| T2-4 | CLI: `cp` | yes | none | H |
| T2-5 | CLI: `create` | yes | none | H (pairs T1-2) |
| T2-6 | CLI: `push` / `signin` | yes | none | OOS |
| T2-7 | HTTP: `/api/pull` `/api/delete` `/api/copy` `/api/create` `/api/blobs/*` | yes | none | H |
| T2-8 | OpenAI: `/v1/embeddings`, tools in chat, `GET /v1/models/{id}` | yes | partial | H |
| T2-9 | Anthropic `/v1/messages` | yes | none | A |
| T2-10 | Sampler params (`mirostat*`, `min_p`, `typical_p`, `tfs_z`, `repeat_last_n`, `presence/frequency_penalty`, `num_keep`, `penalize_newline`, `num_predict`) | yes | `temp`/`top_k`/`top_p`/`rep_pen`/`seed` only | H |

### Tier 3 — Config & Mac platform gaps

| ID | Gap | Ollama | KrillLM today | Gate |
|----|-----|--------|---------------|------|
| T3-1 | CORS origins | `OLLAMA_ORIGINS` | none | H |
| T3-2 | Flash Attention toggle | `OLLAMA_FLASH_ATTENTION` | standard MLX attention | A |
| T3-3 | Env surface | full `OLLAMA_*` set | ~9 `KRILL_*` | H (subset) |
| T3-4 | GUI app / menubar / auto-update / login-item / `launchctl setenv` | yes | CLI-only single binary | OOS |
| T3-5 | First-party SDKs / `ollama launch` integrations | yes | OpenAI SDK partial | A |

## 4. Workstreams

Each workstream lists scope, the touched modules, and acceptance criteria.
File paths are current as of `c17356d` — verify before editing.

### WS-A — Wire compatibility (Tier 0, unblocks everything)

**A1. Ollama-compat mode (port flip DEFERRED).**

**Owner decision (2026-05-16): the default port stays `11435` until full
Mac parity is reached.** Flipping the default to `11434` early would make
stock Ollama clients auto-discover KrillLM and then hit missing endpoints —
a half-working "Ollama impostor" is worse than a clean opt-in. So:

- *Now / Phase 1:* keep default `11435`. `krillm serve --port 11434` must
  work so early adopters can opt in and we can run the parity gate against
  `:11434`. Document in `README.md`/`docs/SERVER_API.md` that the default
  flip is intentionally deferred and tracked here (T0-1).
- *Final activation (Phase 4 / DoD):* once the `mac_parity` gate is green,
  flip the `serve` default to `11434` in one PR with a loud release note
  and a one-release deprecation path for `11435`. This is the single
  "drop-in is now real" switch.

Also add `--compat ollama|openai|both` (default `both`) now — this is
independent of the port and safe to ship in Phase 1.
Touch (Phase 1): `Sources/KLMCLI/ServeCommand.swift` (`--compat`, accept
`--port 11434`), `docs/SERVER_API.md`, `README.md` (deferral note).
Touch (final activation): `Sources/KLMRegistry/Config.swift`
(`server_port` default), `Sources/KLMCLI/ServeCommand.swift`,
release notes.

**A2. Discovery endpoints.**
Implement `GET /api/version` (return KrillLM version + a spoofable
`ollama_compat_version` so version-gated clients proceed), `GET /api/ps`
(loaded model, size, context, `expires_at`/`UNTIL`, processor=GPU),
`POST /api/show` (`modelfile`, `parameters`, `template`, `details`,
`model_info`, `capabilities`). These are read-mostly and unblock the
majority of Ollama GUIs/integrations.
Touch: `Sources/KLMServer/Server.swift`, new
`Sources/KLMServer/OllamaCompat.swift`, `Sources/KLMRegistry/Registry.swift`
(model metadata for `show`).

**A3. Model lifecycle HTTP.**
`POST /api/pull` (stream NDJSON progress), `DELETE /api/delete`,
`POST /api/copy`, `HEAD|POST /api/blobs/:digest`. `/api/create` is paired
with WS-C (Modelfile).
Touch: `Sources/KLMServer/Server.swift`, `Sources/KLMRegistry/Puller.swift`
(emit progress events).

**Acceptance (WS-A):** a stock Ollama GUI (e.g. an Ollama-targeting chat
client) connects to `krillm serve` with no config change, lists models,
shows model info, pulls a model with a progress bar, and chats.

### WS-B — Embeddings (Tier 0)

`POST /api/embed` (batch `input`, `truncate`, L2-normalized
`embeddings[][]`), `POST /api/embeddings` (legacy single `prompt` →
`embedding[]`), `POST /v1/embeddings` (OpenAI shape, `encoding_format`,
`dimensions`). Requires an embedding forward path: either a dedicated
embedding model family loader or mean/last-token pooling over an existing
decoder's hidden states. Decide model support list (start: a small
sentence-embedding MLX model from mlx-community).
Touch: new `Sources/KLMEngine/EmbeddingEngine.swift`,
`Sources/KLMCore/ModelLoader.swift` (pooling head),
`Sources/KLMServer/Server.swift`, `Sources/KLMRegistry/AliasMap.swift`
(embedding aliases).
**Acceptance:** `curl /api/embed` and the OpenAI Python SDK
`client.embeddings.create(...)` return correctly-shaped, L2-normalized
vectors; a RAG client (LangChain/LlamaIndex Ollama embedding provider)
indexes and queries successfully.

### WS-C — Modelfile & model customization (Tier 1/2)

Define a KrillLM Modelfile (accept Ollama's syntax verbatim where feasible:
`FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `MESSAGE`, `LICENSE`;
`ADAPTER` for LoRA is OOS for v1, parse-and-warn). Implement:

- `krillm create <name> -f <Modelfile>` + `POST /api/create`.
- `krillm show <name>` (+ `--modelfile/--parameters/--template/--system`)
  and `POST /api/show` from WS-A2 share one metadata serializer.
- `krillm cp <src> <dst>` + `POST /api/copy`.
- Persist custom models as manifests referencing base blobs (no weight
  copy) plus an overrides blob (system/template/params).

Touch: new `Sources/KLMRegistry/Modelfile.swift`,
`Sources/KLMRegistry/ModelManifest.swift` (overrides field),
`Sources/KLMCLI/{CreateCommand,ShowCommand,CpCommand}.swift`,
`Sources/KLMTokenizer/TokenizerWrapper.swift` (template override resolution).
**Acceptance:** a Modelfile that sets `SYSTEM` + `PARAMETER temperature` +
`TEMPLATE` round-trips through `create` → `show` → `run`/`/api/chat` with
the overrides applied; `ollama show`-shaped JSON validates against clients.

### WS-D — Generation parity: tools, JSON, sampling, context (Tier 0/1/2)

**D1. Tool/function calling.** Implement `tools[]` + `tool_calls` +
`role:"tool"` on `/api/chat`, `/v1/chat/completions`, and (WS-F)
`/v1/messages`. Replace the stub in `Sources/KLMCore/ToolParser.swift`
with model-family-aware tool-call extraction (chat-template
`tools` injection + structured parse of the model's tool-call syntax).
Streaming tool-call deltas required for agent clients.

**D2. Structured output.** `format:"json"` (constrained/guided JSON) and
`format:<JSON schema>` on `/api/generate` & `/api/chat`; map OpenAI
`response_format` → same path. Implement via grammar/logit-mask sampling in
`Sources/KLMSampler/Sampler.swift` (new constrained-decoding module).

**D3. Sampler params (T2-10).** Add `mirostat`, `mirostat_tau`,
`mirostat_eta`, `min_p`, `typical_p`, `tfs_z`, `repeat_last_n`,
`presence_penalty`, `frequency_penalty`, `penalize_newline`, `num_keep`,
and correct `num_predict` (-1 = infinite) to `SamplingParams` and the
request decoders.

**D4. Context override (T1-3).** Honor `num_ctx` per request and
`KRILL_CONTEXT_LENGTH` / `OLLAMA_CONTEXT_LENGTH`; clamp to model max with a
warning rather than fixed silent cap.

Touch: `Sources/KLMCore/ToolParser.swift`,
`Sources/KLMSampler/Sampler.swift` (+ new `ConstrainedSampler.swift`),
`Sources/KLMEngine/InferenceEngine.swift`,
`Sources/KLMServer/Server.swift`.
**Acceptance:** an agent loop (e.g. an OpenAI-SDK function-calling sample)
completes a multi-turn tool call against `/v1/chat/completions`; a
JSON-schema request returns schema-valid output; `num_ctx` and the new
sampler params measurably change behavior in tests.

### WS-E — Serving: keep-alive, concurrency, queue (Tier 1)

- Per-request `keep_alive` (duration string / int seconds / `0` / negative)
  overriding a `KRILL_KEEP_ALIVE` (default `5m`) with auto-eviction timer
  in serve mode; empty-prompt request preloads.
- `krillm stop <model>` + reuse `/v1/models/unload`; `/api/ps` reflects
  `expires_at`.
- `KRILL_NUM_PARALLEL` (per-model in-flight, default 1),
  `KRILL_MAX_LOADED_MODELS`, `KRILL_MAX_QUEUE` (503 when exceeded). Start
  with a request queue + serialized execution; true batching can be a
  follow-up (cross-link to speedup plan).

Touch: `Sources/KLMServer/Server.swift` (scheduler/queue),
new `Sources/KLMServer/ModelScheduler.swift`,
`Sources/KLMCLI/StopCommand.swift`, `Sources/KLMRegistry/Config.swift`.
**Acceptance:** model auto-unloads after `keep_alive`; `/api/ps` shows
countdown; N concurrent clients are queued not dropped (until
`MAX_QUEUE`), and a load test does not corrupt KV/prefix cache.

### WS-F — Anthropic compat + reasoning (Tier 1/2, advisory)

`POST /v1/messages` (system, multi-turn, base64 vision, tools,
`tool_result`, streaming, `thinking`) so Claude Code / Anthropic-SDK
clients work via `ANTHROPIC_BASE_URL`. Add `think` / `reasoning_effort`
plumbing for reasoning-capable models, returning `message.thinking`.
Touch: new `Sources/KLMServer/AnthropicCompat.swift`,
`Sources/KLMEngine/InferenceEngine.swift` (thinking segmentation).
**Acceptance:** Claude Code configured with
`ANTHROPIC_BASE_URL=http://localhost:11434` completes a tool-using session.

### WS-G — Config & Mac platform (Tier 3)

- `KRILL_ORIGINS` CORS allowlist (mirror `OLLAMA_ORIGINS` semantics,
  default localhost) — required for browser-extension clients.
- Accept `OLLAMA_*` env aliases (`OLLAMA_HOST`, `OLLAMA_MODELS`,
  `OLLAMA_KEEP_ALIVE`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_NUM_PARALLEL`,
  `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_MAX_QUEUE`, `OLLAMA_FLASH_ATTENTION`,
  `OLLAMA_KV_CACHE_TYPE`) as fallbacks for the `KRILL_*` equivalents so
  existing user environments and `launchctl setenv` setups work unchanged.
- Flash Attention path in MLX (advisory; cross-links speedup plan).

Touch: `Sources/KLMRegistry/Config.swift` (env alias table),
`Sources/KLMServer/Server.swift` (CORS), `docs/SERVER_API.md`.
**Acceptance:** an `OLLAMA_HOST`/`OLLAMA_MODELS` environment drives KrillLM
identically; a browser-extension client passes CORS preflight.

## 5. Phased Delivery

Each phase = one or more PRs, gated and documented like the speedup plan.

- **Phase 1 — "It connects" (WS-A, WS-B).** `--compat` flag, discovery +
  lifecycle endpoints, embeddings. Default port stays `11435`;
  `--port 11434` works for opt-in + gate runs. Outcome: an opted-in Ollama
  client (pointed at `:11434`) connects, lists, pulls, chats, embeds
  unchanged. Highest ROI.
- **Phase 2 — "It's useful" (WS-D D1/D2/D4, WS-C).** Tool calling, JSON/schema
  output, `num_ctx`, Modelfile + `create`/`show`/`cp`. Outcome: agentic and
  RAG clients work; custom models persist.
- **Phase 3 — "It scales & matches knobs" (WS-E, WS-D D3, WS-G).** Keep-alive
  + auto-evict + concurrency/queue, full sampler params, CORS + `OLLAMA_*`
  env aliases.
- **Phase 4 — "Ecosystem polish + activation" (WS-F, WS-D streaming tool
  deltas, advisory items).** Anthropic `/v1/messages`, reasoning/thinking,
  Flash Attention. **Final step, gated on `mac_parity` green: flip the
  default `serve` port `11435 → 11434`** (T0-1 / WS-A1) — the single switch
  that makes the drop-in real, with a loud release note + `11435`
  deprecation.

## 6. Parity Gate

Add `tools/parity_gate.py` (sibling to `tools/release_gate.py`) plus a
`make parity-gate` target that runs a fixture suite of real Ollama-client
request shapes against `krillm serve` and asserts response-shape parity.
Profiles mirror the speedup gate:

- `mac_parity` (defensible release profile): all `H` rows in §3 pass;
  `A` rows may be skipped with a logged advisory; `OOS` excluded.
- `strict_parity`: every `H` and `A` row passes, no advisory skips.

Track status in [`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md)
alongside the speed gates. A production tag requires *both* the speedup
`release_candidate` gate and the `mac_parity` gate green.

## 7. Non-Goals (explicit OOS)

- `ollama.com` registry, `push`, `signin/signout`, cloud models, web search.
- GGUF / llama.cpp backend; KrillLM stays MLX-safetensors (T1-8). We may add
  GGUF *import-by-conversion* later, tracked separately.
- Electron GUI app, menubar, auto-updater, login-item (T3-4). KrillLM stays
  a CLI/server single binary; a thin menubar wrapper is a separate product
  decision, not parity.
- Linux/Windows, Vulkan, multi-GPU scheduling (`OLLAMA_SCHED_SPREAD`,
  `OLLAMA_GPU_OVERHEAD`).
- LoRA `ADAPTER` in Modelfile v1 (parse-and-warn only).

## 8. Risks & Open Questions

- **Constrained JSON decoding in MLX** (D2) is the highest-uncertainty item;
  needs a grammar/logit-mask design that does not regress decode TPS or
  break the prefix/int8 cache contracts.
- **Tool-call extraction** is model-family specific; the stubbed
  `ToolParser.swift` must become family-aware without bloating per-family
  code (mirror the architecture-detection approach in `ModelLoader`).
- **Concurrency vs prefix/KV cache**: the persistent prefix cache and int8
  KV path assume single-flight today; WS-E must add isolation or a per-slot
  cache, coordinated with the speedup plan owner.
- **Port change is breaking** for current KrillLM users — *resolved by
  deferral*: default stays `11435` until the `mac_parity` gate is green,
  then flips to `11434` in Phase 4 with a release note + one-release
  `11435` deprecation (WS-A1).
- **Embedding model support scope** (WS-B): which MLX models, and whether to
  add a pooling head to decoder models, needs an owner decision.

## 9. Definition of Done (Mac Parity)

A user can: install the `krillm` binary, `krillm serve`, and point any
Ollama-targeting Mac tool (CLI, GUI, OpenAI SDK, Anthropic SDK / Claude
Code, LangChain/LlamaIndex Ollama providers) at it **without changing the
client's host, port, or request code**, getting correct chat, tool-calling,
JSON, embeddings, model management, and lifecycle behavior — with the
`mac_parity` gate green and KrillLM's performance/memory wins intact.
