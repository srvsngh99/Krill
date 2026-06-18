# Krill ↔ Ollama macOS Parity Plan

Local handoff for the next agent/session.

Last updated: 2026-05-17
Base branch: `main`
Base commit: `c17356d` (merged PR #16)
Machine target: Apple Silicon (M-series), macOS 14+

## 0. Status (2026-05-17)

**Phase 1 + Phase 2 tool calling COMPLETE — `make parity-gate` is GREEN
(10/10) on both `mac_parity` and `strict_parity` (branch
`feat/ollama-parity-phase1`, PR #18).**

Shipped:
- WS-A wire compat: `--compat ollama|openai|both`; `GET /api/version`,
  `GET /api/ps`, `POST /api/show`; `POST /api/pull` (NDJSON),
  `DELETE /api/delete`, `POST /api/copy`, `HEAD|POST /api/blobs/:digest`;
  `GET /v1/models/{id}`.
- WS-B embeddings: dedicated BERT/MiniLM/BGE encoder
  (`EmbeddingModel.swift` + `EmbeddingEngine.swift`), `bert` family +
  aliases, `POST /api/embed`, `POST /api/embeddings`, `POST /v1/embeddings`.
  Verified live: `all-minilm` 384-d, L2-normalized, semantically correct
  (cos(dog,puppy)=0.72 vs cos(dog,stocks)=0.04).
- WS-D D1 tool/function calling: model-agnostic sentinel convention
  (`ToolCalling.swift`), `tools[]` + `tool_calls` + `role:"tool"` on
  `/v1/chat/completions` and `/api/chat`, robust extraction (missing
  close tag / backticks / fenced / bare JSON; balanced-brace scanner).
  Verified live on llama-3.2-1b: OpenAI `finish_reason:tool_calls` with
  string args; Ollama `done_reason:tool_calls` with object args;
  multi-turn tool-result round-trip.

Default port flipped to `11434` in 0.4.0 (T0-1 done; `mac_parity` gate
green 18/18 on 2026-05-28). `11435` honored for one release.

> **Update (post-0.4.0): the default port has since changed to 57455
> (unique, coexists with Ollama); `--port 11434` gives the Ollama
> drop-in. See CHANGELOG.**

Also shipped (2026-05-17, same branch):
- WS-G: CORS (`KRILL_ORIGINS`/`OLLAMA_ORIGINS`, OPTIONS preflight +
  `Access-Control-*` on JSON responses, origin allowlist) and the
  `OLLAMA_*` env-alias table (`OLLAMA_HOST` incl. host:port,
  `OLLAMA_MODELS`, `OLLAMA_CONTEXT_LENGTH`, `OLLAMA_KEEP_ALIVE`,
  `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_MAX_QUEUE`,
  `OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`) with `KRILL_*`
  winning. Closes **T3-1**, **T3-3**.
- WS-D D3 (partial): accept the full sampler-param surface; `min_p`
  implemented as a functional GPU logit filter; `num_predict:-1` =
  generate-until-EOS; `presence_penalty`/`frequency_penalty` accepted.
  Closes **T2-10** at the API+min_p level.

- WS-D D2 structured output (T1-1): Ollama `format:"json"` / JSON-schema
  and OpenAI `response_format` (`json_object` / `json_schema`) accepted
  on `/api/chat`, `/api/generate`, `/v1/chat/completions`. Guided-prompt
  injection + tolerant JSON extraction (`StructuredOutput.swift`); true
  grammar-constrained decoding is the tracked follow-up (plan §8).

- WS-C Modelfile (T1-2, T2-1/4/5): `Modelfile.swift` parser
  (FROM/PARAMETER/SYSTEM/TEMPLATE/LICENSE/MESSAGE; ADAPTER parse-warn;
  triple-quoted blocks). `krill create`/`show`/`cp` CLI;
  `POST /api/create` (NDJSON). `Registry.createModel` references base
  weights via symlink (no copy) + `ModelOverrides` on the manifest.
  `SYSTEM` override applied at serve time; `show`/`/api/show` reflect
  `system`/`parameters`/`template`/`license`.

- WS-E keep-alive (T1-4, T2-3): per-request `keep_alive` (duration
  string / int / `0` / negative), `KRILL_KEEP_ALIVE` default,
  background auto-unload evictor (`KeepAlive.swift` actor), `krill
  stop` CLI, `/api/ps` `expires_at` from the live deadline.

- WS-F Anthropic compat (T2-9, advisory): `POST /v1/messages`
  (`AnthropicCompat.swift`) — system string/blocks, `tool_use`/
  `tool_result` flattened onto the shared `<tool_call>` convention,
  `thinking` segmentation, non-streaming + a valid Anthropic SSE event
  sequence. Claude Code / Anthropic SDK via `ANTHROPIC_BASE_URL`.

- WS-D D4 context override (T1-3): `num_ctx` (Ollama options/top-level)
  and `KRILL_CONTEXT_LENGTH`/`OLLAMA_CONTEXT_LENGTH` default honored —
  the engine truncates the prompt to the most-recent N tokens with a
  stderr warning (no decode-loop/cache-contract change).

- WS-E concurrency queue (T1-5): `GenerationQueue` actor serializes
  inference (`KRILL_NUM_PARALLEL` slots, default 1) so concurrent
  clients are *queued, not dropped* and the single-flight engine +
  prefix/int8-KV caches are not entered in parallel; beyond
  `KRILL_MAX_QUEUE` → HTTP 503. The slot is held for the whole token
  loop and is wired into *every* engine-touching path - OpenAI
  stream/non-stream, Ollama `/api/chat` + `/api/generate`
  stream/non-stream, tool, Anthropic, embeddings - with the streaming
  head written only after the slot is acquired, and `defer`-released on
  every exit (no deadlock). Serialization correctness is covered by the
  `GenerationQueue` unit tests; a manual loaded-model run confirmed 4
  overlapping `/v1/chat/completions` are answered and serialized with
  none dropped. The `parity_gate.py` `T1-5` row only checks
  concurrent-request *endpoint stability* (it runs model-less, so it
  short-circuits before the queue) and is labelled as such - it is not
  presented as a serialization proof.

`make parity-gate` now **GREEN 18/18** on both profiles (incl. the
advisory `T2-9`/`T1-5` rows under `strict_parity`).

- WS-D D3 stateful penalties (T2-10, **applied**): `repetition_penalty`,
  `presence_penalty`, `frequency_penalty`, `repeat_last_n`, and
  `mirostat` v1/v2 (`mirostat_tau`/`eta`) are now applied in the decode
  loop via an O(window) scatter. **Zero-overhead on the default path** —
  `penaltiesActive` gates *all* history tracking + extra GPU work, so a
  request that sets no penalty is byte-for-byte the prior hot path and
  the release-critical speed/memory gate is provably unaffected.
  Speculative decoding transparently falls back to the standard loop
  when penalties are active. Live-verified: frequency+presence penalty
  cuts repetition ratio 0.889 → 0.093 on llama-3.2-1b.

- WS-C runtime `PARAMETER` application (**applied**): a created model's
  Modelfile `PARAMETER`s (temperature, top_p, top_k, repeat_penalty,
  min_p, presence/frequency_penalty, mirostat, repeat_last_n, num_ctx,
  num_predict, seed) are applied at serve time as *defaults* — an
  explicit client value still wins. Base weights are referenced via
  per-file symlinks (no copy) and load identically to the base.
  Live-verified: a created model with `PARAMETER seed`/`temperature`
  decodes deterministically and identically to its base.

**Scope honesty — remaining engine-internal follow-ups (the plan's own
"highest-uncertainty / delicate" items, §8):** WS-D D2
grammar-constrained decoding (guided-prompt + extraction today, not a
token-level logit-mask grammar — §8 explicitly flags this as the
highest-uncertainty item); WS-C `TEMPLATE` override at decode (Go-style
template; round-trips via `show`/`/api/show`, not yet re-rendered at
generation); WS-E *batched* multi-slot decode (the serialized queue +
`MAX_QUEUE`→503 + `NUM_PARALLEL` knob are implemented and gated; true
KV-batched concurrent decode and `MAX_LOADED_MODELS>1` multi-model
residency remain — the plan's WS-E acceptance explicitly permits
serialized-first with batching as the follow-up). Each is a depth
refinement of an already-working, gated feature, tracked for a pass
before the DoD `11435→11434` port flip — not a missing endpoint.
`mac_parity` GREEN means the gated drop-in essentials pass — not that
every plan row is done.

## 1. Goal

Make Krill a **drop-in replacement for Ollama on macOS / Apple Silicon**:
any tool, GUI, SDK, or agent that today points at a local Ollama server
should work unchanged when pointed at `krill serve`, while Krill keeps its
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

Krill today exposes `/v1/chat/completions`, `/v1/completions`,
`/v1/models`, `/v1/models/load`, `/v1/models/unload`, `/v1/status`,
`/api/chat`, `/api/generate`, `/api/tags`, `/healthz`, `/metrics`; CLI
`run` / `pull` / `list` / `rm` / `serve` / `bench` / `quantize` / `version`;
config via `~/.krill/config.toml` + `KRILL_*` env vars; MLX-safetensors
models pulled from HuggingFace.

Where Krill already **leads** Ollama (do not regress these):

- Persistent on-disk prefix cache (Ollama does not expose this).
- Native speculative decoding (adaptive K 2–6).
- int8 KV cache composable with prefix cache (PR #11).
- Single binary, no background daemon required.
- ~3× lower resident memory and ~5× faster TTFT on Gemma 4 E2B.

## 3. Parity Gap Matrix

Tiers reflect "how badly this breaks a drop-in Ollama replacement on Mac."
`H` = hard parity gate, `A` = advisory, `OOS` = out of scope for parity.

### Tier 0 — Breaks drop-in compatibility immediately

| ID | Gap | Ollama | Krill today | Gate |
|----|-----|--------|---------------|------|
| T0-1 | Default port | `11434` | `11434` (flipped in 0.4.0; `11435` honored one release) | H ✓ |
| T0-2 | Embeddings | `/api/embed`, `/api/embeddings`, `/v1/embeddings` | none | H |
| T0-3 | Discovery endpoints | `/api/version`, `/api/ps`, `/api/show` | none | H |
| T0-4 | Tool / function calling | native + OpenAI/Anthropic compat | `--tools` rejected, parser stubbed | H |

### Tier 1 — Major feature gaps

| ID | Gap | Ollama | Krill today | Gate |
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

| ID | Gap | Ollama | Krill today | Gate |
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

| ID | Gap | Ollama | Krill today | Gate |
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

**A1. Ollama-compat mode (port flip DONE in 0.4.0).**

**Owner decision (2026-05-16): the default port stays `11435` until full
Mac parity is reached.** Flipping the default to `11434` early would make
stock Ollama clients auto-discover Krill and then hit missing endpoints —
a half-working "Ollama impostor" is worse than a clean opt-in. So:

- *Now / Phase 1:* keep default `11435`. `krill serve --port 11434` must
  work so early adopters can opt in and we can run the parity gate against
  `:11434`. Document in `README.md`/`docs/SERVER_API.md` that the default
  flip is intentionally deferred and tracked here (T0-1).
- *Final activation (Phase 4 / DoD) — DONE in 0.4.0:* the `mac_parity`
  gate went green (18/18, 2026-05-28), so the `serve` default flipped to
  `11434` with a loud release note and a one-release deprecation path for
  `11435` (`--port 11435` / `KRILL_PORT=11435` still honored). This was the
  single "drop-in is now real" switch.

Also add `--compat ollama|openai|both` (default `both`) now — this is
independent of the port and safe to ship in Phase 1.
Touch (Phase 1): `Sources/KLMCLI/ServeCommand.swift` (`--compat`, accept
`--port 11434`), `docs/SERVER_API.md`, `README.md` (deferral note).
Touch (final activation): `Sources/KLMRegistry/Config.swift`
(`server_port` default), `Sources/KLMCLI/ServeCommand.swift`,
release notes.

**A2. Discovery endpoints.**
Implement `GET /api/version` (return Krill version + a spoofable
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
client) connects to `krill serve` with no config change, lists models,
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

Define a Krill Modelfile (accept Ollama's syntax verbatim where feasible:
`FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `MESSAGE`, `LICENSE`;
`ADAPTER` for LoRA is OOS for v1, parse-and-warn). Implement:

- `krill create <name> -f <Modelfile>` + `POST /api/create`.
- `krill show <name>` (+ `--modelfile/--parameters/--template/--system`)
  and `POST /api/show` from WS-A2 share one metadata serializer.
- `krill cp <src> <dst>` + `POST /api/copy`.
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
- `krill stop <model>` + reuse `/v1/models/unload`; `/api/ps` reflects
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
**Acceptance:** an `OLLAMA_HOST`/`OLLAMA_MODELS` environment drives Krill
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
request shapes against `krill serve` and asserts response-shape parity.
Profiles mirror the speedup gate:

- `mac_parity` (defensible release profile): all `H` rows in §3 pass;
  `A` rows may be skipped with a logged advisory; `OOS` excluded.
- `strict_parity`: every `H` and `A` row passes, no advisory skips.

Track status in [`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md)
alongside the speed gates. A production tag requires *both* the speedup
`release_candidate` gate and the `mac_parity` gate green.

## 7. Non-Goals (explicit OOS)

- `ollama.com` registry, `push`, `signin/signout`, cloud models, web search.
- GGUF / llama.cpp backend; Krill stays MLX-safetensors (T1-8). We may add
  GGUF *import-by-conversion* later, tracked separately.
- Electron GUI app, menubar, auto-updater, login-item (T3-4). Krill stays
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
- **Port change is breaking** for current Krill users — *resolved by
  deferral*: default stays `11435` until the `mac_parity` gate is green,
  then flips to `11434` in Phase 4 with a release note + one-release
  `11435` deprecation (WS-A1).
- **Embedding model support scope** (WS-B): which MLX models, and whether to
  add a pooling head to decoder models, needs an owner decision.

## 9. Definition of Done (Mac Parity)

A user can: install the `krill` binary, `krill serve`, and point any
Ollama-targeting Mac tool (CLI, GUI, OpenAI SDK, Anthropic SDK / Claude
Code, LangChain/LlamaIndex Ollama providers) at it **without changing the
client's host, port, or request code**, getting correct chat, tool-calling,
JSON, embeddings, model management, and lifecycle behavior — with the
`mac_parity` gate green and Krill's performance/memory wins intact.
