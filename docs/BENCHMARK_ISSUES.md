# Benchmark Issues — found 2026-06-04, to revisit/fix

Open gaps surfaced while running the fresh KrillLM-vs-Ollama benchmark suite
(`docs/BENCHMARKS.md`) on Apple M4 Pro, KrillLM `9a21941`, Ollama 0.24.0. None
are merge-blocking for what shipped; they are the punch-list for "best inference
engine on macOS." Ordered by impact on that goal.

---

## 1. [KrillLM bug] Server HTTP API does not ingest AUDIO (CLI does)

**Severity: high** (voice is a flagship differentiator; it works in the CLI but
not over the API that real clients use).

- `krillm run gemma-4-e2b --audio speech.wav "Transcribe"` → **exact transcript**
  ("The weather in Tokyo today is sunny with a high of 25 degrees"), prefill ~117
  tokens (audio frames ingested).
- The HTTP server, same model, advertises `audio` capability (`/api/show` →
  `["completion","vision","audio","tools"]`) but every audio request comes back
  **text-only** ("Please provide the audio…"), prefill ~88 (no audio frames):
  - `/api/generate` with top-level `"audio":"<b64>"` + `"audio_format":"wav"` — ignored.
  - `/api/chat` with message-level `"audio":"<b64>"` — accepted (no error) but not ingested.
  - `/api/chat` content-block `input_audio` form — rejected ("content must be a string", see #5).
- **Vision over HTTP works** (`/api/generate` `images:[…]` → correct answer,
  prefill 274), so the multimodal prefill plumbing exists; audio is the gap.
- **Where to look:** `Sources/KLMServer/ServerParsing.swift` parses `json["audio"]`
  into `media.audio` + `media.audioFormat`, and `Server.swift` decodes it to a
  temp file (`DecodedMedia.audioPath`) and loads `audioData`. Trace whether
  `audioData` actually reaches the engine's audio-prefill path on the
  `/api/generate` + `/api/chat` handlers (it does for the CLI `run` path). Likely
  the server generate path drops audio where the CLI passes it through, OR the
  audio decode (WAV → log-mel) isn't invoked server-side. Compare the CLI
  `RunCommand` audio wiring vs the server handler.
- **Gate:** add a server-path audio test (e.g. in `MultimodalEndpointsTests` or a
  smoke test) so this can't regress silently once fixed.

## 2. [Comparison caveat / opportunity] Ollama `gemma4:e2b` returns EMPTY multimodal output

**Severity: medium** (makes our Gemma-4 multimodal win look one-sided; verify it's
real and broaden coverage so the benchmark is unimpeachable).

- Ollama 0.24 `gemma4:e2b` processes the image AND audio (prefill tokens rise to
  ~289 / ~120) but emits **empty content** for both vision and voice, on
  `/api/generate` and `/api/chat`, at multiple `num_predict`. KrillLM answers
  correctly on the same weights.
- This is a **Gemma-4n-specific quirk in Ollama**, not a general Ollama-vision
  failure — Ollama's `qwen2.5vl` / `llava` paths answer fine.
- **Action:** to make the vision comparison bulletproof, also benchmark a model
  where Ollama vision WORKS (e.g. `qwen2.5vl:3b` vs KrillLM `Qwen2.5-VL-3B`) and
  show KrillLM wins on *latency* there, not just "Ollama returns empty." That
  removes the "you just picked a model Ollama is broken on" rebuttal.

## 3. [Harness gap] Concurrency benchmark doesn't capture Ollama p99 TTFT

**Severity: medium** (the tail-latency win is one of the strongest "by miles"
claims and we currently can't quote it fresh).

- `tools/krillm_concurrent_benchmark.py` reported Ollama p99 TTFT as `—` while
  KrillLM's populated (11 → 89 ms across N=1→8). Prior data claimed ~14x p99 TTFT
  advantage; we can't reproduce it until the harness parses Ollama's streaming
  TTFT (first-token timestamp) the same way it does KrillLM's.
- **Where to look:** the harness's per-request timing for the Ollama arm — ensure
  it streams (`"stream":true`) and records the first-chunk time, or reads Ollama's
  reported timings. KrillLM agg throughput at N=8 was 1.83x; the TTFT axis is the
  more dramatic one and should be quantified.

## 4. [Model-bound, document] Multi-step agentic tool-call decision is flaky on 3B models

**Severity: low** (not an engine bug; affects the "agentic superiority" narrative).

- On a 2-tool agentic prompt ("weather in Tokyo, in Fahrenheit" → get_weather →
  convert), BOTH engines were inconsistent on small models: gemma4-e2b (KrillLM
  made calls + reached correct 77°F but hallucinated the first tool name;
  Ollama made zero calls); qwen2.5-3b (Ollama called correctly, KrillLM declined
  on that exact prompt although it calls correctly when prompted directly).
- Single-shot tool calling is **4/4 valid+exact on BOTH** for qwen2.5-3b, so the
  parsing/format is fine — it's the small model's *decision* that's prompt-sensitive.
- **Action:** for a clean agentic demo, use a stronger tool caller (Qwen2.5-7B/14B,
  Llama-3.1-8B) and/or document the prompt-sensitivity. Consider whether KrillLM's
  chat-template / tool-system-prompt differs from Ollama's in a way that changes
  the call decision (see #6).

## 5. [API parity] `/api/chat` rejects OpenAI content-block array form

**Severity: low-medium** (limits multimodal-over-chat ergonomics; blocks the
`input_audio` path for #1).

- `/api/chat` with `messages[].content` as an array of `{type:text|image_url|
  input_audio}` blocks returns "Field 'messages[0].content' must be a string".
  Vision works via the top-level/message `images` array, but the OpenAI-style
  content-block form (and thus `input_audio`) isn't accepted on the Ollama-compat
  chat endpoint.
- **Where to look:** `Sources/KLMServer/ServerParsing.swift` already has
  content-block parsing (`case "input_audio"`, `case "image_url"` around
  L520-560) — confirm which endpoint/path enforces "content must be a string" and
  whether the content-block branch is reachable from `/api/chat`.

## 6. [Investigate] Same-weights tool-call *decision* differs KrillLM vs Ollama

**Severity: low** (subtle; could be chat-template or tool-prompt formatting).

- On one agentic prompt, Ollama qwen2.5:3b chose to call `get_weather` while
  KrillLM qwen2.5-3b returned plain text (no call), with identical weights, greedy.
  Different tool/chat-template rendering between the two engines can flip the
  model's decision.
- **Action:** diff KrillLM's rendered tool prompt (the actual token stream sent to
  the model) against Ollama's for the same request, and align if KrillLM's is
  weaker at eliciting calls.

---

## Quick repro pointers
- Suite: `tools/bench_suite.py` (text/vision/voice/tools, hot+cold).
- Concurrency: `tools/krillm_concurrent_benchmark.py`.
- Assets used: `/tmp/klmbench/red.png` (red box), `/tmp/klmbench/speech.wav`
  (macOS `say` → `afconvert`). Regenerate per `docs/BENCHMARKS.md`.
