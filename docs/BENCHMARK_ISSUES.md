# Benchmark Issues — found 2026-06-04, to revisit/fix

Open gaps surfaced while running the fresh KrillLM-vs-Ollama benchmark suite
(`docs/BENCHMARKS.md`) on Apple M4 Pro, KrillLM `9a21941`, Ollama 0.24.0. None
are merge-blocking for what shipped; they are the punch-list for "best inference
engine on macOS." Ordered by impact on that goal.

---

## 0. [RESOLVED - serial path] Shared-prefix (partial-prefix) KV reuse for the agentic/RAG workload

**Severity: critical.** The agentic/RAG moat (reuse a long shared context -
system prompt, tool schemas, retrieved docs - across calls) was not working: a
request that shared a long prefix with a recent one re-prefilled the ENTIRE
context every time, so KrillLM was at a large prefill disadvantage on the exact
workload agents hammer.

**Corrected diagnosis (it is NOT model-size-dependent).** Controlled
reproduction with a cleared cache shows the prefix cache only ever reused KV for
a BYTE-IDENTICAL full prompt; it had no shared-prefix (partial) reuse. That
limitation is independent of model size - it fails the same way on 3B and 14B.
The earlier "works on 3B, fails on 14B / KV-budget eviction" read compared two
different workloads (identical prompts on 3B vs varying-tail prompts on 14B).

Measured before the fix (cleared cache, same shared ~480-token scaffold):

| scenario | req1 cold | req2 | req3 |
|---|---|---|---|
| 3B, IDENTICAL prompt (full match) | 566 ms | 10 ms | 10 ms |
| 3B, shared prefix + DIFFERENT tail | 546 ms | 533 ms | 534 ms |
| 14B, IDENTICAL prompt (full match) | 2553 ms | 39 ms | 38 ms |
| 14B, shared prefix + DIFFERENT tail | 2552 ms | 2496 ms | 2497 ms |

Full-match caching worked on both sizes; the varying-tail (real agentic) case
re-prefilled on both.

**Fix (this PR): longest-common-prefix (LCP) reuse on the serial path.** When a
request shares a prefix with a recent in-memory prefill, the cached prefix KV is
restored and only the diverging suffix is prefilled (llama.cpp's behaviour). The
attention infra already supported it: `createCachedCausalMask(newLen:cacheLen:)`
builds the `[suffix, prefix+suffix]` mask and RoPE already applies the cache
offset, so this is an orchestration change in `InferenceEngine` plus a
`PrefixCache.lookupLongestPrefix`. Scoped to the fp16 cache and TEXT-only
requests, and to full-attention families (Gemma 4's sliding-window mask is
excluded; full-match hits stay enabled for it). Multi-turn chat benefits too:
each turn stores its full prompt, so the next turn reuses the whole prior turn.

Measured after the fix (shared prefix + DIFFERENT tail):

| | req1 cold | req2 reuse | req3 reuse |
|---|---|---|---|
| 3B | 576 ms | **44 ms** | 45 ms |
| 14B | 2555 ms | **175 ms** | 175 ms |

14B repeated-context prefill is now ~175 ms, at parity with Ollama's cached
~194 ms (was ~2500 ms). Output is byte-identical to a cold full prefill under
greedy decoding (gated by `PrefixCachePartialReuseLiveTests`).

**Still open (follow-up): share one prefix across CONCURRENT streams.** The
serial fix covers sequential requests and multi-turn chat. The batched
`ContinuousBatcher` path (8 concurrent agents on one scaffold) does not yet do
LCP reuse across rows; that is the next increment for the concurrent agentic
bench. Tracked as a follow-up, not in this PR.

**Isolation (unchanged, still accurate):** JSON/grammar-constrained decode
overhead is only ~20%, NOT the cause. 14B single-stream decode is healthy
(~1.08x Ollama). The collapse was purely the un-cached repeated prefill, now
fixed for the serial path.

---

## 0b. [finding] 30B-A3B MoE is unstable on 24GB (Metal assertions)

KrillLM runs `Qwen3-Coder-30B-A3B` natively and produces excellent code, but at
~22GB peak it intermittently throws `failed assertion _status <
MTLCommandBufferStatusCommitted` (GPU command-buffer / memory pressure). Some runs
complete, some crash mid-generation. **Practical stable serving ceiling on 24GB is
~14B; 30B-A3B is KrillLM-solo-only and flaky; 31B/35B will thrash.** Not a code
bug per se — it's the box. Revisit on a 36GB+ Mac. (If the assertion recurs at
lower memory, investigate the Metal command-buffer lifecycle under pressure.)

---

## 1. [RESOLVED - not reproducible] Server HTTP API audio ingestion

**Status: NOT A BUG on current `main`.** Re-checked 2026-06-04 against a freshly
built `.build/release/krillm` (and, separately, the Homebrew v0.4.0 binary):
audio **is** ingested over HTTP. The original benchmark probe must have hit an
environmental error (stale/other binary, wrong port, or a server with no model
loaded); the code path threads `audioData` end-to-end on every handler.

Reproduction (gemma-4-e2b, `/tmp/klmbench/speech.wav`, prompt "Transcribe this
audio"):

| Path                                            | prompt_eval_count | response |
|-------------------------------------------------|-------------------|----------|
| CLI `krillm run --audio`                        | 110               | exact transcript |
| `/api/generate` top-level `"audio"`+`audio_format` | 110            | exact transcript |
| `/api/chat` message-level `"audio"`             | 110               | exact transcript |
| text-only baseline (same prompt, no audio)      | 13                | - |

The ~97-token jump (13 -> 110) is the audio encoder frame run reaching prefill -
exactly the signal the benchmark used to (wrongly) conclude "not ingested." Both
Ollama-compat handlers route audio through `BatchScheduler.submit` → serial →
`engine.generate(messages:…, audioData:)`, identical to the CLI's
`generate(prompt:…)` path.

- **Vision over HTTP also works** (`/api/generate` `images:[…]` → correct answer),
  confirming the shared multimodal prefill plumbing.
- **Regression gates added** (this PR), so the wiring can't silently regress:
  - `MultimodalEndpointsTests.testOllamaChatRequestAcceptsAudioPerMessage` -
    CI-runnable; locks `/api/chat` message-level `audio` → `request.media.audio`
    (the one parse→media link the benchmark flagged) + a two-clip rejection.
  - `NativeAudioRoutingTests.testLiveNativeAudioInflatesPromptTokens` -
    env-gated (`KLM_GEMMA4_MODEL_PATH`); asserts audio inflates
    `GenerationStats.promptTokens` past the text-only baseline by the encoder
    frame count, the exact prefill-token signal the benchmark measured.
- **One real residual:** the OpenAI **content-block array** form
  (`messages[].content: [{type:input_audio}|{type:image_url}|{type:text}]`) is
  rejected on the Ollama `/api/chat` endpoint with "content must be a string"
  (it already works on OpenAI `/v1/chat/completions`). Tracked and fixed under
  **#5** below.

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
