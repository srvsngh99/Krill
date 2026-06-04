# Benchmark Issues — found 2026-06-04, to revisit/fix

Open gaps surfaced while running the fresh KrillLM-vs-Ollama benchmark suite
(`docs/BENCHMARKS.md`) on Apple M4 Pro, KrillLM `9a21941`, Ollama 0.24.0. None
are merge-blocking for what shipped; they are the punch-list for "best inference
engine on macOS." Ordered by impact on that goal.

---

## 0. [RESOLVED - serial + concurrent batched] Shared-prefix (partial-prefix) KV reuse for the agentic/RAG workload

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
requests, and to families with the standard per-layer cache. **Gemma 4 was
initially excluded** (its cross-layer KV-sharing layout, not a mask difference)
**but is now supported on the serial fp16 path** — the shared layers needed only
to rotate the suffix Q at its true positions `[LCP, count)` instead of their
empty-cache offset 0 (`Gemma4Attention`; gated by `Gemma4PartialReuseLiveTests`,
byte-exact vs cold). gemma-4-e2b shared-prefix prefill drops 1001 ms → 158 ms.
The int8-KV serial path and the concurrent batched path still exclude Gemma 4
(see docs/BACKLOG.md). Multi-turn chat benefits too:
each turn stores its full prompt, so the next turn reuses the whole prior turn.

Measured after the fix (shared prefix + DIFFERENT tail):

| | req1 cold | req2 reuse | req3 reuse |
|---|---|---|---|
| 3B | 576 ms | **44 ms** | 45 ms |
| 14B | 2555 ms | **175 ms** | 175 ms |

14B repeated-context prefill is now ~175 ms, at parity with Ollama's cached
~194 ms (was ~2500 ms). Output is byte-identical to a cold full prefill under
greedy decoding (gated by `PrefixCachePartialReuseLiveTests`).

**Follow-up DONE: shared-prefix reuse across CONCURRENT streams.** The batched
`ContinuousBatcher` per-row prefill now does the same LCP reuse as the serial
path (`makeBatchedPrefillRow`): each row, on a full-match miss, restores the
longest cached prefix it shares with a recent prefill and forwards only its
suffix. The batched decode already tolerates a row whose cache is shorter than
its prompt (`epochBaseLen` drives the ragged left-pad mask + per-row offsets),
so this is a per-row prefill change only. Measured (qwen2.5-3b,
`KRILL_NUM_PARALLEL=4`, ~440-token shared scaffold): the cold prefill is ~441 ms;
4 concurrent requests sharing that scaffold then prefill in 13 / 43 / 44 / 111 ms
instead of each re-prefilling. Bit-exact vs a cold decode, gated by
`BatchedDecodeLiveTests.testBatchedPartialPrefixReuseMatchesColdDecode`. fp16 /
text-only / non-Gemma-4, same as the serial path.

**Isolation (unchanged, still accurate):** JSON/grammar-constrained decode
overhead is only ~20%, NOT the cause. 14B single-stream decode is healthy
(~1.08x Ollama). The collapse was purely the un-cached repeated prefill, now
fixed on both the serial and the concurrent batched paths.

---

## 0b. [BOX LIMIT confirmed on retry] 30B-A3B MoE not viable on 24GB

KrillLM runs `Qwen3-Coder-30B-A3B-Instruct-4bit` natively and produces excellent
code, but the 16 GB (4-bit) weights leave no headroom on a 24 GB host. **Re-tested
2026-06-04 on latest `main`, both paths:**
- **CLI** (`krillm run`): generates correct code, but **segfaults mid-generation
  ~1 in 5** longer runs under repeated load (exit 139; RAM-free collapses just
  before the crash). The earlier `failed assertion _status <
  MTLCommandBufferStatusCommitted` is the same memory-pressure failure surfacing
  as either a Metal command-buffer assertion or a raw segfault.
- **Server** (load once, no reload churn): does NOT crash and returns **correct**
  content, but is **unusably slow - ~118 s for a 40-token request** (the model
  swaps; first request after load is the worst). `eval_count` reports 0 on this
  thrashing path (a minor stats artifact; the content is real).

The 256 MB MLX recycling-pool cap (`KRILL_MLX_CACHE_LIMIT_MB`,
`MLXMemoryConfig`) is already applied; it bounds intermediate buffers, not the
16 GB of resident weights, so it cannot help here. **Not a code bug - it is the
box.** Practical stable serving ceiling on 24 GB is ~14 B; 30B-A3B is
solo-only-and-flaky / thrashes when served. Revisit on a 36 GB+ Mac.

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

## 2. [RESOLVED] Vision comparison broadened to a model Ollama renders

**Done.** Ollama 0.24 `gemma4:e2b` processes the image AND audio (prefill tokens
rise to ~289 / ~120) but emits **empty content** for both vision and voice, on
`/api/generate` and `/api/chat`, at multiple `num_predict`. That makes a
Gemma-4-only vision comparison look one-sided. This is a Gemma-4n-specific quirk
in Ollama, not a general Ollama-vision failure: its `qwen2.5vl` / `llava` paths
answer fine.

**Bulletproof comparison on a model Ollama renders** (red-box image, HOT, greedy,
best-of-3 total latency):

| | total latency | answer |
|---|---|---|
| KrillLM `Qwen2.5-VL-3B-Instruct-4bit` | 607 ms | correct (recognizes red) |
| Ollama `qwen2.5vl:3b` | 660 ms | correct (NOT empty) |

KrillLM is ~1.09x on single-stream total latency here - near parity, the same
memory-bandwidth-bound story as text decode, with the dramatic multimodal win
remaining the Gemma-4 correctness case (Ollama empty). The point is that the
comparison now holds on a model where Ollama actually answers, removing the "you
just picked a model Ollama is broken on" rebuttal.

`tools/bench_suite.py` `bench_vision` now also flags EMPTY engine output
explicitly (mirroring `bench_voice`) and prints a NOTE to re-run on a rendering
model when the baseline returns nothing. Reproduce:
`tools/bench_suite.py --krill-model Qwen2.5-VL-3B-Instruct-4bit --ollama-model qwen2.5vl:3b --image IMG.png`.

## 3. [RESOLVED - not reproducible; ambiguity hardened] Concurrency benchmark Ollama p99 TTFT

**The harness already captures Ollama p99 TTFT.** `one_request` streams both arms
(`stream:true`) and records the first chunk carrying a `response` field; both
KrillLM and Ollama `/api/generate` emit NDJSON with per-chunk `response`, so the
same parser populates both. Re-run against Ollama qwen2.5:14b: p99 TTFT 189 ms at
N=1, 621 ms at N=2: populated, not blank. The original blank cell was the Ollama
arm not being run (or its streams failing), not a parser gap.

**Hardened so the ambiguity can't recur silently:** each per-N result now records
`ttft_samples` (how many successful streams yielded a first-token time), and the
summary prints an explicit `note:` when an arm produced streams but captured zero
TTFT samples, so a blank TTFT cell is now always either a populated number, an arm
that was not run, or an explained parse gap, never a silent blank.

To quote the tail-latency comparison fresh, run with BOTH arms, e.g.:
`tools/krillm_concurrent_benchmark.py --krillm-url URL --ollama-host URL --concurrency-sweep 1,2,4,8`.

## 4. [Partly addressed via #6, otherwise model-bound] Multi-step agentic tool-call decision on 3B models

**Severity: low** (not an engine bug; affects the "agentic superiority" narrative).

- On a 2-tool agentic prompt ("weather in Tokyo, in Fahrenheit" -> get_weather ->
  convert), small models are decision-sensitive. Part of the qwen2.5-3b
  divergence was a KrillLM tool-prompt mismatch, now fixed (see #6): with the
  official Qwen tool format, qwen2.5-3b picks the correct first call
  (`get_weather`) on these prompts where the old generic prompt jumped straight
  to the converter.
- Single-shot tool calling is **4/4 valid+exact on BOTH** for qwen2.5-3b; the
  residual flakiness is the tiny model's reasoning, not the format.
- **Remaining guidance:** for a clean agentic demo, prefer a stronger tool caller
  (Qwen2.5-7B/14B, Llama-3.1-8B). The format-level divergence is closed by #6.

## 5. [RESOLVED] `/api/chat` now accepts the OpenAI content-block array form

**Fixed.** `/api/chat` with `messages[].content` as an array of
`{type:text|image_url|input_audio}` blocks used to return "Field
'messages[0].content' must be a string"; it now accepts the same content blocks
the OpenAI `/v1/chat/completions` endpoint does.

The OpenAI and Ollama chat parsers in `Sources/KLMServer/ServerParsing.swift`
now share one `parseContentBlocks` helper (previously the block-parsing logic
lived only in `openAIMessages`; `ollamaMessages` required a string). A
content-block clip routes its image (`data:` URL) / audio (base64 + format) into
the request media payload, alongside Ollama's own message-level `images`/`audio`
fields; a `data:`-less image URL and a second audio clip (across blocks or vs a
message-level `audio`) are rejected just as on the OpenAI path.

Verified end-to-end on `/api/chat` (gemma-4-e2b): `input_audio` block ->
prompt_eval_count 110 + exact transcript; `image_url` block -> prompt_eval_count
274 + correct answer. This also delivers the `input_audio`-over-chat path noted
in #1. Gated by `MultimodalEndpointsTests` (content-block accept + the two
rejection cases).

## 6. [RESOLVED] Tool-call *decision* divergence was a tool-prompt mismatch on Qwen

**Root cause found and fixed.** For Qwen 2.5/3, KrillLM was injecting a generic
Hermes tool prompt ("You can call tools. The available tools are listed as JSON
schemas...") instead of the **official Qwen tool block the model was fine-tuned
on** - the chat template's `# Tools` section with the schemas inside
`<tools></tools>` XML tags. swift-transformers drops the `tools` Jinja variable,
so KrillLM has to reproduce that block as message text (the Llama and Mistral
paths already do this for their families); the Qwen path was still on the
generic prompt. Ollama renders the official block, so the two engines were
feeding the model materially different tool instructions, which can flip a
borderline call decision.

**Fix:** a dedicated `injectQwen` reproduces the official `# Tools` / `<tools>`
block verbatim (the call/result sentinels are unchanged - they already match the
Hermes convention, so the parser is untouched). Demonstrated effect on
qwen2.5-3b, 2-tool agentic prompts, greedy:

| prompt | old (generic Hermes) | new (official Qwen) |
|---|---|---|
| "weather in Tokyo, in Fahrenheit" | `celsius_to_fahrenheit` (wrong: nothing to convert yet) | `get_weather` (correct first step) |
| "How hot is it in Dubai right now in Fahrenheit?" | `celsius_to_fahrenheit` | `get_weather` |

Direct single-tool prompts stayed 4/4 (no regression). Gated by
`ServerTests.testQwenInjectionMirrorsOfficialTemplate` /
`testQwenInjectionAddsSystemTurnWhenAbsent`.

---

## Quick repro pointers
- Suite: `tools/bench_suite.py` (text/vision/voice/tools, hot+cold).
- Concurrency: `tools/krillm_concurrent_benchmark.py`.
- Assets used: `/tmp/klmbench/red.png` (red box), `/tmp/klmbench/speech.wav`
  (macOS `say` → `afconvert`). Regenerate per `docs/BENCHMARKS.md`.
