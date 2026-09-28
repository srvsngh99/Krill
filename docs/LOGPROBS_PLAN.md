# Logprobs Support Plan (OpenAI/Ollama `logprobs` + `top_logprobs`)

Created: 2026-09-28
Status: Phase 1 implemented on `feat/logprobs-phase1` (chat `logprobs` +
`top_logprobs`, non-streaming + streaming, plain decode path; spec/batch
fallback). Phases 2-3 (this doc's §7) not started. See §4/§7 for the
settled decisions and open-question resolutions recorded during
implementation.
Base branch: `main`
Base commit: `db1a53f`
Owner: unassigned

## 1. Problem

Krill's server is an OpenAI/Ollama-compatible API surface (`docs/ARCHITECTURE.md`
§"Serving and agent surfaces"), but today it actively **rejects** the
`logprobs` family of parameters instead of ignoring or supporting them:

- `POST /v1/chat/completions` 400s on `logprobs` and `top_logprobs`
  (`Sources/KrillServer/ServerParsing.swift:142-147`,
  `unsupportedOpenAIChatFields`).
- `POST /v1/completions` 400s on `logprobs` and `echo`
  (`Sources/KrillServer/ServerParsing.swift:154-157`,
  `unsupportedOpenAICompletionFields`).
- Ollama's `/api/chat` and `/api/generate` have no `logprobs`/`top_logprobs`
  handling at all (`unsupportedOllamaChatFields` is empty, but the fields are
  simply never read).

A 400 on a standard, widely-used field is worse than silently ignoring it:
it breaks any OpenAI-SDK client, eval harness, or router that sends
`logprobs` by default, before the first token comes back. This is the same
class of problem `stream_options` was for opencode/OpenAI SDK clients before
it was accepted (see the comment right above `unsupportedOpenAIChatFields`).

This is a **Krill product feature**, not a favour to another project. It
closes an API-parity gap and unlocks real Krill use cases:

- **OpenAI SDK / client compatibility** — any client that sets `logprobs`
  by default currently cannot talk to Krill at all.
- **Evaluation harnesses** — lm-eval-style multiple-choice scoring picks the
  answer letter by comparing the logprobs of a handful of candidate tokens,
  not by sampling.
- **Classification / routing with confidence** — a token's logprob is a
  cheap, standard confidence signal for "is the model sure," used to decide
  whether to escalate to a bigger model or ask a follow-up.
- **Uncertainty display** — surfacing per-token confidence in a UI (the kind
  of overlay `krill run`/`krill code` or a phone console could show).
- **LLM-as-judge scoring** — reading the logprob of "yes"/"no" or a score
  token is a standard, cheaper alternative to parsing free text.

One motivating *consumer*, mentioned here and nowhere else in this plan: a
side project (Jugnu) wants a local teacher model's token probabilities for
distillation labelling. That is one caller among several — the API design
below is OpenAI/Ollama parity, full stop, and must not bend to that one
consumer's convenience.

## 2. What "done" looks like

A client that sends `logprobs: true` (and optionally `top_logprobs: N`) to
`/v1/chat/completions` gets back the standard OpenAI shape, at no cost to a
client that does not ask for it — the existing zero-overhead-by-default
decode path (see `docs/PITFALLS.md` #10 and the release-gate discipline in
`docs/RELEASE_GATE_DECODE_PROPOSAL.md`) must stay exactly as fast when
`logprobs` is absent.

Phase 1 (this plan's minimum shippable slice) covers the plain, single-request,
non-batched, non-speculative decode path only. Batching, speculative decoding,
and prompt/echo logprobs are later phases with an explicit, documented
fallback in the meantime (see §7).

## 3. API surface

### 3.1 `POST /v1/chat/completions` (OpenAI)

Request fields, matching OpenAI's documented shape:

| Field | Type | Notes |
|---|---|---|
| `logprobs` | `bool`, default `false` | Whether to return log probabilities for the generated tokens. |
| `top_logprobs` | `int`, 0–20 | Only meaningful when `logprobs: true`. Number of most-likely alternates to report at each position. **Verify against current OpenAI docs before implementing** — WebFetch to `platform.openai.com` was blocked (403) in this research pass, and two different search snippets disagreed (an old "0–5" limit vs. the "0–20" this plan assumes, which matches what the task brief specified and what Ollama's own docs list — see §3.3). Confirm the current OpenAI-documented max before coding the range check. |

Response shape (added to each `choices[]` entry):

```json
"logprobs": {
  "content": [
    {
      "token": "Hello",
      "logprob": -0.31,
      "bytes": [72, 101, 108, 108, 111],
      "top_logprobs": [
        {"token": "Hello", "logprob": -0.31, "bytes": [72,101,108,108,111]},
        {"token": "Hi",    "logprob": -1.82, "bytes": [72,105]}
      ]
    }
  ]
}
```

- `content` has one entry per **generated** token (not prompt tokens — that
  is `echo`, phase 3).
- `top_logprobs` on a content entry is present whenever the request set
  `top_logprobs > 0`; omitted (or empty) otherwise. Order is highest logprob
  first, and the sampled token is included among them if it makes the cut
  (OpenAI's documented behavior — verify token-inclusion edge case: if the
  sampled token is NOT in the model's own top-N, OpenAI still lists it
  separately as the `content[i]` entry itself, while `top_logprobs` shows the
  actual top-N regardless).
- Emit `"logprobs": null` on `choices[]` when the request did not ask for it
  (matches OpenAI's own default), not an absent key — check what existing
  Krill response builders do for other optional fields for the house style
  (`Sources/KrillServer/Server.swift:1422-1444` uses plain dictionary literals
  with `NSNull()` for explicit nulls elsewhere, e.g. `message["content"] =
  NSNull()` at line 1419).

### 3.2 `POST /v1/completions` (legacy)

- `logprobs: int` (not boolean — legacy completions has always used an
  integer meaning "top N", 0 disables). Currently rejected at
  `ServerParsing.swift:155`.
- `echo: bool` — when true, also return logprobs for the **prompt** tokens.
  Flagged explicitly in the task as a phase-2/3 option because of its cost
  (see §6.4); do not block phase 1 on it.
- Response shape is the older flat form: `choices[].logprobs = {tokens:
  [...], token_logprobs: [...], top_logprobs: [{...}], text_offset: [...]}`
  — a different shape from the chat endpoint's `content[]` array. Verify the
  exact current field names against OpenAI's docs before implementing; this
  plan did not manage to pull the legacy-completions reference successfully
  (see the verification note in §3.1).

### 3.3 Streaming (SSE)

Every `chat.completion.chunk` needs a `choices[0].logprobs` entry for the
token(s) in that chunk, same per-token shape as the non-streaming case, one
entry in `content[]` per chunk (Krill's chat SSE path emits at most one
delta-worth of text per chunk today — see `sseChunk` at
`Sources/KrillServer/ServerFormatting.swift:6` and its caller loop at
`Sources/KrillServer/Server.swift:1638-1679`). `sseChunk` currently takes
`(id, content, finishReason)` as plain strings and hand-formats JSON directly
(no `JSONSerialization`, deliberately — `docs/PITFALLS.md` #10 says why); a
`logprobs` parameter needs to thread through here without regressing that
hot-path discipline.

Two wrinkles specific to Krill's streaming path, both real, both need a
decision before coding:

- **The reasoning filter.** `StreamingReasoningFilter` (used at
  `Server.swift:1636,1673`) strips `<think>...</think>` content from the
  visible stream. If `logprobs` is requested on a reasoning model, decide
  whether logprobs are reported for suppressed "thinking" tokens too, or
  only for the tokens that make it into the visible answer. OpenAI's own
  reasoning-model behavior here is inconsistent across model families and
  not cleanly documented — treat as an open question (§8), and the simplest
  correct-by-construction default is: report a logprobs entry only for
  tokens actually emitted to `content` in the response, mirroring what the
  client actually sees.
- **Tool-call replies are NOT streamed token-by-token today.** When the
  response contains `tool_calls`, `Server.swift:1471-1499` assembles the
  full result first and emits it as a *single* SSE chunk (comment: "emit the
  assembled result as one chunk"). Logprobs for a tool-call turn is either
  out of scope for phase 1, or piggybacks on that single-chunk assembly path
  using the same per-token data collected during generation.

### 3.4 Ollama `/api/chat` and `/api/generate`

Ollama's own current API (`docs.ollama.com/api/chat`, confirmed via live
fetch during this research) documents:

- `logprobs: bool` and `top_logprobs: int` (0–20, default 0) as request
  options, on **both** `/api/chat` and `/api/generate`.
- A response `logprobs` array with `token`, `logprob`, `bytes`,
  `top_logprobs` fields — i.e. Ollama copied OpenAI's chat shape rather than
  inventing its own.

Krill's Ollama-dialect request structs (`unsupportedOllamaChatFields` is
empty at `ServerParsing.swift:159`; `unsupportedOllamaGenerateFields` at
:161-163 does not mention logprobs either) do not currently parse or reject
these fields — they are simply dropped on the floor. Wire them the same way
as the OpenAI dialect once the engine plumbing exists; this should be close
to free once §5's engine work lands, since both dialects funnel into the
same `InferenceEngine.generate(messages:)` (`Sources/KrillEngine/
InferenceEngine.swift:785`).

### 3.5 Error behavior

- `top_logprobs` outside 0–20 (or whatever range §3.1 confirms): reject with
  `ServerRequestError.invalidValue`, matching the existing pattern for other
  ranged fields (grep `ServerParsing.swift` for `invalidValue` — the
  temperature/top_p validators are the template to copy).
- `top_logprobs` set without `logprobs: true`: OpenAI's behavior is to
  ignore `top_logprobs` silently in this case (it does nothing without
  `logprobs`) — verify, but do not 400 for it; that would be a stricter
  parity break than the field it replaces.
- `logprobs` sent as a non-bool on the chat endpoint, or a non-int on the
  legacy completions endpoint: `ServerRequestError.invalidType`, matching
  existing validators.

## 4. Semantics decisions

### 4.1 Raw distribution vs. post-sampling-transform distribution

**This is the single most consequential design decision in this plan**, and
the plan recommends resolving it before writing any code.

Krill's sampler (`Sources/KrillSampler/Sampler.swift:187-242`,
`sampleFrom`) runs, in order: an optional grammar mask, then (if not
greedy) temperature scaling, then top-k (`topKFilter`, line 286), then
top-p (`topPFilter`, line 308), then min-p (`minPFilter`, line 296) — each
filter sets rejected logits to `-1e9` — then a final categorical draw. Greedy
requests (`temperature <= 0`) skip straight to `argMax` (line 191-193) and
never reach the filters at all.

If Krill reports logprobs computed **after** top-k/top-p/min-p truncation,
the reported distribution is not a real probability distribution over the
vocabulary — it is the truncated remainder, renormalized or not depending on
how it's read, and it will disagree with `mlx_lm`/HF `transformers` computing
logprobs from the model's raw logits. If Krill reports logprobs computed
**before** temperature scaling, a `temperature: 2.0` request that
dramatically flattens the actual sampling distribution will report logprobs
that don't reflect what was actually drawn — also a foot-gun for eval
harnesses that assume "the reported logprob is the probability this token
was drawn."

**Recommendation: report logprobs from the raw model logits: a plain
log-softmax of the forward pass, BEFORE temperature scaling, before
top-k/top-p/min-p truncation, and before repetition/presence/frequency
penalties.** (An earlier draft of this plan recommended post-temperature;
that was changed during review for the reasons below.)

- **Greedy requests make post-temperature undefined.** Evaluation harnesses
  and judges almost always call with `temperature: 0`, and Krill's greedy
  path skips temperature entirely (`Sampler.swift:191-193`). A
  post-temperature definition has nothing to report at T=0, or divides by
  zero. Raw logprobs are well defined at every temperature.
- **Reproducibility.** Raw logprobs are a pure function of (model, prompt,
  position). They match what `mlx_lm` / HF `transformers` compute from the
  same logits, which makes the numeric-parity test in section 6 a
  straightforward comparison with a tolerance instead of a
  sampler-configuration-dependent one.
- **Common server practice.** vLLM's default logprobs mode is the raw model
  distribution, before any sampling processing (its `logprobs_mode` option,
  default `raw_logprobs`); verify the current default before implementing.
  Consumers that move between Krill, vLLM and Ollama should see the same
  meaning.
- **Truncation and penalties are decoding heuristics, not the model's
  belief.** Top-k/top-p/min-p zero out most of the mass. Penalties depend on
  history (`Sampler.swift:138-167`, `applyPenalties`), so penalty-adjusted
  logprobs could not be reproduced from the prompt alone.
- **The cost is that the reported logprob is not always the exact
  probability the token was drawn with** when temperature != 1 or truncation
  is active. Document this in the API notes. If a user later needs
  "as-sampled" logprobs, add an opt-in, vLLM-style `logprobs_mode`
  (`raw` | `processed`) in a later phase. Keep `raw` as the default.

**Still to verify before implementing:** OpenAI's own documentation on this
point. This plan's WebFetch to `platform.openai.com` failed (403). The
OpenAI cookbook (`developers.openai.com/cookbook/examples/using_logprobs`)
loaded during research and is a good source to re-fetch. If OpenAI documents
a different contract, record the difference here, but keep raw as the
default for the greedy and reproducibility reasons above.

### 4.2 Token string and `bytes` for multi-byte/partial UTF-8 tokens

Krill already has a **known, documented, real gap** directly relevant to
this field: `Sources/KrillTokenizer/TokenizerWrapper.swift:565-589`
(`recoverByteFallback`) explicitly notes that SentencePiece byte-fallback
tokens (`<0xHH>`) that together encode a multi-byte UTF-8 character **cannot
currently be assembled from a single-token decode** — the comment says
"Only single-BYTE (ASCII, < 0x80) fallbacks are recovered... A multi-byte
character arrives as several `<0xHH>` tokens that cannot be assembled
without carrying state across decode calls, so those keep the previous
behaviour" (i.e. the token decodes to an empty string today, see
`decodeForOutput` at line 629-634).

This is exactly the case OpenAI's `bytes` field exists to handle: a client
is supposed to be able to reconstruct the true output from the byte arrays
even when an individual token's `token` string is not valid UTF-8 on its
own. Two implementation requirements follow:

1. `bytes` must be derived independently of `decode(token:)` /
   `decodeForOutput(token:)` — from the raw token piece (via
   `tokenizer.convertIdToToken`, already used at
   `TokenizerWrapper.swift:583`) parsed as its true byte value(s), not from
   whatever (possibly-empty, possibly-lossy) string the current decode path
   produces. A non-byte-fallback token's `bytes` is simply the UTF-8 bytes of
   its decoded string.
2. `token` (the string field) is allowed by the OpenAI spec to be a
   lossy/replacement-character rendering when the raw bytes aren't valid
   UTF-8 alone — so `decodeForOutput`'s existing (documented, deliberate)
   limitation does not need to be fixed as a prerequisite, but the `bytes`
   field must not inherit that limitation, or Krill will emit factually
   wrong byte arrays, not just a cosmetically wrong string.

### 4.3 Chat-template special tokens

Special/structural tokens (turn markers, tool-call sentinels, the Gemma 4
media markers suppressed by `outputSuppressedTokenIDs`,
`TokenizerWrapper.swift:611-623`) can appear in the raw generated token
stream even when they're stripped from visible output. Decide: does
`logprobs.content[]` include an entry for a suppressed/special token that
never reaches the visible `content` string? Recommend **no** — the array
should stay index-aligned with the visible text a client can see, same
reasoning as the reasoning-filter question in §3.3. This needs to be
implemented consistently at whatever single point aggregates "the tokens
that make up this response" — check whether such a point already exists or
needs to be introduced as part of this work (see §5.1).

## 5. Engine work

### 5.1 Where logits are available today, and what has to change

The engine's raw forward logits are available, per decode step, in exactly
these places — and in **every one of them, the token is currently selected
and the logits are discarded** without ever computing a log-softmax:

| Path | File:line | How the token is chosen today |
|---|---|---|
| Plain single-request decode | `InferenceEngine.swift:1915-1924` (`sampler.sampleArray(logits, ...)`) | Via `Sampler`, which returns only the token ID (`Sampler.swift:110-241`) |
| Draft-model speculative decode (verify step) | `SpeculativeDecoder.swift:165,293` (`argMax(targetLogits, axis: -1)`) | Raw `argMax`, bypasses `Sampler` entirely |
| Continuous-batcher, all-greedy fast path | `ContinuousBatcher.swift:381` (`argMax(logits, axis: -1)`) | Raw `argMax` over the whole batch in one call |
| Continuous-batcher, per-row (mixed greedy/sampled) path | `ContinuousBatcher.swift:459` (`row.sampler.sample(logits[i..<(i+1)], recent:)`) | Via `Sampler`, one row at a time, returns only an `Int` |
| Continuous-batcher, n-gram/prompt-lookup speculative verify | `ContinuousBatcher.swift:663` (`argMax(bl, axis: -1)`) | Raw `argMax`, bypasses `Sampler` |
| Stage-B fixed-cohort batched decode | `BatchedDecode.swift:456` (`argMax(bl, axis: -1)`) | Raw `argMax`, bypasses `Sampler` |

Two structural facts fall out of this table:

1. **`Sampler`'s public API only ever returns a token ID.** `sample()` /
   `sampleArray()` (`Sampler.swift:110-179`) throw away the logits array
   after drawing from it. Any logprobs support needs a new method (or an
   additional return value) that optionally also returns the chosen token's
   logprob and the top-N alternates' logprobs — computed from the *pre-filter*
   logits per §4.1, i.e. a log-softmax of the raw forward-pass logits,
   taken before penalties, temperature scaling and the
   `topKFilter`/`topPFilter`/`minPFilter` truncation. This also covers
   greedy requests, which skip temperature entirely.
2. **Three of the six paths never go through `Sampler` at all** — they call
   `argMax` directly on the full forward output. This is good news for
   Phase 2 (the raw logits are sitting right there in every one of those call
   sites, nothing needs to be threaded further), but it means Phase 1's
   `Sampler`-only change does not cover speculative decoding or the batched
   paths — see §7 for the explicit fallback.

`TokenEvent` (`Sources/KrillEngine/TokenEvent.swift:4-23`) is the value that
carries a decoded token out of the engine to every server dialect; it
currently has no field for logprob data at all. This is the natural place to
add an optional field (e.g. `logprob: TokenLogprobInfo?`), populated only
when the request asked for it, so a request that doesn't ask for logprobs
pays for nothing extra downstream of the engine either.

### 5.2 Cost, and keeping the default path at zero cost

A log-softmax + top-k over the full vocabulary (Krill's largest is Gemma 4's
262144-entry vocab, per `docs/ARCHITECTURE.md`'s multimodal table) is one
extra `MLX.softmax` + `MLX.log` + a top-k sort per decode step — not free,
but a single elementwise+reduce op on an array MLX already produced this
step; the dominant per-step cost remains the transformer forward pass. The
hard requirement, matching the precedent already set for penalties
(`Sampler.swift:95`, gated by `needsHistory`) and for the WS-D D3 penalty
work (`docs/OLLAMA_MAC_PARITY_PLAN.md` §"WS-D", "Zero-overhead on the
default path"), is: **this computation must be behind an `if wantLogprobs`
branch that a request without `logprobs` never enters**, verified by the
release-gate benchmark (§6.3) showing no `text_decode_ratio` regression
(`docs/RELEASE_GATE_DECODE_PROPOSAL.md`) when logprobs are off.

`InferenceEngine.generate(messages:)` (`InferenceEngine.swift:785-799`) is
the single funnel every server dialect calls through; add
`wantLogprobs: Bool = false` and `topLogprobs: Int = 0` parameters here,
threaded down to wherever `Sampler` is invoked, defaulting to off so every
existing call site (CLI, `krill code`, all five server dialects) is
unaffected without being touched.

### 5.3 Per-decode-path plan

| Path | Phase-1 behavior | Later-phase behavior |
|---|---|---|
| Plain single-request decode (`InferenceEngine.swift` main loop) | **Full support.** Extend `Sampler` to optionally return per-step logprob + top-N, wire through `TokenEvent`. | — |
| Draft-model speculative decode (`SpeculativeDecoder.swift`) | **Falls back**: a `logprobs`-requesting request disables spec (`useSpeculative: false`), same mechanism already used to decline spec for non-greedy/penalty/int8-KV requests (`docs/SPECULATIVE_DECODING.md` §"When the spec path is skipped"). | Phase 2: compute logprobs from `targetLogits` at the accept/reject step (`SpeculativeDecoder.swift:165,293`) — the target model's full logits are already computed for verification, so the accepted tokens' logprobs are a log-softmax away, no extra forward pass needed. |
| Continuous batcher / Stage-B batched decode (`ContinuousBatcher.swift`, `BatchedDecode.swift`) | **Falls back**: a `logprobs`-requesting row is not admitted into the batched/continuous pool; it runs the plain single-request path instead (this is a server-level scheduling decision — `BatchScheduler.swift:82,151` already gates batching on `numParallel >= 2` and `supportsBatchedDecode`, so "exclude this one row" is an extension of an existing admission check, not a new concept). | Phase 2: extend the per-row `Sampler` call (`ContinuousBatcher.swift:459`) the same way as the plain path; the two `argMax`-only fast paths (`:381` all-greedy, `:663` n-gram spec verify, `BatchedDecode.swift:456`) need their own log-softmax-on-`bl` addition since they never touch `Sampler`. |
| Prefix cache | **No change needed for phase 1/2** (generated-token logprobs don't touch cached *prompt* KV, only the live forward). Relevant only for `echo` — see §5.4. | — |
| Compiled decode | **Not applicable.** `docs/COMPILED_DECODE_PROBE.md` records this as investigated and closed — "the production decode path stays on the uncompiled growing `KVCache`." No production compiled-decode path exists to integrate with. | — |

### 5.4 Prefix cache and `echo` (prompt logprobs)

`echo`/prompt-logprobs needs a logprob for **every prompt token**, which
needs that token's logits computed with the *preceding* prompt as context —
i.e. a full forward over the whole prompt with no shortcuts. This directly
conflicts with the prefix-cache fast path: on a full cache hit,
`InferenceEngine.swift:1527-1541` explicitly truncates the cache and
re-forwards **only the last prompt token** ("On a full cache hit we already
have KV for the entire prompt. We truncate the last position and re-forward
that single token to get logits without duplicating a KV entry"); on a
partial hit, only the divergent suffix is forwarded
(`InferenceEngine.swift:1542-1552`). Neither path ever produces logits for
the cached prefix positions.

So `echo: true` requests must either (a) bypass the prefix cache entirely
(`usePrefixCache: false`, already a parameter on `generate()`) and pay for a
full prompt forward, or (b) store logits during prefill and cache them
alongside the KV entries — a materially bigger change to `PrefixCache`
(`Sources/KrillCache/PrefixCache.swift`) that is not justified for a
phase-3, explicitly-lower-priority feature. Recommend (a): simple, correct,
and the cost is opt-in (only paid by a request that asks for `echo`).

## 6. Tests and acceptance

1. **Unit tests — parsing.** New cases in whatever file currently covers
   `ServerParsing` request parsing (check `Tests/KrillServerTests/
   ServerTests.swift` and `ServerFormattingTests.swift` first — no dedicated
   `ServerParsingTests.swift` exists today, confirm the right home before
   adding a new file): `logprobs`/`top_logprobs` accepted on chat, range
   validation (0–20 or whatever §3.1 confirms), `logprobs`/`echo` accepted
   on completions, Ollama dialect parity.
2. **Unit tests — response shape and streaming.** Golden-JSON tests for the
   non-streaming `choices[].logprobs.content[]` shape and for the streaming
   per-chunk shape, mirroring the existing pattern in
   `Tests/KrillServerTests/ServerFormattingTests.swift`.
3. **Numeric correctness against `mlx_lm`.** For a small local model, run
   the same prompt through `mlx_lm.generate` (or a short `mlx_lm` /
   `mlx.core` script computing `log_softmax(logits)[token]` directly) and
   assert Krill's reported logprob for each generated token matches within a
   stated tolerance (suggest `1e-3` in log-space for fp16/bf16 models,
   loosened for int4/nvfp4-quantized checkpoints — quantization noise is
   real and already characterized elsewhere in this repo, e.g.
   `docs/RELEASE_GATE_DECODE_PROPOSAL.md`'s dtype-mismatch allowance flag).
   This is the test that actually proves §4.1's semantics decision was
   implemented correctly, not just parsed.
4. **Benchmark gate — no regression when logprobs are off.** Run the
   existing release-gate benchmark (`make bench-release-gate`, backed by
   `tools/release_gate.py`, hard-gated on `text_decode_ratio_floor >= 1.0x`
   per `docs/RELEASE_GATE_DECODE_PROPOSAL.md`) before and after this change
   with `logprobs` never requested, and confirm no regression. This is the
   literal enforcement mechanism for the "zero cost by default" requirement
   in §5.2 — do not skip it.
5. **OpenAI-SDK round-trip test.** A short script using the real `openai`
   Python SDK (or Node SDK) pointed at a local `krill serve`, requesting
   `logprobs=True, top_logprobs=5`, asserting the SDK parses the response
   into its typed `ChatCompletionTokenLogprob` objects without error. This
   catches shape mistakes unit tests miss (wrong nesting, wrong field
   names) because the SDK is stricter than a hand-rolled JSON assertion.
6. **`krill-qa`-style human check.** Extend the `krill-qa` skill's scenario
   set (`~/.claude/skills/krill-qa/krill_qa.py`, see its `infer_scenarios`)
   with a logprobs scenario: request `logprobs`+`top_logprobs` in a real
   multi-turn chat and confirm the values look sane (not NaN, not all
   identical, roughly monotonic with an obviously-more-likely vs.
   less-likely continuation) — the skill's own instructions are explicit
   that automated `ok` flags are "a first pass, not gospel" and a human
   (Claude) judgment pass is the point.

## 7. Phasing

**Phase 1 — chat `logprobs` + `top_logprobs`, plain decode path only.**
Scope: §3.1 (chat request/response shape), §3.3 (streaming), §4 (all
semantics decisions — these have to be settled once, up front, not
re-litigated per phase), §5.1-5.3 (Sampler extension + `TokenEvent` +
`InferenceEngine.generate` plumbing), speculative decode and batching both
explicitly fall back to plain decode when `logprobs` is requested (§5.3).
Tests: items 1-3, 5-6 from §6, plus the phase-1 slice of item 4 (must not
regress since phase 1 touches the hot path with an `if` branch).
**Effort estimate: medium** — one clear vertical slice (parse → engine →
response), but §4.1's semantics decision and §4.2's byte-fallback fix both
need to be right, not just parsed-and-plumbed; budget real time for the
`mlx_lm` numeric-parity test, since that is what actually validates the
work.

**Phase 2 — speculative decode + batched decode support.**
Scope: §5.3's "later-phase behavior" column — extract logprobs from
`SpeculativeDecoder`'s existing `targetLogits` at the verify step, and from
`ContinuousBatcher`'s three non-`Sampler` `argMax` call sites plus its
per-row `Sampler` call. Ollama dialect wiring (§3.4) can land here too, since
it reuses phase 1's engine plumbing once it exists.
**Effort estimate: medium-large** — the logic is a bounded log-softmax
addition at each site, but there are five distinct call sites (§5.1's
table) across two files with different data layouts (per-row vs. fully
batched), each needing its own careful review against the "does this
regress the non-logprobs path" gate.

**Phase 3 — `echo` / prompt logprobs.**
Scope: §3.2 (legacy completions `echo`), §5.4 (bypass prefix cache for echo
requests). Deliberately last: named explicitly in the task as a
cost/priority tradeoff, and the legacy `/v1/completions` endpoint is a
smaller share of real traffic than chat.
**Effort estimate: small-medium** — mechanically simple (disable prefix
cache, run a full forward, log-softmax every position) once phase 1's
per-token logprob machinery exists; the work is mostly in the different
response shape (§3.2) and re-verifying numeric parity against `mlx_lm` for
prompt (not just generated) tokens.

### Open questions (carry into implementation, don't guess)

- Confirm the real current OpenAI `top_logprobs` max (0–20 vs. an older
  0–5) — this research pass could not get a clean read of
  `platform.openai.com` (403 on WebFetch); a second attempt or a different
  source (the OpenAI Python SDK's own type stubs, which pin the literal
  range, would be authoritative and fast to check) should resolve it before
  the range-validation code is written.
- Confirm OpenAI's exact documented pre/post-temperature logprobs semantics
  (§4.1) rather than relying on this plan's inference from behavior —
  `developers.openai.com/cookbook/examples/using_logprobs` is a good
  starting point and did load successfully during this research.
- Reasoning-model logprobs (§3.3, §4.3): should suppressed/thinking tokens
  ever be reported, even behind a Krill-specific extension flag? Not
  standard OpenAI behavior to copy from directly since reasoning-model
  logprobs handling is inconsistent industry-wide; punt to "no" unless a
  concrete consumer asks.
- Legacy `/v1/completions` response field names (§3.2) — this plan did not
  manage to pull a clean current reference for the older flat
  `token_logprobs`/`text_offset` shape; confirm against the OpenAI API
  reference or a working client library before implementing phase 3.
- Whether `top_logprobs` entries should include the sampled token even when
  it falls outside the model's actual top-N (§3.1) — verify against a real
  OpenAI response if the team has API access, since documentation is thin on
  this specific edge case.

### Resolutions (recorded during Phase 1 implementation, 2026-09-28)

- **`top_logprobs` range is 0-20**, confirmed against the OpenAI Python SDK's
  own type stubs (`openai/types/chat/completion_create_params.py`:
  `top_logprobs: Optional[int]` — "An integer between 0 and 20 ..."). Matches
  this plan's assumption; no change needed.
- **Sampled token vs. top-N**: `content[i]` is always the sampled token with
  its own raw logprob (even if outside the model's actual top-N);
  `top_logprobs` is always the true top-N of the raw distribution,
  independent of whether it contains the sampled token. `top_logprobs` is
  `[]` (not omitted) when the request's `top_logprobs` is 0 or absent, to
  match the OpenAI SDK's `ChatCompletionTokenLogprob.top_logprobs: List[...]`
  (a required, non-optional list in the response model).
- **`top_logprobs` without `logprobs: true`**: parsing skips the field
  entirely (no validation, no error, whatever its value/type) exactly when
  `logprobs` is not `true` — matches "silently ignored," including for a
  wrong-typed or out-of-range value in that case.
- **Reasoning-model / suppressed-token logprobs (§3.3, §4.3)**: resolved to
  the conservative default this plan already recommended — a token gets a
  `content[]` entry iff it was emitted as *visible* `content` text, i.e. iff
  it survived `StreamingReasoningFilter` and is not one of
  `outputSuppressedTokenIDs`. If the filter holds text back and releases it
  later, that text's logprob entries travel with it (not dropped, not
  duplicated) — the aggregation counts entries by visible-text emission, not
  by raw decode step.
- **Tool-call replies**: `"logprobs": null` on both the non-streaming choice
  and the single assembled SSE chunk for a tool-call turn. Not computed at
  all in Phase 1 (no partial data collected and discarded).
- **`bytes` derivation (§4.2)**: implemented as
  `KrillTokenizer.rawTokenBytes(for:)`, independent of `decodeForOutput`.
  A new `usesByteLevelPieces: Bool` property is resolved ONCE at load from
  `tokenizer.json`'s `pre_tokenizer`/`decoder` being (or containing, through
  a `Sequence`) a `ByteLevel` stage — not guessed per piece, because a raw
  piece made only of Latin-1-range characters is genuinely ambiguous between
  "literal SentencePiece text" and "GPT-2 byte-level-BPE encoding of a
  different byte" on its own. Order of checks in `rawBytes(forPiece:
  isByteLevelBPE:)`: (1) `<0xHH>` byte-fallback piece -> `[0xHH]`, checked
  regardless of tokenizer style; (2) if `usesByteLevelPieces`, every
  character of the piece round-trips through a GPT-2 byte<->unicode table
  (Qwen/Llama-3-family) -> those bytes (falls back to plain UTF-8 if a
  character is outside the table, which should not happen for a genuine
  byte-level-BPE piece); (3) if not byte-level and the piece contains `▁`
  (SentencePiece) -> `▁` becomes a space, rest is UTF-8 bytes of the piece;
  (4) otherwise, plain UTF-8 bytes of the piece text. The byte-unicode table
  is a small duplicate of `KrillCore/WhisperTokenizer.swift`'s
  `makeByteDecoder()` (KrillTokenizer does not depend on KrillCore). The
  lossy `token` string for a partial-UTF-8 byte-fallback token uses
  `String(decoding:as: UTF8.self)` (U+FFFD on invalid sequences), never the
  empty string `decodeForOutput` would give.

## 8. Registry: add Qwen3.5-4B (separate, small work item)

Unrelated to logprobs mechanically, bundled into this plan only because it
was flagged alongside it. **Do not block phase 1 above on this landing
first or vice versa — they touch disjoint files.**

The `.qwen35` `ModelFamily` case already exists and already runs
Ornith-9B, Qwythos-9B, and Qwen3.8-27B natively
(`Sources/KrillRegistry/AliasMap.swift:184-217`). Per
`docs/ADDING_MODELS.md`'s opening note, adding another model of an
**existing** family needs no runtime code change — the heavy "new family"
checklist further down that doc (the `ModelManifest`/`ModelCapabilities`/
`ModelAdapter`/`ModelProfiles` five-switch exercise) does not apply here.
This is purely a new `ResolvedModel` entry (or entries) in `AliasMap.swift`,
following the exact pattern of the `qwen3.8-27b` entry at lines 215-217, plus
a README.md model-list line (see the existing `qwen3.8-27b` line, `README.md`
around line 165, per `docs/ARCHITECTURE.md`'s note that the README model
section is "the user-facing summary").

Candidate repos exist on `mlx-community` as of this research (confirmed via
web search, not independently verified against the actual checkpoint):
`mlx-community/Qwen3.5-4B-MLX-4bit`, `-MLX-8bit`, and `-MLX-bf16`. Whether to
mirror one under `srv-sngh` first (as Ornith/Qwythos/Qwen3.8-27B all are) or
point directly at `mlx-community` is a judgment call the implementing session
should make by checking whether the checkpoint needs any Krill-specific
re-packaging (vision tower stripped, MTP head stripped, etc. — see the
`qwen3.8-27b` entry's comment for what that repacking looked like last
time). **Verify the repo actually loads with Krill's native `qwen35`
architecture detection before publishing the alias** — a 4B model may have
config differences (layer count, `mrope_section`, GatedDeltaNet ratio) from
the 9B/27B members already supported; do not assume it "just works" without
a real load test.

## 9. Out of scope (this plan)

- Grammar-constrained decoding interaction with logprobs (a grammar-masked
  token's logprob under the mask vs. under the raw distribution) — not asked
  for, not obviously well-defined, leave for a follow-up if a consumer needs
  it.
- Embeddings/reranker endpoints — logprobs is a generation-only concept.
- Any change to `krill run`/`krill code`'s terminal UI to *display*
  logprobs — this plan is API-surface only; a UI consumer is a separate,
  later piece of work.
- Licensing of any model mentioned in §8 — out of scope for this repo's
  process by standing instruction.

## 10. How to start

Read, in this order:

1. This file.
2. `Sources/KrillServer/ServerParsing.swift` (lines 1-160 for the request
   structs and the two `unsupported*Fields` sets this plan removes entries
   from).
3. `Sources/KrillEngine/TokenEvent.swift` (the whole file — it's short — to
   see exactly what a decode step currently carries out of the engine).
4. `Sources/KrillSampler/Sampler.swift` (the whole file — also short — this
   is where §4.1's semantics decision gets implemented).
5. `Sources/KrillEngine/InferenceEngine.swift` around lines 738-830
   (`generate()`'s two overloads) and 1900-1930 (the plain decode loop's
   sample call) to see the exact plumbing path for phase 1.
6. `docs/SPECULATIVE_DECODING.md` in full, and
   `Sources/KrillEngine/ContinuousBatcher.swift` lines 1-60 and 600-700, for
   phase 2 context even if not implementing it yet.
7. `docs/PITFALLS.md` #10 and `docs/RELEASE_GATE_DECODE_PROPOSAL.md`, so the
   "must not regress the hot path" requirement is understood before writing
   the first line of engine code, not discovered at benchmark time.

**Branch and PR rules** (there is no `AGENTS.md` or `CLAUDE.md` in this
repo — this repo's process lives in `docs/RELEASING.md` and the CI
workflows):

- `main` is protected in practice by the CI/PR flow: `.github/workflows/
  swift-tests.yml` runs on every `pull_request` and on `push` to `main`;
  `docs/RELEASING.md`'s proven flow is branch → PR → CI green → merge, "one
  PR, not a separate SHA-fix PR." This session found no committed local git
  hook enforcing a from-`main` block (the `.git/hooks/` directory here has
  only the stock `.sample` files) — treat the PR/CI requirement as the real
  gate regardless, and **branch first**:
  ```bash
  git checkout main && git pull && git status   # must be clean
  git checkout -b feat/logprobs   # or feat/logprobs-phase1, etc.
  ```
- Build/test before opening a PR: `make test` (runs `swift build
  --build-tests`, builds the metallib, then `swift test`). `make
  bench-release-gate` for the regression check in §6 item 4.
- Do not use `--no-verify`, do not force-push, do not commit to `main`
  directly — standard repo hygiene, not anything specific to this feature.
