# Logprobs Support Plan (OpenAI/Ollama `logprobs` + `top_logprobs`)

Created: 2026-09-28
Status: Phase 1 implemented on `feat/logprobs-phase1` (chat `logprobs` +
`top_logprobs`, non-streaming + streaming, plain decode path; spec/batch
fallback). Phases 2-3 (this doc's §7) not started. See §4/§7 for the
settled decisions and open-question resolutions recorded during
implementation. A code-review pass on the PR found and fixed a critical
under-reporting bug in the logprobs aggregator, an unnecessary host round
trip in the sampler, and two harness bugs in the parity script (see the
end of §4/§7's resolutions) — see the PR description for the full list and
final numbers. A 2026-09-29 follow-up fix (`fix/logprobs-qwen35`) closed a
real Phase-1 gap found in production use: `logprobs.content` was always `[]`
for every qwen3_5-family model (qwen3.5-4b, Ornith-9B, Qwythos-9B,
Qwen3.8-27B) because they route through a native VL decode runtime that
never threaded `wantLogprobs` — see the "Verification results
(2026-09-29)" subsection after §7's Resolutions for the root cause, the
numeric parity re-run, and the no-slowdown re-check. A 2026-09-30 follow-up
(`feat/logprobs-ollama-completions`) closed this plan's remaining
Phase-1-documented gap for `logprobs`/`top_logprobs`: Ollama `/api/chat` +
`/api/generate`, and legacy `/v1/completions`'s `logprobs` (int, 0-5; NOT
`echo`, still Phase 3) — see the "Ollama + legacy completions (2026-09-30)"
section after §7's Verification results for the pinned wire formats and
test evidence.
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
- **Reasoning-model / suppressed-token logprobs (§3.3, §4.3)**: a token gets
  a `content[]` entry iff it reaches the visible answer, i.e. iff it
  survived `StreamingReasoningFilter` and is not one of
  `outputSuppressedTokenIDs`. The FIRST implementation of this (landed with
  the rest of Phase 1) used a same-call/same-length shortcut that under-
  reported in two real, non-rare cases — flagged in code review and fixed
  in the same PR before merge: (1) any token the filter HOLDS while
  disambiguating a possible tag prefix (it buffers on a bare `<`, which
  shows up constantly in code — `x < y`, `<div>`, generics) got silently
  dropped along with its neighbour, and (2) a token whose own
  `decodeForOutput` text is empty (a byte-fallback/partial-UTF-8 piece of a
  multi-byte character) never got an entry at all, defeating the `bytes`
  field's whole purpose. `LogprobsAggregator` (`Sources/KrillServer/
  LogprobsFormatting.swift`) now keeps a FIFO of pending (tokenId, info,
  own text) tuples and drains it against `StreamingReasoningFilter`'s own
  emit/discard decisions — reconstructed via a new `pendingUTF8Length`
  introspection property plus the actual emitted text, in UTF-8 byte units
  (not `Character` counts, which are not additive under concatenation for
  scripts like Devanagari) — so every visible token gets exactly one entry,
  in the right order, including ones only resolved at `finish()` (a
  `max_tokens`-truncated stream ending mid-hold no longer loses that
  token). Streaming: a chunk that resolves entries but has no text of its
  own (an empty-decode token) is now still sent, carrying just the
  entries. Covered by 11 `LogprobsAggregatorTests` (English; code with
  `<`/`x < y`/an HTML tag; Devanagari; emoji; a `<think>` block; text held
  and flushed at end-of-stream; empty-decode tokens; suppressed tokens) and
  a real end-to-end check (`tools/logprobs_e2e_check.py`) against
  `llama-3.2-1b` confirming `bytes`-concat reproduces `content` byte-for-
  byte and streaming/non-streaming `content` match exactly, for both a
  Hindi and a code-with-`<` prompt.
- **Sampler cost (§5.2)**: the first implementation of `sampleWithLogprobs`
  paid a full-vocabulary host round trip every step (`asArray(Float.self)`
  then rebuilding an `MLXArray`) purely to get an object independent of
  `applyPenalties`' in-place scatter — flagged in review as unnecessary for
  a vocab up to 262k wide. Fixed by building the raw log-softmax graph from
  the pre-penalty logits BEFORE `applyPenalties` runs: MLX operations
  capture their input's value at the call site into a new, independent
  result object, so this ordering alone protects the reported logprob from
  the scatter's later in-place mutation of the SAME Swift object (the real
  cause of the aliasing bug the host round trip was working around) — no
  copy, host or device, needed. One combined `eval()` now covers the
  chosen token, its logprob, and the top-N.
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
- **Numeric parity investigation (§6 item 3)**: the first parity run
  (`tools/logprobs_parity.py` against `llama-3.2-1b` 4-bit) reported a
  sampled-token max diff of 2.34e-2 and a top-N alternate max diff of 0.95
  nats, with no noise floor to compare against — not itself conclusive.
  Investigation found two harness bugs, both fixed: (1) the script
  teacher-forced its walk through the model with **mlx_lm's own greedy
  pick**, not Krill's actual sampled id, so any single divergence (an
  ordinary near-tie flip from fp noise) put every later position on a
  genuinely different context than the one Krill's own forward pass
  conditioned on, manufacturing large compounding diffs unrelated to
  Krill's own correctness; (2) `build_bytes_to_id` resolved a vocabulary
  piece's bytes with plain last-wins dict assignment, silently mis-
  resolving any byte-string two pieces happen to share — fixed with
  explicit collision detection (ambiguous byte strings are excluded, not
  guessed). The script also now measures mlx_lm's own intrinsic noise floor
  (one full-sequence forward vs. incremental KV-cache decode, both inside
  mlx_lm, no Krill involved) as the yardstick to compare Krill against.
  Re-run after both fixes, on `llama-3.2-1b` 4-bit (128256-token vocab, 40
  generated tokens, no byte collisions found): floor max diff 1.5e-2 nats;
  Krill vs. mlx_lm-incremental sampled-token max diff 2.52e-2 (1.68x the
  floor — within the ~2x expectation), median diff 5.6e-4 (excellent
  agreement typically). Alternates (non-sampled top-N entries) showed
  larger diffs (median 4.3e-2, max 0.60 nats) — expected from log-softmax's
  own sensitivity, not a separate bug: a dominant/sampled token's logprob
  is comparatively self-stabilizing (its own logit drives most of the
  `logsumexp` it's subtracted from), while a low-probability alternate's
  logprob is driven by the DIFFERENCE of two independently-noisy logits
  with no such cancellation, at positions where the probability itself is
  already tiny, so the same underlying per-step logit noise shows up
  larger in the reported nats. A second run against `qwen3-0.6b` 4-bit
  (151669-token vocab) initially showed a catastrophic 28.5-nat mismatch at
  the very first generated token — traced to a THIRD harness gap, not a
  Krill bug: `apply_chat_template`'s default `enable_thinking` differs from
  what Krill actually used, so the reconstructed prompt itself was wrong
  (Qwen3's template inserts an empty `<think>\n\n</think>\n\n` scaffold
  under `enable_thinking=False`, changing every position's context). Once
  called with the matching kwarg, Qwen3-0.6b's diff (max 0.099, 0.89x its
  own floor of 0.111) was, if anything, tighter than the floor. Neither an
  unquantized (bf16/fp16) model comparison nor a permanent harness fix for
  per-family chat-template kwargs was completed: the working machine had
  only ~1.3 GB of free disk at investigation time (task instructions bar
  downloading anything over 3 GB and, more fundamentally, filling the disk
  further was unsafe regardless of the model's size), and a mid-investigation
  benchmark side-comparison actually ran the disk to 0 bytes free via a
  `.build` directory copy, confirming the margin was real. Re-run the
  bf16/fp16 comparison from §6 item 3, and generalize `logprobs_parity.py`
  to auto-detect per-family template kwargs, once disk headroom allows.

### Verification results (2026-09-29)

**Bug fixed: `logprobs.content` was always `[]` for every qwen3_5-family
model.** Root cause (`Sources/KrillEngine/InferenceEngine.swift`, the
`Qwen35VLForConditionalGeneration` intercept in `generate(messages:)`,
originally added around line 894): `qwen3.5-4b`'s checkpoint config.json
carries `vision_config`/`image_token_id` even though it is used purely as a
text model (confirmed: `python3 -c "import json; print('vision_config' in
json.load(open('config.json')))"` on the local blob -> `True`).
`ArchitectureDetection`'s `qwen3_5` rule (`Sources/KrillCore/
ArchitectureDetection.swift:280-297`) treats that key's presence as "this
checkpoint needs the VL loader" and calls `loadQwen35VL`, building a
`Qwen35VLForConditionalGeneration`. Because of that, `generate(messages:)`
routes EVERY request for this model - image or text-only - through the
dedicated `generateQwen35VL` / `Qwen35VLRuntime.generate` native runtime
(the same runtime Ornith-9B/Qwythos-9B/Qwen3.8-27B use), a completely
separate decode loop from the generic dense path Phase 1's `wantLogprobs`
plumbing was wired into. Neither `generateQwen35VL` nor `Qwen35VLRuntime.
generate` accepted `wantLogprobs`/`topLogprobs` before this fix, and
`Qwen35VLRuntime`'s `onToken` callback only ever passed a bare token id, so
every `TokenEvent` for these models had `logprob: nil` regardless of the
request - the SSM-cache guard (`hasSSMCacheSpec`) the original bug note
suspected is real but irrelevant here: it never even runs for a request
that took this earlier VL intercept. Fixed by threading `wantLogprobs`/
`topLogprobs` through `generateQwen35VL` into `Qwen35VLRuntime.generate`,
which now calls `Sampler.sampleWithLogprobs` (same one-step-behind
"compute at sample time, attach at yield time" convention as the generic
loop's `pendingLogprobInfo`) when requested, and passes the resulting
`TokenLogprobInfo?` to `onToken`.

**Other paths audited for the same gap** (native, model-family-specific
decode runtimes that bypass the generic loop the same way qwen3_5 did):
`generateQwen25VL`/`Qwen25VLRuntime` (Qwen 2.5-VL, always routes here per
its own comment "Image and text-only requests both route here"),
`generateLlamaVision`/`MllamaRuntime` (Llama-3.2-Vision, same "always"
routing), `generateLocateAnything` (LocateAnything-3B, same), and
`generateMuseGlimmer` (Muse Glimmer, image requests only - its text-only
path already uses the generic loop and is unaffected). All four have the
IDENTICAL structural gap: `onToken: { token in ... }` closures with no
logprob threading, and none of their `generate...` functions accept
`wantLogprobs`/`topLogprobs`. None of these were fixed in this change - the
reported bug and its repro are entirely about qwen3_5, and fixing four more
independent native runtimes (each its own file, its own `Output` struct,
its own sampling call sites) is out of scope for this PR. Recorded here as
a known, real, same-class gap for a follow-up: a request with
`logprobs: true` against any Llama-3.2-Vision, Qwen 2.5-VL, or
LocateAnything-3B checkpoint, or an IMAGE request against Muse Glimmer,
still silently returns `logprobs.content: []` today.

**Regression test**: `Tests/KrillEngineTests/Qwen35LogprobsTests.swift`,
checkpoint-gated on `KRILL_QWEN35_MODEL_PATH`/`KRILL_ORNITH_MODEL_PATH`
(same pattern as `Qwen35VLSmokeTests`). Asserts every generated token
carries a `TokenLogprobInfo` with the requested `top_logprobs` count, and
that a greedy request's sampled token is exactly its own raw distribution's
top-1 alternate (catches a shallow "populate something non-nil" fix, not
just presence). Both new tests pass against the real `qwen3.5-4b` 4-bit
checkpoint post-fix; both assertions fail against the pre-fix code (every
`event.logprob` was `nil`).

**Task 1 end-to-end verification**, real `qwen3.5-4b` served via
`krill serve --model qwen3.5-4b`:
- Exact repro from `docs/LOGPROBS_QWEN35_EMPTY.md`
  (`max_tokens:1, top_logprobs:5, temperature:0`, `KRILL_ENABLE_THINKING=0`)
  now returns `choices[0].logprobs.content` with 1 entry (was `[]`).
- `tools/logprobs_e2e_check.py` (bytes-concat == visible content, streaming
  `content` == non-streaming `content`) against `qwen3.5-4b`,
  `KRILL_ENABLE_THINKING=0`: **all checks pass** (Hindi prompt: 6 entries;
  code-with-`<` prompt: 15 entries).
- Same script with `KRILL_ENABLE_THINKING=1` and `max_tokens` raised to 400
  (a copy of the script with `max_tokens=64` -> `400`, since the default
  budget is consumed entirely by the reasoning block on some prompts - see
  below): code-with-`<` prompt passes (15 entries, streaming ==
  non-streaming, bytes-concat exact). The Hindi prompt produced `content:
  ''` / `finish_reason: length` even at `max_tokens: 900` - reproduced
  IDENTICALLY with `logprobs` entirely absent from the request, i.e. this
  specific model+prompt combination spends its whole token budget inside
  `<think>...</think>` and never reaches a visible answer. Pre-existing
  reasoning-verbosity/token-budget behavior, unrelated to logprobs and not
  a regression from this fix.
- No regression on `llama-3.2-1b` (4-bit, unaffected dense-family
  request): same e2e script, all checks pass (6 and 14 entries).

**Task 2 numeric parity** (`tools/logprobs_parity.py`, extended in this
change to report median abs diff alongside max, and to accept an optional
`enable_thinking` CLI arg so the harness can match Krill's actual template
kwarg instead of silently taking the tokenizer's own default - the "third
harness gap" the 2026-09-28 investigation flagged as unresolved). Greedy,
`top_logprobs=5`, 40 generated tokens per model, prompt "Explain in two
sentences why the sky is blue.":

| Model | Floor max / median | Krill sampled max / median | Krill alt max / median | Ratio (sampled/floor) | Byte collisions |
|---|---|---|---|---|---|
| `Qwen3-0.6B-bf16` (`enable_thinking=0`) | 1.537e-1 / 7.721e-3 | 7.337e-2 / 1.915e-3 | 3.016e-1 / 1.815e-2 | 0.48x | 0 |
| `Llama-3.2-1B-Instruct-bf16` | 2.076e-2 / 2.756e-4 | 5.777e-2 / 3.948e-4 | 7.500e-1 / 6.260e-2 | 2.78x | 0 |
| `qwen3.5-4b` (4-bit, `enable_thinking=0`, post-fix) | 9.206e-2 / 2.114e-3 | 4.041e-2 / 8.440e-4 | 3.750e-1 / 6.219e-2 | 0.44x | 0 |

`Qwen3-0.6B-bf16` and `qwen3.5-4b` both land BELOW their own mlx_lm
self-consistency floor (ratio < 1) - excellent agreement, and for
`qwen3.5-4b` specifically this also confirms the fix is not merely
"populates a field" but numerically correct (matches the SSM-hybrid raw
distribution `mlx_lm` itself computes).

`Llama-3.2-1B-Instruct-bf16` needed one prerequisite fix to run at all: it
crashed with `Fatal error: [scaled_dot_product_attention] Mask type must
promote to output type bfloat16` on EVERY generation request (including the
load-time warmup pass), unrelated to logprobs - reproduced identically with
`logprobs` absent. Root cause: `LlamaModelInner.callAsFunction`
(`Sources/KrillCore/LlamaModel.swift`, ~line 52) built its causal mask via
`createCachedCausalMask(newLen:cacheLen:)` without passing `dtype:`, so it
defaulted to `.float16` - fine for the already-supported int4-quantized
`llama-3.2-1b` alias (whose dequantized compute dtype is already float16,
so the default happened to match), but wrong for a genuinely bf16
checkpoint, where MLX's fused SDPA kernel requires the mask dtype to
promote to the attention output's dtype and float16 does not promote to
bfloat16. Fixed by passing `dtype: x.dtype`, matching the pattern every
other family's call site already uses (`DeepSeekModel`/`Glm4Model`/
`MixtralModel`/`Qwen3MoEModel`/etc. all pass `x.dtype`/`h.dtype`). Zero
behavior change for the existing int4 `llama-3.2-1b` alias (`x.dtype`
resolves to the same `.float16` the old hardcoded default gave).

Regarding the plan's original "suggest 1e-3 in log-space" target (§6 item
3): `Llama-3.2-1B-Instruct-bf16`'s own mlx_lm self-consistency FLOOR
(full-sequence vs incremental, no Krill involved) is 2.076e-2 - already
20x looser than 1e-3. This is an intrinsic property of bf16 arithmetic
(7 mantissa bits vs float16's 10), not a Krill defect: the 1e-3 target was
calibrated before anyone had tried a genuinely bf16 (as opposed to
int4-dequantized-to-fp16) checkpoint. Krill's own diff-to-floor ratio
(2.78x) is close to the same ~2x-ish range already accepted for the
llama-3.2-1b 4-bit run recorded in the Resolutions section above (1.68x),
and both `median abs diff` values (3.948e-4) are tight; the `max` statistic
is dominated by a small handful of positions out of 40 and swings easily.
No fix attempted beyond the SDPA mask dtype crash above - closing the
remaining gap to the floor would mean computing in float32 throughout
(a materially bigger, riskier change, and arguably WRONG - it would no
longer report what the bf16 model actually computed). Diagnosis, not a
bug: report as-is per the task's own instruction for this case.

**Task 3 no-slowdown check** (`krill bench <model> --runs 3 --gen-len 128`,
alternating baseline `7f8a506` (A) and this fix (B), A/B/A/B/A/B, in a
scratch worktree removed afterward). The machine was NOT actually idle
during this run - a long-running `kreach-crawler` background process (an
unrelated project, pre-existing, left running per this session's
instructions) and several other concurrent agent sessions were competing
for CPU/memory throughout, which shows up directly in the numbers below as
a large mid-run step change affecting BOTH builds identically (strong
evidence the swing is environmental, not code-related).

`llama-3.2-1b` (4-bit), decode tok/s per round: A 26.6, B 22.9, A 18.8,
B 22.9, A 21.8, B 19.4. A mean 22.4 (range 7.8), B mean 21.7 (range 3.5) -
prefill and TTFT show the same magnitude of run-to-run spread. The A-B mean
difference (0.7 tok/s) is far smaller than the intra-group spread; no
directional regression.

`qwen3.5-4b` (4-bit), decode tok/s per round: A1 6.7, B1 6.4 (both hit the
same severe ambient slowdown - 15+ second TTFT on both builds, a swap/
contention event unrelated to either binary), A2 32.9, B2 36.1, A3 35.3,
B3 36.6. Restricting to the three rounds after the machine settled (A2/A3
vs B2/B3): A mean decode 34.1 tok/s, B mean 36.4 tok/s - B is NOT slower;
prefill (A 118.7-88.6, B 117.5-118.8) and TTFT (A 7367-4315ms, B
4360-4308ms) tell the same story. No slowdown from this fix when logprobs
is not requested, on either model.

**Logprobs-ON decode tok/s, for reference** (fix build, `krill serve` +
streaming HTTP client measuring inter-chunk timing, `max_tokens: 128`,
greedy, `top_logprobs: 5` when on - NOT the same measurement method as the
`krill bench` numbers above, which call `model.forward`/`Sampler.sample`
directly and do not go through `InferenceEngine`/the server at all, so the
absolute numbers are not comparable across the two tables):

| Model | logprobs off | logprobs on | Overhead |
|---|---|---|---|
| `llama-3.2-1b` (4-bit) | ~163.8 tok/s | ~64.2 tok/s | ~2.6x slower |
| `qwen3.5-4b` (4-bit) | ~33.3 tok/s | ~32.7 tok/s | ~2% slower |

The overhead is real and expected when `logprobs` IS requested (a
128k-261k-vocab log-softmax + top-N gather every step, plus a materially
larger SSE payload per chunk) - it is NOT a violation of §5.2's "zero cost
by default" requirement, which is specifically about the case logprobs is
absent (verified above). The overhead is proportionally much larger on
`llama-3.2-1b` because its per-step forward pass is tiny (1B params), so a
fixed extra cost dominates; on `qwen3.5-4b`'s much heavier 4B hybrid-SSM
forward pass the same fixed cost is nearly invisible.

## Ollama + legacy completions (2026-09-30)

Closes this plan's Phase-1 known gap ("Legacy `POST /v1/completions`...and
the Ollama `/api/chat`/`/api/generate` dialects do not parse these fields
yet") for the Ollama dialect and the legacy completions endpoint's
`logprobs` (not `echo` — that stays Phase 3, §5.4, unchanged). Scope: server
+ parsing + formatting + tests + docs only, per the task brief for this
follow-up; `Sources/KrillEngine`/`Sources/KrillSampler` were not touched
(another agent's concurrent work), and none was needed — both surfaces
reuse the exact same `InferenceEngine.generate(wantLogprobs:topLogprobs:)`
plumbing and `LogprobsAggregator` Phase 1 already built.

### Pinned wire formats (verified against primary sources before coding)

**Ollama `/api/chat` + `/api/generate`** — `docs.ollama.com/api/chat`,
`docs.ollama.com/api/generate` (WebFetch, 2026-09-30), cross-checked against
the Go source `github.com/ollama/ollama/api/types.go` and the real `ollama`
Python package (`pip install ollama`, version 0.6.3, installed trivially
into `/Users/sourav/.krill/venv` — its `_types.py` matches the Go source
exactly):

- Request: `logprobs: bool`, `top_logprobs: int` (0-20) — **top-level
  fields, NOT inside `options`** (confirmed by both the docs pages and the
  Python client's `ChatRequest`/`GenerateRequest` field list; `options` only
  holds sampling knobs). Ollama's docs don't state a default; Krill treats
  absent as `false`/`0`, same as every other boolean/int field here.
- Response, Go struct (`api/types.go`):
  ```go
  // ChatResponse / GenerateResponse
  Logprobs []Logprob `json:"logprobs,omitempty"`

  type TokenLogprob struct {
      Token   string  `json:"token"`
      Logprob float64 `json:"logprob"`
      Bytes   []int   `json:"bytes,omitempty"`
  }
  type Logprob struct {
      TokenLogprob
      TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
  }
  ```
  Three things this pins that the docs pages alone left ambiguous:
  1. `logprobs` is a **top-level field of the response object** (a sibling
     of `message`/`response`, `done`, etc.), not nested inside `message`.
  2. **Every one of these fields is `omitempty`** — an empty `top_logprobs`
     list is omitted per-entry, and the whole `logprobs` array is omitted
     from the response/chunk when there's nothing to report. This is the
     opposite of the OpenAI chat endpoint's convention (always-present
     `logprobs: null`/`"content": []`), confirmed as deliberate by checking
     three independent sources (docs, Go source, Python client) agreeing.
  3. The Python client's own `TokenLogprob`/`Logprob` pydantic models
     (`ollama/_types.py`) do NOT declare a `bytes` field at all — but
     pydantic's default `extra="ignore"` means sending `bytes` anyway (to
     match the real Go server) doesn't break the client; verified by an
     actual round-trip (see Tests below) rather than assumed.
  - Streaming placement: neither the docs nor the Go source say whether
    `logprobs` is per-chunk or cumulative, or whether it appears on the
    final `done: true` object. Krill's choice (documented, not guessed):
    each NDJSON line's `logprobs` covers only the token(s) newly resolved
    in that line (same per-chunk-not-cumulative convention as the chat SSE
    path), and the final `done: true` line never carries `logprobs` (there
    is nothing new to report there — any trailing held-token entries are
    flushed onto their own preceding line first, mirroring the chat SSE
    path's tail-chunk handling).

**Legacy OpenAI `/v1/completions`** — the OpenAI Python SDK's own type
stubs in `/Users/sourav/.krill/venv` (authoritative, matches the task
brief's instruction to check them over the docs site):
  - `openai/types/completion_create_params.py`: `logprobs: Optional[int]`,
    doc comment: *"Include the log probabilities on the `logprobs` most
    likely output tokens, as well the chosen tokens... The API will always
    return the `logprob` of the sampled token, so there may be up to
    `logprobs+1` elements in the response... **The maximum value for
    `logprobs` is 5.**"* This resolves §3.2's still-open "0-5 vs 0-20"
    question: it's **5**, a real and deliberate difference from chat's
    `top_logprobs` (0-20), not a typo carried over from the newer endpoint.
  - `openai/types/completion_choice.py`:
    ```python
    class Logprobs(BaseModel):
        text_offset: Optional[List[int]] = None
        token_logprobs: Optional[List[float]] = None
        tokens: Optional[List[str]] = None
        top_logprobs: Optional[List[Dict[str, float]]] = None

    class CompletionChoice(BaseModel):
        finish_reason: Literal["stop", "length", "content_filter"]
        index: int
        logprobs: Optional[Logprobs] = None
        text: str
    ```
    This pins the one shape-level surprise: `top_logprobs` is a list of
    **`{token: logprob}` STRING-KEYED DICTS**, one per generated-token
    position — not chat's separate `{token, logprob, bytes}` object array.
    The "`logprobs`+1" comment resolves this plan's open question about
    whether the sampled token is folded into the same structure as the
    alternates (yes, here — unlike chat, where it's a separate top-level
    field of the `content[i]` entry either way).

### Implementation (`Sources/KrillServer/` only)

- `ServerParsing.swift`: `ServerCompletionRequest.logprobs: Int?` (nil =
  not requested, distinct from an explicit `0`); new
  `legacyCompletionsLogprobsValue` validator (0-5, `invalidType`/
  `invalidValue` matching the existing pattern); removed `"logprobs"` from
  `unsupportedOpenAICompletionFields` (`"echo"` stays rejected — Phase 3).
  `ServerChatRequest.wantLogprobs`/`topLogprobs` (already existed for the
  OpenAI dialect) are now also populated by `ollamaChatRequest`; new
  `ServerGenerateRequest.wantLogprobs`/`topLogprobs`, populated by
  `ollamaGenerateRequest`. Both Ollama parsers read the fields **top-level**
  (never from `optionsObject(from:)`), and both apply the same
  "`top_logprobs` silently ignored without `logprobs: true`" convention
  Phase 1 established for the OpenAI dialect (a Krill-wide house rule now,
  not restated per dialect in the docs).
- `LogprobsFormatting.swift`: `ollamaLogprobEntryJSON` (strips an empty
  `top_logprobs` key to match Go's `omitempty`) and `ollamaLogprobsArrayJSON`
  (maps a `LogprobsAggregator`'s `entries` through it) for the Ollama shape;
  `legacyCompletionLogprobsJSON` builds the flat `{tokens, token_logprobs,
  top_logprobs, text_offset}` object from the same `entries` — all three
  reuse the chat endpoint's `logprobsContentEntry`/`entries` machinery
  unchanged, so reasoning-block exclusion, suppressed-token exclusion, and
  `bytes` derivation (§4.2) are identical across every dialect by
  construction, not by parallel re-implementation.
- `Server.swift`:
  - `handleCompletions` (`/v1/completions`): builds a `LogprobsAggregator`
    only when `request.logprobs != nil` (same "pay nothing when off"
    pattern as `handleNonStreamingCompletion`); the `logprobs` key is added
    to the response ONLY in that case — never `NSNull()` — so a request
    that never sets `logprobs` gets byte-for-byte the same response as
    before this change existed. This endpoint has no streaming support at
    all (`stream: true` already 400s at parse time, pre-existing and
    unrelated to logprobs), so there is no streaming case to implement here.
  - `handleOllamaChat` / `handleOllamaGenerate`: the streaming path's bare
    `StreamingReasoningFilter` is replaced with a `LogprobsAggregator(...,
    enabled: wantLogprobs)` — a pure passthrough to the same filter when
    disabled, so the off-path emits byte-identical NDJSON lines via the
    exact same fast manual-string-building code as before (the
    JSONSerialization-based path only runs when `wantLogprobs` is true, a
    genuinely new code path with no old behavior to match). The final
    `done: true` line never carries `logprobs`; a resolved-but-empty-text
    chunk (an empty-decode token) still gets its own line so its entries
    aren't lost, mirroring the chat SSE tail-chunk rule. Non-streaming:
    `response["logprobs"]` is set only when `!entries.isEmpty` (Go
    `omitempty`, and incidentally also the simplest way to guarantee
    byte-identity when off, since the key is never touched at all in that
    case).
  - `handleToolChat`: `wantLogprobs` is now `request.wantLogprobs` for
    BOTH dialects (was `style == .openAI && request.wantLogprobs`). Ollama
    tool-call replies **omit** `logprobs` entirely on a `tool_calls` turn
    (the Go `omitempty` equivalent of chat's `logprobs: null`); a
    plain-content reply (tools offered, not used) gets the real entries,
    same rule as chat. The streaming tool-chat path needed no code change
    for Ollama — it already re-serializes the whole assembled `response`
    dict as one NDJSON line, so a `logprobs` key set on that dict rides
    along automatically.
  - `BatchScheduler.submit`'s existing `if wantLogprobs { return serial() }`
    gate (Phase 1) applies here unchanged: Ollama/legacy-completions
    requests route through the identical `runGenerate(...)` →
    `engines.scheduler(for:)?.submit(wantLogprobs:topLogprobs:)` call every
    other dialect uses, so a logprobs request on any of these three
    endpoints was already routed to the plain decode path with no
    endpoint-specific gating code needed — confirmed by reading
    `BatchScheduler.swift` rather than assumed, and covered by the existing
    (unmodified, still-passing) `BatchSchedulerTests.swift`.

### Tests / verification (real numbers)

**Unit** (`Tests/KrillServerTests/ServerTests.swift`,
`ServerFormattingTests.swift`): parsing acceptance/range/type/ignored-when-
disabled for both Ollama dialects and legacy completions (incl. confirming
Ollama's fields are top-level, not read from `options`; `raw: true` and
`echo: true` remain rejected exactly as before); golden-shape tests for
`ollamaLogprobEntryJSON`/`ollamaLogprobsArrayJSON`/
`legacyCompletionLogprobsJSON` including the `{}`-when-`logprobs:0` case and
the "sampled token folded in, not duplicated when already a top alternate"
case. `make test`: **1744 tests, 135 skipped, 0 failures** (includes
`AgentSessionTests`, which this session's investigation confirmed carries a
known ordering race under CPU contention per this repo's own memory notes —
0 failures on this run, re-run alone if a future run shows flakes there).

**Real server** (`krill serve` from this branch's release build, port
57480, both `llama-3.2-1b` 4-bit and `qwen3.5-4b` 4-bit — the same
qwen3_5-family model the 2026-09-29 fix targeted, confirming this follow-up
composes cleanly with that fix), via a new reusable script,
`tools/logprobs_ollama_completions_e2e_check.py`, plus ad hoc scripts for
the tool-chat cases: for both models, greedy, a code-with-`<` prompt (the
same reasoning-filter stress case Phase 1 used) —

- `/v1/completions`: `tokens` joined == `text` exactly; all four parallel
  arrays (`tokens`/`token_logprobs`/`top_logprobs`/`text_offset`) the same
  length; `text_offset` a correct cumulative sum of each token's own
  length; every position's `top_logprobs` dict contains the sampled token
  at the same value as `token_logprobs[i]` and never exceeds `logprobs+1`
  entries; a request without `logprobs` has NO `logprobs` key at all
  (checked as key absence, not null). `logprobs: 0` end-to-end: verified
  `top_logprobs` is `[{}]` (not omitted, not populated) with
  `token_logprobs` still populated — llama-3.2-1b: 14 tokens; qwen3.5-4b:
  15 tokens, both passed every check.
- `/api/chat` and `/api/generate`, non-streaming AND streaming: `bytes`-
  concat of every entry equals the UTF-8 bytes of the returned
  `message.content`/`response` exactly; streamed content equals
  non-streaming content; streamed entry count equals non-streaming entry
  count; a request without `logprobs` has no `logprobs` key. `/api/generate`
  with a `system` override + `logprobs: true` together: works (no
  rejection, no crash). Both models, both endpoints, both stream modes: all
  checks passed (llama-3.2-1b: 14 entries per endpoint; qwen3.5-4b: 15).
- **Cross-endpoint numeric parity** (the test that actually proves these
  three surfaces compute the SAME thing chat does, not just a
  similarly-shaped one): for the identical single-turn prompt, the first
  generated token's raw logprob from `/v1/completions`, `/api/chat`,
  `/api/generate`, and `/v1/chat/completions` (reference) — **exact 0.0
  nats difference**, all four, both models. (Getting this exact match
  required running the `/v1/chat/completions` reference call LAST in the
  script, after the other three had already warmed Krill's prefix cache
  with the identical prompt — running it first/cold showed a ~7e-3 nat
  diff, the SAME "full-prefill vs cache-hit-reforward" noise class this
  plan's Resolutions section already documents for `mlx_lm`
  full-sequence-vs-incremental runs, not a logprobs bug. Noted in the
  script's own comments so a future reader doesn't mistake cache warmth
  for correctness.)
- **Tool-chat** (`/v1/chat/completions` and `/api/chat`, both streaming and
  non-streaming, against `qwen3.5-4b`): `tool_choice: "required"` (OpenAI)
  → `logprobs: null`, single SSE chunk; `tool_choice: "none"` (OpenAI) →
  real entries. Ollama's `tool_choice` field is not parsed at all
  (pre-existing, unrelated to logprobs — every Ollama tool request runs
  `.auto`), so the equivalent check asserts the INVARIANT instead of
  forcing a branch: whichever way the model goes, a `tool_calls` `done_
  reason` OMITS `logprobs` entirely and a plain-content reply carries
  populated entries — observed both ways across the two prompts tried, both
  correct.
- **OpenAI SDK round-trip**: `client.completions.create(model=...,
  logprobs=5)` against `qwen3.5-4b` parses into the typed `Logprobs` object
  with no SDK validation error (`tokens`, `token_logprobs`, `text_offset`,
  and `top_logprobs: List[Dict[str, float]]` all present and correctly
  typed); a request without `logprobs` gets `choices[0].logprobs is None`.
  Streaming was not tested for this endpoint since Krill's `/v1/completions`
  has no streaming support to test (pre-existing, see above) — the task's
  "stream and non-stream" instruction does not apply here for a reason
  unrelated to this change.
- **`ollama` Python package**: trivially installable into
  `/Users/sourav/.krill/venv` (`pip install ollama`, no extra deps beyond
  what was already there) — not skipped. `ollama.Client(host=...).chat(
  model=..., logprobs=True, top_logprobs=3)` and `.generate(...,
  logprobs=True, top_logprobs=2)` against `qwen3.5-4b` both parse into
  typed `Logprob`/`TokenLogprob` objects with no error; a request without
  `logprobs` gets `response.logprobs is None`.
- **Benchmark gate**: not re-run for this follow-up. The change adds no new
  branch to the plain decode path's per-token hot loop (Ollama/legacy
  completions reuse Phase 1's `LogprobsAggregator`/engine plumbing
  unchanged); the only new code executes inside each endpoint's own
  request-handling `Task`, gated the same `if wantLogprobs`/`if let
  logprobsAgg` way Phase 1 already verified against the release-gate
  benchmark for the chat endpoint. Re-run `make bench-release-gate` before
  a release if a stricter proof is wanted.

### Limits / not done

- Legacy `/v1/completions` `echo` (prompt logprobs) — still Phase 3, needs
  the prefix-cache bypass in §5.4, unrelated to this follow-up's scope.
- Legacy `/v1/completions` streaming — does not exist in Krill at all
  (`stream: true` already rejected before this change); not something this
  follow-up could add without expanding scope well beyond "wire up
  logprobs", so left alone and documented rather than silently worked
  around.
- Ollama `tool_choice` is not parsed at all (pre-existing gap, confirmed
  while writing the tool-chat test above) — every Ollama tool request runs
  under `.auto`. Not this follow-up's bug to fix; noted here since it
  shaped how the tool-chat logprobs test had to be written (an invariant
  check instead of a forced-branch assertion).
- The four native VL/multimodal decode runtimes flagged as a known gap in
  the 2026-09-29 verification section (Qwen 2.5-VL, Llama-3.2-Vision,
  LocateAnything-3B, Muse Glimmer image requests) are unaffected by this
  follow-up either way: they still return `logprobs.content: []`/no
  `logprobs` key on EVERY endpoint (chat, Ollama, legacy completions) for
  the same reason as before — their `generate...` functions never accept
  `wantLogprobs` at all. Confirmed by re-reading, not re-tested against a
  real checkpoint (no new regression risk introduced, since this follow-up
  touches no engine code).

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
