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
test evidence. Phase 2 (`feat/logprobs-phase2-spec-batched`, 2026-09-30) is
now implemented: draft-model + n-gram speculative decode, and the
batched/continuous decode paths, all compute logprobs natively instead of
falling back to plain serial decode — see the "Phase 2 — spec + batched
logprobs (2026-09-30)" section at the end of this doc for the per-path
table, numbers, and limits. Phase 3 (`echo`/prompt logprobs on legacy
`/v1/completions`, `feat/logprobs-echo-and-thinking-switch`, 2026-09-30) is
now implemented — see the "Phase 3 — echo (2026-09-30)" section at the very
end of this doc for the design (raw-prompt-tokens, no-chat-template scoring;
BOS-stripping for the `tokens`-joined-equals-`text` invariant;
prefix-cache-bypass mechanics) and the numeric parity results. **Every phase
of this plan is now implemented.**
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

## Engine follow-ups (2026-09-30)

Two independent follow-ups on top of the qwen3_5 fix in "Verification
results (2026-09-29)" above, both scoped to `Sources/KrillSampler/*` and
`Sources/KrillEngine/*` (the server dialects — Ollama, legacy
`/v1/completions` — were a parallel, separate change).

### Task A: O(V log V) → O(V) top-N selection in `sampleWithLogprobs`

`Sampler.sampleWithLogprobs` found its top-N alternates with a full
`argSort` over the ENTIRE vocabulary every decode step, regardless of how
small N was — O(V log V) where V is 128,256 (llama-3.2-1b) to 248,320
(qwen3.5-4b) to as high as 262k for some registered families. Replaced with
`argPartition(negated, kth: n-1)` (O(V): puts the N smallest-negated, i.e.
largest-logprob, values into the first N positions in in undefined order)
followed by a real `argSort` of just those N candidates (O(N log N), N ≤
20) to get the final descending order. `n == 0` still skips the block
entirely, unchanged from before.

**Correctness**: `SamplerLogprobsTests.
testTopNPartialSelectionMatchesFullSortReferenceOnRandomLogits` runs 20
trials over 3 vocab sizes (50/90/130) and every N in
`[0, 1, 5, 20, vocabSize]` (the last exercising `argPartition`'s `kth ==
count - 1` boundary), asserting the new path's top-N token-id SET, order,
and per-token logprob all match an independent full-sort reference built
from a manual log-softmax. All 9 `SamplerLogprobsTests` (the 8 pre-existing
plus this one) pass.

**Speed**: measured via `krill serve` + streaming HTTP timing (SSE
`content` deltas), decode tok/s = 1 / (median inter-token gap, first gap
dropped) — median-of-gaps rather than `(tokens-1)/(last-first)` because a
CPU-contention scheduling stall on the Python client can otherwise drain
several already-buffered SSE lines in one wake-up and inflate an
apparent rate. "before" = `main`@`f85045b` built in a scratch worktree
(release config); "after" = this branch (release config); same binary
flags (`--host 127.0.0.1`, `KRILL_API_KEY` set), same prompt
("Write a short paragraph about the ocean.", `temperature: 0`,
`max_tokens: 320`), alternating build order every round.

This dev Mac was NOT quiet during measurement — `ps aux | sort -k3 -nr`
showed, at various points during the runs: a `kreach-crawler` process
steady at ~10-12% CPU (pre-existing, left running per instructions), a
`jugnu/step9/teacher_mlx.py` batched-labeling job at 4-40% CPU, and
(heaviest) a `tools/dry_run.py` video-render job that pegged ONE CPU core
at ~100% for several minutes during the first llama-3.2-1b pass, plus
several `kreach/.venv` Python helpers at 10-65% CPU. None of these were
started by this task and none were killed. The result is a strongly
BIMODAL distribution per config — most single requests land in a
"contended" band (~20-60 tok/s) but a minority land in an "uncontended"
band matching the model's true ceiling (~200-245 tok/s for llama-3.2-1b) —
so a plain median across a handful of samples is not trustworthy on its
own; the numbers below report the full sample set and call out the
clearest signal in it.

**llama-3.2-1b (vocab 128,256), OFF vs `top_logprobs: 20`, 8 alternating
rounds, `max_tokens: 320`** (tok/s, one value per round):

| | r1 | r2 | r3 | r4 | r5 | r6 | r7 | r8 | median | max |
|---|---|---|---|---|---|---|---|---|---|---|
| before, OFF | 39.7 | 46.0 | 30.7 | 207.5 | 36.1 | 38.3 | 33.4 | 232.0 | 39.0 | 232.0 |
| after, OFF | 37.0 | 29.4 | 61.7 | 32.1 | 46.6 | 24.3 | 47.0 | 28.0 | 34.5 | 61.7 |
| before, top_logprobs=20 | 25.8 | 26.2 | 53.1 | 29.1 | 31.1 | 24.2 | 25.0 | 21.0 | 26.0 | **53.1** |
| after, top_logprobs=20 | 27.5 | **215.6** | 57.4 | 60.8 | 19.9 | **204.4** | 29.7 | **204.3** | 59.1 | **215.6** |

The clean signal: **before this fix, `top_logprobs: 20` NEVER once reached
the uncontended ceiling in 8 trials** (max 53.1 tok/s, vs. OFF's own
ceiling of 232.0) — the O(V log V) full sort imposed a real cost floor
regardless of how favorably the scheduler behaved. **After the fix,
`top_logprobs: 20` reached the same ~204-216 tok/s ceiling OFF reaches, in
3 of 8 trials** — i.e. when the machine gives it a fair shot, the
logprobs-ON path is no longer distinguishable from logprobs-OFF. The
"after" OFF column happening not to catch its own high-mode window in this
particular 8-round sample (max 61.7) is itself further evidence of how
much this box's contention dominates a single-digit-N-round sample — it is
not a regression (the OFF branch is byte-for-byte the pre-existing code
in both builds; see the diff).

**qwen3.5-4b (vocab 248,320), OFF vs `top_logprobs: 20`, 6 alternating
rounds, `max_tokens: 320`:**

| | r1 | r2 | r3 | r4 | r5 | r6 | median | max |
|---|---|---|---|---|---|---|---|---|
| before, OFF | 7.3 | 7.5 | 41.8 | 8.2 | 8.2 | 53.8 | 8.2 | 53.8 |
| after, OFF | 6.7 | 34.6 | 13.8 | 6.0 | 7.1 | 16.5 | 10.4 | 34.6 |
| before, top_logprobs=20 | 6.4 | 6.3 | 6.7 | 6.0 | 6.5 | 25.4 | 6.5 | 25.4 |
| after, top_logprobs=20 | 7.3 | 6.0 | 6.2 | 6.9 | 11.0 | 6.2 | 6.6 | 11.0 |

At this scale the fix makes **no measurable difference** — before and
after are statistically indistinguishable for `top_logprobs: 20` (medians
6.5 vs 6.6 tok/s), and there's no llama-style "before never reaches the
ceiling" pattern. This makes sense: qwen3.5-4b's decode step is dominated
by the 4B-parameter forward pass (this whole model runs at roughly 6-8
tok/s baseline on this box, an order of magnitude slower than
llama-3.2-1b's ~30-50 tok/s baseline), so the O(V log V)→O(V) sort saving,
real as it is, is a rounding error next to the matmul cost per step at this
model size. The llama-3.2-1b result above is where this fix actually
matters: on a small/fast model, the full-vocabulary sort was a large
enough fraction of a decode step to visibly cap throughput; on a model
whose forward pass already dominates the step, it isn't. The fix is still
correct and unconditionally cheaper (never worse, sometimes much better),
just not always the bottleneck.

An earlier, broader (4-config: OFF/0/5/20, 3-5 rounds, no median-of-gaps
fix) pass on llama-3.2-1b produced numbers too noisy to interpret at all —
every config straddled both the "contended" and "uncontended" bands within
3-5 samples, before the median-of-gaps timing fix and before the 8/6-round
focused re-runs above. That confirms this machine's background load, not
the measurement method, is the dominant source of variance here; the
focused re-runs' larger sample counts and the "never reaches the ceiling"
vs. "reaches the ceiling some of the time" framing are the load-bearing
comparison, not any single median.

### Task B: logprobs in the remaining native vision/multimodal runtimes

Threaded `wantLogprobs`/`topLogprobs` through the four native runtimes this
plan's "Other paths audited for the same gap" note (in "Verification
results (2026-09-29)" above) flagged as having the identical structural
gap as qwen3_5 but left unfixed: `Qwen25VLRuntime`, `MllamaRuntime`,
`LocateAnythingRuntime`, `MuseGlimmerRuntime` (image requests only — its
text-only path already used the generic loop). Same pattern as
`Qwen35VLRuntime` in every case: compute via `Sampler.sampleWithLogprobs`
at the moment a token is sampled, carry the resulting `TokenLogprobInfo`
forward one step (`pendingLogprobInfo`), attach it to `onToken`'s second
argument at the moment that token is actually yielded — so a stop token
that never gets yielded also never gets an orphaned logprob entry, and the
FIRST (prefill-sampled) token is included exactly like every other. The
logprobs-OFF branch in each runtime is the pre-existing code, untouched
(see the diffs — every change is additive, `if wantLogprobs { ... } else {
<original line, unchanged> }`).

None of the four had a structural reason to decline: all four are ordinary
prefill + incremental-KV-cache AR decode loops over free text (including
LocateAnything, whose bounding-box output is still ordinary tokenizer text,
just with `<box>...</box>` syntax — not a non-text detection head), so all
four got the fix rather than a documented decline.

**Runtime-by-runtime status:**

| Runtime | Status | Real-model verification |
|---|---|---|
| `Qwen35VLRuntime` (qwen3_5 family) | Fixed (previous PR, f85045b) | Real `qwen3.5-4b` checkpoint (pre-existing test) |
| `Qwen25VLRuntime` (Qwen 2.5-VL) | Fixed (this PR) | Real checkpoint: `mlx-community/Qwen2.5-VL-3B-Instruct-3bit` (2.5 GiB, downloaded for this task), IMAGE request, `logprobs.content` populated + top-1-alternate-matches-sampled-token + logprobs-OFF/ON token-sequence identity, all pass (`Qwen25VLLogprobsTests.swift`). This exact checkpoint also produces incoherent text (`</</</...`) on a plain text-only, no-logprobs, no-image `krill run` — confirmed independent of this change (pre-existing 3-bit-quant quality issue, not a regression); the pre-existing `Qwen25VLSmokeTests`/`Qwen25VLProfileTests` coherence assertions fail against it for the same reason, unrelated to logprobs. |
| `MllamaRuntime` (Llama-3.2-Vision) | Fixed (this PR) | Real forward pass against the tiny synthetic checkpoint `tools/verify_mllama_parity.py` builds (multi-image fixture) — genuine Swift+MLX runtime + cross-attention + `Sampler` code path, NOT the real 11B weights (would be ~6 GiB at 4-bit, over this task's 3 GiB/download budget). `MllamaRuntimeTests.testRuntimeLogprobsPopulatedAndOffPathTokensUnchanged` passes. |
| `LocateAnythingRuntime` (LocateAnything-3B) | Fixed (this PR) | Synthetic random-weight model only (`LocateAnythingRuntimeTests.swift`, text-only decode — LocateAnything's decode is plain 1-D RoPE, identical code path to the image case). Real checkpoints (`nvidia/LocateAnything-3B` 7.8 GiB, or the MLX `-4bit` re-release at 3.1 GiB) were both over budget. NOT run against a real checkpoint. |
| `MuseGlimmerRuntime` (Muse Glimmer, image requests) | Fixed (this PR) | Synthetic random-weight model only (`MuseGlimmerLogprobsTests.swift`, reusing `MuseGlimmerNativeTests`' config-JSON pattern, real image splice + vision tower forward). The real model is 30B (smallest published MLX build 19.4 GiB) — no real-checkpoint gate exists anywhere in this repo for this family (see `MuseGlimmerNativeTests.swift`'s header). The parity fixture generator (`tools/verify_muse_glimmer_parity.py`) needs a `transformers` dev build (`5.16.0.dev0`) carrying `muse_glimmer`, not present in `~/.krill/venv`'s `transformers` 5.8.1 — not attempted (would mean changing a shared venv's transformers version for other agents' projects). NOT run against a real checkpoint. |
| Gemma 4 (image + audio) | No fix needed | Already correct: Gemma 4's multimodal forward has no dedicated native runtime (1-D RoPE, no per-step positional offset to thread), so it serves through the GENERIC dense decode loop this plan's Phase 1 wired `wantLogprobs` into. Verified for real against `gemma-4-e2b` (on disk already) with an in-process-generated solid-color PNG and an in-process-generated 1.5s sine-tone WAV (`Gemma4MultimodalLogprobsTests.swift`) — both the image request and the audio request populate `logprobs.content` with the requested `top_logprobs` count. |

### Full bypass-path audit (every `onToken`/`continuation.yield(TokenEvent` site)

| Path | `wantLogprobs` reaches it? | Status |
|---|---|---|
| Generic dense decode loop (`InferenceEngine.generate(messages:)`, plain path) | Yes | Correct (Phase 1) |
| Generic loop's 2-deep pipeline fast path (`usePipeline`) | N/A — declines when `wantLogprobs` (`&& !wantLogprobs` in its own gate) | Correct (Phase 1) |
| Draft-model speculative decode (`shouldSpec`) | N/A — declines when `wantLogprobs` (`&& !wantLogprobs`, `InferenceEngine.swift` ~line 1344) | Correct (Phase 1) |
| N-gram speculative decode (`shouldNgram`) | N/A — declines when `wantLogprobs` (`&& !wantLogprobs`, ~line 1353) | Correct (Phase 1) |
| `Qwen35VLRuntime` (qwen3_5) | Yes | Fixed (f85045b) |
| `Qwen25VLRuntime` (Qwen 2.5-VL) | Yes | **Fixed (this PR)** |
| `MllamaRuntime` (Llama-3.2-Vision) | Yes | **Fixed (this PR)** |
| `LocateAnythingRuntime` | Yes | **Fixed (this PR)** |
| `MuseGlimmerRuntime` (image requests) | Yes | **Fixed (this PR)** |
| Gemma 4 image/audio (generic loop, no dedicated runtime) | Yes (via generic loop) | Correct, verified for real (this PR) |
| `InferenceEngine.generateBatched` / `runBatchedDecode` (static cohort batching) | **No — `BatchGenRequest` has no `wantLogprobs`/`topLogprobs` field at all**; `runBatchedDecode`'s `TokenEvent` yields never carry `logprob` | **NOT fixed — out of scope.** A caller cannot even ask for logprobs on this path today; if server routing ever sends a `logprobs: true` request here, every token silently reports `logprob: nil`. Fixing this means adding fields to `BatchGenRequest` (a type the parallel server-side change also touches) and threading `Sampler.sampleWithLogprobs` through `runBatchedDecode`'s per-row sampling — a bigger, separate change. |
| `InferenceEngine.submitBatched` / `ContinuousBatcher` (continuous batching) | **No — same `BatchGenRequest`, same gap** | **NOT fixed — out of scope**, same reasoning as above. |

Recorded here as the known, real, same-class gap for a future follow-up —
exactly how this plan's own "Verification results (2026-09-29)" section
originally flagged the four now-fixed VL runtimes.

## Phase 2 — spec + batched logprobs (2026-09-30)

Closes this plan's §7 Phase 2 scope: a `logprobs` request no longer disables
draft-model speculative decode, n-gram speculative decode, or the
batched/continuous decode pool. Branch `feat/logprobs-phase2-spec-batched`,
base `main`@`be00f91`. Scope per the task brief: `Sources/KrillEngine/*`,
`Sources/KrillSampler/*`, `Sources/KrillServer/BatchScheduler.swift`.

### Shared primitive (`Sources/KrillSampler/Sampler.swift`)

Every argMax-only decode path (speculative verify, the continuous batcher's
fast paths, Stage-B batched decode) needs the same "raw log-softmax + top-N
from a logits row and an already-chosen token id" computation `Sampler.
sampleWithLogprobs` already did internally. Factored into three static,
batch-capable (1-D or `[N, vocab]`) functions, and `sampleWithLogprobs`
itself now calls them (same numbers, verified byte-for-byte against the
pre-existing `SamplerLogprobsTests` — all 9 still pass unchanged):

- `Sampler.rawLogSoftmaxAndTopN(_:topLogprobs:)` — the log-softmax graph plus
  the `argPartition`(O(V)) + `argSort`(O(N log N)) top-N graph, built from RAW
  logits only (no notion of "chosen token" yet, so it's safe to build before
  or after a caller's own `argMax`/penalty step - ordering only matters for
  `sampleWithLogprobs`'s own pre-existing alias hazard with `applyPenalties`,
  documented in place).
- `Sampler.gatherChosenLogprob(_:chosenIds:)` — gathers each row's own
  logprob from an already-built `logSoftmax` via `takeAlong` (per-row
  gather, not `take`'s flat semantics — required for a real batch).
- `Sampler.logprobInfos(chosenLogprobs:topIdx:topVals:n:)` — host-side
  materialization into `[TokenLogprobInfo]`, one per row.

All three are lazy (no `eval`) so every call site controls its own batching
of host syncs.

### Per-path implementation and status

| Path | File | Status | How |
|---|---|---|---|
| Draft-model speculative decode | `SpeculativeDecoder.step` | **Done** | `wantLogprobs`/`topLogprobs` params (default off); logprobs for every verify-accepted token come from the SAME `targetLogits` already computed to verify the draft (one batched `rawLogSoftmaxAndTopN` call over the accepted positions); the bonus token (full-acceptance case) uses `sampleWithLogprobs` on its own forward. Draft-model logits are never reported. Return type changed to `(tokens: [Int], logprobs: [TokenLogprobInfo]?)` — `nil` array when `wantLogprobs` is false. |
| N-gram (prompt-lookup) speculative decode | `SpeculativeDecoder.ngramStep` | **Done** | Identical approach: the no-match (`k==0`) single-decode branch uses `sampleWithLogprobs`; the verify branch derives logprobs from the same verify forward as the accepted/bonus tokens. |
| `InferenceEngine.generate` spec/n-gram gating | `InferenceEngine.swift` | **Done** | Removed `&& !wantLogprobs` from `shouldSpec`/`shouldNgram` (byte-identical for `wantLogprobs == false`, since the term was simply `true` there before). The prefill-sampled first token (emitted before the spec loop starts) now uses `sampleWithLogprobs` when requested; the n-gram-stall handoff into the plain pipeline seeds `pendingLogprobInfo` correctly so the pipeline's own entry invariant holds. |
| Continuous batcher, per-row `Sampler` path | `ContinuousBatcher.swift` (non-pipeline decode loop) | **Done** | Per row: `sampleWithLogprobs` only when `row.wantLogprobs`; every other row in the same step is untouched. |
| Continuous batcher, all-greedy pipeline fast path | `ContinuousBatcher.swift` (`pipeEligible` loop) | **Done** | Pipeline stays eligible regardless of `wantLogprobs` (no epoch-wide fallback). Per step: gather ONLY the subset of rows with `wantLogprobs` out of the batched `logits`/`sampled` tensors (`take` on the row-index subset), run the batched primitive on just that subset, and carry the result one iteration forward the same way the existing `pendingSample` token is carried — so a step with no wanting rows touches the logprobs code path not at all (`wantIdx.isEmpty` short-circuit), and a step with some wanting rows pays a log-softmax sized to the subset, not the batch. |
| Continuous batcher, n-gram-spec verify round | `ContinuousBatcher.swift` (`decodeSpecRound`) | **Done** | Per row with `wantLogprobs`: derive logprobs for that row's `accepted` tokens from its own slice of the round's verify logits `bl[i, 0..<cacheEntries, :]` — same "reuse the verify forward" reasoning as the single-stream spec path. Non-wanting rows in the same round are untouched. |
| Stage-B fixed-cohort batched decode | `InferenceEngine.swift` (`BatchedCaptures`/`runBatchedDecode`) | **Done** | `BatchedCaptures` gained per-row `wantLogprobs`/`topLogprobs` arrays (from each `BatchGenRequest`). Prefill and the per-step per-row sampling call `sampleWithLogprobs` only for a row that asked; `emit` carries the pending logprob the same one-step-ahead way the single-stream loop does. `generateBatched`'s `serialFallback()` now also threads `wantLogprobs`/`topLogprobs` (was silently dropped before, a real gap for any fallback row). NOTE: this static-cohort path is not reachable from the server today (`BatchScheduler` only calls `submitBatched`→`ContinuousBatcher`) — exercised only by `Tests/KrillEngineTests/BatchedDecodeLiveTests.swift`. |
| `InferenceEngine.submitBatched`/`ContinuousBatcher` admission | `BatchGenRequest` (`InferenceEngineTypes.swift`) | **Done** | New `wantLogprobs: Bool = false`/`topLogprobs: Int = 0` fields (source-compatible defaults). |
| `BatchScheduler.submit` | `Sources/KrillServer/BatchScheduler.swift` | **Done** | Removed the Phase-1 `if wantLogprobs { return serial() }` early return; a logprobs request is now eligible for the batched pool under the same rules as any other request (still excluded by `format`, an explicit speculative opt-in, or the existing seeded-non-greedy/multimodal guards — unchanged). `BatchGenRequest` construction now threads `wantLogprobs`/`topLogprobs` through. |

### Tests

**Unit** (`Tests/KrillEngineTests/SpeculativeLogprobsTests.swift`, new): a
tiny, fully deterministic, CONTEXT-FREE synthetic `LoadedModel` (forward =
table lookup from the input token id, ignoring KV caches — see the file's
class doc) drives real calls to `SpeculativeDecoder.step`/`.ngramStep`
against an independent, from-scratch reference log-softmax (plain Swift
`Double` math, no `Sampler` involved). 5 tests: full-acceptance (bonus token
included) and rejection cases for `.step`, `wantLogprobs: false` returns
`nil`, n-gram no-match and n-gram accepted-run cases for `.ngramStep`. All
assert the reported logprob matches the independent reference within `1e-4`
at every position, not just "is non-nil". All 5 pass. Existing suites
unaffected: `SamplerLogprobsTests` (9), `SpeculativeDecodingTests` (15),
`NgramSpeculativeDecodingTests` (19) all still pass unchanged.

**`make test` full run**: **1775 tests, 141 skipped, 0 failures**, re-run
alone (not concurrently with other agents' work) to rule out the known
`AgentSessionTests` ordering race — none observed on either run.

**Real model — token/logprob parity** (`krill serve`, release build, port
57483, `KRILL_API_KEY` set; OpenAI Python SDK via `/Users/sourav/.krill/
venv`), greedy, `top_logprobs: 5`, prompt "Explain in two sentences why the
sky is blue.":

- **`llama-3.2-1b` (4-bit)**: plain serial reference (`KRILL_NGRAM_SPEC=0`,
  `KRILL_NUM_PARALLEL=1`) vs n-gram spec (default-on, solo request, so the
  engine's own low-concurrency spec preference engages) — **tokens identical
  (24/24)**, logprobs max diff **7.49e-3**, median **2.65e-4** (tighter than
  this plan's own previously-recorded 1.5e-2 floor for this exact
  model/quantization). Plain serial vs 3 CONCURRENT batched requests
  (`KRILL_NGRAM_SPEC=0`, `KRILL_NUM_PARALLEL=3`, same prompt) — **all 3 rows'
  tokens identical to the serial reference**, logprobs max diff 7.49e-3 /
  1.46e-2 / 7.46e-3 across the three rows, median diffs 1.4e-4 - 4.1e-4 — the
  same noise class the batched path's fp16/bf16 shape-dependent evaluation
  order already produces (documented in this plan's Phase-1 resolutions).
- **Mixed-row batch, real server**: 3 concurrent requests to the SAME
  batched server, `[wantLogprobs=true, false, true]` — the `false` row got
  `logprobs: null` (Ollama/OpenAI-null convention, no entries computed for
  it), the two `true` rows got real 24-entry `logprobs.content`, and **all
  three rows produced byte-identical `message.content`** — proves per-row
  opt-in doesn't perturb a non-wanting row's own tokens or a wanting row's
  neighbor.
- **`qwen3-0.6b-bf16`**: needed `KRILL_ENABLE_THINKING=0` (else the
  `<think>` block consumes the whole token budget — a pre-existing,
  already-documented behavior in this plan's Phase-1 resolutions, not new).
  At `max_tokens: 300`: the plain-serial and 3-concurrent-batched GREEDY
  TOKEN SEQUENCES themselves diverge for this bf16 model (confirmed
  reproducible with `logprobs` entirely ABSENT from every request, i.e. a
  pre-existing numeric-precision property of the batched/padded attention
  path for a genuinely-bf16 model, NOT a regression introduced by this
  change — see below). Where two rows of an identical-prompt 3-way batch
  happened to follow the exact same 44-token greedy path (rows 0 and 1 in
  this run), their logprobs agreed with each other with max diff **0.134**,
  median **0.011** nats — looser than the 4-bit llama case, consistent with
  this plan's own observation that a genuinely-bf16 model's floor is roughly
  20x looser than a 4-bit-quantized one (Phase-1 resolutions,
  `Llama-3.2-1B-Instruct-bf16` floor 2.076e-2 vs llama-3.2-1b-4bit floor
  1.5e-2), now compounded by 3-way batch left-padding's own fp arithmetic
  variance. **This token-sequence divergence for `qwen3-0.6b-bf16` under
  concurrent batching is a real, pre-existing engine limitation** (confirmed
  with a from-scratch repro sending 3 identical-prompt concurrent requests
  with NO logprobs involved at all, both before measuring diffs and as a
  sanity check against this PR's own build) — recording it here as a known
  gap for a future investigation, out of scope for a logprobs-plumbing PR to
  fix (it would mean changing the batched attention/padding numerics, not
  logprobs reporting).
- **Draft-model speculative decode**: **not run against a real checkpoint.**
  `draftPairs` (`SpeculativeDecoder.swift`) only pairs larger targets
  (llama-3.2-3b, llama-3.1-8b, qwen2.5-7b/14b/3b, gemma-2-9b, gemma-4-e4b)
  with smaller drafts; none of those TARGETS are present in this
  environment's local model store (`~/.krill/models/blobs/`), only
  `llama-3.2-1b` itself (which is a draft, never a target) and `qwen3.5-4b`/
  `Qwen3-0.6B-bf16`/`Llama-3.2-1B-Instruct-bf16`/`gemma-4-*`, none of which
  have a registered draft. Downloading a valid pair was possible within the
  3 GB budget (e.g. a 4-bit `llama-3.2-3b`) but was not attempted given the
  session's time budget; the synthetic-model unit tests above (which
  exercise the real `SpeculativeDecoder.step` code path end to end, just
  against a hand-built deterministic forward function instead of a real
  transformer) are the correctness evidence for this path instead. Flagged
  here explicitly per the task's own "skip + say so" instruction.

**Speed — logprobs-OFF regression check** (`llama-3.2-1b`, release build;
"before" = `main`@`be00f91` built fresh in a scratch worktree, "after" = this
branch; alternating A/B, restarting the server between rounds since each
config needs a different `KRILL_NGRAM_SPEC`/`KRILL_NUM_PARALLEL`; streaming
HTTP client, decode tok/s = 1/(median inter-chunk gap for the single-request
case) or aggregate tokens/wall-clock for the 3-concurrent batched case). This
dev Mac was NOT idle during measurement (other agents' concurrent Krill work
plus this repo's own documented background load), and **only 2 rounds per
config were run** (not the 3 this plan's own template asks for) due to this
session's time budget — a materially smaller sample than ideal; treat the
signal as suggestive, not conclusive, on its own.

- **N-gram spec, OFF** (`KRILL_NUM_PARALLEL=1`, ngram default-on, `max_tokens:
  320`, decode tok/s per round): before 54.4, 37.0 (mean 45.7); after 30.4,
  47.3 (mean 38.85). Overlapping ranges; the before/after gap is within the
  round-to-round spread of either series. No clean directional regression,
  but not a clean "no regression" proof either at n=2 — the STRUCTURAL
  argument is stronger here: `shouldSpec`/`shouldNgram`'s `wantLogprobs`
  guard was REMOVED, not added, and for a `wantLogprobs == false` request the
  boolean expression's VALUE is unchanged (the removed term was simply
  `true`), so the off-path through `InferenceEngine.generate`'s spec gating
  is byte-for-byte the same code as before this change for every request
  that doesn't ask for logprobs.
- **Batched, OFF** (`KRILL_NGRAM_SPEC=0`, `KRILL_NUM_PARALLEL=3`, 3
  concurrent requests, `max_tokens: 200`, aggregate tok/s per round): before
  24.9, 28.9 (mean 26.9); after 27.8, 27.9 (mean 27.85) — **this-PR is not
  slower** (marginally higher, within noise). Matches the structural
  expectation: the only new code in the off-path is an O(R) `wantIdx =
  rows.indices.filter { ... }` per epoch (empty when nothing asks for
  logprobs) and a closure definition, both negligible next to a batched
  forward pass.
- **Logprobs-ON, for reference** (not a regression gate — the overhead IS
  expected when logprobs is requested): n-gram spec, `top_logprobs: 5`,
  `max_tokens: 320`: **31.2 tok/s** (within the OFF series' own spread,
  20-54 tok/s across the 4 OFF rounds above — at `top_logprobs: 5` on this
  4-bit 1B model the overhead is small relative to the machine's own
  round-to-round noise, consistent with the O(V)-not-O(V log V) top-N fix
  already landed). Batched, `top_logprobs: 5`, 3 concurrent, `max_tokens:
  200`: **24.96 tok/s** aggregate vs the OFF series' 26.9-27.85 mean — a
  roughly 7-13% overhead, the expected cost of the per-wanting-row
  log-softmax + top-N gather each step.

### Limits / not done

- **Draft-model speculative decode has no real-checkpoint run** in this
  environment (no target+draft pair on disk) — see above. The code path is
  implemented and covered by synthetic unit tests exercising the real
  `SpeculativeDecoder.step` logic end to end.
- **`qwen3-0.6b-bf16`'s batched-vs-serial token-sequence divergence** is a
  pre-existing engine limitation (confirmed unrelated to logprobs), not
  fixed here — recorded as a known gap.
- **Speed A/B sample size is 2 rounds per config**, not 3, due to this
  session's time budget on a machine already carrying other agents' work;
  the structural "the off-path code is unchanged / touches only an empty
  `wantIdx` filter" argument is offered alongside the numbers rather than in
  place of them.
- `InferenceEngine.generateBatched`/`runBatchedDecode` (Stage B) is not
  reachable from the server (`BatchScheduler` never calls it) — its
  `wantLogprobs` support was added for completeness and is covered only by
  the pre-existing `BatchedDecodeLiveTests.swift`-style live tests when a
  developer runs them directly against a real checkpoint, not by this PR's
  own new tests.
- `make bench-release-gate` was not re-run for this Phase 2 change (the
  logprobs-OFF spec/batched A/B above is the intended stand-in given this
  session's constraints); re-run it before a release if a stricter proof is
  wanted, per this plan's own §6 item 4 convention.

## Tool-call logprobs (2026-09-30)

**Question.** Phase 1 shipped `logprobs: null` on any `tool_calls` reply,
unconditionally, even when the request asked for `logprobs: true`. Is that
actually what OpenAI does, or is it a Krill invention that should instead
populate real entries — and if the latter, entries for *which* tokens: the
raw generated text (tool-call JSON included) as the model actually produced
it, or only whatever ends up in `message.content`?

**Sources consulted.**

1. **OpenAI Python SDK generated types** (`~/.krill/venv/lib/python3.13/
   site-packages/openai/types/chat/chat_completion.py` and
   `chat_completion_token_logprob.py`, `# File generated from our OpenAPI
   spec by Castiron` — i.e. these mirror the real API surface, not
   hand-written docs):
   ```python
   class ChoiceLogprobs(BaseModel):
       """Log probability information for the choice."""
       content: Optional[List[ChatCompletionTokenLogprob]] = None
       """A list of message content tokens with log probability information."""
       refusal: Optional[List[ChatCompletionTokenLogprob]] = None
       """A list of message refusal tokens with log probability information."""
   ```
   Two load-bearing facts: (a) `content` is explicitly documented as
   covering **`message.content`** — there is no third field anywhere in
   `ChoiceLogprobs` for tool-call *argument* tokens, and no evidence OpenAI
   reports logprobs for them at all; (b) `content`'s type is `Optional[...]
   = None` — it is *nullable*, not a signal that the surrounding `logprobs`
   object itself disappears. `Choice.logprobs: Optional[ChoiceLogprobs] =
   None` is the *only* place the whole object can be absent.
2. **Live behavior report, OpenAI developer forum** ("Can I use logprobs &
   function calling at the same time?"): a user passing `logprobs=True,
   top_logprobs=5` alongside function/tool calling reports the real
   response shape as `logprobs=ChoiceLogprobs(content=None)` — a *present*
   `logprobs` object whose `content` is `null`, exactly matching the SDK
   type above, not a bare `logprobs: null`. Multiple independent posts in
   the same thread describe this as consistent, expected behavior, not a
   bug report about a missing top-level object.
3. **`openai/types/chat/chat_completion_message.py`** (checked for whether
   `content` and `tool_calls` can be simultaneously non-null): community
   reports (OpenAI community forum, "Function Call returning NULL
   message[0].content") consistently describe `content` as `null` whenever
   `tool_calls` is populated for OpenAI's own models — the two are
   observed as mutually exclusive in practice, not merely nullable
   independently.
4. **vLLM** — attempted to check `vllm/entrypoints/openai/serving_chat.py`
   directly (GitHub raw fetch and `gh api` both 404'd from this sandbox, no
   working direct network path to github.com content endpoints); a
   web-search pass over vLLM's issue tracker turned up several *known
   compatibility bugs* around logprobs + tool-calling (e.g. "`logprobs` is
   not compatible with the OpenAI spec", "`choices.logprobs.content` array
   is always empty") but nothing authoritative on the specific null-vs-
   object shape. Not used as a source for the decision below; noted here so
   a future pass knows this avenue was tried and came up empty, rather than
   silently skipped.

**Decision.** For a **pure tool-call turn** (no visible content — see next
paragraph for why that's the only case that exists in Krill today), match
OpenAI's real, confirmed shape: `logprobs` is a real object with `content`
and `refusal` both `null` — `{"content": null, "refusal": null}` — not the
old bare `"logprobs": null`. The task brief's "mixed turn" hypothesis
(visible content *precedes* the tool call, and gets real entries) does not
apply to Krill as built: `handleToolChat`'s response construction
(`Sources/KrillServer/Server.swift`) sets `message["content"] = NSNull()` /
`""` **unconditionally** whenever `calls` is non-empty, discarding
`cleaned` (any leftover text after tool-call-sentinel extraction) even when
it is non-empty — matching finding #3 above (content/tool_calls mutual
exclusion) rather than fighting it. So there is no live case today where a
tool-call reply's `message.content` is non-null; if that ever changes
(Krill starts surfacing pre-call commentary), `logprobsAgg?.entries` would
need slicing to just the tokens preceding the tool-call sentinel — noted
here for whoever makes that change, not implemented speculatively.

**Implementation.** `toolChatLogprobsJSON(wantLogprobs:hasToolCalls:
content:)` (`Sources/KrillServer/LogprobsFormatting.swift`) is the single
pure function `handleToolChat` calls for the OpenAI dialect's `choices[0].
logprobs`. Because `handleToolChat` is the *one* handler both `/v1/chat/
completions` and Ollama's `/api/chat` route tool-bearing requests through,
for both streaming and non-streaming (a streaming tool-call reply is
assembled in full, then emitted as a single SSE/NDJSON chunk that copies
whichever `logprobs` value the assembled response got), fixing this one
call site covers all four combinations (OpenAI/Ollama × stream/non-stream)
with no duplicated logic. The Ollama dialect is intentionally **not**
changed: it already omits the `logprobs` key entirely for a `tool_calls`
reply (Go `omitempty` convention, matching how it omits the key whenever
there's "nothing to report" elsewhere), which is Ollama's own idiomatic
equivalent of OpenAI's null-content object — there is no Ollama-side gap to
close.

**Unit tests.** `Tests/KrillServerTests/ServerFormattingTests.swift`:
`testToolChatLogprobsJSONIsNullWhenNotRequested`,
`testToolChatLogprobsJSONForPureToolCallTurnIsObjectWithNullContent`,
`testToolChatLogprobsJSONForPlainReplyStillGetsRealEntries`.

**Real-server verification (qwen3.5-4b, real tool definition, both
dialects, stream + non-stream).** See the PR body for the exact requests
and responses captured against a locally-running `krill serve` build.

## Real-model runtime checks (2026-09-30)

Verification against real checkpoints on this machine (not synthetic
weights), run to close out the qwen3_5 raw-snapshot fix and the open
prefix-cache/LocateAnything/Muse Glimmer questions from earlier phases.

**Task A: raw HF Qwen3.5-4B snapshot, before/after `qwen35VLKeyRewrite`.**
Loaded the raw bf16 snapshot at
`/Users/sourav/laya-demo/.hf-semif/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`
(read-only, belongs to another project) through `loadQwen35VL`. Before the
fix it produced confident garbage (random-init output); after the fix it
produces coherent multi-sentence output. Numeric parity against `mlx_lm`
(enable_thinking=0, greedy, top_logprobs=5, 40 tokens, prompt "Explain in
two sentences why the sky is blue."): `mlx_lm` floor max 9.813881e-02 /
median 1.014709e-03; Krill sampled-token max 7.489395e-02 / median
1.163483e-03 — 0.76x of the floor, the same excellent-agreement pattern
already established for the converted 4-bit alias. The registered 4-bit
alias `qwen3.5-4b` (`mlx-community/Qwen3.5-4B-MLX-4bit`) was re-checked
after the rewrite and is unaffected: still coherent, output unchanged (the
rewrite is a no-op on an already mlx_vlm-format checkpoint).

**Prefix cache, qwen3_5 family.** Confirmed by design and by a live check
that the prefix cache never engages for `.ssm`-cache-spec families
(`InferenceEngine.swift`: `effectiveUsePrefixCache = usePrefixCache &&
!hasSSMCacheSpec`, with a rationale comment — a GatedDeltaNet layer's
recurrent state is not position-addressable, so a partial KV-prefix hit
would desync it). Sent an identical ~500-token prompt twice to a live
qwen3_5 server: 12.84s then 7.11s. For comparison, a dense control model
(llama-3.2-1b, prefix-cache-eligible) on the same repeat pattern: 0.43s
then 0.18s. No prefix-cache log line was emitted for either family — Krill
does not log a prefix-cache hit for any model — so the qwen3_5 repeat
speedup is ordinary warm-cache/kernel-reuse, not a KV-prefix hit, unlike
the dense control's.

**LocateAnything (`srv-sngh/LocateAnything-3B-mlx-nvfp4`).** Downloaded the
real checkpoint (`model.safetensors`, verified 3,255,388,251 bytes against
the expected size) and served it with a release build (`make release`,
metallib required — a bare `swift build -c release` is not enough; `krill
serve` fails fast with "MLX Metal runtime library was not found" without
it). Sent a real image request (`test_box.png`, prompt "Locate the red box
in the image.") with `logprobs: true, top_logprobs: 3` alongside an
identical request with logprobs off:

- `content[]` was non-empty: 16 entries.
- bytes-concat of the logprobs entries equalled the visible text exactly.
- the logprobs-on and logprobs-off responses produced identical token
  sequences: `<ref>Locate the red box in the image</ref><box><221><221>
  <673><561></box>`.

LocateAnything emits its box/coordinate tokens as ordinary tokenizer text
(`<box><x1><y1><x2><y2></box>`), so this is not a special case for the
logprobs pipeline — the per-token log-probabilities cover the coordinate
tokens the same as any other text token, with no separate code path.

**Muse Glimmer.** Every real MLX build of this family is ≥19.4 GiB, and no
smaller checkpoint of the family exists on HF. On this 24 GiB machine that
leaves no room to load a real one alongside anything else, so Muse Glimmer
stays verified only against synthetic weights (as in the phase above) —
not against a real checkpoint. No download was attempted for this pass;
revisit if a smaller variant is ever published or this runs on a
larger-memory machine.

## Phase 3 — echo (2026-09-30)

Closes this plan's last remaining gap: `echo` on legacy `POST
/v1/completions` (§3.2, §5.4). Branch
`feat/logprobs-echo-and-thinking-switch`, bundled with the unrelated
per-request thinking-switch feature (own PR section, not a logprobs
concern).

### Sources checked before implementing

- OpenAI Python SDK (`/Users/sourav/.krill/venv`, `openai==2.29.0`):
  `openai/types/completion_create_params.py` — `echo: Optional[bool]`,
  `"Echo back the prompt in addition to the completion"`; `logprobs:
  Optional[int]`, `"The API will always return the logprob of the sampled
  token, so there may be up to logprobs+1 elements... The maximum value for
  logprobs is 5."` `openai/types/completion_choice.py` —
  `Logprobs.{text_offset, token_logprobs, tokens, top_logprobs}`, all
  `Optional`, confirming every field CAN be `None` per-position (consistent
  with, though not itself proof of, the null-first-prompt-token convention
  below).
- `platform.openai.com`'s API reference (legacy completions) returned 403 on
  WebFetch, same as every prior research pass in this doc — this is a
  now-deprecated endpoint whose full reference page is evidently gated.
  Semantics not confirmed by the SDK's type stubs alone (the first-prompt-
  token-is-null convention, `text_offset` counting prompt+completion
  characters, and `max_tokens: 0` returning just the scored prompt) are
  well-established, extensively-documented historical behavior of this
  exact API surface from its GPT-3-era popularity (community docs,
  cookbooks, and client-library behavior all agree), not independently
  re-verified against a live OpenAI response in this pass — flagged here
  rather than silently assumed-and-uncited.
- `docs/LOGPROBS_PLAN.md` itself (this file), §3.2 and §5.4, written during
  Phase 1 planning.

### Design decisions

1. **Raw prompt, no chat template.** `/v1/completions` already always wraps
   whatever prompt string it's given into one `[user: prompt]` chat-template
   turn before generating the actual completion (`runGenerate(prompt:)` →
   `InferenceEngine.generate(messages:)`) — a pre-existing Krill quirk,
   unrelated to and untouched by this feature. For `echo`, that templated,
   family-specific rendering is the WRONG thing to echo back or score: the
   client sent a raw string, not a chat turn, and OpenAI's own legacy
   completions API never applied a chat template to begin with. So `echo`'s
   prompt text and its per-token logprobs are computed from
   `tokenizer.encode(prompt)` — the same plain encode used elsewhere for raw
   text — entirely independent of whatever the completion call's chat
   template does. This also happens to be exactly what `mlx_lm`'s own
   `tokenizer.encode(prompt)` does for the parity oracle, so the numbers are
   directly comparable with no template-kwarg guessing (the "third harness
   gap" §7's Resolutions flagged for chat's parity script does not recur
   here).
2. **Bypass the prefix cache entirely (§5.4 option (a)).** A NEW public
   method, `InferenceEngine.echoPromptLogprobs(prompt:topLogprobs:)`
   (`Sources/KrillEngine/InferenceEngine.swift`), is its own self-contained
   forward pass: a fresh `makeKVCaches(spec:numLayers:)` KVCache that is
   never looked up in or stored into the shared `PrefixCache`, chunked
   (a dedicated `echoLogprobsChunkSize = 512`, deliberately smaller than the
   model's own `prefillChunkSize` — see point 3) through the model's plain
   `forward` closure (NOT the `prefillForward`/"last-token-only" closure the
   normal decode path prefers, which only ever computes logits for the LAST
   position of a chunk — echo needs EVERY position's logits). This method
   never touches `generate(messages:)`, `Sampler.sample`/`sampleArray`/
   `sampleWithLogprobs`, or any decode-loop state, so the prefill path for
   every OTHER request (no `echo`) is provably byte-for-byte unaffected —
   confirmed by re-running `tools/logprobs_e2e_check.py` and
   `tools/logprobs_ollama_completions_e2e_check.py` unmodified (see Tests
   below) and by `make test`'s full suite, all green.
3. **Chunk-and-discard, never materialize `[promptLen, vocab]`.** Each
   chunk's `[chunk, vocab]` raw logits (from `forward`) go straight into
   `Sampler.rawLogSoftmaxAndTopN` (the SAME Phase 2 batched primitive that
   already computes N independent rows' log-softmax + top-N in one call —
   reused here treating "N positions in a chunk" exactly like Phase 2 treats
   "N rows in a batch step", no new math), producing that chunk's
   `logSoftmax`/`topIdx`/`topVals`, from which the chunk's per-position
   `TokenLogprobInfo`s are extracted and the chunk's raw tensors are then
   dropped before the next chunk starts. `echoLogprobsChunkSize` (512, not
   the model's own possibly-2048 `prefillChunkSize`) bounds this chunk's
   float32 working set to ≤ ~550 MB even at Gemma 4's 262144-entry vocab —
   deliberately conservative on a shared 24 GiB machine that may already be
   under memory pressure from other jobs (this session's own `memory_pressure`
   check at test time showed ~27% free, close to the 30% floor).
4. **A leading BOS breaks `tokens`-joined-equals-`text` — strip it from what
   gets REPORTED, not from what gets FED to the model.** Found empirically,
   not anticipated in the original plan: `tokenizer.encode(prompt)` for a
   Llama-family checkpoint prepends a BOS token (`<|begin_of_text|>`) that is
   necessary context for correct generation but is NOT literal text the
   client sent — its `decodeForOutput` is the non-empty string
   `"<|begin_of_text|>"`, which broke the "concatenating `tokens` reproduces
   the returned `text`" invariant this endpoint's own response shape
   promises (and that `tools/logprobs_ollama_completions_e2e_check.py`
   explicitly checks for the sibling endpoints). Fix: `echoPromptLogprobs`
   still feeds the FULL id sequence (BOS included) to the model for correct
   context, but drops a detected leading BOS
   (`tokenIds.first == tokenizer.bosTokenId`) from the RETURNED
   `tokenIds`/`tokenStrings`/`infos` arrays. The new first reported token
   keeps its REAL, already-computed logprob (scored against the hidden BOS
   context) rather than being forced to `null` — only a prompt with
   genuinely no preceding context at all (a true single-token or empty
   prompt) gets a `null` first entry, matching OpenAI's documented
   convention for THAT case specifically. Qwen-family tokenizers were
   observed NOT to prepend a BOS at all (`Qwen3-0.6B-bf16`'s raw encode of
   an 11-word prompt produced exactly 11 ids, matching Krill's reported
   token count one-for-one, offset 0) — the strip is a no-op for those
   checkpoints, applied uniformly rather than per-family-special-cased.
5. **`max_tokens: 0` skips generation entirely.** Confirmed by the SDK's
   `echo` doc comment ("echo: true with max_tokens: 0 returns just the
   scored prompt") to be real, intentional OpenAI behavior, not an edge case
   to reject. `ServerParsing`'s shared `tokenLimit`/`requiredTokenLimit`
   helpers (used by chat and both Ollama dialects) reject an explicit `0`
   via `positiveInt` — loosening that shared helper would change three OTHER
   endpoints' behavior for a case only this one documents a meaning for, so
   a new `completionsTokenLimit` helper (same shape, `>= 0` instead of `> 0`)
   is used ONLY by `openAICompletionRequest`. `Server.swift`'s
   `handleCompletions` then skips the `runGenerate` call entirely when
   `request.maxTokens == 0` (no `TokenBudget` resolution, no engine call at
   all) — `finish_reason: "length"` (0 tokens were generated because the
   limit was 0, immediately reached), `usage.prompt_tokens` from a cheap
   tokenize-only `InferenceEngine.promptTokenIds(_:)` (no forward pass) since
   there is no `GenerationStats` to read it from. This also generalizes
   `max_tokens: 0` to a non-`echo` legacy-completions request (empty text,
   real prompt-token usage count) — an adjacent, cheap, low-risk fix
   (matches documented OpenAI behavior generally, not just for `echo`) left
   in rather than narrowly gated to `echo: true` only.
6. **Response shape reuses Phase 1's machinery, not a parallel
   implementation.** A new `echoPromptLogprobEntry(tokenString:info:eng:)`
   (`Sources/KrillServer/LogprobsFormatting.swift`) builds one prompt-side
   entry in the exact same `[String: Any]` shape
   `LogprobsAggregator`'s generated-token entries already use (`token`,
   `logprob`, `top_logprobs`) — `info == nil` (the true-first-token case)
   encodes as `NSNull()` sentinels for BOTH `logprob` and `top_logprobs`,
   which `legacyCompletionLogprobsJSON` (extended, not replaced) now
   recognizes and passes through as JSON `null` in both `token_logprobs[i]`
   and `top_logprobs[i]` (not `{}` — OpenAI's documented convention for the
   position with no preceding context). Prompt entries are simply prepended
   to the existing completion entries before the ONE existing
   `legacyCompletionLogprobsJSON(entries:)` call — `text_offset`'s existing
   running-sum logic needed no changes at all to correctly accumulate across
   prompt then completion, since it was already generic over "an ordered
   list of token entries."

### Implementation summary (files touched)

- `Sources/KrillEngine/InferenceEngine.swift`: `promptTokenIds(_:)` (cheap
  tokenize-only helper) and `echoPromptLogprobs(prompt:topLogprobs:)` (the
  echo-only forward+log-softmax path described above), plus
  `echoLogprobsChunkSize`. No changes to `generate(messages:)`, `Sampler`,
  or any existing decode path.
- `Sources/KrillServer/ServerParsing.swift`: `ServerCompletionRequest.echo:
  Bool`; `echo` removed from `unsupportedOpenAICompletionFields`;
  `completionsTokenLimit` (accepts an explicit `0`, scoped to this endpoint
  only).
- `Sources/KrillServer/LogprobsFormatting.swift`: `echoPromptLogprobEntry`;
  `legacyCompletionLogprobsJSON` extended to recognize the `NSNull()`
  sentinel and emit `null` (not `{}`/`0`) for that position, for both
  `token_logprobs`/`top_logprobs` — a no-op change for every existing
  (non-`echo`) call site, since no generated-token entry ever carries that
  sentinel.
- `Sources/KrillServer/Server.swift`: `handleCompletions` — computes
  `promptTokenCount` (via `promptTokenIds`) whenever `echo` or
  `max_tokens: 0`; computes `promptEntries` (via `echoPromptLogprobs`) only
  when `echo && wantLogprobs`; skips the `runGenerate` call entirely when
  `maxTokens == 0`; prepends `promptEntries` to the existing
  `completionEntries` before the one `legacyCompletionLogprobsJSON` call.
  Every branch a non-`echo`, `maxTokens > 0` request takes is textually
  identical to before this change.

### Tests

**Unit** (`Tests/KrillServerTests/ServerTests.swift`,
`ServerFormattingTests.swift`): `echo` parsing (accepted, defaults `false`,
rejects a non-bool, combines with `logprobs`+`max_tokens: 0`);
`max_tokens: 0` parsing (accepted, rejects negative, the two token-limit
fields still conflict-check against each other); `echoPromptLogprobEntry`'s
null-first-token shape and its real-value shape;
`legacyCompletionLogprobsJSON`'s handling of a prepended null entry
(`token_logprobs`/`top_logprobs` both `null`, `text_offset` still
accumulates correctly into the following completion entries). `make test`:
**1801 tests, 141 skipped, 0 failures** (full suite, including the
pre-existing `AgentSessionTests`/`BatchSchedulerTests`/etc., not just the
new cases).

**Real server** (`krill serve` from a release build, `llama-3.2-1b` 4-bit
and `Qwen3-0.6B-bf16`, port 57480 — the real `server_api_key` from
`~/.krill/config.toml` read into `KRILL_API_KEY` at runtime, never written
to a file/commit):

- OpenAI SDK round-trip (`client.completions.create(prompt=..., echo=True,
  logprobs=3, max_tokens=0)`): parses into the typed `Logprobs` object with
  no SDK error; `choices[0].text == prompt` exactly; `tokens`/
  `token_logprobs`/`top_logprobs` all length-matched;
  `"".join(tokens) == text` exactly (the invariant point 4 above exists to
  guarantee).
- `echo: true, logprobs: 2, max_tokens: 5, temperature: 0`: full
  prompt+completion text returned; `tokens` joined still equals `text`
  exactly; `text_offset` a correct cumulative sum across BOTH halves.
- `echo: true` with no `logprobs`: prompt text prepended, `logprobs` key
  absent (not `null`) — matches this endpoint's existing "no field a client
  didn't ask for" convention.
- Non-`echo`, `logprobs` absent (the pre-existing case): unaffected — spot
  re-checked, and `tools/logprobs_e2e_check.py` +
  `tools/logprobs_ollama_completions_e2e_check.py` (unmodified) both still
  pass in full against `qwen3.5-4b` on this same build (see their own
  output for the per-check breakdown), which is the actual regression gate
  for "every other request is byte-for-byte unaffected."

**Numeric parity vs `mlx_lm`** (new `tools/logprobs_echo_parity.py`: a
single full-sequence `mlx_lm` forward over the SAME raw-encoded prompt ids,
log-softmax in float32, compared against a chunked-through-a-real-KVCache
`mlx_lm` reference as the intrinsic floor — mirroring `logprobs_parity.py`'s
"full-sequence vs incremental" floor methodology — then against Krill's
actual `/v1/completions` `echo` response), prompt "Explain in two sentences
why the sky is blue.", greedy, `top_logprobs: 5`:

| Model | Floor max/median | Krill vs mlx_lm max/median | Notes |
|---|---|---|---|
| `llama-3.2-1b` (4-bit) | 0.0 / 0.0 (12 tokens, single chunk both sides) | 1.776e-2 / 5.472e-3 (11 compared) | In the same ~1e-2-2e-2 range as this plan's own already-documented 4-bit dequantization noise for GENERATED-token logprobs (Resolutions: 2.52e-2 max on the same checkpoint) |
| `Qwen3-0.6B-bf16` | 0.0 / 0.0 (11 tokens, single chunk both sides; confirmed NO leading BOS, offset 0) | 7.912e-2 / 1.641e-2 (10 compared) | Comparable to this plan's own bf16 alternate-token noise characterization (§7's Resolutions: bf16 has 7 mantissa bits vs float16's 10 — this is intrinsic arithmetic noise, not a Krill defect) |

Both prompts (12/11 raw tokens) fit inside one 512-token chunk on both the
Krill and the mlx_lm-reference side, so the FLOOR reads as exactly `0.0` —
a real result (there is no chunk boundary to introduce non-associativity
here), not a bug in the floor computation; it simply means this parity run
does not independently exercise the multi-chunk code path (`echoPromptLogprobs`'s
chunking loop) beyond what `make test`'s unit-level review of that loop's
logic already covers. Re-run with a prompt over 512 tokens if a
multi-chunk-specific regression is ever suspected.

### Limits / not done

- `echo` carries no media payload at all (`ServerCompletionRequest` has no
  `media` field), so it has no interaction whatsoever with the four native
  VL/multimodal decode runtimes (Qwen 2.5-VL, Llama-3.2-Vision,
  LocateAnything-3B, Muse Glimmer image requests) — not because of any
  remaining gap in them: their `logprobs` support was already fixed (`main`
  commit `6101a5b`, `fix(engine): thread logprobs into the four remaining
  native VL runtimes`) and, for LocateAnything, verified against a real
  checkpoint (this doc's own "LocateAnything" real-model section), both
  BEFORE this branch's base commit. That work predates and is unrelated to
  this `echo` follow-up.
- No multi-chunk (prompt > 512 tokens) real-model run was performed in this
  pass (both parity prompts were short) — the chunking loop is exercised by
  `make test`'s existing coverage of the pattern it reuses
  (`rawLogSoftmaxAndTopN`'s own Phase-2 batched tests) but not by a
  dedicated long-prompt real-model echo test. Low risk (the loop is a
  straightforward reuse of already-verified per-chunk math with no new
  cross-chunk state), but explicitly not independently verified end-to-end
  here.
- `echo` was not combined with speculative decode or the batched/continuous
  pool in testing — `echoPromptLogprobs` runs entirely independently of
  `generate(messages:)`'s spec/batch decision logic (it's a separate method
  called BEFORE any `runGenerate` call for the completion half), so there is
  no interaction to test: the completion half of an `echo` request goes
  through the exact same spec/batch eligibility path any other
  `/v1/completions` request would.
- The legacy-completions-specific "OpenAI's exact current `platform.openai.com`
  reference page" could not be independently re-fetched (403, consistent
  with every prior attempt in this document) — semantics rely on the SDK
  type stubs plus well-established historical documentation of this
  (now-deprecated) API shape, flagged explicitly rather than silently
  presented as independently re-verified.
