# 0005. Logprobs semantics (OpenAI/Ollama `logprobs`, `top_logprobs`, `echo`)

Status: adopted. Date: 2026-09-30. Owner: unassigned. Scope: what a reported
logprob *means* across every Krill surface that can report one — chat,
Ollama, legacy completions, streaming, tool calls, and prompt (`echo`)
scoring — and why. Implementation history and numeric verification live in
`docs/LOGPROBS_PLAN.md`; this record is the semantics contract, not the plan.

---

## 0. TL;DR

- **Raw, pre-sampling log-softmax, in float32, from the forward-pass
  logits** — before temperature, top-k/top-p/min-p, repetition/presence/
  frequency penalties, and grammar masking. Not the "as-sampled" probability.
- **Entries only for visible tokens.** Reasoning/thinking, special, and
  suppressed tokens never get a `content[]`/array entry.
- **`bytes` comes from the raw token piece**, never from the decoded display
  text.
- **Zero cost when not requested** — no extra forward pass, no extra
  log-softmax, on the hot path a non-logprobs request already takes.
- **Tool-call turns report no per-token data**: OpenAI gets
  `{content: null, refusal: null}`, Ollama omits the key.
- **JSON `null` on an optional request field means "absent/default"**, same
  as every other optional field.
- **`echo` scores the raw prompt** — no chat template, bypasses the prefix
  cache, chunks every 512 positions, and drops a leading BOS from the
  reported arrays.
- **Range ceilings differ by surface**: `top_logprobs` 0–20 on chat/Ollama,
  `logprobs` 0–5 on legacy `/v1/completions` — a real OpenAI API difference,
  not a Krill inconsistency.

---

## Context

Krill's server previously rejected the entire `logprobs` family with a 400
on every surface (`docs/LOGPROBS_PLAN.md` §1). Closing that gap required
picking an actual meaning for "the logprob of a token" before writing any
code, because the obvious alternatives disagree with each other and with
standard practice:

- Krill's sampler (`Sources/KrillSampler/Sampler.swift`, `sampleFrom`) runs
  an optional grammar mask, then (if not greedy) temperature, top-k, top-p,
  min-p, then draws — each filter sets rejected logits to `-1e9`. Greedy
  requests skip straight to `argMax` and never run the filters at all.
- A logprob computed *after* those filters is not a real distribution over
  the vocabulary (it's the truncated, possibly-unnormalized remainder), and
  disagrees with `mlx_lm`/HF `transformers`, which score raw logits.
- A logprob computed *before* temperature has nothing well-defined to report
  differently at `temperature: 0` (greedy) — there's no "after" to contrast.
- Every one of Krill's decode paths (plain, draft-model/n-gram speculative,
  batched/continuous) computes logits differently and at different call
  sites, so "where do we grab the numbers from" had to be answered per path,
  consistently, without regressing the paths' existing zero-logprobs-cost
  fast lanes (`docs/PITFALLS.md` #10).

## Decision

**1. Raw pre-sampling log-softmax in float32, before penalties, temperature,
top-k/p/min-p, and grammar.** Implemented by building the log-softmax graph
from the logits as they exist *before* `applyPenalties`' in-place scatter
runs — MLX captures the input's value at the call site into a new,
independent result, so no copy (host or device) is needed to protect the
reported number from the later mutation
(`Sources/KrillSampler/Sampler.swift:264`, `sampleWithLogprobs`;
`Sources/KrillSampler/Sampler.swift:352`, `rawLogSoftmaxAndTopN`, the shared
primitive reused by the spec/batched paths). This is greedy-defined
(well-defined at `temperature: 0`, where most eval/judge traffic actually
runs), reproducible as a pure function of `(model, prompt, position)`, and
matches `mlx_lm`/HF `transformers` and vLLM's own default `logprobs_mode`
(`raw_logprobs`). The cost: the reported number is not always the exact
probability the token was drawn with when temperature != 1 or a
top-k/top-p/min-p filter is active — documented in `docs/SERVER_API.md`
rather than hidden.

**2. Entries only for visible tokens.** A token gets a `content[]` (or
Ollama/legacy array) entry **iff** it survives `StreamingReasoningFilter`
and is not one of `outputSuppressedTokenIDs` — reasoning/thinking tokens,
special tokens, and suppressed tokens get none. `LogprobsAggregator`
(`Sources/KrillServer/LogprobsFormatting.swift:101`) keeps a FIFO of pending
`(tokenId, info, ownText)` tuples and drains it against the filter's own
emit/discard decisions, so a token the filter HOLDS while disambiguating a
tag prefix (e.g. a bare `<`) or a token whose own decoded text is empty (a
byte-fallback/partial-UTF-8 piece) still gets exactly one correctly-ordered
entry — including ones only resolved at `finish()`.

**3. `bytes` is derived from the raw token piece, not the decoded text.**
`KrillTokenizer.rawTokenBytes(for:)`
(`Sources/KrillTokenizer/TokenizerWrapper.swift:735`, exposed through
`Sources/KrillEngine/InferenceEngine.swift:113` and the
`LogprobsFormatting.swift:20` protocol) resolves, in order: (1) a
`<0xHH>` byte-fallback piece → `[0xHH]`; (2) for a byte-level-BPE tokenizer
(resolved once at load from `tokenizer.json`), every character of the piece
round-tripped through the GPT-2 byte↔unicode table; (3) for a SentencePiece
tokenizer, `▁` → space plus the piece's UTF-8 bytes; (4) otherwise, the
piece's plain UTF-8 bytes. The displayed `token` string uses
`String(decoding:as: UTF8.self)` (lossy, `U+FFFD` on an incomplete
sequence) for a partial-UTF-8 byte-fallback token, but `bytes` is never
lossy — it is what the aggregator's own real end-to-end check
(`tools/logprobs_e2e_check.py`) confirms reproduces the visible text
byte-for-byte when concatenated.

**4. Zero cost when not requested.** Every decode path (plain,
speculative, batched/continuous) only enters the logprobs code at all for a
row/request that asked for it — a step with no wanting rows never touches a
`wantIdx`-gated subset op (`Sources/KrillSampler/Sampler.swift`'s shared
primitive; `ContinuousBatcher.swift`'s `pipeEligible` loop). No fallback to
a slower path is needed for this, since Phase 2 made every path compute
logprobs from logits it already has for its own purposes (verification
logits for spec decode; the batched forward's own logits for batched
decode).

**5. Tool-call turns report no per-token data.** OpenAI gets a real
`logprobs` object with both `content` and `refusal` explicit `null` —
`{"content": null, "refusal": null}` — confirmed against the OpenAI Python
SDK's `ChoiceLogprobs` type, not a bare `logprobs: null`. Ollama omits the
`logprobs` key entirely for a tool-call reply, matching its own `omitempty`
convention elsewhere. Rationale: a tool-call reply's *underlying* token
stream is the raw pre-extraction text, not the structured `tool_calls` the
client sees, and Krill's `message.content` is always `null` for a
tool-call reply regardless of logprobs — there is no stable, OpenAI-shaped
"content" to attach entries to.

**6. JSON `null` on an optional request field means absent.** An explicit
`null` for `logprobs`, `top_logprobs`, `echo`, `chat_template_kwargs`, or
`chat_template_kwargs.enable_thinking` is treated exactly like omitting the
field — "use the default" — not an error and not a distinct third state.
This matches the convention already established for other optional fields
before this work (`docs/LOGPROBS_PLAN.md`'s "treat explicit JSON null as
absent for optional request fields" commit) and was extended to every new
logprobs/thinking field rather than inventing a one-off rule.

**7. `echo` scores the raw prompt.** `echo: true` on legacy
`/v1/completions` scores the prompt **as the client sent it, with no chat
template applied** — a deliberate divergence from how that same request's
actual completion is generated (which does wrap the prompt in a chat
template, pre-existing, unrelated behavior). This requires bypassing the
prefix cache entirely (`usePrefixCache: false`): a prefix-cache hit only
re-forwards the last prompt token or the divergent suffix
(`InferenceEngine.swift:1527-1552`), never producing logits for the cached
prefix positions, which `echo` needs for *every* prompt position. The
chunking loop (`chunkedPromptLogprobs`,
`Sources/KrillEngine/GenerationSupport.swift:133`, called from
`InferenceEngine.echoPromptLogprobs` at `InferenceEngine.swift:209`) scores
512 positions at a time, threading the same `KVCache` across chunks so a
chunk boundary changes nothing about the result. A leading BOS token the
tokenizer adds (not literal client text) is stripped from the reported
`tokens`/`token_logprobs`/`top_logprobs`/`text_offset` arrays so `tokens`
joined always reproduces the returned `text` exactly, while the new first
reported token still gets a real logprob (scored against that hidden BOS
context) rather than a forced `null`.

**8. `top_logprobs` range is 0–20 for chat/Ollama; `logprobs` range is 0–5
for legacy `/v1/completions`.** Confirmed against the OpenAI Python SDK's
own type stubs (`completion_create_params.py` for both chat's
`top_logprobs` and legacy completions' `logprobs`, whose doc comment states
"The maximum value for `logprobs` is 5" explicitly) rather than guessed —
this is a real, deliberate difference between the two OpenAI endpoints, not
a typo carried from one to the other.

## Alternatives considered

| Option | What it is | Why not |
|---|---|---|
| Post-sampling-transform logprobs (report what was actually sampled from) | Logprob computed after temperature/top-k/top-p/min-p/penalties | Undefined at `temperature: 0` (the single most common eval/judge setting); not reproducible from `(model, prompt, position)` alone since penalties depend on generation history; disagrees with `mlx_lm`/HF `transformers`/vLLM's default. |
| Opt-in `logprobs_mode` (`raw` \| `processed`), vLLM-style, from day one | Let the client choose raw vs. post-transform | Real added API surface and engine complexity for a mode nobody asked for yet; deferred — can be added later as a strict addition without breaking `raw` as default. |
| Report entries for every generated token, including reasoning/suppressed ones, behind a flag | Exposes thinking-token logprobs for debugging | Not standard OpenAI behavior to copy; reasoning-model logprobs handling is inconsistent industry-wide; no concrete consumer asked for it — punted until one does. |
| Derive `bytes` from the decoded display text | Simpler, one code path instead of a tokenizer-aware byte table | Breaks silently for byte-fallback/partial-UTF-8 tokens, where decoded text is lossy (`U+FFFD`) or empty — exactly the positions where a consumer needs real bytes most. |
| Keep the Phase-1 fallback (disable speculative/batched decode when `logprobs` is requested) permanently | Simpler: one code path ever computes logprobs | Silently slower for any `logprobs` request against a model that would otherwise use spec/batched decode — a real regression for the eval/judge use case this feature exists for. Replaced in Phase 2 once each path's own logits were shown to be enough (no extra forward pass needed). |
| Fold the sampled token into `top_logprobs` only when it's already in the top-N (OpenAI ambiguity) | Smaller response when the sampled token is an outlier | Real OpenAI responses always include the sampled token as `content[i]` itself regardless of top-N membership, with `top_logprobs` separately always the true top-N — matching that, not inventing a smaller shape, keeps client-side parsing identical to talking to real OpenAI. |

## Why this over the others

Reproducibility and greedy-correctness dominated every other concern: the
primary consumers named in `docs/LOGPROBS_PLAN.md` §1 (eval harnesses,
LLM-as-judge scoring, OpenAI-SDK client compatibility) all run at or near
`temperature: 0` and need a number that is a pure function of the model and
position, comparable across runs and across engines (`mlx_lm`, vLLM). A
scheme that depends on sampler configuration (temperature, truncation,
penalties) cannot give that, and would need a from-scratch definition at
`temperature: 0` where "after" nothing has happened yet. Matching vLLM's
own default means a consumer that already targets vLLM's raw-logprobs mode
needs no special-casing for Krill.

## Consequences

- A reported logprob is the model's raw belief, not the probability of the
  sampling event for `temperature != 1` or an active top-k/top-p/min-p
  filter — documented in `docs/SERVER_API.md`'s Logprobs section so a
  consumer does not assume otherwise.
- Every decode path (plain, speculative, batched/continuous) carries its
  own small, reviewed addition of this same raw-log-softmax primitive
  rather than one shared code path, because each has a different logits
  layout (per-row vs. fully batched) — more surface area to review, but no
  path silently falls back to a slower one.
- `echo` pays a full, uncached prompt forward, chunked at 512 positions —
  opt-in cost, only paid by a request that asks for it.
- A future opt-in `logprobs_mode` (`raw` \| `processed`) remains a strict,
  additive extension if a concrete consumer ever needs as-sampled
  probabilities; `raw` stays the default either way.

## Testing / verification

- Numeric parity against `mlx_lm` (`tools/logprobs_parity.py`) for
  generated-token logprobs on `llama-3.2-1b` (4-bit) and `qwen3-0.6b`
  (bf16), each measured against `mlx_lm`'s own full-sequence-vs-incremental
  noise floor rather than a bare tolerance — see
  `docs/LOGPROBS_PLAN.md`'s "Resolutions" and "Verification results
  (2026-09-29)" sections for the numbers.
- `LogprobsAggregatorTests` (`Tests/KrillServerTests/`) — 11 cases covering
  English, code containing `<`, Devanagari, emoji, a `<think>` block, a
  token held and flushed at end-of-stream, empty-decode tokens, and
  suppressed tokens.
- `tools/logprobs_e2e_check.py` — real end-to-end check against
  `llama-3.2-1b` confirming `bytes`-concat reproduces `content` byte-for-byte
  and that streaming/non-streaming `content` match exactly, for a Hindi and
  a code-with-`<` prompt.
- `EchoChunkingTests` (`Tests/KrillEngineTests/EchoChunkingTests.swift`) —
  a synthetic, context-dependent model exercised at chunk sizes 3, 4, 7, and
  ≥ prompt length, checked for cross-chunk-size agreement and against an
  independently computed (plain Swift, no MLX) ground truth; both tests
  confirmed to fail under two deliberately reintroduced bugs before
  confirming they pass on the real code.
- Phase 2 spec/batched logprobs: synthetic, context-free `LoadedModel` unit
  tests exercising `SpeculativeDecoder.step` and `ContinuousBatcher`'s
  per-step logic end to end; a real-checkpoint speed A/B (logprobs-off)
  confirming no regression to the existing zero-overhead-by-default path.
- Known gaps, not yet closed: no real-checkpoint run for draft-model
  speculative decode (no draft+target pair on disk) or for Muse Glimmer
  (smallest published build 19.4 GiB, exceeds this machine's budget); no
  real multi-hundred-token `echo` run against `mlx_lm` (memory-blocked this
  session) — see `docs/LOGPROBS_PLAN.md`'s Status header for the full list.

## Key files

- `Sources/KrillSampler/Sampler.swift` — `sampleWithLogprobs`,
  `rawLogSoftmaxAndTopN` (shared raw log-softmax primitive).
- `Sources/KrillServer/LogprobsFormatting.swift` — `LogprobsAggregator`,
  the `rawTokenBytes` protocol requirement, OpenAI/Ollama/legacy response
  shaping.
- `Sources/KrillTokenizer/TokenizerWrapper.swift` — `rawTokenBytes(for:)`.
- `Sources/KrillEngine/GenerationSupport.swift` — `chunkedPromptLogprobs`.
- `Sources/KrillEngine/InferenceEngine.swift` — `echoPromptLogprobs`,
  prefix-cache bypass mechanics.
- `Sources/KrillServer/ServerParsing.swift` — request field parsing,
  explicit-`null`-as-absent handling, range validation.
- `docs/LOGPROBS_PLAN.md` — implementation plan, phasing, and all dated
  verification sections.
- `docs/SERVER_API.md` — the user-facing Logprobs/Thinking reference.
