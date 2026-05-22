# WS2: Speculative Decoding

Status: in progress (correctness + plumbing landed; the `>= 1.5x` decode
target is not met and is structurally unreachable on M-series, so since
2026-05-22 `text_decode_ratio` is advisory in both gate profiles with a
hard `>= 1.0x` floor - see `docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md`).
Detailed usage and result: [../SPECULATIVE_DECODING.md](../SPECULATIVE_DECODING.md)

## What landed in this PR

- Cache-aware causal mask (`createCachedCausalMask`) used by all dense
  text models and Gemma 4; the spec verify path no longer hits a runtime
  shape mismatch when K > 1 with a non-empty cache.
- Draft cache prefill on the prompt, attributed to `prefillDuration`
  rather than decode time.
- Draft cache truncate-on-reject and bonus-forward sync mirroring the
  target. Without these, the draft cache poisons the next round.
- Greedy-only guard on the spec path (temperature == 0, top-p >= 1,
  top-k <= 0, min-p == 0, no penalties / mirostat, fp16 KV).
- `--draft-model <alias|path|auto>` on `krillm run` and `krillm serve`
  (also `KRILL_DRAFT_MODEL` for serve). `auto` consults `draftPairs`.
- `GenerationStats.speculative` (rounds / accepted / final_k /
  acceptance_rate) wired into CLI stats line and the
  `krillm_vs_ollama_benchmark.py` JSON report.
- New unit tests for the greedy guard, curated-pair lookup, and
  adaptive-K reset semantics.

## What the `>= 1.5x` decode target still does not reach

The `text_decode_ratio >= 1.5x` target is not met. It is no longer a
hard *gate* in either profile (advisory since 2026-05-16 for
`release_candidate`, 2026-05-22 for `strict`; the hard guarantee is the
`>= 1.0x` floor) - but the `>= 1.5x` aspiration this workstream targets
remains unreached. Benchmarked on two target sizes (3B and 8B) on
M-series 4-bit MLX with the smallest available mlx-community drafter
(llama-3.2-1b), bracketed by low and high acceptance prompts:

| Target / prompt                | KrillLM no-spec | KrillLM spec | K | acceptance |
| ------------------------------ | --------------- | ------------ | - | ---------- |
| llama-3.2-3b   / story         | 104.3 tok/s     | 74.6         | 2 | 0.47       |
| llama-3.1-8b   / story         |  50.2 tok/s     | 40.9         | 2 | 0.50       |
| llama-3.1-8b   / technical     |  50.2 tok/s     | 44.1         | 5 | 0.73       |

Cross-engine Ollama numbers are tracked in
`docs/SPECULATIVE_DECODING.md` but the prompt-tokens count differs
across engines (39-52% delta), so the load-bearing comparison here
is KrillLM-vs-KrillLM (spec-on vs spec-off, identical encoder).

Output sha256 is identical with and without spec on every pair
(greedy parity verified); the throughput regression is empirically
out of reach on every prompt and K setting tested, including the
high-acceptance 0.73 run. See `docs/SPECULATIVE_DECODING.md` for
the break-even framework that maps the observed gap to per-round
overhead (alpha + beta > 0.67 in the model, with the fitted beta ~ 0.78 and alpha ~ 0.125) rather than insufficient
acceptance.

Cost model fit to the 8B / 1B K=5 acceptance-0.73 run gives
`alpha ~ 1/8`, `beta ~ 0.78`. Asymptotic ceiling at infinite K and
100% acceptance is `1 / (alpha + beta) ~ 1.10x`. Strict 1.5x requires
`alpha + beta < 0.67`, which the current engine + model pair cannot
reach. See `docs/SPECULATIVE_DECODING.md` for the derivation.

Three credible unlocks (all out of scope for this PR and the
WS2 follow-ups that landed):

1. Drive beta lower (eval syncs, MLX kernel-launch overhead).
   beta ~ 0.55 at high K and acceptance would lift the ceiling to
   ~1.47x; would not alone clear strict.
2. A ~70B target paired with a 1B draft (RAM-infeasible on
   M-series at 4-bit, ~40 GB just for the target).
3. Tree attention / Medusa-style multi-branch verify - lifts the
   effective r above the K+1 single-sequence ceiling. Cleanest path
   to >= 1.5x at the current alpha/beta. Several weeks of work,
   custom attention masks, separate workstream.

The `>= 1.0x` hard `text_decode_ratio_floor` is unaffected in both gate
profiles (`release_candidate` and, since 2026-05-22, `strict`): KrillLM
no-spec is 1.087-1.10x faster than Ollama on both the 3b and 8b targets,
so the floor holds with margin.

## Goal

Make KrillLM consistently exceed Ollama on decode throughput while preserving
greedy output correctness.

Both gate profiles now accept a hard non-regression floor
(`text_decode_ratio_floor >= 1.0x`) and treat the `>= 1.5x` decode target as
advisory: `release_candidate` since 2026-05-16, and `strict` since
2026-05-22 (owner-accepted; see `docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md`),
because the `>= 1.5x` target is structurally unreachable on M-series with
available draft models. This workstream is how a genuine `>= 1.5x` decode
ratio, and the re-promotion of the target back to hard, would be earned.

## Current Problem

Dense one-token-at-a-time decode is weight-bandwidth bound. On tiny 4-bit
Gemma 4 E2B, Ollama/llama.cpp Metal kernels are competitive enough that
micro-optimizations are unlikely to produce a durable `>= 1.5x` ratio.

## Candidate Approaches

- Self-speculative decoding for Gemma 4.
- Draft model with matching tokenizer/vocab.
- Medusa-style heads if a compatible checkpoint path exists.
- Better integration of existing `SpeculativeDecoder` with Gemma 4 cache,
  tokenizer, and greedy verification.

## Key Files

```text
Sources/KLMEngine/SpeculativeDecoder.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCache/KVCache.swift
Sources/KLMCache/QuantizedKVCache.swift
tools/gemma4_multimodal_benchmark.py
tools/release_gate.py
docs/RELEASE_GATE_DECODE_PROPOSAL.md
```

## Implementation Phases

1. Measure current per-token decode with clean server-mode reports.
2. Select drafting strategy.
3. Implement draft path without breaking greedy parity.
4. Track accepted tokens, rejected tokens, draft depth, and effective tok/s.
5. Extend benchmark reports with speculative metadata.
6. Promote strict decode gate only after results have margin.

## Acceptance

- Greedy output matches baseline for deterministic prompts.
- Cache state remains correct after accepted and rejected draft tokens.
- Benchmark report records draft metadata.
- `text_decode_ratio_floor >= 1.0x` holds under both gate profiles on the
  accepted report (KrillLM never decodes slower than Ollama). The
  `>= 1.5x` target is the tracked advisory aspiration; it re-promotes to
  a hard gate per the re-promotion contract once genuinely earned.
- No regression to text wall time, text TTFT, memory, or Gemma 4 image path.

## Non-Goals

- Do not *silently* hide decode misses. A gate-semantics change (such as
  the 2026-05-22 strict advisory demotion) must be owner-accepted,
  recorded in a proposal doc, and keep the miss visible as an advisory
  WARN plus a caveat - never a quiet relaxation.
- Do not use speculative decoding when it changes deterministic output.
- Do not make Gemma 4-only assumptions leak into unrelated families.
