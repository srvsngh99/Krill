# WS2: Speculative Decoding

Status: in progress (correctness + plumbing landed; strict 1.5x not yet).
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

## What still does not pass

The `text_decode_ratio >= 1.5x` strict gate. Benchmarked on two target
sizes (3B and 8B) on M-series 4-bit MLX with the smallest available
mlx-community drafter (llama-3.2-1b):

| Target            | KrillLM no-spec | KrillLM spec | Ollama  |
| ----------------- | --------------- | ------------ | ------- |
| llama-3.2-3b      | 104.3 tok/s     | 74.6 tok/s   | 94.7    |
| llama-3.1-8b      | 50.2 tok/s      | 40.9 tok/s   | 46.2    |

Output sha256 is identical with and without spec on every pair
(greedy parity verified); the throughput regression is a structural
property of MLX batched-forward cost on this hardware. See
`docs/SPECULATIVE_DECODING.md` for the derivation (verify is linear
in K, so spec wins requires `expected_accepted > ~1.1 * K`, which we
measure below; raising K only inflates verify cost without
proportional gain).

The three credible unlocks (all out of scope for this PR and the
WS2 follow-ups that landed):

1. Upstream mlx-swift kernel changes that make batched forward
   sublinear in K.
2. A ~70B target paired with a 1B draft (RAM-infeasible on
   M-series at 4-bit, ~40 GB just for the target).
3. Tree attention / Medusa-style multi-branch verify - several
   weeks of work, custom attention masks, separate workstream.

Strict gate stays advisory; release_candidate `>= 1.0x` hard floor is
unaffected (KrillLM no-spec is 1.087-1.10x faster than Ollama on both
3b and 8b targets).

## Goal

Make KrillLM consistently exceed Ollama on decode throughput while preserving
greedy output correctness.

The current `release_candidate` profile accepts a hard non-regression floor
(`text_decode_ratio_floor >= 1.0x`) and treats the `>= 1.5x` decode target as
advisory. Strict keeps `text_decode_ratio >= 1.5x` hard. This workstream is
how the strict decode gap closes.

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
- `text_decode_ratio >= 1.5x` under strict on the accepted report.
- No regression to text wall time, text TTFT, memory, or Gemma 4 image path.

## Non-Goals

- Do not hide decode misses by changing the gate.
- Do not use speculative decoding when it changes deterministic output.
- Do not make Gemma 4-only assumptions leak into unrelated families.
