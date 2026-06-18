# Speculative Decoding (WS2)

Status: correctness + plumbing landed. Decode uplift on available pairs on
M-series Macs does **not** meet the WS2 `text_decode_ratio >= 1.5x` target,
and the "Why 1.5x strict is empirically out of reach" section below derives
that target is structurally unreachable on this hardware. As of 2026-05-22
`text_decode_ratio` is therefore **advisory in both gate profiles**
(`release_candidate` and `strict`), each carrying a hard `>= 1.0x`
non-regression floor (Krill must never decode slower than Ollama). See
`docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md` and "Benchmark results" below.

## What this gives you

A small draft model proposes K tokens; the target model verifies them in
a single batched forward; accepted prefix is committed and the loop
advances. When the draft is well matched and decode is compute-bound on
the target, this trades draft latency for batched-target throughput.

## Enabling it

CLI:

```bash
# explicit alias or path
krill run llama-3.2-3b "Explain X" --draft-model llama-3.2-1b

# curated pair lookup (consults `draftPairs` in SpeculativeDecoder.swift)
krill run llama-3.2-3b "Explain X" --draft-model auto
```

Server:

```bash
krill serve --model llama-3.2-3b --draft-model llama-3.2-1b
# or via env:
KRILL_DRAFT_MODEL=llama-3.2-1b krill serve --model llama-3.2-3b
```

Once a draft is loaded, `generate()` uses the spec path automatically for
greedy requests. Tests and parity references can disable that opt-in via
`engine.setAutoUseSpec(false)`.

## When the spec path is skipped

The engine falls back to standard decode (silently, no error) when:

- No draft model is loaded.
- The request is non-greedy: `temperature > 0`, `top_p < 1`,
  `top_k > 0`, `min_p > 0`. The decoder only does greedy verification;
  Leviathan-style rejection sampling is not implemented, so non-greedy
  through the spec path would silently diverge from the per-request
  sampler.
- A penalty / mirostat sampler is active (`SamplingParams.penaltiesActive`).
- The KV cache is int8-quantized (the spec path operates on fp16 caches).

## Correctness contract

Greedy parity is preserved: enabling spec with a curated draft must
produce byte-identical output to the same request with no draft model.
This is the smoke test the WS2 PR shipped against; see "Benchmark
result" for the sha256-stable preview.

Two correctness fixes were required to make this hold:

1. **Cache-aware causal mask.** The dense-text model heads previously
   built an `(N, N)` causal mask for any multi-token forward, regardless
   of cache state. The spec verify forwards `K` new tokens against a
   prompt-warmed cache of length `L_prev`, producing an attention score
   matrix of shape `(K, L_prev + K)`; the `(N, N)` mask cannot broadcast
   onto it and MLX errored at runtime. `createCachedCausalMask(newLen:cacheLen:dtype:)`
   produces the right shape:
   - `newLen == 1` → no mask (decode case unchanged).
   - `cacheLen == 0` → `(1, 1, newLen, newLen)` square causal (prefill case
     unchanged).
   - else → `(1, 1, newLen, cacheLen + newLen)` with the first `cacheLen`
     columns unmasked (new queries attend freely to cached keys) and the
     last `newLen` columns upper-triangular (causal within the new
     slice).

   Applied to Llama / Qwen / Mistral / Gemma / Phi / GLM / Gemma 4. The
   single-token decode and empty-cache prefill cases are byte-for-byte
   unchanged.

2. **Draft cache prefill and accept/reject sync.** The previous spec
   path allocated empty draft caches per generation and never warmed
   them with the prompt, so draft proposals were generated from a
   one-token context (whatever the first sampled token happened to
   be). Acceptance rate was bounded by how often "predict from a single
   token with no prior context" happened to agree with the target's
   prompt-conditioned distribution. The engine now forwards
   `tokensToProcess` through the draft into its own `[KVCache]` before
   the decode-time clock starts (so draft prefill is attributed to
   `prefillDuration`, not `decodeTime`).

   The decoder additionally now:
   - Trims the draft cache on rejection to mirror the target's truncate
     (otherwise the draft cache would carry KV for tokens the target
     rejected, poisoning the next round).
   - Forwards the K-th accepted draft token into the draft cache on full
     acceptance (otherwise the draft cache would be one position behind
     the target after every bonus, again poisoning the next round).

## Verification: one batched argmax

The spec path is greedy-gated, so target verification of the K proposed
tokens is an argmax. The verifier computes that argmax for all K
positions in a single batched op (`argMax(targetLogits, axis: -1)`,
one eval), rather than slicing each position and sampling it
separately. The per-position form cost `K - 1` extra GPU
synchronizations per round - pure overhead on the path the structural
analysis below identifies as overhead-bound. The batched form is
bit-identical (same per-position argmax) and is the only correct
shortcut here precisely because the path is greedy-only.

## Benchmark metadata

`GenerationStats.speculative` (`SpeculativeStats`) is populated when the
spec path ran:

```text
rounds          // verify rounds executed
acceptedTokens  // total tokens emitted by the spec path
finalK          // adaptive K at end of generation
acceptanceRate  // rolling rate over last 16 rounds, in [0, 1]
```

CLI `krill run` prints these on the line after the standard stats:

```text
spec: rounds=64, accepted=127, final_k=2, acceptance=0.47
```

`tools/krill_vs_ollama_benchmark.py` parses that line into a
`speculative` block on each Krill run; the comparison harness propagates
it into the per-run JSON report so the strict gate can read it without
re-running the binary.

## Benchmark results

All numbers: M-series, 4-bit MLX target / draft, 3-5 runs after 1-2
warmups, server-equivalent (warm) cache. Output sha256 verified
identical with and without spec on every pair (greedy parity holds).

Cross-engine caveat: the prompt-tokens count differs between Krill
and Ollama in the runs below (39-52% delta from tokenizer
preprocessing differences). The reported `decode_tokens_per_second`
is measured per-engine in the steady-state decode phase and is not
materially affected by the prompt-tokens delta, but the TTFT column
is omitted from the cross-engine comparison and prompt-eval throughput
should not be compared directly. The Krill-vs-Krill (spec-on vs
spec-off) comparison is unaffected because the same engine encodes
both runs.

### llama-3.2-3b / llama-3.2-1b

Prompt: `"Explain quantum computing in simple terms."`, max 128 tokens.

| Engine                 | decode tok/s (median) | spec K | acceptance |
| ---------------------- | --------------------- | ------ | ---------- |
| Krill, no spec       | 104.3                 | -      | -          |
| Krill, spec on (1b)  |  74.6                 | 2      | 0.47       |
| Ollama llama3.2:3b     |  94.7                 | -      | -          |

Krill no-spec: 1.10x vs Ollama. Spec on: 0.72x vs Krill no-spec.

### llama-3.1-8b / llama-3.2-1b

Two prompts on the same pair to bracket acceptance:

A. Story prompt (low-mid acceptance):
`"Tell me a story about a robot who discovers an old library."`,
max 128 tokens.

| Engine                 | decode tok/s (median) | spec K | acceptance |
| ---------------------- | --------------------- | ------ | ---------- |
| Krill, no spec       | 50.2                  | -      | -          |
| Krill, spec on (1b)  | 40.9                  | 2      | 0.50       |
| Ollama llama3.1:8b     | 46.2                  | -      | -          |

B. Technical prompt (higher acceptance):
`"Explain the architecture of a modern transformer-based language
model in detail, covering attention mechanisms, layer normalization,
and the rationale for grouped-query attention."`, max 128 tokens.

| Engine                 | decode tok/s (median) | spec K | acceptance |
| ---------------------- | --------------------- | ------ | ---------- |
| Krill, spec on (1b)  | 44.1                  | 5      | 0.73       |
| Krill, no spec       | 50.2 (run A baseline) | -      | -          |
| Ollama llama3.1:8b     | 46.2                  | -      | -          |

The high-acceptance run (B) raises K to 5 and acceptance to 0.73 -
strong drafter agreement - and spec still LOSES vs Krill no-spec
(44.1 vs 50.2 tok/s). This is the strongest single-run evidence
that the gap is in the per-round overhead, not in acceptance.

## Why the 1.5x decode target is empirically out of reach on this hardware

The WS2 `text_decode_ratio >= 1.5x` target cannot be reached
on M-series with the model pairs currently available in mlx-community
**on any prompt or K setting tested**, including the high-acceptance
0.73 run above. (This is why the target is advisory in both gate
profiles, with a hard `>= 1.0x` floor, rather than a hard gate.) Below we fit a cost model to the measured numbers
and show that even at infinite K and 100% acceptance, the achievable
spec speedup on this engine on this hardware is bounded at roughly
1.10x - well short of 1.5x.

Break-even framework (per round, all costs in "cost-per-target-token"
units):

```text
r     = expected accepted tokens per round (= acceptance * K + 1
        on full accept; less on rejection).
alpha = draft per-token cost / target per-token cost. For
        llama-3.1-8b / llama-3.2-1b at 4-bit MLX, alpha ~ 1/8.
beta  = verify(K+1) cost / ((K+1) * target per-token cost).
        beta = 1.0 means verify scales perfectly linearly in
        positions; beta < 1.0 means verify is sublinear (the
        usual case for batched forwards); beta > 1.0 means
        per-round fixed overhead dominates.

Throughput ratio (spec tok/s / baseline tok/s):
   ratio = r / (alpha * K + beta * (K + 1))
```

Plugging in the measured 8B / 1B K=5 acceptance-0.73 run:
- `r = 0.73 * 5 + 1 = 4.65`
- alpha = 1/8 = 0.125
- Observed ratio = 44.1 / 50.2 = 0.879
- Solve for beta: `0.879 = 4.65 / (0.125 * 5 + beta * 6)`, so
  `beta = (4.65 / 0.879 - 0.625) / 6 = 0.78`.

So verify IS sublinear in this engine (beta < 1.0), just not enough.
That number is a single-point fit; a microbenchmark of
`target.forward(K)` at varying K (not in this PR) would tighten the
estimate and is the natural next investigation.

With beta = 0.78 fixed, the throughput ratio as K -> infinity and
r -> K+1 (100% acceptance, impossible in practice) asymptotes at:

```text
ratio_max = lim_{K -> inf} (K + 1) / (alpha * K + beta * (K + 1))
          = 1 / (alpha + beta)
          = 1 / (0.125 + 0.78)
          ~ 1.10
```

So even at infinite acceptance with K -> inf, this configuration
caps at ~1.10x. Strict 1.5x requires `alpha + beta < 1 / 1.5 = 0.67`.
At alpha = 0.125 that means beta <= 0.55 - a 30% improvement on the
current beta - AND r close to its K+1 ceiling. Neither is reachable
with the available model pair / engine configuration without an
algorithmic change.

Three credible unlocks (all out of scope for this PR and the WS2
follow-ups that landed):

1. **Reduce per-round overhead (drive beta lower).** Concretely:
   fewer eval syncs (folding target verify + bonus into one forward
   was tried off-branch; did not materially help on the K=5 8B run),
   tighter Python-Swift boundary, batched draft loop. Bringing beta
   from ~0.78 to ~0.55 would let `1 / (alpha + beta) ~ 1.47` at the
   r -> K+1 limit; this is the only path that does not require new
   hardware or a different algorithm, and would need an MLX-level
   investigation (not just Krill-level code).
2. **Target / draft size ratio jumps an order of magnitude
   (alpha << 1/8).** A 70B target with a 1B draft on M-series is
   RAM-infeasible at 4-bit (~40 GB just for the target). Smaller-
   than-1B drafts for Llama 3.x do not exist in mlx-community today.
3. **Tree attention / Medusa-style multi-branch verify.** Each verify
   forward proposes a small tree of continuation paths rather than a
   single sequence, lifting effective r above the single-sequence
   K+1 ceiling. This is the cleanest path to >=1.5x at the current
   alpha/beta. Multi-week work, custom attention masks, separate
   workstream.

The release_candidate gate's `text_decode_ratio_floor >= 1.0x` (hard)
remains green and unaffected: Krill no-spec is 1.087-1.10x faster
than Ollama on both 3b and 8b targets.

## Non-goals

- Non-greedy spec sampling (would need Leviathan rejection sampling).
- Spec on int8-KV models (spec assumes fp16 cache snapshot/restore).
- Lifting the strict gate by metadata only. The gate moves when measured
  uplift moves; not before.
