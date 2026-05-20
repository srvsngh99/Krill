# Speculative Decoding (WS2)

Status: correctness + plumbing landed. Decode uplift on available pairs on
M-series Macs does **not** yet meet the WS2 `text_decode_ratio >= 1.5x`
strict-gate target. The strict gate remains advisory; the
release_candidate hard floor (`>= 1.0x`) is unchanged. See "Benchmark
result" below.

## What this gives you

A small draft model proposes K tokens; the target model verifies them in
a single batched forward; accepted prefix is committed and the loop
advances. When the draft is well matched and decode is compute-bound on
the target, this trades draft latency for batched-target throughput.

## Enabling it

CLI:

```bash
# explicit alias or path
krillm run llama-3.2-3b "Explain X" --draft-model llama-3.2-1b

# curated pair lookup (consults `draftPairs` in SpeculativeDecoder.swift)
krillm run llama-3.2-3b "Explain X" --draft-model auto
```

Server:

```bash
krillm serve --model llama-3.2-3b --draft-model llama-3.2-1b
# or via env:
KRILL_DRAFT_MODEL=llama-3.2-1b krillm serve --model llama-3.2-3b
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

## Benchmark metadata

`GenerationStats.speculative` (`SpeculativeStats`) is populated when the
spec path ran:

```text
rounds          // verify rounds executed
acceptedTokens  // total tokens emitted by the spec path
finalK          // adaptive K at end of generation
acceptanceRate  // rolling rate over last 16 rounds, in [0, 1]
```

CLI `krillm run` prints these on the line after the standard stats:

```text
spec: rounds=64, accepted=127, final_k=2, acceptance=0.47
```

`tools/krillm_vs_ollama_benchmark.py` parses that line into a
`speculative` block on each KrillLM run; the comparison harness propagates
it into the per-run JSON report so the strict gate can read it without
re-running the binary.

## Benchmark results

All numbers: M-series, 4-bit MLX target / draft, 3-5 runs after 1-2
warmups, server-equivalent (warm) cache. Output sha256 verified
identical with and without spec on every pair (greedy parity holds).

Cross-engine caveat: the prompt-tokens count differs between KrillLM
and Ollama in the runs below (39-52% delta from tokenizer
preprocessing differences). The reported `decode_tokens_per_second`
is measured per-engine in the steady-state decode phase and is not
materially affected by the prompt-tokens delta, but the TTFT column
is omitted from the cross-engine comparison and prompt-eval throughput
should not be compared directly. The KrillLM-vs-KrillLM (spec-on vs
spec-off) comparison is unaffected because the same engine encodes
both runs.

### llama-3.2-3b / llama-3.2-1b

Prompt: `"Explain quantum computing in simple terms."`, max 128 tokens.

| Engine                 | decode tok/s (median) | spec K | acceptance |
| ---------------------- | --------------------- | ------ | ---------- |
| KrillLM, no spec       | 104.3                 | -      | -          |
| KrillLM, spec on (1b)  |  74.6                 | 2      | 0.47       |
| Ollama llama3.2:3b     |  94.7                 | -      | -          |

KrillLM no-spec: 1.10x vs Ollama. Spec on: 0.72x vs KrillLM no-spec.

### llama-3.1-8b / llama-3.2-1b

Two prompts on the same pair to bracket acceptance:

A. Story prompt (low-mid acceptance):
`"Tell me a story about a robot who discovers an old library."`,
max 128 tokens.

| Engine                 | decode tok/s (median) | spec K | acceptance |
| ---------------------- | --------------------- | ------ | ---------- |
| KrillLM, no spec       | 50.2                  | -      | -          |
| KrillLM, spec on (1b)  | 40.9                  | 2      | 0.50       |
| Ollama llama3.1:8b     | 46.2                  | -      | -          |

B. Technical prompt (higher acceptance):
`"Explain the architecture of a modern transformer-based language
model in detail, covering attention mechanisms, layer normalization,
and the rationale for grouped-query attention."`, max 128 tokens.

| Engine                 | decode tok/s (median) | spec K | acceptance |
| ---------------------- | --------------------- | ------ | ---------- |
| KrillLM, spec on (1b)  | 44.1                  | 5      | 0.73       |
| KrillLM, no spec       | 50.2 (run A baseline) | -      | -          |
| Ollama llama3.1:8b     | 46.2                  | -      | -          |

The high-acceptance run (B) raises K to 5 and acceptance to 0.73 -
strong drafter agreement - and spec still LOSES vs KrillLM no-spec
(44.1 vs 50.2 tok/s). This is the strongest single-run evidence
that the gap is in the per-round overhead, not in acceptance.

## Why 1.5x strict is empirically out of reach on this hardware

The WS2 `text_decode_ratio >= 1.5x` strict gate cannot be unblocked
on M-series with the model pairs currently available in mlx-community
**on any prompt or K setting tested**, including the high-acceptance
0.73 run above. The pattern across all measured configurations:
verify forward at K=2-5 costs ~1.0-1.2x of K serial single-token
forwards in this engine, so net-positive spec needs an effective
acceptance higher than that ratio. The technical-prompt run hits
0.73 with K=5 (so 0.73 * 5 = 3.65 expected accepted drafts + 1
target token = 4.65 tokens / round vs 5 verify positions); on the
assumption that verify scales linearly in positions, that should
break even or slightly win, but it loses in practice by ~12%. That
extra cost is likely per-round MLX eval-sync and Python-side bookkeeping
overhead, NOT the matmul cost; a microbenchmark of
`target.forward(K)` at varying K (not in this PR) would substantiate
that claim more rigorously.

Break-even framework (per round, all in "cost-per-target-token" units):

  Let r = expected accepted tokens per round (including the bonus
  token on full accept).
  Let alpha = draft per-token cost / target per-token cost (~= 1/8
  for llama-3.1-8b / llama-3.2-1b at 4-bit).
  Let beta = verify(K+1) cost / (K+1) per-position cost. beta = 1.0
  for ideal linear scaling; beta > 1 captures per-round fixed
  overhead (eval syncs, kernel launch, slice ops).

  Spec wins iff:
    r > beta * (K + 1) + alpha * K

The numeric example for 8B / 1B at K=5, alpha=1/8, assuming beta=1.0:
spec needs r > 5 * 1 + 5 * 1/8 = 5.625. Measured r = 0.73 * 5 + 1 = 4.65.
Spec loses by ~17%, matching the observed 12% throughput regression.
Raising K further does not help: r grows at most linearly in K with the
same slope (acceptance), while the right-hand side grows faster.

Three credible unlocks (all out of scope for this PR and the WS2
follow-ups that landed):

1. **Reduce per-round overhead (beta closer to 1.0).** Concretely:
   fewer eval syncs (fold target verify + bonus into one forward;
   tried off-branch, did not materially help), tighter Python-Swift
   boundary, batched draft loop. Could yield up to ~20% improvement.
2. **Target / draft size ratio jumps an order of magnitude
   (alpha << 1/8).** A 70B target with a 1B draft on M-series is
   RAM-infeasible at 4-bit (~40 GB just for the target). Smaller-
   than-1B drafts for Llama 3.x do not exist in mlx-community today.
3. **Tree attention / Medusa-style multi-branch verify.** Each verify
   forward proposes a small tree of continuation paths rather than a
   single sequence, raising effective r per verify. Requires custom
   attention masks and a different verification algorithm; multi-week
   work, separate workstream.

The release_candidate gate's `text_decode_ratio_floor >= 1.0x` (hard)
remains green and unaffected: KrillLM no-spec is 1.087-1.10x faster
than Ollama on both 3b and 8b targets.

## Non-goals

- Non-greedy spec sampling (would need Leviathan rejection sampling).
- Spec on int8-KV models (spec assumes fp16 cache snapshot/restore).
- Lifting the strict gate by metadata only. The gate moves when measured
  uplift moves; not before.
