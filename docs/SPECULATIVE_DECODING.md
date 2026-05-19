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

## Benchmark result (this PR)

Target / draft: `llama-3.2-3b` / `llama-3.2-1b` (both 4-bit MLX).
Prompt: `"Explain quantum computing in simple terms."`, max 128 tokens,
5 measured runs after 2 warmups, server-equivalent (warm) cache, M-series.

| Engine                 | decode tok/s (median) | TTFT (median) |
| ---------------------- | --------------------- | ------------- |
| KrillLM, no spec       | 104.3                 | 86 ms         |
| KrillLM, spec on (1b)  |  74.6                 | 114 ms        |
| Ollama llama3.2:3b     |  94.7                 | 78 ms         |

Spec acceptance: 0.47 with adaptive K saturating at 2. Output sha256 is
identical with and without spec (greedy parity verified).

Result: on this pair on M-series, spec **regresses** decode throughput
because per-round verify overhead exceeds the gain from accepting <1
extra token per round at K=2. KrillLM's no-spec decode is already 1.10x
faster than Ollama on this model, but the WS2 `text_decode_ratio >= 1.5x`
strict gate is not unblocked by this PR.

What would unlock 1.5x strict (future work):
- Larger target where decode is compute-bound (e.g. 8B+), so verify's
  amortization wins by more.
- A draft model with a higher size ratio to the target (1:10 or smaller)
  and shared tokenizer/vocab so proposals land more often.
- Higher acceptance: tune `minK`/`maxK`/window or evaluate tree
  attention to verify multiple branches per forward.

## Non-goals

- Non-greedy spec sampling (would need Leviathan rejection sampling).
- Spec on int8-KV models (spec assumes fp16 cache snapshot/restore).
- Lifting the strict gate by metadata only. The gate moves when measured
  uplift moves; not before.
