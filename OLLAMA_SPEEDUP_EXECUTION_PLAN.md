# KrillLM Ollama Speedup Execution Plan

Local handoff for the next agent/session.

Last updated: 2026-05-11 (post PR #12 merge)
Base branch: `main`
Base commit: `2f3386a` (merged PR #12)
Machine target: Apple Silicon M4 Pro, 24 GB RAM

## Current Position

Two follow-up PRs have shipped on top of the PR #9 release-readiness baseline.
The product goal — beating Ollama by 1.5x to 3x with honest inputs — is closer,
but Workstreams 1, 2, and 3 below are still required before we can promote
audio out of `out_of_scope` and prefill TPS out of advisory.

### What landed since PR #9

- **PR #11 (`perf: compose int8 KV cache with prefix cache`).** int8 KV cache
  and the persistent prefix cache now coexist on the Gemma 4 path. New
  `QuantizedKVSnapshot` carries uint8 K/V plus fp16 scales/zeros, so prefix
  replay avoids dequant→requant. `PrefixCache` gains `storeQuantized` /
  `lookupQuantized` with a dtype-tagged key schema (v3) and a separate
  `.q8.safetensors` on-disk suffix; fp16 and int8 entries are isolated.
  `InferenceEngine` removes the previous int8→no-prefix-cache restriction
  and handles Gemma 4's KV-sharing-suffix correctly.
- **PR #12 (`feat: release_candidate gate profile + kv_cache_dtype in
  reports`).** The release gate now distinguishes hard-gated, advisory, and
  out_of_scope metrics via `tools/release_gate.py --profile <name>`. `strict`
  (default) preserves original behavior; `release_candidate` is the
  defensible path to a release tag. Missing hard metrics now fail the gate.
  `memory_ratio` is advisory until the benchmark records peak memory.
  `tools/gemma4_multimodal_benchmark.py` records `benchmark.kv_cache_dtype`
  from `KRILL_KV_CACHE_DTYPE`; the gate echoes it in the report.

### Achieved (cumulative)

- Server multimodal is wired for Gemma 4.
- Gemma 4 image-only requests use the native Swift SigLIP2 path.
- Gemma 4 audio requests use the persistent `mlx-vlm` bridge.
- Chat image conditioning is fixed and covered by a live regression.
- Multimodal prefix-cache keys include media hashes (schema v2).
- int8 KV cache is opt-in for Gemma 4 and live parity passes.
- **int8 KV + prefix cache compose (PR #11).** Live test
  `QuantizedPrefixCacheLiveTests/testInt8PrefixCacheReplayMatchesColdRun`
  asserts cold/warm runs produce identical greedy tokens.
- **Release gate has profile semantics (PR #12).** Hard / advisory /
  out_of_scope per metric, missing-hard-metric fail, dtype tag in reports.
- OpenAI and Ollama server request shapes are covered.
- Public docs distinguish current support from remaining release blockers.

Latest verified baseline:

```text
make test                       -> 128 tests, 9 skipped, 0 failures
make release                    -> passed
live int8 KV parity             -> passed
live int8 + prefix cache replay -> passed
live image-conditioning regr.   -> passed
python3 -m unittest tools.test_release_gate -> 7 tests pass
```

### Current gate verdict (`.build/benchmarks/v4-mm.json`)

```text
strict profile                  -> exit 1 (unchanged; same FAILs as before)
release_candidate profile       -> exit 0 (with --allow-dtype-mismatch)
                                   on the v4-mm.json snapshot
```

Per-metric breakdown under `release_candidate`:

```text
HARD  text_decode_ratio   1.5030x   target >= 1.5     PASS
HARD  text_wall_ratio     0.5172x   target <= 0.67    PASS
HARD  text_ttft_ratio     0.1173x   target <= 0.67    PASS
HARD  image_wall_ratio    0.5593x   target <= 0.67    PASS

ADV   text_prefill_ratio  1.4498x   target >= 1.5     WARN (advisory)
ADV   image_prefill_ratio 1.0385x   target >= 1.5     WARN (advisory)
ADV   memory_ratio        N/A       target <= 1.0     N/A  (not yet measured)

SKIP  audio_wall_ratio    3.9265x   target <= 0.67    out_of_scope
SKIP  audio_prefill_ratio  N/A      target >= 1.5     out_of_scope

geometric mean speedup            1.418x
```

A `release_candidate` PASS on the existing snapshot is honest **only** because
prefill TPS, memory, and audio are explicitly opted out with documented
rationale (see `docs/BENCHMARKING.md`). Promoting to a production release
requires re-promoting these to hard one at a time.

Do not tag a production release until either:

1. `strict` exits 0 against the accepted report, or
2. `release_candidate` is the agreed gate, every advisory is reviewed, every
   `out_of_scope` is documented, and the report is committed alongside the
   release notes.

## Goal For The Next PR

Pick **one** of the remaining Workstreams (1, 2, 3) below and ship it. The
order I'd recommend, by leverage and scope:

1. **Wire peak-memory sampling** into `gemma4_multimodal_benchmark.py` so
   `memory_ratio` populates and can re-promote to hard. Smallest scope; fully
   retires the contract documented in PR #12.
2. **Workstream 3, narrow slice:** profile text_prefill, find one optimization
   that pushes the ratio from 1.45x to ≥1.5x. Then re-promote
   `text_prefill_ratio` to hard under `release_candidate`. Image prefill stays
   advisory until the metric is redefined to count vision-encoder time.
3. **Workstream 2:** add Gemma 4-compatible self-speculative decoding so
   `text_decode_ratio` consistently exceeds 1.5x with margin. Larger scope;
   requires a draft path that doesn't break greedy parity.
4. **Workstream 1:** port Gemma 4's audio Conformer to native Swift+MLX so
   `audio_*` can leave `out_of_scope`. Largest scope; multi-week effort.

The product claim remains: KrillLM should beat Ollama by 1.5x to 3x on the
accepted benchmark matrix, with equivalent inputs and honest metadata.

## Main Workstreams

### 1. Native Audio In Swift

Current audio path:

```text
HTTP/CLI request -> PythonFallback actor -> persistent mlx-vlm sidecar -> MLX Python
```

This is much better than per-call subprocess startup, but it still loses badly:

```text
audio_wall_ratio = 3.9265x
target           <= 0.67x
```

Required work:

- Port Gemma 4 audio encoder path to Swift+MLX.
- Implement the Conformer audio tower and relative position behavior correctly.
- Implement audio token expansion and projection into language model space.
- Route audio-only and image+audio requests through native Swift when available.
- Keep bridge fallback behind explicit capability or config guard.
- Add live audio quality smoke tests against deterministic fixtures.

Acceptance:

- Native audio path produces coherent output for the sine/silence fixtures.
- Server audio no longer uses `PythonFallback` when native audio is enabled.
- `audio_wall_ratio <= 0.67x` or a revised audio gate is justified and passing.

Key files:

```text
Sources/KLMCore/AudioEncoder.swift
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMEngine/PythonFallback.swift
Sources/KLMServer/Server.swift
Tests/KLMEngineTests/Gemma4SmokeTests.swift
Tests/KLMServerTests/MultimodalEndpointsTests.swift
tools/gemma4_multimodal_benchmark.py
```

### 2. Gemma 4 Speculative Decoding

Text decode barely passes in the multimodal gate and is weaker in the standalone
text benchmark. To consistently reach 1.5x to 3x, we need a Gemma 4-compatible
drafting strategy.

Options:

- Find or produce a small Gemma 4-compatible draft model with matching vocab.
- Implement self-speculative decoding.
- Add Medusa-style draft heads if a compatible checkpoint path is available.
- Improve the existing speculative decoder contract so it can handle Gemma 4
  caches cleanly.

Constraints:

- Do not use Gemma 2 as a Gemma 4 drafter unless vocab/token semantics are
  proven compatible.
- Keep quality deterministic for greedy comparisons.
- Track accepted tokens, rejected tokens, and effective generated tok/s.

Acceptance:

- `text_decode_ratio` consistently exceeds `1.5x` on both standalone text and
  multimodal text tasks.
- No quality regression in greedy smoke tests.
- Spec path is benchmarked with cache mode and draft metadata recorded.

Key files:

```text
Sources/KLMEngine/SpeculativeDecoder.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMCache/KVCache.swift
Sources/KLMCore/Gemma4Model.swift
Tests/KLMEngineTests/SpeculativeDecodingTests.swift
tools/krillm_vs_ollama_benchmark.py
tools/gemma4_multimodal_benchmark.py
```

### 3. Prefill And Kernel Optimization

Remaining failures:

```text
text_prefill_ratio  = 1.4498x, target >= 1.5
image_prefill_ratio = 1.0385x, target >= 1.5
```

For text prefill, we are close. For image prefill, the metric is ambiguous
because the vision cache moves work out of the measured prefill window while
improving user-visible wall time.

Required work:

- Profile prompt prefill by layer and kernel.
- Measure MLX graph materialization and kernel dispatch overhead.
- Investigate fused RMSNorm, attention, and MLP kernels where practical.
- Keep correctness tests around logits, generated prefix, and cache behavior.
- Separate true language prefill from media encoder/projector time in reports.

Acceptance:

- `text_prefill_ratio >= 1.5x`, or a documented reason that the gate should use
  TTFT/wall time for short prompts instead.
- `image_prefill_ratio` is either improved to target or replaced by a better
  multimodal metric with clear rationale.

Key files:

```text
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCore/VisionEncoder.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMSampler/Sampler.swift
tools/release_gate.py
tools/gemma4_multimodal_benchmark.py
docs/BENCHMARKING.md
docs/RELEASE_READINESS_REMEDIATION.md
```

### 4. Release Gate Semantics

Status: implemented. The gate now distinguishes hard-gated, advisory, and
out-of-scope metrics via an explicit profile selected at the command line.

Done:

- `tools/release_gate.py --profile <name>` (default `strict`). The new
  `release_candidate` profile hard-gates user-visible latency
  (text_decode, text_wall, text_ttft, image_wall) and treats prefill TPS
  metrics and memory as advisory (text_prefill, image_prefill, memory).
  Audio metrics are out_of_scope until Workstream 1 lands.
- Missing hard metrics fail the gate; we cannot claim a metric passes
  without measuring it.
- `memory_ratio` is advisory under `release_candidate` until
  `gemma4_multimodal_benchmark.py` records `peak_memory_gb_median` for both
  engines. Re-promote to hard once the benchmark emits it.
- Out-of-scope skips are recorded in `scope_skipped_metrics[]` with a
  human-readable reason so the omission stays auditable; advisory failures
  print a `WARN` glyph and are tagged `[advisory]` but do not break the gate.
- The gate report records `profile` and `kv_cache_dtype` at top level.
- `tools/gemma4_multimodal_benchmark.py` records `benchmark.kv_cache_dtype`
  (sourced from `KRILL_KV_CACHE_DTYPE`).
- `docs/BENCHMARKING.md` documents both profiles, the per-metric kind, and
  the rationale for each downgrade.

Verification:

```text
release_gate.py .build/benchmarks/v4-mm.json
  → exit 1 (strict; unchanged behavior, same failures)

release_gate.py .build/benchmarks/v4-mm.json
  --profile release_candidate --allow-dtype-mismatch
  → exit 0
```

The release_candidate run on v4-mm.json passes the four user-latency hard
gates (text_decode 1.50x, text_wall 0.52x, text_ttft 0.12x, image_wall 0.56x)
with three advisory readings (text_prefill 1.45x WARN, image_prefill 1.04x
WARN, memory N/A) and two documented audio skips. Geometric mean speedup
1.418x. The dtype mismatch (KrillLM bf16 vs Ollama Q4_K_M) remains a
separate, explicit caveat that the operator must opt into via
`--allow-dtype-mismatch`.

### 5. Quantized KV Save/Restore

Status: implemented. int8 KV and prefix cache now compose on the Gemma 4 path.

Done:

- `QuantizedKVSnapshot` carries the raw uint8 K/V plus the fp16 scales/zeros
  so persistence preserves the quantized form (no dequant→requant round trip).
- `QuantizedKVCache.quantizedSnapshot()`, `restoreQuantized(_:)`, and
  `truncate(to:)` mirror the fp16 cache contract.
- `PrefixCache.storeQuantized` / `lookupQuantized` write to a distinct
  `<key>.q8.safetensors` filename and the cache key schema gains a dtype tag
  (v3) so int8 and fp16 entries can never collide.
- `InferenceEngine` removes the `&& !useInt8KV` block on prefix cache and
  dispatches to the quantized snapshot/restore path when int8 is active.
  Truncate-and-re-forward keeps the prompt length stable on a full hit.
- Unit coverage: `Tests/KLMCoreTests/QuantizedPrefixCacheTests.swift`
  (round-trip; cross-dtype isolation in both directions; truncate+update).
- Live coverage:
  `Tests/KLMEngineTests/QuantizedPrefixCacheLiveTests.swift`
  runs the same prompt twice with `kvCacheDtype: "int8"` and a shared
  prefix cache, asserting cold/warm greedy tokens match.

Remaining for the gate report:

- Benchmark reports should record `kv_cache_dtype` and prefix-cache
  hit/miss state per run. The current `gemma4_multimodal_benchmark.py`
  records `cache_mode` but not the KV dtype — wire that through next.

## Benchmark Rules For Next Agent

Do not publish speed claims unless reports are attached or committed.

Required benchmark metadata:

- Git commit.
- Machine and OS.
- KrillLM binary path.
- Ollama version and model.
- Exact KrillLM model path.
- Quantization/dtype metadata.
- Prompt and media SHA256.
- Runs and warmups.
- Seed, temperature, top-p, max tokens.
- Cache mode per result.
- Whether comparison is bit-identical, dtype-class equivalent, or
  quantization-class equivalent.

Use warm-server comparisons for release gating. Do not mix repeated CLI process
startup against a warm Ollama daemon.

Useful commands:

```bash
make test
make release

KLM_GEMMA4_MODEL_PATH=/Users/sourav/.krillm/models/blobs/gemma-4-e2b \
CLANG_MODULE_CACHE_PATH=.build/clang-module-cache \
swift test --filter QuantizedKVCacheIntegrationTests/testInt8AndFp16ProduceSimilarGreedyPrefix --skip-build

KLM_GEMMA4_MODEL_PATH=/Users/sourav/.krillm/models/blobs/gemma-4-e2b \
CLANG_MODULE_CACHE_PATH=.build/clang-module-cache \
swift test --filter MultimodalEndpointsTests/testTwoDifferentImagesProduceDifferentOutputs --skip-build

/Users/sourav/.krillm/venv/bin/python3 tools/release_gate.py \
  .build/benchmarks/v4-mm.json \
  --output .build/benchmarks/next-pr-release-gate.json
```

## Suggested Next PR Breakdown

If multiple agents work in parallel, split ownership as follows:

1. Audio agent:
   - Owns `AudioEncoder`, Gemma 4 audio projection, native audio routing, and
     audio live tests.

2. Decode/speculation agent:
   - Owns speculative decoding, drafter compatibility, and decode benchmarks.

3. Prefill/kernel agent:
   - Owns profiling, prefill optimizations, and any MLX/Metal kernel changes.

4. Gate/docs agent:
   - Owns `release_gate.py`, benchmark semantics, docs, and report validation.

Do not let agents edit the same files in conflicting ways without coordination.
The riskiest overlap is `InferenceEngine.swift`, `Gemma4Model.swift`, and
`gemma4_multimodal_benchmark.py`.

## Definition Of Done For Release Candidate

- `make test` passes.
- `make release` passes.
- Live Gemma 4 text/image/audio tests pass on the local checkpoint.
- Accepted release benchmark report exits `0`.
- Docs and README match the implemented behavior.
- Release notes include benchmark artifacts and caveats.
- No production claim says "faster than Ollama" beyond what the accepted report
  proves.
