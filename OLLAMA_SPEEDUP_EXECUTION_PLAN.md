# KrillLM Ollama Speedup Execution Plan

Local handoff for the next agent/session.

Last updated: 2026-05-13 (PR #13 in progress on `feat/peak-memory-sampling`)
Base branch: `main`
Base commit: `2f3386a` (merged PR #12); PR #13 sits on top
Machine target: Apple Silicon M4 Pro, 24 GB RAM

## Current Position

Two follow-up PRs have shipped on top of the PR #9 release-readiness baseline.
The product goal — beating Ollama by 1.5x to 3x with honest inputs — is closer,
but Workstreams 1, 2, and 3 below are still required before we can promote
audio out of `out_of_scope` and prefill TPS out of advisory.

### What landed since PR #9

- **PR #13 (`feat: peak-memory sampling + release_candidate memory hard-gate`,
  in progress on `feat/peak-memory-sampling`).**
  `gemma4_multimodal_benchmark.py` samples each engine's process-tree
  memory from a daemon thread (50 ms poll). On macOS the per-PID number
  is `phys_footprint` from `proc_pid_rusage(RUSAGE_INFO_V2)` — the same
  figure Activity Monitor reports, which counts resident mmap'd pages
  (KrillLM's safetensors weights). Other platforms fall back to RSS.
  Each measured run records `peak_memory_gb` and `peak_memory_basis`.
  `release_gate.py` promotes `memory_ratio` to `hard` under
  `release_candidate`, with an automatic downgrade to advisory whenever
  quantization classes differ; `memory_ratio` is excluded from the
  geometric-mean speedup headline because footprint is not a speed
  dimension. New unit tests: `tools/test_memory_sampling.py` (17) and
  six new gate tests covering the conditional downgrade. The fresh
  `.build/benchmarks/v5-mm.json` shows the canonical comparison is
  actually class-equal (KrillLM affine 4-bit MLX vs Ollama Q4_K_M
  GGUF) — v4's "bf16-vs-Q4" framing was a metadata bug from invoking
  the bench with a registry name instead of the local model path.
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
python3 -m unittest tools.test_release_gate    -> 13 tests pass (was 7)
python3 -m unittest tools.test_memory_sampling -> 14 tests pass (new)
```

### Current gate verdict (`.build/benchmarks/v5-mm.json`, refreshed 2026-05-13)

```text
strict profile                            -> exit 1
release_candidate --allow-dtype-mismatch  -> exit 1
```

Per-metric breakdown under `release_candidate`:

```text
HARD  text_wall_ratio     0.6351x   target <= 0.67    PASS
HARD  text_ttft_ratio     0.2126x   target <= 0.67    PASS
HARD  image_wall_ratio    0.6656x   target <= 0.67    PASS  (margin <1%)
HARD  text_decode_ratio   1.1738x   target >= 1.5     FAIL
HARD  memory_ratio        1.1447x   target <= 1.0     FAIL  (4-bit-vs-4-bit;
                                                          auto-downgrade does
                                                          not apply)

ADV   text_prefill_ratio  1.2231x   target >= 1.5     WARN (advisory)
ADV   image_prefill_ratio 0.8899x   target >= 1.5     WARN (advisory)

SKIP  audio_wall_ratio    3.7868x   target <= 0.67    out_of_scope
SKIP  audio_prefill_ratio  N/A      target >= 1.5     out_of_scope

geometric mean speedup     1.157x   (memory_ratio excluded from headline)

KrillLM phys_footprint:    text 9.611 GB / image 9.611 GB / audio 10.481 GB
Ollama  phys_footprint:    text 8.396 GB / image 8.495 GB / audio 8.511 GB
```

The release-candidate gate is currently red on `text_decode_ratio` and
`memory_ratio`. Two notes on interpreting the regression vs the v4-mm
snapshot:

- **The v4 "bf16-vs-Q4" framing was wrong.** v4 was invoked with
  `--krill-model gemma-4-e2b` (registry name); `krill_quantization()`
  could not load the local config (HuggingFace 401) and recorded
  `quantization_class: "unknown"`, which made the gate's
  cross-quantization auto-downgrade trigger spuriously. With the local
  path the model is correctly identified as 4-bit affine MLX, the
  comparison is class-equal with Ollama Q4_K_M, and `memory_ratio`
  is genuinely hard-gateable.
- **KrillLM's text decode is unchanged.** v4 measured 110.30 tok/s; v5
  measures 110.36 tok/s. The ratio dropped from 1.50x → 1.17x because
  Ollama happened to run faster (73 → 94 tok/s) on v5's daemon. Both
  numbers are honest; the gate is sensitive to Ollama variance.

Do not tag a production release until either:

1. `strict` exits 0 against the accepted report, or
2. `release_candidate` is the agreed gate, every advisory is reviewed, every
   `out_of_scope` is documented, every hard miss is closed (currently
   `text_decode_ratio` and `memory_ratio`), and the report is committed
   alongside the release notes.

## Goal For The Next PR

Item 1 ("wire peak-memory sampling") is in flight on PR #13 — the harness
records peak RSS/MLX-Metal peak per run and the gate hard-gates
`memory_ratio` under `release_candidate` (with the documented
quant-class-mismatch auto-downgrade). Once PR #13 merges, pick **one** of
the remaining Workstreams (1, 2, 3) below and ship it. Recommended order
by leverage and scope:

1. **Workstream 3, narrow slice:** profile text_prefill, find one optimization
   that pushes the ratio from 1.45x to ≥1.5x. Then re-promote
   `text_prefill_ratio` to hard under `release_candidate`. Image prefill stays
   advisory until the metric is redefined to count vision-encoder time.
2. **Workstream 2:** add Gemma 4-compatible self-speculative decoding so
   `text_decode_ratio` consistently exceeds 1.5x with margin. Larger scope;
   requires a draft path that doesn't break greedy parity.
3. **Workstream 1:** port Gemma 4's audio Conformer to native Swift+MLX so
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
  metrics as advisory (text_prefill, image_prefill).
  Audio metrics are out_of_scope until Workstream 1 lands.
- Missing hard metrics fail the gate; we cannot claim a metric passes
  without measuring it.
- `memory_ratio` is **hard** under `release_candidate` (PR #13). It
  auto-downgrades to advisory whenever quantization classes differ
  (KrillLM bf16 vs Ollama Q4_K_M today), via
  `release_gate.py:resolve_metric_kinds`; the downgrade is recorded in
  `scope.memory_ratio` and added to `caveats`. `strict` keeps it hard
  regardless of dtype. Re-promotion to a fully hard verdict happens
  automatically once a quantization-class-equal comparison is supplied.
- `memory_ratio` is excluded from the geometric-mean speedup headline
  (`SPEEDUP_EXCLUDED_METRICS`) because footprint is not a speed dimension
  and folding the bf16-vs-Q4 ratio into a perf headline would understate
  the speed result for reasons unrelated to speed.
- Out-of-scope skips are recorded in `scope_skipped_metrics[]` with a
  human-readable reason so the omission stays auditable; advisory failures
  print a `WARN` glyph and are tagged `[advisory]` but do not break the gate.
- The gate report records `profile` and `kv_cache_dtype` at top level.
- `tools/gemma4_multimodal_benchmark.py` records `benchmark.kv_cache_dtype`
  (sourced from `KRILL_KV_CACHE_DTYPE`) and a `memory_sampling` block
  describing how peak memory was measured (RSS poll interval, resolved
  PIDs, basis legend, notes).
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

v4-mm.json was the historical snapshot; see the "Current gate verdict"
section above for the refreshed v5-mm.json results that include
peak-memory sampling and the corrected (class-equal) quantization
identification.

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

Pass the **full local path** to `--krill-model` (e.g.
`/Users/.../models/blobs/gemma-4-e2b`), not the registry name. The bench
reads `quantization_class` from the local `config.json`; with a registry
name it falls back to a HuggingFace lookup that 401s for private/local
models and records `quantization_class: "unknown"`, which makes the
gate's class-equality check (and therefore the `memory_ratio`
auto-downgrade) misfire. v4-mm.json's `quantization.comparison.class_equal:
false` was an artifact of this; v5-mm.json shows the actual
class-equal 4-bit-vs-4-bit comparison.

Pass `--krillm-server-pid <pid>` when sampling memory under
`--krillm-image-mode native_server` if the auto-detected `pgrep -f
'krillm.*serve'` would also match a wrapper shell (Claude harness, tmux,
etc.). The override gives a clean per-process number; the auto-detect
is a few-MB shell wrapper away from clean.

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
