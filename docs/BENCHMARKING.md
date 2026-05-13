# Benchmarking Guide

## Quick Start

```bash
# Build release binary
make release

# Text-only KrillLM vs Ollama (CLI subprocess mode)
make bench-compare KRILL_MODEL=llama-3.2-1b OLLAMA_MODEL=llama3.2:1b

# Server-mode (warm vs warm, eliminates process startup)
krillm serve --model llama-3.2-1b --port 11435 &
make bench-compare KRILLM_URL=http://127.0.0.1:11435

# Gemma4 multimodal (text/image/audio)
make setup-mlx-vlm
make bench-gemma4-multimodal

# Release gate (pass/fail evaluation)
make bench-release-gate
```

## Benchmark Modes

### CLI Mode (default)
- Spawns `krillm run` subprocess per request
- Includes process startup, model loading overhead
- Use for: measuring end-user experience

### Server Mode (`--krillm-url` / `KRILLM_URL`)
- Benchmarks against a running KrillLM HTTP server
- Model stays loaded, no process startup
- Use for: fair comparison against Ollama daemon (both warm)

## Tools

### krillm_vs_ollama_benchmark.py
Reproducible KrillLM vs Ollama comparison.

**Key flags:**
- `--krillm-url URL` — server mode (no subprocess)
- `--runs N` — measured runs (default 5)
- `--warmup N` — warmup runs (default 2)
- `--max-tokens N` — tokens per run (default 32)
- `--output PATH` — JSON report path

**Report fields:**
- `decode_tokens_per_second` — decode throughput
- `prefill_tokens_per_second` — prefill throughput
- `wall_time_s` — total wall time (includes network for server mode)
- `ttft_ms_wall` — time to first token (wall clock)
- `ttft_ms_server` — server-side TTFT (server mode only)
- `output_sha256` — determinism verification

**Cache warning:** Server mode with repeated prompts benefits from prefix cache. Reports include `prefix_cache_active: true` and warn when prefill speed varies >3x across runs.

### gemma4_multimodal_benchmark.py
Gemma4 text/image/audio comparison.

**Key flags:**
- `--krillm-url URL` — server mode (text, image, and audio supported on Gemma 4)
- `--engine both|krillm|ollama` — single-engine or comparison
- `--runs N` / `--warmup N`

**Server mode notes:** image and audio payloads are sent as base64 in the standard Ollama shape. Audio runs through the `mlx-vlm` bridge, so the server needs `mlx-vlm` installed for audio benchmarks (`make setup-mlx-vlm`).

### release_gate.py
Evaluates benchmark reports against performance thresholds.

**All thresholds (must match `tools/release_gate.py`):**
| Metric | Target | Direction |
|--------|--------|-----------|
| text_decode_ratio | >= 1.5x | Higher is better |
| text_wall_ratio | <= 0.67x | Lower is better |
| text_ttft_ratio | <= 0.67x | Lower is better |
| text_prefill_ratio | >= 1.5x | Higher is better |
| image_wall_ratio | <= 0.67x | Lower is better |
| image_prefill_ratio | >= 1.5x | Higher is better |
| audio_wall_ratio | <= 0.67x | Lower is better |
| audio_prefill_ratio | >= 1.5x | Higher is better |
| memory_ratio | <= 1.0x | Lower is better |

**Output:** JSON report + colored terminal summary with per-metric pass/fail, geometric mean speedup, worst metric, bottleneck classification.

### Gate profiles

`release_gate.py --profile <name>` selects which metrics are hard-gated. The
profile is recorded in the gate report so audit trails are unambiguous.

| Profile | Behavior |
|---------|----------|
| `strict` (default) | Every threshold is hard-gated. Preserves the original behavior; existing CI invocations do not need to change. |
| `release_candidate` | Hard-gates user-visible latency metrics. Treats prefill TPS and memory as advisory. Scopes audio metrics out until native Swift audio (Workstream 1) lands. |

**Per-metric kind under `release_candidate`:**

| Metric | Kind | Rationale |
|--------|------|-----------|
| text_decode_ratio   | hard | Core decode throughput claim. |
| text_wall_ratio     | hard | User-visible total latency. |
| text_ttft_ratio     | hard | User-visible first-token latency. |
| text_prefill_ratio  | advisory | Wall time and TTFT already gate user latency; prefill TPS is noisy on short prompts. |
| image_wall_ratio    | hard | User-visible total latency on image prompts. |
| image_prefill_ratio | advisory | Vision encoder cache lifts work out of the measured prefill window, so this bucket understates the user win that image_wall already captures. Re-promote once the metric excludes cache mode or is redefined. |
| memory_ratio        | hard (auto-downgraded to advisory when quantization classes differ) | The benchmark now samples each engine's process-tree RSS; the gate hard-gates the ratio so KrillLM cannot quietly regress its footprint. When the comparison crosses quantization classes (e.g. KrillLM bf16 vs Ollama Q4_K_M) the metric is dominated by the weight format, not the runtime, so the gate auto-downgrades it to advisory and records the downgrade in `scope.memory_ratio` and the caveats. It re-promotes to hard automatically the moment a quantization-class-equal report is supplied. |
| audio_wall_ratio    | out_of_scope | Audio runs through the mlx-vlm sidecar; native Swift audio is Workstream 1. |
| audio_prefill_ratio | out_of_scope | Same as audio_wall. |

**Peak memory sampling.** `gemma4_multimodal_benchmark.py` samples the
resident set size (RSS) of each engine's process tree from a background
thread (default 50 ms poll) while every measured request runs and records
the peak. Ollama is sampled via `pgrep ollama` (covers the daemon plus the
`ollama runner` child); the KrillLM server is sampled via
`pgrep -f 'krillm.*serve'`; the krillm CLI subprocess is sampled by its own
PID. Pass `--ollama-pids` / `--krillm-server-pid` to override auto-detection,
or `--sample-memory off` to skip it. The KrillLM bridge path keeps using
mlx-vlm's `GenerationResult.peak_memory` (MLX Metal allocator peak); both
bases are documented under `memory_sampling.basis` in the report. Per-run
results carry `peak_memory_gb` and `peak_memory_basis`. RSS is reported in
decimal GB to match mlx-vlm's basis.

**`memory_ratio` is excluded from the geometric-mean speedup headline.**
Footprint is not a speed dimension; folding a quantization-class-driven
memory ratio into the perf headline would understate the speed result for
reasons unrelated to speed. Memory still appears as its own evaluation.

Out-of-scope metrics appear in the gate report under `scope_skipped_metrics`
with the documented reason — they are **not** silently dropped. Advisory
metrics are evaluated and printed (with a `WARN` glyph and `[advisory]` tag)
but never break the gate. Only hard failures set `gate: "fail"`.

**Missing hard metrics fail the gate.** If a hard-gated metric is absent
from the report (e.g. the benchmark never recorded it), the gate verdict is
`fail`. A claim of "release candidate" must be backed by an actual
measurement; an unobserved metric is treated as a failure, not a pass.

Use `--profile release_candidate` when validating that a release candidate
matches the user-latency claim defined in
[`OLLAMA_SPEEDUP_EXECUTION_PLAN.md`](../OLLAMA_SPEEDUP_EXECUTION_PLAN.md) §4.
Use the default `strict` profile when running CI gates or pre-release sweeps
where every metric must still meet its original threshold.

### KV cache dtype in reports

The benchmark harness records `benchmark.kv_cache_dtype` (sourced from
`KRILL_KV_CACHE_DTYPE`, default `fp16`). The gate echoes it in the gate
report's `kv_cache_dtype` field and on the terminal summary header so int8
and fp16 runs are never confused for one another.

## Release Readiness Status

This build is a release-readiness baseline plus a documented
release-candidate path, not yet a production release.

- **`strict` gate** (default) currently exits `1` against the accepted
  multimodal report. `text_prefill_ratio`, `image_prefill_ratio`, and
  `audio_wall_ratio` fail their hard thresholds.
- **`release_candidate` gate** exits `0` on the same report (with
  `--allow-dtype-mismatch`) because prefill TPS is advisory, audio is
  out_of_scope, and `memory_ratio` is auto-downgraded to advisory while the
  comparison spans quantization classes (KrillLM bf16 vs Ollama Q4_K_M).

Run `make bench-release-gate` for the latest per-metric results. The gate
report at `.build/benchmarks/release-gate.json` contains exact ratios, the
profile in use, the worst metric, bottleneck classification, and the
KV cache dtype the run used. See
[`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md) for
the full plan, the per-metric promotion contract, and acceptance criteria.

Key gaps as of the last reviewed run:

- `text_prefill_ratio` (1.45x) is advisory under `release_candidate`;
  re-promote once a drafter, fused kernel, or eval-cadence change pushes
  it past 1.5x.
- `image_prefill_ratio` (1.04x) is advisory because the vision-encoder
  cache lifts work out of the measured prefill window;
  `image_wall_ratio` already passes hard at 0.56x.
- `memory_ratio` is now sampled (process-tree RSS for native paths and
  Ollama, MLX Metal allocator peak for the bridge) and hard-gated under
  `release_candidate`. It auto-downgrades to advisory when the engines'
  quantization classes differ, which is the case on the canonical
  bf16-vs-Q4_K_M snapshot. It re-promotes to hard once a
  quantization-class-equal comparison is run.
- Audio benchmarks still run through `mlx-vlm`; native Swift audio is
  Workstream 1 of the execution plan and `audio_*` is `out_of_scope`
  under `release_candidate` until that ships.
- Server multimodal benchmarks now exercise real media payloads (image
  native, audio bridge).
- int8 KV cache and the prefix cache compose end-to-end on Gemma 4 (PR
  #11). Reports record `benchmark.kv_cache_dtype` so int8 vs fp16 runs
  are never confused.

## Cache-Hit Benchmark Caveats

Benchmark prompts may hit the prefix cache, especially in server mode with repeated prompts. After warmup, measured runs may show near-zero prefill cost (TTFT ~11ms). Reports include `prefix_cache_active: true` and warn when prefill speed varies >3x across runs.

When reporting benchmark results, every report must label its `cache_mode` as one of:
- **cold**: first request, no cache (measures true prefill)
- **warm**: model loaded, no prefix cache hit (measures warm prefill)
- **cache_hit**: repeated prompt, prefix cache hit (measures cache restore + decode only)

Release criteria must state which `cache_mode` is being compared. At least one cold-path benchmark must be included whenever prefill performance is part of the claim. Do not mix cache-hit numbers with cold-path numbers in the same comparison.

## Apples-to-Apples Comparison Rules

- Do not compare text-only placeholder runs against real-media runs. Server-mode multimodal comparisons must send real media payloads to both engines (image via the native Swift SigLIP2 path, audio via the `mlx-vlm` bridge until Workstream 1 lands). Any metric that the harness skips, or that the active gate profile classifies as `out_of_scope`, must be surfaced explicitly in the gate report (`scope_skipped_metrics` with reason) — never silently substituted with text-only numbers.
- KrillLM and Ollama runs in the same report must use the same prompts, media assets, max-token budgets, sampling settings, `cache_mode`, and `kv_cache_dtype` (recorded in the report under `benchmark.kv_cache_dtype`).

## Non-Negotiable Rules

1. Benchmark persistent warm paths, not repeated CLI process startup
2. Separate cold-load, warm prefill, decode, TTFT, and total wall time
3. Record exact model artifact, dtype, quantization, git commit, hardware
4. Use the same prompts/assets/settings for both engines
5. Mark comparisons honestly (bit-identical vs class-equivalent)
6. Never publish speed claims without matching reports

## Known Limitations

- **Prefill**: MLX Swift prefill is 3-5x slower than Ollama's llama.cpp Metal kernels. This is a framework limitation, not a KrillLM bug.
- **Server overhead**: HTTP streaming adds ~17% overhead vs CLI (reduced by direct JSON formatting).
- **Prefix cache**: Repeated-prompt benchmarks show inflated TTFT improvement. Label cache-hit results explicitly.
