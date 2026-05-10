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
| `release_candidate` | Hard-gates user-visible latency and memory metrics. Treats prefill TPS metrics as advisory. Scopes audio metrics out until native Swift audio (Workstream 1) lands. |

**Per-metric kind under `release_candidate`:**

| Metric | Kind | Rationale |
|--------|------|-----------|
| text_decode_ratio   | hard | Core decode throughput claim. |
| text_wall_ratio     | hard | User-visible total latency. |
| text_ttft_ratio     | hard | User-visible first-token latency. |
| text_prefill_ratio  | advisory | Wall time and TTFT already gate user latency; prefill TPS is noisy on short prompts. |
| image_wall_ratio    | hard | User-visible total latency on image prompts. |
| image_prefill_ratio | advisory | Vision encoder cache lifts work out of the measured prefill window, so this bucket understates the user win that image_wall already captures. Re-promote once the metric excludes cache mode or is redefined. |
| memory_ratio        | hard | Peak memory must not regress vs Ollama. |
| audio_wall_ratio    | out_of_scope | Audio runs through the mlx-vlm sidecar; native Swift audio is Workstream 1. |
| audio_prefill_ratio | out_of_scope | Same as audio_wall. |

Out-of-scope metrics appear in the gate report under `scope_skipped_metrics`
with the documented reason — they are **not** silently dropped. Advisory
metrics are evaluated and printed (with a `WARN` glyph and `[advisory]` tag)
but never break the gate. Only hard failures set `gate: "fail"`.

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

This build is a release-readiness baseline, not a production release. Release benchmark gates currently **fail**. Run `make bench-release-gate` for the latest per-metric results. The gate report at `.build/benchmarks/release-gate.json` contains exact ratios, the worst metric, and bottleneck classification. See [`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md) for the full plan and acceptance criteria.

Key gaps as of the last reviewed run:
- Text decode ratio does not meet the 1.5x threshold.
- Prefill ratios are below target (MLX framework limitation).
- Image benchmarks via the multimodal harness currently exercise the mlx-vlm bridge path; the native Swift image path is what the CLI uses end-to-end and should be benchmarked directly when measuring native performance.
- Audio benchmarks exercise the mlx-vlm bridge because native audio is not implemented.
- Server multimodal benchmarks now exercise real media payloads (image native, audio bridge).

## Cache-Hit Benchmark Caveats

Benchmark prompts may hit the prefix cache, especially in server mode with repeated prompts. After warmup, measured runs may show near-zero prefill cost (TTFT ~11ms). Reports include `prefix_cache_active: true` and warn when prefill speed varies >3x across runs.

When reporting benchmark results, every report must label its `cache_mode` as one of:
- **cold**: first request, no cache (measures true prefill)
- **warm**: model loaded, no prefix cache hit (measures warm prefill)
- **cache_hit**: repeated prompt, prefix cache hit (measures cache restore + decode only)

Release criteria must state which `cache_mode` is being compared. At least one cold-path benchmark must be included whenever prefill performance is part of the claim. Do not mix cache-hit numbers with cold-path numbers in the same comparison.

## Apples-to-Apples Comparison Rules

- Do not compare KrillLM text-only placeholder runs against Ollama real-media runs. The server-mode multimodal harness skips image/audio for KrillLM specifically because the server does not accept media payloads — that skip must be visible in release-gate output, not silently substituted with text-only numbers.
- KrillLM and Ollama runs in the same report must use the same prompts, media assets, max-token budgets, sampling settings, and `cache_mode`.

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
