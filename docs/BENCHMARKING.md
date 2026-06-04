# Benchmarking Guide

## Quick Start

```bash
# Build release binary
make release

# Text-only KrillLM vs Ollama (CLI subprocess mode)
make bench-compare KRILL_MODEL=llama-3.2-1b OLLAMA_MODEL=llama3.2:1b

# Server-mode (warm vs warm, eliminates process startup)
krillm serve --model llama-3.2-1b --port 57455 &
make bench-compare KRILLM_URL=http://127.0.0.1:57455

# Gemma4 multimodal (text/image/audio — all native, no Python bridge)
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

**Server mode notes:** image and audio payloads are sent as base64 in the standard Ollama shape. Audio runs on the native Swift+MLX USM path — no Python/`mlx-vlm` dependency (the bridge was removed in WS6 Step 4).

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
| `strict` (default) | Hard-gates every threshold EXCEPT two owner-accepted advisory demotions: `text_decode_ratio` (advisory >=1.5x + hard `text_decode_ratio_floor` >=1.0x; `docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md`) and `image_prefill_ratio` (advisory, no floor; `docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md`). Both are structural microbenchmark mismeasurements; every other metric stays hard, so `strict` remains the uncompromised reference. |
| `release_candidate` | Hard-gates user-visible latency metrics and class-equal peak memory. Treats prefill TPS as advisory. Scopes audio metrics out until native Swift audio (Workstream 1) lands. |

**Per-metric kind under `release_candidate`:**

| Metric | Kind | Rationale |
|--------|------|-----------|
| text_decode_ratio   | **advisory** (>=1.5x) + synthetic **hard `text_decode_ratio_floor` >=1.0x** in BOTH profiles | Owner-accepted for `release_candidate` 2026-05-16 (`docs/RELEASE_GATE_DECODE_PROPOSAL.md`) and for `strict` 2026-05-22 (`docs/RELEASE_GATE_STRICT_DECODE_PROPOSAL.md`). Dense decode is per-token weight-read-bandwidth bound; on tiny 4-bit Gemma 4 e2b llama.cpp's Metal kernels are at parity, and the user-visible "1.5x faster" claim is carried by text_wall/text_ttft (hard). The floor still guarantees KrillLM never decodes slower than Ollama (a missing decode value hard-fails too). The >=1.5x target is structurally unreachable on M-series (see `docs/SPECULATIVE_DECODING.md`); it re-promotes to hard >=1.5x when speculative decoding sustains >=1.5x with greedy parity OR a long-output decode task is added. |
| text_wall_ratio     | hard | User-visible total latency. |
| text_ttft_ratio     | hard | User-visible first-token latency. |
| text_prefill_ratio  | advisory | Wall time and TTFT already gate user latency; prefill TPS is noisy on short prompts. |
| image_wall_ratio    | hard | User-visible total latency on image prompts. |
| image_prefill_ratio | **advisory in BOTH profiles, no floor** | Owner-accepted for `strict` 2026-05-22 (`docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md`); already advisory under `release_candidate`. The vision-encoder cache lifts SigLIP2 forward + projector cost out of the measured prefill window, so this prefill-TPS bucket divides non-comparable denominators and is structurally `< 1.0x` by design — a measurement artifact, not a regression. Unlike `text_decode_ratio` it carries **no `<metric>_floor`** (a `>= 1.0x` floor would be meaningless here); the hard `image_wall_ratio` carries the user-visible image guarantee. Re-promotes to hard `>= 1.5x` once the benchmark separates vision-encoder time from language-model prefill time. |
| memory_ratio        | hard (auto-downgraded to advisory when quantization classes differ) | The benchmark now samples each engine's process-tree memory; the gate hard-gates the ratio so KrillLM cannot quietly regress its footprint. When the comparison crosses quantization classes (e.g. one engine bf16 vs the other Q4_K_M) the metric is dominated by the weight format, not the runtime, so the gate auto-downgrades it to advisory and records the downgrade in `scope.memory_ratio` and the caveats. The canonical Gemma 4 e2b comparison is class-equal (KrillLM affine 4-bit MLX vs Ollama Q4_K_M GGUF, both `4-bit` class), so memory is hard-gated on it. |
| audio_wall_ratio    | hard | WS6: native Swift+MLX audio is the only audio path and benchmarks faster than Ollama (~0.53× wall). |
| audio_prefill_ratio | hard | WS6: native audio prefill ~2.4× Ollama. |

**Peak memory sampling.** `gemma4_multimodal_benchmark.py` samples each
engine's process-tree memory from a background thread (default 50 ms poll)
while every measured request runs and records the peak. On macOS the
per-PID number is `phys_footprint` from
`proc_pid_rusage(RUSAGE_INFO_V2)` — the same figure Activity Monitor's
"Memory" column and `vmmap -summary` report, which counts resident
mmap'd pages (the safetensors weights KrillLM relies on) with proper
apportionment. Non-Darwin platforms fall back to RSS from `ps`, which
under-counts mmap'd pages but is the best portable substitute. Ollama
is sampled via `pgrep ollama` (covers the daemon plus the `ollama runner`
child); the KrillLM server is sampled via `pgrep -f 'krillm.*serve'`;
the krillm CLI subprocess is sampled by its own PID. Pass `--ollama-pids`
/ `--krillm-server-pid` to override auto-detection, or `--sample-memory
off` to skip it. The KrillLM bridge path keeps using mlx-vlm's
`GenerationResult.peak_memory` (MLX Metal allocator peak). All three bases
are documented under `memory_sampling.basis` in the report and the actual
basis used for each run appears in `peak_memory_basis`. Values are decimal
GB throughout, matching mlx-vlm's basis.

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

### MLX Metal buffer cache cap (`KRILL_MLX_CACHE_LIMIT_MB`)

mlx-swift recycles freed intermediate buffers in a pool sized from Metal's
`recommendedMaxWorkingSetSize` (≈16 GB on a 24 GB M4 Pro). Those pages are
resident and counted by `phys_footprint` / `RSIZE` — the figure the
benchmark samples for `memory_ratio` — even though MLX treats them as
"free", so an uncapped pool can grow into the multi-GB range under
sustained load. KrillLM caps it on every native model load via
`MLXMemoryConfig` (`Sources/KLMCore/MLXMemoryConfig.swift`):

- Default: **256 MB** — covers Gemma 4 e2b's fixed-size decode-step buffers
  so the hot loop still recycles (no decode regression).
- `KRILL_MLX_CACHE_LIMIT_MB=<N>` overrides the cap in MB.
- `KRILL_MLX_CACHE_LIMIT_MB=0` disables the cap (legacy unbounded
  behavior).

When benchmarking memory, always pass the clean `--krillm-server-pid`
override (see the memory-sampling section). The historical v5 ~9.6 GB
KrillLM reading was the combination of an *uncapped* pool and a
*contaminated* (non-`--krillm-server-pid`) process-tree sample; with the
cap and clean sampling, KrillLM text/image phys_footprint is ~2.85–3.0 GB.

## Release Readiness Status

This build is a release-readiness baseline plus a documented
release-candidate path, not yet a production release.

- **`release_candidate` gate** exits `0` (**GATE: PASS**) against the
  accepted report `.build/benchmarks/v6-mm.json` (PR #16). User-visible
  latency (text TTFT ~5x, text wall ~1.57x faster, native vision/image
  wall ~1.77x faster) and
  class-equal `memory_ratio` (0.32–0.84) hard-pass, plus the hard
  `text_decode_ratio_floor ≥1.0x`. `text_decode_ratio`'s ≥1.5x target is
  advisory (printed as WARN at ~1.19x — **no claim KrillLM hit 1.5x
  decode**). Voice/audio is native (WS6) and `hard` in both profiles —
  faster than Ollama (~0.53× wall, ~2.4× prefill). Prefill is advisory.
- **`strict` gate** (default) exits `0` on the post-native-audio multimodal
  report since 2026-05-22. Native Swift audio (WS1) made `audio_*`
  hard-pass; the two remaining structural microbenchmark misses are
  owner-accepted advisory demotions — `text_decode_ratio` (advisory + hard
  `>=1.0x` floor) and `image_prefill_ratio` (advisory, no floor). Every
  other metric stays hard, so `strict` remains the uncompromised reference
  and is the gate required for a production tag.

Post-merge PR #18 note: a text-only `llama-3.2-1b` vs `llama3.2:1b`
server sanity run produced strong decode numbers, but the report failed
`release_candidate` on TTFT/wall time and had a prompt-token mismatch.
It is useful regression signal, not a release benchmark artifact. The next
release PR must rerun the full Gemma 4 text/vision/audio benchmark after
restoring the local Gemma 4 checkpoint and attach the generated report.

Run `make bench-release-gate` for the latest per-metric results. The gate
report at `.build/benchmarks/release-gate.json` contains exact ratios, the
profile in use, the worst metric, bottleneck classification, and the
KV cache dtype the run used. See
[`RELEASE_READINESS_REMEDIATION.md`](RELEASE_READINESS_REMEDIATION.md) for
the full plan, the per-metric promotion contract, and acceptance criteria.

Key release gaps as of the last reviewed run:

- `text_decode_ratio` (~1.19x on v6-mm) is advisory under BOTH profiles
  (with the hard >=1.0x floor). KrillLM decodes ~102 tok/s
  vs Ollama's ~86 tok/s on the accepted Gemma 4 E2B report, where
  llama.cpp's hand-tuned Metal decode kernels are genuinely competitive.
  Closing it to ≥1.5x is owned by Workstream 2 (Gemma 4-compatible
  speculative decoding) and is a documented multi-week follow-up.
  User-visible latency still wins decisively (text TTFT ~5x, text wall
  ~1.57x, native vision wall ~1.77x).
- `text_prefill_ratio` (1.22x) is advisory under `release_candidate`;
  re-promote once a drafter, fused kernel, or eval-cadence change
  pushes it past 1.5x.
- `image_prefill_ratio` (~0.9–1.1x) is advisory under BOTH profiles
  (no floor) since 2026-05-22: the vision-encoder cache lifts SigLIP2 +
  projector cost out of the measured prefill window, so the metric divides
  non-comparable denominators. The hard `image_wall_ratio` (~0.50x) carries
  the user-visible image guarantee. See
  `docs/RELEASE_GATE_IMAGE_PREFILL_PROPOSAL.md`.
- `memory_ratio` is **hard** and now **passing** (0.32–0.84 across 5
  runs; canonical 0.322). PR #16 root-caused the historical 1.14x / ~9.6
  GB reading to an uncapped MLX buffer pool plus contaminated sampling;
  with `KRILL_MLX_CACHE_LIMIT_MB` (default 256 MB) and clean
  `--krillm-server-pid` sampling, KrillLM text/image phys_footprint is
  ~2.85–3.0 GB vs Ollama's ~8.2–8.4 GB. Both engines remain 4-bit class
  so the comparison is genuinely hard-gated.
- Audio benchmarks run on the native Swift+MLX USM path (WS6; the
  `mlx-vlm` bridge was removed). `audio_*` is `hard` in both profiles:
  on the WS6 native-audio report KrillLM audio wall ~0.15 s vs Ollama
  ~0.29 s and prefill ~2.4× Ollama — faster, not at-parity-only.
- Server multimodal benchmarks exercise real media payloads (image and
  audio both native).
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

- Do not compare text-only placeholder runs against real-media runs. Server-mode multimodal comparisons must send real media payloads to both engines (image via the native Swift SigLIP2 path, audio via the native Swift+MLX USM path). Any metric that the harness skips, or that the active gate profile classifies as `out_of_scope`, must be surfaced explicitly in the gate report (`scope_skipped_metrics` with reason) — never silently substituted with text-only numbers.
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
