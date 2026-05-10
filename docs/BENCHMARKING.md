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
- `--krillm-url URL` — server mode (text only, skips image/audio)
- `--engine both|krillm|ollama` — single-engine or comparison
- `--runs N` / `--warmup N`

**Server mode limitation:** KrillLM server does not accept image/audio payloads via HTTP yet. Server benchmark skips media tasks.

### release_gate.py
Evaluates benchmark reports against performance thresholds.

**Default thresholds:**
| Metric | Target | Direction |
|--------|--------|-----------|
| text_decode_ratio | >= 1.5x | Higher is better |
| text_wall_ratio | <= 0.67x | Lower is better |
| text_ttft_ratio | <= 0.67x | Lower is better |
| text_prefill_ratio | >= 1.5x | Higher is better |
| image_wall_ratio | <= 0.67x | Lower is better |
| audio_wall_ratio | <= 0.67x | Lower is better |

**Output:** JSON report + colored terminal summary with per-metric pass/fail, geometric mean speedup, worst metric, bottleneck classification.

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
