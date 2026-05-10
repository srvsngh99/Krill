# Release Readiness Remediation Plan

Date: 2026-05-10
Baseline commit: `4a2b6e6` (`Release readiness baseline and Gemma4 fixes`)

## Status

The release-readiness baseline has been merged to `main`, but it is not a
production release candidate yet. Treat the current state as a functional
baseline for follow-up work:

- Native Gemma4 text now produces coherent output.
- Native Gemma4 image runs end to end and identifies the red-box fixture.
- Gemma4 audio is routed through the `mlx-vlm` Python bridge.
- Server timing fields and benchmark/release-gate tooling exist.
- Server image/audio payloads are not supported yet.
- Release benchmark gates still fail.

Do not tag or publish a release until the acceptance criteria in this document
pass on the target M4 Pro 24 GB machine.

## Latest Verification

Commands run locally from the repository root:

```bash
make test
make release

.build/release/krillm run gemma-4-e2b "Say hello." --max-tokens 12 --temp 0
.build/release/krillm run gemma-4-e2b "What is shown in this image? Answer briefly." \
  --image .build/benchmarks/assets/gemma4-red-box.png --max-tokens 16 --temp 0
.build/release/krillm run gemma-4-e2b "What sound is in this audio? Answer briefly." \
  --audio .build/benchmarks/assets/gemma4-tone-5s.wav --max-tokens 24 --temp 0

python3 tools/krillm_vs_ollama_benchmark.py \
  --krillm-url http://127.0.0.1:11438 \
  --krill-model gemma-4-e2b \
  --ollama-model gemma4:e2b \
  --runs 5 \
  --warmup 2 \
  --output .build/benchmarks/review-rereview-text-server.json \
  --timeout 300

/Users/sourav/.krillm/venv/bin/python3 tools/release_gate.py \
  .build/benchmarks/review-rereview-text-server.json \
  --output .build/benchmarks/review-rereview-text-gate.json

/Users/sourav/.krillm/venv/bin/python3 tools/gemma4_multimodal_benchmark.py \
  --engine both \
  --runs 2 \
  --warmup 1 \
  --output .build/benchmarks/review-rereview-gemma4-multimodal.json \
  --timeout 300

/Users/sourav/.krillm/venv/bin/python3 tools/release_gate.py \
  .build/benchmarks/review-rereview-gemma4-multimodal.json \
  --output .build/benchmarks/review-rereview-gemma4-multimodal-gate.json
```

Results:

| Check | Result |
| --- | --- |
| `make test` | Passed, 81 tests |
| `make release` | Passed |
| Native Gemma4 text smoke | Coherent output |
| Native Gemma4 image smoke | Identifies red-box fixture |
| Gemma4 audio smoke | Runs through `mlx-vlm` bridge |
| Text/server release gate | Failed |
| Multimodal release gate | Failed |

Text/server benchmark medians:

| Metric | KrillLM | Ollama | Ratio |
| --- | ---: | ---: | ---: |
| Wall time | 0.3384 s | 0.5380 s | 0.6289x |
| Decode throughput | 98.57 tok/s | 87.65 tok/s | 1.1246x |
| Prefill throughput | 1389.85 tok/s | 1938.30 tok/s | 0.7170x |

Multimodal gate summary:

| Metric | Result |
| --- | ---: |
| Geometric mean speedup | 0.429x |
| Image wall ratio | 2.2921x |
| Audio wall ratio | 1.2525x |
| Text wall ratio | 0.6043x |
| Text decode ratio | 1.4413x |
| Image prefill ratio | 0.0237x |
| Audio prefill ratio | 0.0592x |

## Release Blockers

### 1. Release Gates Fail

The configured release gate requires substantial speedups over Ollama. Current
results do not meet those thresholds:

- Text wall time passes.
- Text decode and text prefill do not meet the 1.5x threshold.
- Image/audio wall time are slower than Ollama in the multimodal harness.
- Image/audio prefill are far below target.

Required outcome:

- `tools/release_gate.py` exits `0` for the accepted release benchmark report.
- The report is committed or attached to the release notes with machine,
  model, quantization, warmup, run count, and cache mode.

### 2. Server Multimodal Is Not Implemented

The HTTP server does not accept image/audio payloads for KrillLM generation.
The benchmark script correctly skips image/audio in `--krillm-url` mode to avoid
invalid comparisons.

Required outcome:

- Either implement server image/audio payload support end to end, or explicitly
  declare server multimodal out of scope for this release.
- If implemented, benchmark KrillLM and Ollama with equivalent media payloads.
- Add API tests for accepted and rejected image/audio request shapes.

### 3. Documentation Does Not Match Behavior

`README.md` still describes Gemma4 media as routed through `mlx-vlm` and native
image as unsupported. Current behavior is:

- Gemma4 text: native Swift.
- Gemma4 image: native Swift.
- Gemma4 audio: `mlx-vlm` bridge.
- Server image/audio: unsupported.

Required outcome:

- Update `README.md`, `docs/ARCHITECTURE.md`, `docs/BENCHMARKING.md`,
  `docs/GEMMA4_INTERNALS.md`, and `docs/SERVER_API.md` so they agree.
- Clearly separate CLI support from server/API support.
- Clearly separate native support from Python bridge support.

### 4. Audio Fixture And Quality Criteria Are Weak

The generated audio fixture produced different interpretations:

- KrillLM/`mlx-vlm`: click or sharp percussive sound.
- Ollama: dog barking.

This makes quality comparison ambiguous.

Required outcome:

- Replace or augment the audio fixture with deterministic, semantically obvious
  samples.
- Add expected-answer checks or manual review rubric for text, image, and audio.
- Record output previews and hashes in benchmark reports.

### 5. Cache-Affected Results Need Explicit Labels

The prefix cache threshold is now low enough for repeated benchmark prompts to
hit cache. This is a valid optimization, but it must not be mixed with cold-path
results.

Required outcome:

- Benchmark reports must label cold, warm, and cache-hit runs.
- Release criteria must state which mode is being compared.
- At least one cold-path benchmark must be included for prefill performance.

## Work Plan For Next PR

1. Documentation correction
   - Update all public docs to match the current support matrix.
   - Add a release status table with CLI/server/native/bridge distinctions.

2. Benchmark harness hardening
   - Add explicit `cache_mode` to reports: `cold`, `warm`, or `cache_hit`.
   - Add validation that prompt/media token counts are comparable where expected.
   - Make server multimodal skipped status visible in release-gate output.

3. Server multimodal decision
   - Choose one:
     - Implement server image/audio E2E.
     - Or declare server media out of scope and remove it from release gate.
   - Do not benchmark KrillLM text placeholders against Ollama real media.

4. Performance work
   - Improve Gemma4 text decode to at least 1.5x Ollama, or revise the gate with
     a documented rationale.
   - Improve Gemma4 prefill or split cold prefill from repeated-prompt cache-hit
     latency.
   - Profile native image prefill, which is currently much slower than Ollama.

5. Quality and correctness coverage
   - Add CLI E2E smoke tests for Gemma4 text and image where model assets exist.
   - Add a bridge availability test path for audio.
   - Add benchmark output quality checks or fixture-specific expected terms.

## Acceptance Criteria

A follow-up PR can be considered release-ready only when all items below are
true:

- `make test` passes.
- `make release` passes.
- Direct CLI smoke checks pass for:
  - Gemma4 text.
  - Gemma4 image.
  - Gemma4 audio through bridge, unless native audio is implemented.
- `tools/release_gate.py` exits `0` for the accepted text benchmark.
- If multimodal is in release scope, `tools/release_gate.py` exits `0` for the
  accepted multimodal benchmark.
- Docs accurately state:
  - Native versus bridge support.
  - CLI versus server support.
  - Known unsupported paths.
  - Benchmark caveats.
- Release notes include benchmark report paths or attached artifacts.
- No benchmark uses non-equivalent KrillLM/Ollama inputs.

## Recommended Release Language Until Fixed

Use this language if publishing internal builds before full release readiness:

> This build is a release-readiness baseline, not a production release. It
> includes Gemma4 native text/image improvements, audio via `mlx-vlm`, server
> timing fields, and benchmark gates. Server multimodal remains unsupported and
> current benchmarks do not yet meet the release speedup threshold.
