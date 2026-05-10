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

## Post-Remediation Measurements (2026-05-10)

After this PR (server multimodal implemented, multimodal benchmark routed
through `native_server` mode against a long-running KrillLM daemon, decode
loop pipelined to overlap GPU forward with CPU tokenizer/yield):

Text/server benchmark medians (`postpr-text-server.json`):

| Metric | KrillLM | Ollama | Ratio | vs baseline |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 0.312 s | 0.564 s | 0.5805x | 0.6289x |
| Decode throughput | 109 tok/s | 88 tok/s | 1.2414x | 1.1246x |
| Prefill throughput | 1626 tok/s | 1877 tok/s | 0.8385x | 0.7170x |

Multimodal `--krillm-image-mode native_server` gate
(`postpr-mm-server-gate.json`):

| Metric | Ratio | Threshold | Status | vs baseline |
| --- | ---: | ---: | --- | ---: |
| text_decode_ratio | 1.2260x | >=1.5 | FAIL | 1.4413x |
| text_prefill_ratio | 1.6008x | >=1.5 | OK | 0.0237x |
| text_ttft_ratio | 0.1406x | <=0.67 | OK | n/a |
| text_wall_ratio | 0.6063x | <=0.67 | OK | 0.6043x |
| image_prefill_ratio | 1.7539x | >=1.5 | OK | 0.0237x |
| image_wall_ratio | 0.7735x | <=0.67 | FAIL | 2.2921x |
| audio_wall_ratio | 12.0280x | <=0.67 | FAIL | 1.2525x |

Big wins: image prefill flipped from 50x slower to 1.75x faster than Ollama
because the benchmark now exercises the native Swift image path. Text prefill
flipped from below Ollama to above the 1.5x threshold. Text TTFT and wall
ratio pass.

Remaining gaps:

1. text_decode at 1.23x is 18% short of the 1.5x target. Requires either a
   vocab-compatible Gemma 4 draft model so speculative decoding can be
   safely enabled, or kernel-level tuning. Out of scope for this PR.
2. image_wall at 0.77x is 15% short of the 0.67x target.
3. audio_wall is 12x slower because audio still routes through the Python
   `mlx-vlm` bridge (now via subprocess from the server, paying Python
   startup cost). Will not be competitive until native audio is implemented.

The release gate still fails. This PR is a measurable step forward, not a
release tag.

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
