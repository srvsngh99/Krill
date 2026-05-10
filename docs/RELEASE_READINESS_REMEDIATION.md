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

After two iterations on this PR. Iteration 1 added server multimodal,
routed multimodal benchmarks through the native Swift image path (instead
of the mlx-vlm bridge), and pipelined the decode loop. Iteration 2 added a
persistent mlx-vlm sidecar (replaces per-call Python subprocess), a
SHA-256-keyed vision encoder cache, and on-GPU sampler chaining in the
decode loop.

Text/server benchmark medians (`v2-text.json`, 5 runs / 2 warmup):

| Metric | KrillLM | Ollama | Ratio | Baseline |
| --- | ---: | ---: | ---: | ---: |
| Wall time | 0.295 s | 0.539 s | 0.5482x | 0.6289x |
| Decode throughput | 110 tok/s | 87 tok/s | 1.2588x | 1.1246x |
| Prefill throughput | 1768 tok/s | 1896 tok/s | 0.9323x | 0.7170x |

Multimodal `--krillm-image-mode native_server` gate (`v3-mm-gate.json`,
4 runs / 2 warmup):

| Metric | Ratio | Threshold | Status | Baseline |
| --- | ---: | ---: | --- | ---: |
| text_decode_ratio | 1.2356x | >=1.5 | FAIL | 1.4413x |
| text_prefill_ratio | 1.3772x | >=1.5 | FAIL | 0.0237x |
| text_ttft_ratio | 0.1162x | <=0.67 | OK | n/a |
| text_wall_ratio | 0.6058x | <=0.67 | OK | 0.6043x |
| image_prefill_ratio | 1.0206x | >=1.5 | FAIL | 0.0237x |
| image_wall_ratio | 0.5689x | <=0.67 | OK | 2.2921x |
| audio_wall_ratio | 3.5913x | <=0.67 | FAIL | 1.2525x |

Big wins:

- **image_wall** flipped from 2.29x slower to 0.57x of Ollama (passes 0.67
  gate) thanks to the vision encoder cache: SigLIP2 forward + projector
  bypass on repeat-image benchmarks, plus the native Swift path.
- **audio_wall** dropped 12.03x -> 3.59x by replacing per-call Python
  subprocess with a long-running mlx-vlm sidecar. Still slower than Ollama
  because Ollama has a true native audio path; full parity needs native
  audio implementation in Swift.
- **image_prefill** flipped from 0.024x (50x slower) to 1.02x because the
  benchmark now exercises the native vision path. The 1.5x prefill gate is
  still missed; iteration 1 saw 1.75x but the vision cache shifts work out
  of prefill, so prefill_tps ratio falls even though wall ratio improves.
- **text_ttft 0.12x**, **text_wall 0.61x**, both well under the 0.67 gate.

Remaining gaps:

1. text_decode at 1.24x is 17% short of 1.5x. The decode loop is already
   pipelined and on-GPU sampling is chained; the next step is a
   vocab-compatible Gemma 4 draft model for speculative decoding, or
   kernel-level work (KV quantization, fused attention/MLP). Out of scope
   for this PR.
2. image_prefill at 1.02x is below the 1.5x gate. Wall ratio is the
   user-facing number and that passes; the prefill_tps metric is
   structurally lower because the vision cache moves work out of prefill
   rather than making prefill itself faster.
3. audio_wall 3.59x. Subprocess startup is gone; the remaining gap is
   mlx-vlm's actual generate cost relative to Ollama's native path. Closing
   it requires native audio in Swift.

The release gate still fails on text_decode, image_prefill, and audio_wall.
Wall-time metrics across text and image now beat Ollama by 1.6x-1.7x.
This PR substantially closes the gap; clearing the full gate is gated on
deeper work.

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
