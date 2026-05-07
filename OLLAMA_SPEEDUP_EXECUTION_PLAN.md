# KrillLM Speedup Execution Plan

This document is a handoff for agents that will execute the next phase of
KrillLM performance work. The target is to make KrillLM consistently faster
than Ollama on Apple Silicon, with a release gate of 1.5x to 3x on the chosen
benchmark matrix.

The current PR branch is:

```text
release-readiness-hardening-20260507
```

The current PR is:

```text
https://github.com/srvsngh99/KrillLM/pull/6
```

## Executive Summary

KrillLM is not yet faster than Ollama in every metric. The main reason is that
Gemma4 image/audio currently routes through a Python bridge:

```text
KrillLM CLI -> PythonFallback -> mlx-vlm -> MLX Python runtime
```

That path is production-safe, but it is not a fully native KrillLM runtime. It
adds Python runtime behavior, uses mlx-vlm's processor/generation stack, and
does not exercise optimized Swift-native multimodal preprocessing.

The required target architecture is:

```text
KrillLM server
  -> native Swift tokenizer/prompt/media preprocessing
  -> native Swift Gemma4 multimodal model path
  -> warmed MLX Swift / Metal kernels
  -> persistent streaming response
```

The work below is organized so multiple agents can execute independently.

## Current Benchmark Facts

These results are from the current branch and local M4 Pro 24 GB machine.

### 4-bit class Gemma4 E2B

Report:

```text
.build/benchmarks/gemma4-e2b-multimodal-4bit.json
```

Comparison:

```text
KrillLM: MLX affine 4-bit, bits=4, group_size=64, mode=affine
Ollama:  GGUF Q4_K_M
```

This is 4-bit-class equivalent, not bit-identical quantization.

Median results:

| Task | KrillLM Decode | Ollama Decode | KrillLM Wall | Ollama Wall |
| --- | ---: | ---: | ---: | ---: |
| Text | 128.11 tok/s | 90.50 tok/s | 0.297s | 0.482s |
| Image | 115.53 tok/s | 92.80 tok/s | 0.710s | 0.308s |
| Audio | 125.18 tok/s | 94.15 tok/s | 0.396s | 0.327s |

Finding:

KrillLM wins decode throughput in this setup, but Ollama wins or is stronger on
image/audio wall time because prefill/media preprocessing is much faster.

### Unquantized precision Gemma4 E2B

Reports:

```text
.build/benchmarks/gemma4-e2b-bf16-mlx.json
.build/benchmarks/gemma4-e2b-f16-ollama.json
```

Comparison:

```text
MLX:    mlx-community/gemma-4-e2b-it-bf16, dtype bfloat16
Ollama: gemma4:e2b-it-bf16, reports quantization F16
```

This is an unquantized precision comparison, but not bit-identical dtype.

Median results:

| Task | MLX/Krill Decode | Ollama Decode | MLX Wall | Ollama Wall |
| --- | ---: | ---: | ---: | ---: |
| Text | 48.79 tok/s | 47.73 tok/s | 0.733s | 0.838s |
| Image | 48.71 tok/s | 48.78 tok/s | 0.912s | 0.484s |
| Audio | 50.79 tok/s | 50.07 tok/s | 0.483s | 0.475s |

Finding:

Decode is effectively tied. Ollama has much faster prefill and better image
wall time. MLX BF16 fits on the 24 GB M4 Pro with about 10.3 GB peak for text
and 11.0 GB peak for image/audio.

## Product Goal

KrillLM should beat Ollama by 1.5x to 3x on the release benchmark matrix.

The target should not be stated as a release claim until the benchmark reports
prove it.

Required metrics:

| Metric | Target |
| --- | --- |
| Text decode tok/s | >= 1.5x Ollama, stretch 3x |
| Text TTFT | <= 0.67x Ollama, stretch <= 0.33x |
| Text wall time | <= 0.67x Ollama |
| Image wall time | <= 0.67x Ollama |
| Audio wall time | <= 0.67x Ollama |
| Image/audio prefill | >= 1.5x Ollama or wall time must still win |
| Peak memory | <= Ollama or justified if faster |
| Model load | tracked separately, not mixed into warm benchmark |

## Non-Negotiable Benchmark Rules

Agents must not publish speed claims without matching benchmark reports.

1. Benchmark persistent warm paths, not repeated CLI process startup.
2. Separate cold-load, warm prefill, decode, TTFT, and total wall time.
3. Run text, image, and audio separately.
4. Record exact model artifact, dtype, quantization, git commit, hardware, OS,
   seed, prompt, media SHA256, run count, and warmup count.
5. Use the same prompts/assets/settings for KrillLM and Ollama.
6. Mark comparisons honestly:
   - bit-identical
   - same dtype class
   - same quantization class
   - local installed artifact comparison
7. Fail benchmark gates when metadata is incompatible unless an explicit
   `--allow-*` flag is passed and the report labels the caveat.

## Current Code Map

Key files:

```text
Sources/KLMCLI/RunCommand.swift
Sources/KLMEngine/PythonFallback.swift
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCore/VisionEncoder.swift
Sources/KLMCore/AudioEncoder.swift
Sources/KLMServer/Server.swift
Sources/KLMServer/ServerParsing.swift
tools/gemma4_multimodal_benchmark.py
tools/krillm_vs_ollama_benchmark.py
Makefile
README.md
```

Important current behavior:

```text
Gemma4 text can fall back to native Swift experimental path.
Gemma4 image/audio require mlx-vlm through PythonFallback.
Native image/audio preprocessing intentionally throws unsupported errors.
Server rejects unsupported image/tool fields instead of silently ignoring them.
```

## Primary Bottlenecks

### 1. Python bridge in Gemma4 media path

Current media path runs through `PythonFallback` and `mlx-vlm`.

Impact:

- Not native Swift.
- Harder to control tokenizer/media preprocessing.
- Harder to optimize prefill.
- Adds Python runtime behavior and dependency footprint.
- Weakens product claim that KrillLM itself is faster.

### 2. Prefill/media preprocessing

Ollama is much faster in prompt/media prefill. This dominates image and audio
wall time even when KrillLM decode is faster.

Impact:

- Image wall time loses badly in unquantized run.
- Audio wall time is roughly tied rather than a clear win.

### 3. Benchmark path mismatch

Some current tests use `mlx-vlm` directly or CLI subprocess behavior, while
Ollama is a warm daemon.

Impact:

- Unfair in both directions.
- Makes it difficult to isolate native KrillLM runtime wins.

### 4. No native Gemma4 image/audio preprocessing

`VisionEncoder.swift` and `AudioEncoder.swift` contain model components and
explicit unsupported preprocessing behavior, but production preprocessing is
not implemented.

Impact:

- Cannot do native media E2E.
- Cannot optimize media tensor prep or caching inside Swift.

## Execution Workstreams

The workstreams below are designed for separate agents. Agents must coordinate
on branch and file ownership to avoid conflicts.

## Agent A: Benchmark Infrastructure Owner

Goal:

Create a benchmark gate that can prove or disprove the 1.5x to 3x target.

Owned files:

```text
tools/gemma4_multimodal_benchmark.py
tools/krillm_vs_ollama_benchmark.py
Makefile
README.md
```

Tasks:

1. Add a first-class `bench-release-gate` target.
2. Add JSON schema-like validation for benchmark reports.
3. Add threshold checks:
   - text decode ratio
   - text wall ratio
   - image wall ratio
   - audio wall ratio
   - prefill ratio
   - memory ratio
4. Add report comparison tooling:
   - compare KrillLM report and Ollama report from separate runs
   - support sequential single-engine runs when disk is constrained
5. Add metadata compatibility checks:
   - model architecture
   - parameters
   - dtype
   - quantization
   - prompt/media SHA256
6. Emit a single release summary:
   - pass/fail per metric
   - geometric mean speedup
   - worst metric
   - bottleneck classification

Acceptance criteria:

- `make bench-release-gate` produces a single JSON report and clear terminal
  pass/fail summary.
- The gate fails if KrillLM does not meet configured thresholds.
- The report says exactly why a comparison is not bit-identical.
- The gate can compare sequential reports generated on low-disk machines.

## Agent B: Persistent KrillLM Server Benchmark Owner

Goal:

Benchmark KrillLM as a persistent warm server, matching Ollama daemon behavior.

Owned files:

```text
Sources/KLMServer/Server.swift
Sources/KLMServer/ServerParsing.swift
tools/gemma4_multimodal_benchmark.py
Makefile
```

Tasks:

1. Add or expose a benchmark-friendly KrillLM HTTP endpoint if needed.
2. Ensure model is loaded once and reused across measured requests.
3. Add warmup endpoint or warmup request sequence.
4. Add per-request timing fields:
   - request received
   - parse complete
   - tokenize complete
   - media preprocess complete
   - prefill complete
   - first token emitted
   - generation complete
5. Make benchmark harness call KrillLM server, not CLI/Python directly.
6. Keep output compatible with OpenAI/Ollama clients where possible.

Acceptance criteria:

- Benchmark can run:

```bash
krillm serve --model <model> --port <port>
make bench-release-gate KRILLM_URL=http://127.0.0.1:<port>
```

- Report includes TTFT, decode, prefill, wall, and server phase timings.
- KrillLM server benchmark no longer pays per-request process startup.

## Agent C: Native Gemma4 Image Preprocessing Owner

Goal:

Replace image `mlx-vlm` preprocessing with native Swift preprocessing.

Owned files:

```text
Sources/KLMCore/VisionEncoder.swift
Sources/KLMCore/Gemma4Model.swift
Tests/KLMCoreTests/Gemma4MultimodalTests.swift
```

Tasks:

1. Read Gemma4 processor config from model directory.
2. Implement image loading:
   - PNG
   - JPEG
   - RGB conversion
3. Implement resize/crop/pad behavior matching Gemma4 processor.
4. Implement normalization and tensor layout expected by Gemma4 vision tower.
5. Integrate image soft-token insertion for `image_token_id`.
6. Add golden tests against Python `mlx-vlm` preprocessing for a fixed image.
7. Add performance tests for preprocessing time and memory allocation.

Acceptance criteria:

- Native Swift can prepare image tensors without Python.
- One fixed image produces shape/token count matching `mlx-vlm`.
- Native image path can run E2E for Gemma4 in Swift.
- Image wall time improves over the Python bridge.

## Agent D: Native Gemma4 Audio Preprocessing Owner

Goal:

Replace audio `mlx-vlm` preprocessing with native Swift preprocessing.

Owned files:

```text
Sources/KLMCore/AudioEncoder.swift
Sources/KLMCore/Gemma4Model.swift
Tests/KLMCoreTests/Gemma4MultimodalTests.swift
```

Tasks:

1. Read Gemma4 audio processor config from model directory.
2. Implement WAV/AIFF loading for benchmark assets.
3. Add resampling to target sample rate.
4. Implement log-mel spectrogram generation.
5. Implement chunking/masking compatible with Gemma4 audio tower.
6. Handle short audio explicitly:
   - pad if model expects minimum chunk length
   - fail with a clear error if impossible
7. Add golden tests against Python `mlx-vlm` for a fixed WAV.
8. Add performance tests for preprocessing time and memory allocation.

Acceptance criteria:

- Native Swift can prepare audio tensors without Python.
- Fixed WAV produces shape/token count matching `mlx-vlm`.
- Native audio path can run E2E for Gemma4 in Swift.
- Audio wall time improves over Python bridge.

## Agent E: Native Gemma4 Model Integration Owner

Goal:

Make the native Swift Gemma4 model path production-grade for text/image/audio.

Owned files:

```text
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMEngine/InferenceEngine.swift
Tests/KLMEngineTests/
Tests/KLMCoreTests/
```

Tasks:

1. Audit `Gemma4Model.swift` against current Gemma4 architecture.
2. Verify text-only logits against Python/MLX for fixed prompts.
3. Add image/audio embedding injection into the native forward path.
4. Ensure attention masks, rope parameters, sliding/full layer handling, and
   cache semantics match Gemma4 requirements.
5. Add model-load compatibility checks for:
   - 4-bit MLX
   - BF16 MLX
6. Remove experimental warning only when golden tests pass.
7. Add conformance tests for:
   - first-token logits
   - deterministic greedy output
   - KV-cache continuation
   - image prompt
   - audio prompt

Acceptance criteria:

- Native Gemma4 text passes deterministic output tests.
- Native Gemma4 image/audio pass smoke tests without Python.
- `PythonFallback` is no longer used for benchmark-critical paths.

## Agent F: Prefill and KV Optimization Owner

Goal:

Close the prefill gap where Ollama currently wins.

Owned files:

```text
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMCore/KVCache.swift
Sources/KLMCore/PrefixCache.swift
Tests/KLMCoreTests/KVCacheTests.swift
Tests/KLMCoreTests/PrefixCacheTests.swift
```

Tasks:

1. Profile prefill phase separately from decode.
2. Preallocate KV cache buffers instead of repeated concatenation/growth.
3. Reuse prefix cache for static system/chat templates.
4. Cache media embeddings for repeated benchmark assets.
5. Avoid unnecessary MLXArray copies.
6. Add memory allocation instrumentation around prefill.
7. Add benchmark with long prompt and repeated prefix.

Acceptance criteria:

- Prefill timing is visible in reports.
- KV cache avoids repeated reallocations in decode loops.
- Repeated-prefix benchmark shows measurable improvement.
- Image/audio wall time improves because media/prompt setup is amortized or
  faster.

## Agent G: Speculative Decoding Owner

Goal:

Get 1.5x to 3x text decode wins where a compatible drafter exists.

Owned files:

```text
Sources/KLMEngine/SpeculativeDecoder.swift
Sources/KLMEngine/InferenceEngine.swift
Tests/KLMEngineTests/SpeculativeDecodingTests.swift
tools/gemma4_multimodal_benchmark.py
```

Tasks:

1. Identify compatible Gemma4 drafter/assistant models.
2. Add model-pair loading for target + drafter.
3. Benchmark deterministic greedy output equivalence.
4. Optimize verify step and KV rollback.
5. Add adaptive draft length tuning by acceptance rate.
6. Report accepted/proposed token ratio.

Acceptance criteria:

- Text decode reaches >= 1.5x Ollama on at least one release target.
- Output remains deterministic for greedy generation.
- Benchmark records draft acceptance stats.

## Agent H: Release Claims and Docs Owner

Goal:

Keep public claims aligned with benchmark truth.

Owned files:

```text
README.md
OLLAMA_SPEEDUP_EXECUTION_PLAN.md
REVIEW_OBSERVATIONS.md
TEST_COVERAGE_PLAN.md
```

Tasks:

1. Update README only after benchmark gate passes.
2. Keep caveats explicit:
   - dtype mismatch
   - quantization mismatch
   - warm vs cold
   - native vs Python bridge
3. Add a "Performance Claims" section with links to reports.
4. Add reproduction commands.
5. Add release checklist.

Acceptance criteria:

- README never claims 1.5x to 3x without a report.
- Every performance claim names the model, dtype/quantization, hardware, and
  benchmark command.

## Suggested Execution Order

The fastest path is:

1. Agent A: harden benchmark gate.
2. Agent B: benchmark persistent KrillLM server.
3. Agent C and D in parallel: native image/audio preprocessing.
4. Agent E: connect native Gemma4 multimodal E2E.
5. Agent F: optimize prefill/KV after timings are available.
6. Agent G: add speculative decoding for text.
7. Agent H: update claims only after gates pass.

## First 72 Hours Plan

Day 1:

- Add release benchmark gate.
- Add persistent server benchmark route.
- Capture phase timings for current paths.
- Re-run baseline with server-style KrillLM and Ollama.

Day 2:

- Implement image preprocessing MVP.
- Implement audio preprocessing MVP.
- Add golden shape/token-count tests against `mlx-vlm`.

Day 3:

- Wire native image/audio into Gemma4 forward path.
- Run image/audio E2E without Python.
- Profile prefill and identify top three allocations/copies.

## Definition of Done for 1.5x to 3x Claim

The speedup claim is allowed only when all of these are true:

1. `make bench-release-gate` passes.
2. KrillLM uses the native Swift/MLX path for benchmarked features.
3. Reports are committed or attached to the release.
4. Benchmark metadata shows compatible model/precision classes.
5. KrillLM beats Ollama on:
   - text decode
   - text wall time
   - image wall time
   - audio wall time
   - TTFT
6. If any metric does not meet 1.5x, the release claim must be narrowed.

## Current Honest Positioning

Until the above work is complete, use this wording:

```text
KrillLM is competitive with Ollama on Gemma4 E2B decode throughput on Apple
Silicon and can exceed Ollama in some local 4-bit-class decode tests. Ollama is
currently stronger on Gemma4 multimodal prefill and some wall-time metrics.
KrillLM's next performance milestone is a fully native Gemma4 multimodal path
with a release gate targeting 1.5x to 3x speedup over Ollama.
```

Do not use:

```text
KrillLM is faster than Ollama in everything.
```

until benchmark gates prove it.
