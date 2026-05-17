# Native Gemma 4 Audio Implementation Plan

Created: 2026-05-17
Status: implementation handoff
Owner: unassigned

## Goal

Replace the current Gemma 4 audio bridge path with a native Swift + MLX
implementation that runs on Metal and can be benchmarked directly against
Ollama.

The current release-candidate scope is text + vision. Voice/audio is
explicitly out of scope because it still routes through `mlx-vlm` and does
not satisfy the strict release gate. Native audio is required before KrillLM
can claim production readiness for voice workflows.

## Current State

Current routing:

```text
CLI/server audio request
  -> PythonFallback
  -> persistent mlx-vlm sidecar
  -> Python MLX implementation
```

Native routing today:

```text
text only    -> Swift text model
image only   -> Swift text model + native SigLIP2 vision encoder
audio input  -> mlx-vlm bridge
image+audio  -> mlx-vlm bridge
```

Important files:

```text
Sources/KLMCLI/RunCommand.swift
Sources/KLMCore/Gemma4Model.swift
Sources/KLMCore/ModelLoader.swift
Sources/KLMCore/VisionEncoder.swift
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMEngine/PythonFallback.swift
Sources/KLMServer/Server.swift
Sources/KLMServer/ServerMultimodal.swift
tools/gemma4_multimodal_benchmark.py
tools/release_gate.py
docs/GEMMA4_INTERNALS.md
docs/BENCHMARKING.md
docs/RELEASE_READINESS_REMEDIATION.md
```

Known unloaded Gemma 4 audio weights:

```text
audio_tower.*       -> BF16, not loaded into Swift modules yet
embed_audio.*       -> QuantizedLinear 4-bit, not loaded yet
```

Relevant token:

```text
<|audio|> token id = 258881
```

Latest local benchmark signal after PR #20 merge:

```text
Report: .build/benchmarks/release-gemma4-mm-server-4x2-2026-05-17.json
Gate:   .build/benchmarks/release-gate-server-4x2-2026-05-17.json

release_candidate: PASS
strict:            FAIL

audio_wall_ratio:  3.6610x slower than Ollama
audio_prefill:     unavailable for KrillLM audio path
```

The strict failure is expected until native audio lands.

## Non-Goals

- Do not remove the `mlx-vlm` bridge immediately. Keep it as a fallback and
  a correctness oracle while native audio is being built.
- Do not change text or image routing unless needed for image+audio
  integration.
- Do not weaken strict release gates to hide audio failures.
- Do not claim production-ready voice/audio until native audio passes quality
  and performance gates.

## Workstream 1: Discover The Audio Architecture

Objective: produce an exact map from Gemma 4 config and weights to Swift
modules.

Tasks:

1. Inspect the local checkpoint:

   ```text
   /Users/sourav/.krillm/models/blobs/gemma-4-e2b
   ```

2. Extract and document:

   ```text
   config.json audio_config
   processor config / feature extractor settings
   audio_tower.* key names and tensor shapes
   embed_audio.* key names and tensor shapes
   audio token expansion rules
   expected mel/log-mel input shape
   sample rate and padding/chunking rules
   ```

3. Compare against `mlx-vlm` / upstream Gemma 4 implementation for:

   ```text
   waveform loading
   resampling
   mono conversion
   STFT window/hop/FFT
   mel filterbank
   log/normalization
   attention mask construction
   audio tower output length
   projection into language hidden size
   ```

Deliverable:

```text
docs/GEMMA4_INTERNALS.md updated with the audio tower architecture,
preprocessing constants, tensor shapes, and weight-key mapping.
```

Acceptance:

- A developer can map every required `audio_tower.*` and `embed_audio.*`
  tensor to a Swift module or an explicit TODO.
- The expected input/output shapes for a deterministic WAV fixture are known.

## Workstream 2: Native Audio Preprocessing

Objective: implement the audio frontend in Swift.

Start narrow:

```text
WAV PCM mono/stereo input only
local file input for CLI
base64 WAV input for server
```

Required components:

```text
WAV decode
stereo to mono conversion
resampling to Gemma 4 sample rate
STFT
mel filterbank
log/normalization
padding/chunking
attention mask or valid-frame metadata
```

Suggested file:

```text
Sources/KLMCore/AudioPreprocessor.swift
```

Testing:

```text
Tests/KLMCoreTests/AudioPreprocessorTests.swift
```

Acceptance:

- Deterministic WAV fixture produces stable shapes and hashes.
- Preprocessor output shape matches `mlx-vlm` for the same fixture.
- Invalid, oversized, and unsupported audio files fail with actionable
  errors before model execution.

## Workstream 3: Native Audio Tower

Objective: implement the Gemma 4 audio encoder and projection path in Swift
using MLX arrays on Metal.

Expected modules:

```text
AudioEncoder
AudioConformerBlock
AudioAttention / relative-position attention
AudioConv / subsampling layers if present in config
AudioProjection / embed_audio projection
```

Suggested file:

```text
Sources/KLMCore/AudioEncoder.swift
```

Model loading changes:

```text
Sources/KLMCore/ModelLoader.swift
```

Load:

```text
audio_tower.*
embed_audio.*
```

Follow existing local patterns:

- Use structured safetensors key loading rather than ad hoc string slicing.
- Reuse `ClippableLinear` where Gemma 4 audio weights use the same
  `linear.weight`, `input_min`, `input_max`, `output_min`, `output_max`
  pattern as vision.
- Preserve quantization metadata for `embed_audio.*`.
- Keep audio tower BF16 unless the checkpoint requires otherwise.

Acceptance:

- The audio tower can run independently on a WAV fixture without touching
  the language model.
- Intermediate tensor shapes match `mlx-vlm`.
- Final projected audio embeddings have the same hidden size as the text
  model (`1536` for Gemma 4 E2B).

## Workstream 4: Multimodal Integration

Objective: route audio embeddings through the existing native multimodal
generation path.

Current native image path:

```text
image bytes
  -> VisionEncoder
  -> MultimodalEmbedder
  -> masked scatter into <|image|> token positions
  -> language model forward
```

Target native audio path:

```text
audio bytes
  -> AudioPreprocessor
  -> AudioEncoder
  -> Audio projection
  -> masked scatter into <|audio|> token positions
  -> language model forward
```

Required changes:

```text
Sources/KLMEngine/InferenceEngine.swift
Sources/KLMCLI/RunCommand.swift
Sources/KLMServer/Server.swift
Sources/KLMServer/ServerMultimodal.swift
```

Routing rules:

1. If loaded Gemma 4 checkpoint has native audio modules, audio-only requests
   use Swift native audio by default.
2. If request has image + audio and both native modules are available, use
   native image + native audio in one Swift generation path.
3. If native audio is unavailable and bridge is installed, bridge fallback
   remains available.
4. If native audio is unavailable and bridge is not installed, return the
   existing clear 503/install hint.

Suggested feature flags:

```text
KRILL_NATIVE_AUDIO=1        # default on after implementation is stable
KRILL_AUDIO_BRIDGE_ONLY=1   # force old bridge path for comparison/debug
```

Acceptance:

- `krillm run gemma-4-e2b "What sound is this?" --audio fixture.wav`
  does not instantiate `PythonFallback` when native audio is enabled.
- Server `/api/generate`, `/api/chat`, and `/v1/chat/completions` route
  audio requests natively when possible.
- Image+audio no longer forces the whole request through Python when native
  image and native audio are both available.

## Workstream 5: Correctness Tests

Add unit tests first, then live tests.

Unit tests:

```text
AudioPreprocessorTests
AudioEncoderShapeTests
ModelLoaderAudioWeightTests
InferenceEngineAudioRoutingTests
ServerAudioRoutingTests
```

Live tests, gated by:

```text
KLM_GEMMA4_MODEL_PATH=/Users/sourav/.krillm/models/blobs/gemma-4-e2b
```

Required live checks:

1. Native audio request produces non-empty output.
2. Audio fixture changes output versus text-only prompt.
3. Two different audio fixtures produce different outputs.
4. Image+audio request conditions on both media inputs.
5. Bridge-forced and native audio outputs are semantically compatible on
   deterministic fixtures.

Quality fixtures:

```text
.build/benchmarks/assets/gemma4-tone-5s.wav
.build/benchmarks/assets/gemma4-silence-2s.wav
.build/benchmarks/assets/gemma4-sine-1khz-5s.wav
```

If these fixtures are not semantically reliable enough, create better ones
with `tools/generate_audio_fixture.py` and update the benchmark rubric.

## Workstream 6: Benchmark And Gate

Run all benchmarks sequentially to avoid memory pressure.

Build:

```text
make release
```

Start server:

```text
.build/release/krillm serve --model gemma-4-e2b --host 127.0.0.1 --port 11435 --compat both
```

Run multimodal benchmark:

```text
/Users/sourav/.krillm/venv/bin/python3 tools/gemma4_multimodal_benchmark.py \
  --krill-model /Users/sourav/.krillm/models/blobs/gemma-4-e2b \
  --ollama-model gemma4:e2b \
  --krillm-url http://127.0.0.1:11435 \
  --krillm-image-mode native_server \
  --runs 4 \
  --warmup 2 \
  --output .build/benchmarks/native-audio-mm.json
```

Run gates:

```text
make bench-release-gate \
  GATE_INPUT=.build/benchmarks/native-audio-mm.json \
  GATE_REPORT=.build/benchmarks/native-audio-release-candidate-gate.json \
  GATE_ALLOW_FLAGS="--profile release_candidate"

make bench-release-gate \
  GATE_INPUT=.build/benchmarks/native-audio-mm.json \
  GATE_REPORT=.build/benchmarks/native-audio-strict-gate.json \
  GATE_ALLOW_FLAGS="--profile strict"
```

Acceptance:

- `release_candidate` still passes.
- `strict` no longer fails audio due to bridge scope.
- Audio metrics are present, not `N/A`.
- `audio_wall_ratio <= 0.67x`, or a revised threshold is justified in a
  separate release-gate proposal and accepted.
- Benchmark report identifies the KrillLM audio path as native, not bridge.

## Definition Of Done

Native audio is done only when all are true:

1. Audio preprocessing and audio tower are implemented in Swift + MLX.
2. `audio_tower.*` and `embed_audio.*` load from the local Gemma 4 checkpoint.
3. CLI audio uses native Metal by default when available.
4. Server audio uses native Metal by default when available.
5. Image+audio no longer forces bridge fallback when native modules are
   available.
6. The bridge remains available behind an explicit fallback/debug path.
7. Live Gemma 4 audio tests pass.
8. Multimodal benchmark includes text, vision, and audio with native audio.
9. Release docs and README support matrix are updated.
10. Strict gate audio failures are closed or consciously re-scoped by an
    accepted release-gate proposal.

## Risks

- Gemma 4 audio preprocessing may be more important than the tower itself;
  tiny shape or normalization mismatches can produce plausible but wrong
  answers.
- Audio relative-position attention may differ from text/vision attention.
- The current audio fixture can be ambiguous; quality tests need fixtures
  that support clear expected/forbidden terms.
- Memory accounting changes once audio moves into the long-lived server
  process. Compare peak memory on server-mode reports, not CLI one-offs.
- Mixed image+audio requests may need multiple media scatter regions in one
  prompt. Do not regress image-only cache correctness.

## Agent Kickoff Prompt

Use this prompt when dispatching an implementation agent:

```text
Implement native Gemma 4 audio for KrillLM.

Start by reading docs/NATIVE_GEMMA4_AUDIO_PLAN.md,
docs/GEMMA4_INTERNALS.md, Sources/KLMCore/Gemma4Model.swift,
Sources/KLMCore/VisionEncoder.swift, Sources/KLMCore/ModelLoader.swift,
Sources/KLMEngine/InferenceEngine.swift, Sources/KLMEngine/PythonFallback.swift,
and tools/gemma4_multimodal_benchmark.py.

Goal: replace the mlx-vlm bridge for Gemma 4 audio with a native Swift + MLX
audio preprocessing, audio tower, and audio projection path that runs on
Metal. Keep the bridge as fallback/debug only.

Do not weaken release gates. Add focused unit tests and live tests gated by
KLM_GEMMA4_MODEL_PATH. Update docs after implementation.

Important acceptance:
- CLI and server audio do not use PythonFallback when native audio is
  available.
- Image+audio can run through the native Swift path.
- Fresh text/vision/audio benchmark records native audio metrics.
- release_candidate passes, and strict no longer fails solely because audio
  is bridge-backed or missing metrics.
```
