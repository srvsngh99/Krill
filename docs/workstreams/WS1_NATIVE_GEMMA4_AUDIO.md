# WS1: Native Gemma 4 Audio

Status: done (delivered via PR #21 + PR #22; native USM Conformer audio
default-on, mlx-vlm bridge retired, audio gates hard in release_candidate)
Detailed plan: [../NATIVE_GEMMA4_AUDIO_PLAN.md](../NATIVE_GEMMA4_AUDIO_PLAN.md)

## Goal

Move Gemma 4 voice/audio from the `mlx-vlm` bridge to native Swift + MLX
running on Metal.

## Why This Exists

Audio is currently the main multimodal scope gap:

```text
audio input -> PythonFallback -> persistent mlx-vlm sidecar -> Python MLX
```

That path works, but it is slower than Ollama and is scoped out of the
`release_candidate` gate. It blocks any production claim for voice workflows.

## Deliverables

- Native audio preprocessing.
- Native Gemma 4 audio tower.
- Native `embed_audio.*` projection.
- CLI and server routing that use native audio by default when available.
- Bridge fallback retained for debugging and compatibility.
- Live audio quality tests and fresh text/vision/audio benchmark report.

## Acceptance

- CLI audio does not instantiate `PythonFallback` when native audio is
  available.
- Server `/api/generate`, `/api/chat`, and `/v1/chat/completions` route
  Gemma 4 audio natively.
- Image+audio can run through the native Swift multimodal path.
- Audio metrics are present in benchmark reports.
- Strict gate no longer fails because audio is bridge-backed or missing.

## Agent Kickoff

Start with [../NATIVE_GEMMA4_AUDIO_PLAN.md](../NATIVE_GEMMA4_AUDIO_PLAN.md).
Do not weaken release gates. Keep the bridge as an explicit fallback.
