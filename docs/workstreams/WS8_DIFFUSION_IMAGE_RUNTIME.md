# WS8: Diffusion Image Runtime (Mage-Flow)

Status: SCOPING. No code. This document exists to decide whether Krill takes on
image generation at all, and if so what the smallest honest version costs.

## The ask, and why it is not a model port

`microsoft/Mage-Flow` was raised as "add native support, MLX nvfp4", alongside
two language models. It is not a language model, and the difference is not a
detail — every existing Krill family (dense text, MoE, VLM, embedder, reranker,
ASR/TTS) shares one shape: **tokens in, logits out, sampled autoregressively
against a KV cache**. `LoadedModel` encodes exactly that contract:

```swift
public let forward: (MLXArray, [KVCacheProtocol]?) -> MLXArray   // -> logits
public let vocabSize: Int
```

Mage-Flow has no tokens out, no logits, no vocabulary, and no KV cache. It
denoises a latent image over N scheduler steps and decodes it through a VAE. It
cannot be expressed as a `LoadedModel`, so none of the loader table, the
sampler, the batcher, the prefix cache, or the OpenAI/Ollama chat surface
applies. Adding it means a **second runtime alongside the LM runtime**, not a
row in `architectureRules`.

## What the checkpoint actually contains

`model_index.json` declares a `MageFlowPipeline` with four parts:

| Component | Class | Size | Notes |
|---|---|---|---|
| `transformer` | `MageFlow` (custom, `mage_flow` module) | 8.23 GB | MMDiT, `depth: 12` double-stream blocks, hidden 3072, 24 heads, `qkv_bias`, `rope_type: "msrope"` (3-axis, `axes_dim [16, 56, 56]`) |
| `vae` | `MageVAE` (custom) | 0.35 GB | latent `in/out_channels: 128`, `patch_size: 1` |
| `text_encoder` | `Qwen3VLForConditionalGeneration` | 8.88 GB | conditioning; `context_in_dim: 2560` |
| `scheduler` | `FlowMatchEulerDiscreteScheduler` | — | rectified flow, `static_shift: 6.0`, `schedule_mode: "z-image"` |

Total repo 17.51 GB. Note the transformer and VAE are **custom classes in a
`mage_flow` module**, not stock `diffusers` blocks — so there is no reference
implementation to transliterate from `diffusers` directly; the upstream package
is the spec.

## Work required

Ordered by risk, not by sequence.

1. **MMDiT double-stream block** (highest risk). Two residual streams (image +
   text) with cross-modal attention and per-stream modulation. Nothing in
   `KrillCore` resembles this; `TransformerBlock` is single-stream causal
   pre-norm. New Swift module, no reuse.
2. **`msrope` 3-axis rotary** with `axes_dim [16, 56, 56]`. Krill has 3D mRoPE
   for Qwen2.5-VL (`Qwen25VLPositions`), which is the nearest prior art, but the
   axis split and application differ — expect adaptation, not reuse.
3. **MageVAE decoder.** 128-channel latent -> RGB. Krill has no VAE of any kind.
   Small in parameters (0.35 GB), not small in surface: resnet blocks,
   attention at some resolutions, upsampling stack, and exact padding.
4. **Flow-matching Euler scheduler.** The genuinely cheap part — a sigma
   schedule and an Euler step, plus `static_shift`/`z-image` shift handling.
   Pure arithmetic, easily unit-tested against the Python scheduler.
5. **Qwen3-VL text conditioning.** Krill has a native `qwen3_5` text runtime,
   but this needs the **VL** encoder producing a 2560-dim conditioning sequence,
   and Krill's qwen3_5 support is text-only today. Either port the VL tower or
   accept a dependency.
6. **Image I/O + a non-chat API surface.** Output is pixels. Needs an endpoint
   that is not `/v1/chat/completions` (`/v1/images/generations` is the obvious
   shape), plus PNG encode. Image *editing* additionally needs VAE **encode**
   and a reference-image path.
7. **Quantization.** `CheckpointQuantizer.assertSupportedDense` refuses
   multimodal/vision checkpoints outright, and nvfp4 is group-16 — DiT and VAE
   weights are far more sensitive to 4-bit than LM weights, and diffusion has no
   argmax to hide error behind. Expect the VAE to stay fp16 and the DiT to need
   per-module protection, i.e. the reference-set path, which needs a known-good
   4-bit build to learn from — and none exists.

## Parity strategy

The LM gates do not transfer: there are no logits and no argmax. The oracle must
be **latent-space, step-wise**: run the Python pipeline with a fixed seed and
fixed timesteps, dump the latent after each scheduler step, and assert the Swift
runtime matches per step (cosine + max-abs), then compare the final decoded
image. Divergence in a diffusion loop compounds silently and a plausible-looking
image is not evidence of correctness — per-step latents are the only honest gate.
Fix the noise externally rather than trusting seed parity across RNGs.

## Recommendation

This is a **new modality**, comparable in size to WS5 (second native vision
family) or larger, and it lands nothing for existing users. Three options:

1. **Do it as a proper workstream.** Sequence: scheduler -> VAE decode -> MMDiT
   block -> msrope -> conditioning -> API. Each stage independently gated
   against a Python oracle. fp16 first; treat nvfp4 as a follow-up, not a
   requirement.
2. **Publish weights only.** Convert to MLX and upload so the artifact exists.
   Cheap, but nothing can run it — including Krill — so it is a placeholder, and
   an nvfp4 conversion would ship unvalidated (see quantization risk above).
3. **Decline.** Krill is an LLM engine; image generation is well served
   elsewhere on Apple Silicon (`mflux`, `diffusers` + MPS).

Recommendation: **(1) if image generation is a product goal, otherwise (3)**.
Option 2 is the weakest — it spends the download and the quantization risk
without producing anything runnable or verified.

## Explicitly out of scope until this is decided

Any `architectureRules` row, `ModelFamily` case, or alias for `mage_flow`. A
family entry implies `loadModel` can serve it, and it cannot.
