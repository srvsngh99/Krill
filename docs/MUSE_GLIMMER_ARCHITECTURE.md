# Muse Glimmer 30B: six architectural choices worth stealing

Meta shipped [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)
in August 2026 — Apache 2.0, multimodal, distilled from Muse Spark, and built
for agents that run on a laptop rather than in a datacentre.

We ported it to [Krill](https://github.com/souravai/krill), our native
Swift + MLX inference engine for Apple Silicon. Porting a model is the most
thorough way to read its architecture: you cannot skim a design decision you
have to reimplement. Six choices stood out, and most of them are not in the
model card.

All numbers below are read from the shipped `config.json` and the reference
implementation, not estimated.

---

## 1. Long context is nearly free, and that is the whole design

Muse Glimmer advertises a 131,072-token context. The interesting part is what
that costs.

52 layers, but only **13 are full-attention**. The other 39 are sliding-window
layers permanently capped at 2048 tokens. Combine that with just **2 KV heads**
(against 32 query heads — a GQA ratio of 16) at head_dim 128, and the KV cache
works out to 1 KiB per token per layer:

| context | full-layer KV | sliding KV | **total KV** |
|---:|---:|---:|---:|
| 4,096 | 0.05 GiB | 0.08 GiB | **0.13 GiB** |
| 32,768 | 0.41 GiB | 0.08 GiB | **0.48 GiB** |
| 131,072 | 1.62 GiB | 0.08 GiB | **1.70 GiB** |

**1.7 GiB at the full 131K.** The 39 sliding layers stop growing after 2048
tokens, so context length only scales the 13 global layers.

The practical consequence on consumer hardware: **context is not your
constraint, weights are.** On a 24 GB Mac the 4-bit build is 18.1 GiB of
weights against 1.7 GiB of KV at maximum context. You will run out of room for
the model long before you run out of room for the conversation — which is
exactly backwards from the usual local-inference experience, and it is
deliberate.

## 2. The global layers are the ones without positional encoding

Modern long-context models usually give local layers a short RoPE and global
layers a long-theta RoPE. Muse Glimmer inverts it. From the config:

```json
"layer_types":      ["sliding_attention", "sliding_attention",
                     "sliding_attention", "full_attention", ...]
"layer_rope_theta": [500000.0, 500000.0, 500000.0, 0, ...]
```

Every fourth layer — the full-attention one — has `layer_rope_theta == 0`,
meaning **NoPE**: no positional encoding at all. The reference simply passes
`position_embeddings=None` into those layers.

The 39 sliding layers carry standard RoPE and handle local order. The 13 global
layers see the whole sequence with no positional signal whatsoever, and rely on
the causal mask alone for ordering. NoPE layers are known to extrapolate past
their training length far more gracefully than RoPE ones, so this is a
length-generalisation strategy: keep positions where the window is short and
positions are reliable, drop them where you need to reach 131K.

## 3. Attention has an output gate

Each attention block computes an extra projection from the layer's *normed
input* and uses it to gate the attention output — before `o_proj`:

```
o = sdpa(q, k, v)
o = o * sigmoid(gate_proj(x))     # x = the layer's normed input
x = o_proj(o)
```

That is a full `[hidden, heads*head_dim]` weight per layer (6656 × 4096) spent
purely on deciding how much attention to let through. It is the same idea as
the gate in a GLU feed-forward, applied to attention — the block can suppress
its own attention output per-channel when the context is not useful, which is
plausibly why the model recovers well from bad tool calls in agentic loops.

Note the gate reads the block's *input*, not the attention result. It is
deciding "how much should this token listen at all", not "was that lookup any
good".

## 4. Two RMSNorm formulas in the same model

The four per-layer norms use the Gemma-style centred form, with weights stored
around zero:

```
output = normalize(x) * (1 + w)
```

The final `language_model.norm` uses the plain form:

```
output = normalize(x) * w
```

Mixing these up produces no crash and no shape error — just a model that is
quietly, confidently wrong. The layers are also arranged as a Gemma-3-style
sandwich (norm → attn → post-norm → residual, norm → MLP → post-norm →
residual), and the pre- and post-norms use **different epsilons**: `1e-5` and
`1e-8`. There are three further norms with no weights in the checkpoint at all
— on the queries and keys, on the token embeddings, and on the projected vision
features.

## 5. Query scaling that is additive to the softmax scale

Queries are RMS-normalised per head and then multiplied by `qk_scale_factor`
(3.87) — and the usual `head_dim**-0.5` softmax scale **still applies on top**.
Keys are normalised but not scaled.

Normalising Q and K puts them on a unit sphere, which makes attention logits
depend only on direction; the learned constant then re-introduces a controlled
temperature. It is a training-stability trick that survived into inference, and
it is easy to misread as a replacement for the standard scale rather than an
addition to it.

The head also applies an `output_multiplier` (0.196) and then tanh
softcapping at ±20 — so no logit can ever exceed 20 in magnitude.

## 6. A drafter that reads the target's mind

The repo ships `Muse-Glimmer-30B-assistant`, a 3B "DFlash" speculative-decoding
drafter claiming 3.1× speedup. It is not a standalone small model: its config
declares

```json
"target_layer_ids": [1, 13, 25, 37, 49]
```

It consumes **hidden states from five specific layers of the 30B target**. You
cannot run it alone, and you cannot pair it with a different target. Speculative
decoding usually pairs two independently-trained models and accepts a weak
drafter; this one is trained against the target's internals, which is why five
layers of 3B can keep up with 30B.

---

## The vision tower has a trap in it

The perception encoder is a 1.8B ViT-G/14 with windowed attention (3 of every 4
layers), 2-axis interleaved RoPE, and a `pixel_shuffle` 2×2 merge. Standard
enough. But the learned 32×32 position table is resampled to each image's patch
grid, and *how* it is resampled matters:

`modeling_muse_glimmer.py` **defines its own copy** of
`get_vision_bilinear_indices_and_weights`, shadowing the same-named helper in
`transformers/vision_utils.py`, with this comment:

> the fn is equivalent with `F.grid_sample(inputs, align_corners=False, padding="zeros")`

Two differences from the helper it shadows, and our port hit both:

1. **Half-pixel centres**, not `align_corners=True` — the shared helper's
   docstring says the opposite. Getting this wrong is a ~6% error on the patch
   embeddings.
2. **Zero padding**: a tap whose index falls outside the table contributes
   *nothing*, so the four bilinear corner weights do not sum to 1 near the
   border. Keeping the clamped tap's weight is a ~2.5% error, and it only shows
   up when the patch grid is large relative to the position table.

Neither changes a single tensor shape. Both are invisible without a
logit-parity gate. If you are porting this model, read the model file's version
of that function, never `vision_utils`'.

---

## Status in Krill

Honest state as of 2026-08-11:

- **Native Swift + MLX runtime landed** — text decoder and perception encoder,
  no Python in the inference path.
- **Logit-parity gated** against the transformers reference on synthetic
  checkpoints: prefill, cached decode, vision tower and the full image-feature
  chain all match at argmax + cosine > 0.9999. The gate runs at both a fast tiny
  geometry and at the real tensor geometry (hidden 6656, GQA 32/2, head_dim
  128).
- **The real 4-bit checkpoint loads correctly** — every parameter maps and
  verifies.
- **Cross-verified against a second runtime on the real weights** — Krill vs
  mlx-vlm 0.6.12 over prefill and cached decode: same argmax, cosine > 0.9999.
  Synthetic gates could never have settled this; every isolated axis (geometry,
  depth, quantization, dtype) was green while the model looked broken.
- **Generation works end to end.** `krill run muse-glimmer-30b "What is the
  capital of France? Answer in one word."` answers `Paris` at ~8 tok/s decode.

One protocol bug was real and is fixed: the model addresses each message to a
recipient (`to=self` for its scratchpad, `to=user` for the answer), and Krill
treated the whole stream as visible text — so the reply arrived wrapped in raw
`to=self<|message|>` scaffold. Reasoning now routes to the reasoning channel,
and the checkpoint counts as a reasoning template for the token-budget rule
(at `high` reasoning strength it can spend a 512-token budget entirely on
thinking and hand back an empty answer).

**Image serving works.** An image request routes to `MuseGlimmerRuntime`, which
splices perception-encoder features into the token embeddings at the `<|patch|>`
run and then decodes exactly like the text path (1-D positions, no mRoPE offset
to thread). Text turns keep the generic dense path, so they lose none of its
prefix caching or speculation. Asked to describe a red circle above a blue
rectangle, the 30B answers: *"A red circle sits on top of a blue rectangle
against a light gray background."*

Wiring it up exposed a bug well outside this model: Krill's image decoders
flipped every image vertically. A `CGBitmapContext` stores its backing buffer
top row first — the bottom-left origin governs the DRAWING coordinate system,
not memory layout — but four decoders (Gemma 4, LLaVA, Qwen 2.5-VL, Llama 3.2
Vision) flipped rows on readout anyway. Shapes, colours and histograms all
survive a vertical flip, so it never looked like a bug; it looked like models
being bad at "above" and "below". `ImageDecodeOrientationTests` now pins all
seven decoders against a fixture authored outside CoreGraphics.

Still open: the `<atem:function_calls>` tool dialect is not parsed, which is why
`.tools` is not advertised. That plus the memory envelope — 18.1 GiB of weights
paging on a 24 GB box — is what keeps this `.experimental`.

## What this costs on a 24 GB Mac

The 4-bit build is 18.1 GiB. On a 24 GB machine that leaves little for macOS,
activations and KV. The published MLX build's own card recommends 48 GB.

Measured with `krill bench` (512-token prompt / 4096-token prompt, 32 generated,
warmup + averaged runs) against the rest of the local ladder:

| model | weights | decode @512 | decode @4096 | peak |
|---|---:|---:|---:|---:|
| gemma-4-12b | 6.7 GB | 26.9 tok/s | 24.8 tok/s | 9.9 GB |
| Qwen3-Coder-30B-A3B (MoE) | 16.0 GB | 47.5 tok/s | 12.1 tok/s | 18.1 GB |
| **muse-glimmer-30b** | 18.1 GB | **7.2 tok/s** | **6.0 tok/s** | 17.3 GB |

This is §1 showing up in the measurements. Going from 512 to 4096 tokens — 8x the
context — costs Muse Glimmer **17%** of its decode rate and 1.2 GB of peak
memory, because 39 of 52 layers never look further back than their 2048 window
and carry only 2 KV heads. The 30B MoE next to it, faster than Muse Glimmer by
6.6x at short context, gives up **75%** of its decode rate over the same
stretch. Context really is nearly free here; weights really are the wall.

Prefill is where the dense stack is paid for: 71–79 tok/s against the MoE's
476–513, since every token touches all 30B parameters rather than 3B of active
experts. A 4096-token prompt therefore takes ~52 s to first token.

(An earlier note here claimed 1.0 tok/s. That came from a run competing with a
concurrent sweep; a later 0.1 tok/s reading came from a benchmark harness that
prefilled through the full forward and allocated caches without the family's
`cacheSpec`. Both are superseded by the table above.)

Given §1, the fix is not more RAM for context — it is smaller weights. A ~3-bit
mixed-precision requant lands near 13–14 GiB, which fits with headroom, and the
1.7 GiB full-context KV cache still fits alongside it. That is the build worth
making for consumer hardware.

---

*Ported and measured with [Krill](https://github.com/souravai/krill) — native
Swift + MLX inference on Apple Silicon.*
