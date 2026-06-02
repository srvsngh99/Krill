#!/usr/bin/env python3
"""Build a tiny Llama-3.2-Vision (mllama) checkpoint with mlx-vlm and record its
reference logits, to check the native KrillLM `Llama32VisionForCausalLM` runtime
for logit parity against mlx-vlm on identical weights + vision inputs.

Exercises the full mllama path: the tiled ViT vision tower (Conv2d patch embed,
class token, gated aspect-ratio + position embeddings, local transformer + gated
global transformer, intermediate-layer concatenation), the multi-modal
projector, and the Llama text decoder whose `cross_attention_layers` attend to
the projected vision features (gated cross-attention with q/k RMSNorm).

ALL parameters (including the normally zero-initialized cross-attention and
vision gates) are randomized, so the gated cross-attention actually contributes
and the parity check is meaningful rather than a 0*x tautology.

The real Llama-3.2-11B-Vision is large; a tiny synthetic mllama stands in for
the numerics (both runtimes are MLX, so parity is ~bit-exact).

Usage:
    python3 tools/verify_mllama_parity.py /tmp/krillm-mllama
    KLM_MLLAMA_PARITY_DIR=/tmp/krillm-mllama swift test --filter MllamaParityTests
"""
import json
import os
import sys

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_vlm.models.mllama import Model, ModelConfig
from mlx_vlm.models.mllama.config import TextConfig, VisionConfig

# Tiny dims.
TEXT_HIDDEN = 128
TEXT_LAYERS = 4
CROSS_LAYERS = [1, 3]            # which decoder layers are cross-attention
VOCAB = 320
IMAGE_SIZE = 56
PATCH = 14                       # (56/14)^2 = 16 patches; +1 CLS = 17
VIS_HIDDEN = 64
VIS_LAYERS = 4
VIS_GLOBAL = 2
INTER_INDICES = [1, 3]
MAX_TILES = 4
NUM_TILES = 4                    # use all tiles (aspect_ratio_mask all-ones)
ASPECT_ID = 3
# vision_output_dim = hidden * (1 + len(intermediate_layers_indices))
VIS_OUT = VIS_HIDDEN * (1 + len(INTER_INDICES))   # 64 * 3 = 192


def text_config_dict():
    return {
        "model_type": "mllama", "vocab_size": VOCAB, "hidden_size": TEXT_HIDDEN,
        "intermediate_size": 256, "num_hidden_layers": TEXT_LAYERS,
        "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0, "max_position_embeddings": 2048,
        "cross_attention_layers": CROSS_LAYERS, "tie_word_embeddings": False,
    }


def vision_config_dict():
    return {
        "image_size": IMAGE_SIZE, "patch_size": PATCH, "num_channels": 3,
        "hidden_size": VIS_HIDDEN, "intermediate_size": 128,
        "num_hidden_layers": VIS_LAYERS, "num_attention_heads": 4,
        "max_num_tiles": MAX_TILES, "max_aspect_ratio_id": 8,
        "num_global_layers": VIS_GLOBAL, "norm_eps": 1e-5,
        "vision_output_dim": VIS_OUT, "intermediate_layers_indices": INTER_INDICES,
    }


def build(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)

    text = TextConfig(**text_config_dict())
    vision = VisionConfig(**vision_config_dict())
    cfg = ModelConfig(
        text_config=text, vision_config=vision, model_type="mllama",
        image_token_index=128256, vocab_size=VOCAB)

    mx.random.seed(0)
    model = Model(cfg)
    mx.eval(model.parameters())

    # Randomize EVERY parameter (small scale) so the zero-init gates become
    # non-zero and the gated cross-attention / vision paths are exercised.
    flat = tree_flatten(model.parameters())
    randomized = []
    for i, (k, v) in enumerate(flat):
        r = mx.random.normal(v.shape, scale=0.06, key=mx.random.key(1000 + i))
        randomized.append((k, r.astype(v.dtype)))
    model.update(tree_unflatten(randomized))
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(os.path.join(outdir, "model.safetensors"), weights)

    # Synthetic inputs.
    tokens = [5, 6, 7, 8, 9, 10]
    input_ids = mx.array([tokens])
    pixel_values = mx.random.normal(
        [1, 1, NUM_TILES, 3, IMAGE_SIZE, IMAGE_SIZE], key=mx.random.key(42))
    aspect_ratio_ids = mx.array([[ASPECT_ID]])
    aspect_ratio_mask = mx.ones([1, 1, MAX_TILES], dtype=mx.int32)
    mx.eval(pixel_values)

    out = model(
        input_ids, pixel_values,
        aspect_ratio_ids=aspect_ratio_ids,
        aspect_ratio_mask=aspect_ratio_mask)
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    last = logits[0, -1, :]

    config = {
        "architectures": ["MllamaForConditionalGeneration"],
        "model_type": "mllama",
        "image_token_index": 128256,
        "vocab_size": VOCAB,
        "text_config": text_config_dict(),
        "vision_config": vision_config_dict(),
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    os.makedirs(os.path.join(outdir, "inputs"), exist_ok=True)
    mx.save_safetensors(
        os.path.join(outdir, "inputs", "vision_inputs.safetensors"),
        {
            "pixel_values": pixel_values.astype(mx.float32),
            "aspect_ratio_ids": aspect_ratio_ids.astype(mx.int32),
            "aspect_ratio_mask": aspect_ratio_mask.astype(mx.int32),
        })
    ref = {
        "tokens": tokens,
        "vocab_size": VOCAB,
        "last_token_logits": [float(v) for v in last.tolist()],
        "argmax": int(mx.argmax(last).item()),
    }
    with open(os.path.join(outdir, "reference_logits.json"), "w") as f:
        json.dump(ref, f)

    # Text-only reference (no image): exercises the cross-attention no-image
    # split-query fallback, which the image path does not reach.
    text_out = model(input_ids)
    text_logits = text_out.logits if hasattr(text_out, "logits") else text_out
    mx.eval(text_logits)
    text_last = text_logits[0, -1, :]
    text_ref = {
        "tokens": tokens,
        "vocab_size": VOCAB,
        "last_token_logits": [float(v) for v in text_last.tolist()],
        "argmax": int(mx.argmax(text_last).item()),
    }
    with open(os.path.join(outdir, "reference_text_logits.json"), "w") as f:
        json.dump(text_ref, f)

    print(f"Wrote tiny mllama + reference logits to {outdir}")
    print(f"  tokens={tokens}  vocab={VOCAB}")
    print(f"  argmax(last_token_logits) = {ref['argmax']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    build(sys.argv[1])
