#!/usr/bin/env python3
"""Build a tiny LLaVA-1.5 checkpoint with mlx-vlm and record its reference
logits, to check the native Krill `LlavaForCausalLM` runtime for logit parity
against mlx-vlm on identical weights + pixel inputs.

Exercises the full LLaVA path: the CLIP vision tower (Conv2d patch embed + class
token + position embedding + pre/post LayerNorm + quick-gelu MLP encoder), the
`vision_feature_layer=-2` + drop-CLS feature selection, the multi-modal
projector (linear -> gelu -> linear), the image-token merge into the text
embeddings, and the Llama text backbone.

The real llava-1.5-7b is large; a tiny synthetic LLaVA stands in for the
numerics (both runtimes are MLX, so parity is ~bit-exact).

Usage:
    python3 tools/verify_llava_parity.py /tmp/krill-llava
        KLM_LLAVA_PARITY_DIR=/tmp/krill-llava \\
        swift test --filter LlavaParityTests
"""
import json
import os
import sys

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_vlm.models.llava import Model, ModelConfig, TextConfig, VisionConfig

# Tiny dims. image_size / patch_size = 3 -> 9 patches; +1 CLS = 10 positions.
IMAGE_SIZE = 42
PATCH = 14
N_PATCH = (IMAGE_SIZE // PATCH) ** 2          # 9
IMAGE_TOKEN = 1                               # an id inside the small vocab
VISION_LAYERS = 3                             # feature_layer=-2 -> after layer 1


def build(outdir: str, pytorch_conv: bool = False) -> None:
    os.makedirs(outdir, exist_ok=True)

    text = TextConfig(
        model_type="llama", hidden_size=128, num_hidden_layers=2,
        intermediate_size=256, num_attention_heads=4, num_key_value_heads=4,
        rms_norm_eps=1e-5, vocab_size=320, rope_theta=10000.0,
        max_position_embeddings=2048, tie_word_embeddings=False)
    vision = VisionConfig(
        model_type="clip_vision_model", num_hidden_layers=VISION_LAYERS,
        hidden_size=64, intermediate_size=128, num_attention_heads=4,
        image_size=IMAGE_SIZE, patch_size=PATCH, num_channels=3,
        layer_norm_eps=1e-5)
    cfg = ModelConfig(
        text_config=text, vision_config=vision, model_type="llava",
        image_token_index=IMAGE_TOKEN, vision_feature_select_strategy="default",
        vision_feature_layer=-2, vocab_size=320)

    mx.random.seed(0)
    model = Model(cfg)
    mx.eval(model.parameters())

    weights = dict(tree_flatten(model.parameters()))
    # Optional: emit the CLIP patch-embed Conv2d weight in PyTorch
    # `[out, in, kH, kW]` layout (what a raw HF llava-1.5 checkpoint ships)
    # instead of MLX `[out, kH, kW, in]`, so the loader's transpose path is
    # exercised. The reference logits are computed before this re-layout, so
    # they are unchanged -- the native loader must transpose back to match.
    if pytorch_conv:
        pk = "vision_tower.vision_model.embeddings.patch_embedding.weight"
        weights[pk] = mx.transpose(weights[pk], (0, 3, 1, 2))
    mx.save_safetensors(os.path.join(outdir, "model.safetensors"), weights)

    # input_ids: [BOS, <image>*N_PATCH, some text]; pixel_values [1,3,H,W].
    tokens = [5] + [IMAGE_TOKEN] * N_PATCH + [6, 7, 8, 9]
    input_ids = mx.array([tokens])
    pixel_values = mx.random.normal([1, 3, IMAGE_SIZE, IMAGE_SIZE])
    mx.eval(pixel_values)

    out = model(input_ids, pixel_values, mask=None)
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    last = logits[0, -1, :]

    config = {
        "architectures": ["LlavaForConditionalGeneration"],
        "model_type": "llava",
        "image_token_index": IMAGE_TOKEN,
        "vision_feature_select_strategy": "default",
        "vision_feature_layer": -2,
        "vocab_size": 320,
        "text_config": {
            "model_type": "llama", "hidden_size": 128, "num_hidden_layers": 2,
            "intermediate_size": 256, "num_attention_heads": 4,
            "num_key_value_heads": 4, "rms_norm_eps": 1e-5, "vocab_size": 320,
            "rope_theta": 10000.0, "max_position_embeddings": 2048,
            "tie_word_embeddings": False,
        },
        "vision_config": {
            "model_type": "clip_vision_model", "num_hidden_layers": VISION_LAYERS,
            "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
            "image_size": IMAGE_SIZE, "patch_size": PATCH, "num_channels": 3,
            "layer_norm_eps": 1e-5,
        },
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # pixels live in a subdir so the model weight loader (top-level *.safetensors
    # glob) does not slurp them as a stray model key under strict verify.
    os.makedirs(os.path.join(outdir, "inputs"), exist_ok=True)
    mx.save_safetensors(os.path.join(outdir, "inputs", "pixel_values.safetensors"),
                        {"pixel_values": pixel_values.astype(mx.float32)})
    ref = {
        "tokens": tokens,
        "image_token": IMAGE_TOKEN,
        "vocab_size": 320,
        "last_token_logits": [float(v) for v in last.tolist()],
        "argmax": int(mx.argmax(last).item()),
    }
    with open(os.path.join(outdir, "reference_logits.json"), "w") as f:
        json.dump(ref, f)

    print(f"Wrote tiny LLaVA + reference logits to {outdir}")
    print(f"  tokens={tokens}")
    print(f"  argmax(last_token_logits) = {ref['argmax']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    pytorch = len(sys.argv) > 2 and sys.argv[2] == "pytorch"
    build(sys.argv[1], pytorch_conv=pytorch)
