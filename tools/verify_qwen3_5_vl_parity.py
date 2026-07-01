#!/usr/bin/env python3
"""Qwen3.5-VL vision-tower parity oracle.

Builds mlx_vlm's real Qwen3.5-VL vision tower (`mlx_vlm.models.qwen3_5.vision`,
which subclasses the shared Qwen3-VL tower) with a TINY synthetic config,
generates random weights + a random patch batch, runs the forward, and dumps
weights / input / grid / output to a fixture directory. The Swift
`Qwen35VLVisionParityTests` loads the fixture, runs `Qwen35VLVisionModel`, and
asserts argmax/cosine parity — proving the native port matches the oracle.

Run under the mlx_vlm venv:
    ~/.krill/venv-ornith/bin/python tools/verify_qwen3_5_vl_parity.py <outdir>
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mlx_vlm.models.qwen3_5.vision import VisionModel
from mlx_vlm.models.qwen3_5.config import VisionConfig

TINY = dict(
    model_type="qwen3_5",
    depth=2,
    hidden_size=32,
    intermediate_size=64,
    num_heads=4,
    in_channels=3,
    patch_size=4,
    temporal_patch_size=2,
    spatial_merge_size=2,
    num_position_embeddings=16,  # 4x4 learned grid
    out_hidden_size=48,
    deepstack_visual_indexes=[],
)


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/qwen3_5_vl_parity")
    outdir.mkdir(parents=True, exist_ok=True)

    mx.random.seed(0)
    cfg = VisionConfig.from_dict(TINY) if hasattr(VisionConfig, "from_dict") else VisionConfig(**TINY)
    model = VisionModel(cfg)

    # Random weights (deterministic), materialized into the module.
    params = dict(tree_flatten(model.parameters()))
    new = {k: mx.random.normal(v.shape).astype(v.dtype) * 0.05 for k, v in params.items()}
    from mlx.utils import tree_unflatten
    model.update(tree_unflatten(list(new.items())))
    mx.eval(model.parameters())

    # One image, grid 1x4x6 (h,w multiples of merge_size=2).
    t, h, w = 1, 4, 6
    grid_thw = mx.array([[t, h, w]], dtype=mx.int32)
    n_patches = t * h * w
    patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
    pixel_values = mx.random.normal((n_patches, patch_dim)).astype(mx.float32)
    mx.eval(pixel_values)

    out, deepstack = model(pixel_values, grid_thw)
    mx.eval(out)
    assert deepstack == [], f"expected no deepstack features, got {len(deepstack)}"

    # Dump weights + io as safetensors (so the Swift test loads uniformly via
    # MLX.loadArrays), plus config/grid as json.
    flat = {k: v for k, v in new.items()}
    mx.save_safetensors(str(outdir / "weights.safetensors"), flat)
    mx.save_safetensors(str(outdir / "io.safetensors"),
                        {"pixel_values": pixel_values.astype(mx.float32),
                         "output": out.astype(mx.float32)})
    meta = dict(config=TINY, grid_thw=[[t, h, w]],
                n_patches=n_patches, patch_dim=patch_dim,
                out_shape=list(out.shape))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote fixture -> {outdir}")
    print(f"  weights: {len(flat)} tensors")
    print(f"  pixel_values: {tuple(pixel_values.shape)}  output: {tuple(out.shape)}")
    print(f"  output[0,:6] = {np.array(out[0, :6])}")


if __name__ == "__main__":
    main()
