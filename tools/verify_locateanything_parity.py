#!/usr/bin/env python3
"""LocateAnything-3B vision-path parity oracle (MoonViT tower + mlp1 connector).

LocateAnything-3B ships a custom PyTorch vision tower (`modeling_vit.py`,
`MoonVitPretrainedModel` — the Kimi-VL MoonViT: Conv2d patch embed + learnable
2D-interpolated position embedding + 2D *complex* rotary bidirectional-attention
blocks + a 2x2 patch merger) followed by a top-level `mlp1` connector
(LayerNorm(vit*4) -> Linear(vit*4, llm) -> GELU -> Linear(llm, llm)) that maps
merged vision tokens into the Qwen2.5 text hidden space.

This builds that reference stack with a TINY synthetic config + random weights,
runs the forward on a random patch batch, and dumps weights / input / grid /
outputs to a fixture dir. The Swift `LocateAnythingVisionParityTests` loads the
fixture, runs the native `MoonViTVisionModel` + connector, and asserts
argmax/cosine parity — proving the port matches the NVIDIA reference.

Run:
    ~/.krill/venv/bin/python tools/verify_locateanything_parity.py <outdir>

The reference modeling_vit.py is imported from the staged model dir (default
~/.cache/huggingface/staged/LocateAnything-3B); override with $LA3B_SRC.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file

# --- import the NVIDIA reference MoonViT ---------------------------------
SRC_CANDIDATES = [
    os.environ.get("LA3B_SRC"),
    os.path.expanduser("~/.cache/huggingface/staged/LocateAnything-3B"),
    # scratchpad copy of just the custom code
    "/private/tmp/claude-501/-Users-sourav-Desktop-playground-Krill/"
    "e8f25ff7-901d-4159-90ba-b3fc542cff5d/scratchpad/la3b",
]
SRC = next((p for p in SRC_CANDIDATES if p and Path(p, "modeling_vit.py").exists()), None)
if SRC is None:
    sys.exit("could not find modeling_vit.py; set $LA3B_SRC to the model dir")
sys.path.insert(0, SRC)
from modeling_vit import MoonViTConfig, MoonVitPretrainedModel  # noqa: E402

VIT_HIDDEN = 32
LLM_HIDDEN = 48
TINY = dict(
    model_type="moonvit",
    patch_size=4,
    init_pos_emb_height=4,      # 4x4 learned grid -> forces bicubic interp
    init_pos_emb_width=4,
    num_attention_heads=4,
    num_hidden_layers=2,
    hidden_size=VIT_HIDDEN,
    intermediate_size=64,
    merge_kernel_size=(2, 2),
)


def build_mlp1(vit_hidden: int, llm_hidden: int) -> nn.Sequential:
    # Mirrors LocateAnythingForConditionalGeneration.mlp1 exactly.
    return nn.Sequential(
        nn.LayerNorm(vit_hidden * 4),
        nn.Linear(vit_hidden * 4, llm_hidden),
        nn.GELU(),
        nn.Linear(llm_hidden, llm_hidden),
    )


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/locateanything_parity")
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    cfg = MoonViTConfig(**TINY)
    cfg._attn_implementation = "sdpa"        # no flash_attn on this box
    vit = MoonVitPretrainedModel(cfg).eval()
    mlp1 = build_mlp1(VIT_HIDDEN, LLM_HIDDEN).eval()

    # Deterministic small random weights (match the qwen3_5_vl oracle scale).
    with torch.no_grad():
        for p in list(vit.parameters()) + list(mlp1.parameters()):
            p.copy_(torch.randn_like(p) * 0.05)

    # One image, grid h=6 w=8 (multiples of merge_kernel=2), patch 4x4, C=3.
    h, w = 6, 8
    n_patches = h * w
    patch_dim = 3 * cfg.patch_size * cfg.patch_size      # C*ph*pw = 48
    grid_hws = torch.tensor([[h, w]], dtype=torch.int32)
    # pixel_values as the image processor emits them: (N, C, ph, pw).
    pixel_values = (torch.randn(n_patches, 3, cfg.patch_size, cfg.patch_size)
                    .to(torch.float32))

    with torch.no_grad():
        merged = vit(pixel_values=pixel_values, grid_hws=grid_hws)  # list[ (n/4, vit*4) ]
        vit_out = torch.cat(merged, dim=0)                          # (n/4, vit*4=128)
        conn_out = mlp1(vit_out)                                    # (n/4, llm=48)

    # Flatten pixel_values to (N, C*ph*pw) in C-major order for the Swift side
    # (its patch embed runs as a matmul with the flattened Conv2d weight).
    px_flat = pixel_values.reshape(n_patches, -1).contiguous()

    # --- dump fixture ----------------------------------------------------
    weights = {}
    for name, p in vit.named_parameters():
        weights[f"vision_model.{name}"] = p.detach().to(torch.float32).contiguous()
    # mlp1.{0,1,3} -> keep the checkpoint's own naming.
    for name, p in mlp1.named_parameters():
        weights[f"mlp1.{name}"] = p.detach().to(torch.float32).contiguous()

    save_file(weights, str(outdir / "weights.safetensors"))
    save_file(
        {
            "pixel_values": px_flat.to(torch.float32),
            "vit_output": vit_out.to(torch.float32),
            "connector_output": conn_out.to(torch.float32),
        },
        str(outdir / "io.safetensors"),
    )
    meta = dict(
        config=TINY,
        vit_hidden=VIT_HIDDEN,
        llm_hidden=LLM_HIDDEN,
        grid_hws=[[h, w]],
        n_patches=n_patches,
        patch_dim=patch_dim,
        vit_out_shape=list(vit_out.shape),
        conn_out_shape=list(conn_out.shape),
    )
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote fixture -> {outdir}")
    print(f"  weights: {len(weights)} tensors")
    print(f"  pixel_values(flat): {tuple(px_flat.shape)}")
    print(f"  vit_output: {tuple(vit_out.shape)}  connector_output: {tuple(conn_out.shape)}")
    print(f"  vit_output[0,:6]  = {np.array(vit_out[0, :6])}")
    print(f"  connector[0,:6]   = {np.array(conn_out[0, :6])}")
    # Also print the weight-key tree (the @ModuleInfo(key:) contract).
    print("  --- weight keys ---")
    for k in sorted(weights):
        print(f"    {k}: {tuple(weights[k].shape)}")


if __name__ == "__main__":
    main()
