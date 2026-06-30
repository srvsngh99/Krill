#!/usr/bin/env python3
"""Dump a CLIP-L (DeepEncoder `vision_model`) feature-parity fixture.

The CLIP tower consumes `patch_embeds` from SAM (not raw pixels), so we feed a
FIXED random patch-embed grid [1, 1024, 16, 16] (the standard 1024-image path:
SAM 64x64 -> /2 -> /2 -> 16x16) and dump (input, output) for the Swift port to
match. At 16x16 == 224/14 the position-embedding `get_abs_pos` is a no-op, so
this is a clean ViT forward.

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_clip_reference.py <out.safetensors>
"""
import glob
import importlib.util
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

REPO = "baidu/Unlimited-OCR"


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/uocr_clip_ref.safetensors"
    snap = snapshot_download(REPO)

    mod_path = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/baidu/**/deepencoder.py"),
        recursive=True)[0]
    spec = importlib.util.spec_from_file_location("deepencoder", mod_path)
    de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)

    clip = de.build_clip_l().float().eval()

    # Load vision_model.* weights (strip the `model.vision_model.` prefix).
    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        sd.update(load_file(f))
    pref = "model.vision_model."
    vsd = {k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)}
    missing, unexpected = clip.load_state_dict(vsd, strict=False)
    print(f"CLIP loaded; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing e.g. {missing[:5]}", flush=True)

    torch.manual_seed(0)
    patch_embeds = torch.randn(1, 1024, 16, 16, dtype=torch.float32)
    pixel_values = torch.zeros(1, 3, 1024, 1024, dtype=torch.float32)  # batch-size only
    with torch.no_grad():
        output = clip(pixel_values, patch_embeds)  # [1, 257, 1024]
    print(f"output shape {tuple(output.shape)}", flush=True)

    save_file({"patch_embeds": patch_embeds.contiguous(),
               "output": output.contiguous()}, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
