#!/usr/bin/env python3
"""Dump a SAM-ViT-B (DeepEncoder `sam_model`) feature-parity fixture.

Feeds a FIXED random image [1, 3, 1024, 1024] and dumps (input, output). The
SAM tower returns patch_embeds [1, 1024, 16, 16] (BCHW) — exactly what the CLIP
tower consumes. Used to validate the native Swift SAM port.

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_sam_reference.py <out.safetensors>
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
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/uocr_sam_ref.safetensors"
    snap = snapshot_download(REPO)

    mod_path = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/baidu/**/deepencoder.py"),
        recursive=True)[0]
    spec = importlib.util.spec_from_file_location("deepencoder", mod_path)
    de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)

    sam = de.build_sam_vit_b().float().eval()

    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        sd.update(load_file(f))
    pref = "model.sam_model."
    ssd = {k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)}
    missing, unexpected = sam.load_state_dict(ssd, strict=False)
    print(f"SAM loaded; missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"  missing e.g. {missing[:6]}", flush=True)

    torch.manual_seed(0)
    image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        output = sam(image)  # [1, 1024, 16, 16]
    print(f"output shape {tuple(output.shape)}", flush=True)

    save_file({"image": image.contiguous(), "output": output.contiguous()}, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
