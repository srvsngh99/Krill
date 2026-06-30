#!/usr/bin/env python3
"""Dump the full DeepEncoder vision-feature fixture: SAM -> CLIP -> concat ->
projector, i.e. the [1, 256, 1280] features that get spliced into the LM.

Composition (from modeling_unlimitedocr.py):
    sam_feat  = sam_model(image)                 # [B,1024,16,16]
    clip_feat = vision_model(image, sam_feat)    # [B,257,1024]
    vis = cat(clip_feat[:,1:], sam_feat.flatten(2).permute(0,2,1), dim=-1)  # [B,256,2048]
    features = projector(vis)                    # [B,256,1280]

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_vision_reference.py <out.safetensors>
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
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/uocr_vision_ref.safetensors"
    snap = snapshot_download(REPO)
    mod_path = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/baidu/**/deepencoder.py"),
        recursive=True)[0]
    spec = importlib.util.spec_from_file_location("deepencoder", mod_path)
    de = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(de)

    sam = de.build_sam_vit_b().float().eval()
    clip = de.build_clip_l().float().eval()
    try:
        from addict import Dict as ADict
    except Exception:
        ADict = dict
    projector = de.MlpProjector(ADict(projector_type="linear", input_dim=2048, n_embed=1280)).float().eval()

    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        sd.update(load_file(f))

    def load(mod, pref):
        sub = {k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)}
        m, u = mod.load_state_dict(sub, strict=False)
        print(f"{pref} loaded; missing={len(m)} unexpected={len(u)}", flush=True)

    load(sam, "model.sam_model.")
    load(clip, "model.vision_model.")
    load(projector, "model.projector.")

    torch.manual_seed(0)
    image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        sam_feat = sam(image)                                      # [1,1024,16,16]
        clip_feat = clip(image, sam_feat)                          # [1,257,1024]
        vis = torch.cat((clip_feat[:, 1:],
                         sam_feat.flatten(2).permute(0, 2, 1)), dim=-1)  # [1,256,2048]
        features = projector(vis)                                  # [1,256,1280]
    print(f"features shape {tuple(features.shape)}", flush=True)

    save_file({"image": image.contiguous(), "features": features.contiguous()}, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
