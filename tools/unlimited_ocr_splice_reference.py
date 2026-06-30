#!/usr/bin/env python3
"""Dump the base-view multimodal SPLICE fixture (G5 assembly).

For a single (non-tiled) 1024 image, modeling_unlimitedocr assembles the LM
vision tokens as (lines 551-573):
    g = projector(...)              # [1, 256, 1280]
    g = g.view(16, 16, 1280)
    g = cat([g, image_newline.expand(16,1,1280)], dim=1)   # [16, 17, 1280]
    g = g.view(-1, 1280)                                    # [272, 1280]
    glob_local = cat([g, view_seperator[None,:]], dim=0)    # [273, 1280]

Dumps the vision features (input to the Swift assembly), image_newline,
view_seperator, and the expected [273,1280] assembled sequence.

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_splice_reference.py <out.safetensors>
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
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/uocr_splice_ref.safetensors"
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
        mod.load_state_dict(sub, strict=False)

    load(sam, "model.sam_model.")
    load(clip, "model.vision_model.")
    load(projector, "model.projector.")
    image_newline = sd["model.image_newline"].float()      # [1280]
    view_seperator = sd["model.view_seperator"].float()    # [1280]

    torch.manual_seed(0)
    image = torch.randn(1, 3, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        sam_feat = sam(image)
        clip_feat = clip(image, sam_feat)
        vis = torch.cat((clip_feat[:, 1:], sam_feat.flatten(2).permute(0, 2, 1)), dim=-1)
        feats = projector(vis)                              # [1,256,1280]

        _, hw, n = feats.shape
        h = w = int(hw ** 0.5)
        g = feats.view(h, w, n)
        g = torch.cat([g, image_newline[None, None, :].expand(h, 1, n)], dim=1)  # [16,17,1280]
        g = g.view(-1, n)                                    # [272,1280]
        glob_local = torch.cat([g, view_seperator[None, :]], dim=0)  # [273,1280]

    print(f"features {tuple(feats.shape)} -> assembled {tuple(glob_local.shape)}", flush=True)
    save_file({
        "features": feats.contiguous(),
        "image_newline": image_newline.contiguous(),
        "view_seperator": view_seperator.contiguous(),
        "assembled": glob_local.contiguous(),
    }, out)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
