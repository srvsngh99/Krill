#!/usr/bin/env python3
"""End-to-end (base-view) multimodal prefill reference for Unlimited-OCR.

The HF forward hardcodes .cuda(), so we replicate the exact model forward on CPU:
  embeds = embed_tokens(input_ids)
  vis    = projector(cat(clip[:,1:], sam.flatten)); assemble base [273,1280]
  embeds[images_seq_mask] = vis                      # masked_scatter
  logits = LM(inputs_embeds=embeds).logits[0,-1]
We hand-build a deterministic input_ids (bos + text + 273 <image> tokens), so
Krill can feed the IDENTICAL ids/image and compare the first-token logits.

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_e2e_reference.py <out.safetensors>
"""
import glob
import importlib.util
import json
import os
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from PIL import Image, ImageDraw

# transformers 5.x shim (4.x-era remote code).
import transformers.utils.import_utils as _iu
for _n in ("is_torch_fx_available", "is_torch_fx_proxy"):
    if not hasattr(_iu, _n):
        setattr(_iu, _n, (lambda *a, **k: False))

REPO = "baidu/Unlimited-OCR"
IMAGE_TOKEN_ID = 128815
N_IMG = 273  # 16*17 + 1 for a base 1024 view


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/uocr_e2e_ref.safetensors"
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
        mod.load_state_dict({k[len(pref):]: v.float() for k, v in sd.items() if k.startswith(pref)},
                            strict=False)
    load(sam, "model.sam_model.")
    load(clip, "model.vision_model.")
    load(projector, "model.projector.")
    image_newline = sd["model.image_newline"].float()
    view_seperator = sd["model.view_seperator"].float()

    # Language model (full-causal: inference disables sliding_window).
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    DSConfig = get_class_from_dynamic_module("configuration_deepseek_v2.DeepseekV2Config", REPO)
    DSForCausalLM = get_class_from_dynamic_module("modeling_deepseekv2.DeepseekV2ForCausalLM", REPO)
    lc = cfg.language_config
    lang_cfg = DSConfig.from_dict(lc if isinstance(lc, dict) else lc.to_dict())
    lang_cfg.sliding_window = None
    lm = DSForCausalLM(lang_cfg).float().eval()
    lm.load_state_dict(sd, strict=False)  # model.* + lm_head.* bind; vision keys ignored

    # Build a deterministic OCR-ish image (real text so the prediction is meaningful).
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, line in enumerate(["Invoice 2026", "Total: $42.00", "Thank you"]):
        d.text((20, 40 + i * 60), line, fill=(0, 0, 0))
    mean = (0.5, 0.5, 0.5)
    from PIL import ImageOps
    global_view = ImageOps.pad(img, (1024, 1024), color=tuple(int(x * 255) for x in mean))
    arr = torch.from_numpy(__import__("numpy").asarray(global_view)).float() / 255.0  # [1024,1024,3]
    arr = (arr - 0.5) / 0.5
    images_ori = arr.permute(2, 0, 1).unsqueeze(0).contiguous()  # [1,3,1024,1024]

    # Deterministic input_ids: bos + text + 273 image tokens + text.
    prefix = [100, 200, 300]   # arbitrary in-vocab text ids
    suffix = [400, 500]
    ids = [0] + prefix + [IMAGE_TOKEN_ID] * N_IMG + suffix
    seq_mask = [False] * (1 + len(prefix)) + [True] * N_IMG + [False] * len(suffix)
    input_ids = torch.tensor([ids], dtype=torch.long)
    images_seq_mask = torch.tensor([seq_mask], dtype=torch.bool)

    with torch.no_grad():
        embeds = lm.model.embed_tokens(input_ids)              # [1,L,1280]
        sam_feat = sam(images_ori)
        clip_feat = clip(images_ori, sam_feat)
        vis = torch.cat((clip_feat[:, 1:], sam_feat.flatten(2).permute(0, 2, 1)), dim=-1)
        feats = projector(vis)                                  # [1,256,1280]
        h = w = 16
        g = feats.view(h, w, 1280)
        g = torch.cat([g, image_newline[None, None, :].expand(h, 1, 1280)], dim=1).view(-1, 1280)
        assembled = torch.cat([g, view_seperator[None, :]], dim=0)  # [273,1280]
        embeds = embeds.clone()
        embeds[0].masked_scatter_(images_seq_mask[0].unsqueeze(-1), assembled)
        logits = lm(inputs_embeds=embeds, use_cache=False).logits[0, -1].float()

    print(f"L={input_ids.shape[1]} argmax={int(logits.argmax())} "
          f"top5={[int(i) for i in logits.topk(5).indices.tolist()]}", flush=True)
    save_file({
        "input_ids": input_ids.to(torch.int32).contiguous(),
        "images_ori": images_ori.contiguous(),
        "images_seq_mask": images_seq_mask.to(torch.int32).contiguous(),
        "last_logits": logits.contiguous(),
    }, out)
    with open(out + ".json", "w") as f:
        json.dump({"argmax": int(logits.argmax()), "L": int(input_ids.shape[1])}, f)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
