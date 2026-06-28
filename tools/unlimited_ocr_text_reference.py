#!/usr/bin/env python3
"""Dump TEXT-ONLY reference logits for baidu/Unlimited-OCR's DeepSeek-MoE
language backbone, to gate the native Krill `loadUnlimitedOCRText` path.

No image input: we feed a fixed token id sequence straight into the causal LM
and record the per-position logits. Krill is fed the IDENTICAL ids and must
match. Run on CPU/float32 for a deterministic reference.

Usage:
    ~/.krill/venv/bin/python tools/unlimited_ocr_text_reference.py \\
        <out.json>            # default out: scratchpad/unlimited_ocr_ref.json
"""
import json
import sys

import torch

# The repo's trust_remote_code modeling was written for transformers 4.x; 5.x
# removed several helpers it imports. Shim the removed names BEFORE the dynamic
# module is exec'd, so we can run the reference without downgrading the shared
# venv's transformers (which the mlx tooling depends on).
import transformers.utils.import_utils as _iu
for _name in ("is_torch_fx_available", "is_torch_fx_proxy"):
    if not hasattr(_iu, _name):
        setattr(_iu, _name, (lambda *a, **k: False))

import glob
import os

from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

REPO = "baidu/Unlimited-OCR"
TOKENS = [0, 17, 285, 1001, 4096, 88, 9, 12345]  # fixed, in-vocab (vocab 129280)


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/unlimited_ocr_ref.json"
    print(f"loading {REPO} language backbone (trust_remote_code, cpu/float32)...", flush=True)

    # Build ONLY the DeepSeek language model from `language_config` — the same
    # text backbone Krill's loadUnlimitedOCRText builds — bypassing the vision
    # wrapper so this is an apples-to-apples text oracle.
    cfg = AutoConfig.from_pretrained(REPO, trust_remote_code=True)
    DeepseekV2Config = get_class_from_dynamic_module(
        "configuration_deepseek_v2.DeepseekV2Config", REPO)
    DeepseekV2ForCausalLM = get_class_from_dynamic_module(
        "modeling_deepseekv2.DeepseekV2ForCausalLM", REPO)
    lc = cfg.language_config
    lang_dict = lc if isinstance(lc, dict) else lc.to_dict()
    lang_cfg = DeepseekV2Config.from_dict(lang_dict)
    model = DeepseekV2ForCausalLM(lang_cfg).float().eval()

    # Load just the language weights from the shared checkpoint (vision keys
    # land in `unexpected` and are ignored).
    snap = snapshot_download(REPO)
    sd = {}
    for f in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        sd.update(load_file(f))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    lang_missing = [k for k in missing if not k.startswith(("model.sam_model",
                    "model.vision_model", "model.projector"))]
    if lang_missing:
        print(f"WARNING: {len(lang_missing)} language params unfilled, e.g. {lang_missing[:5]}",
              flush=True)
    print(f"loaded; missing={len(missing)} unexpected={len(unexpected)} "
          f"(vision keys expected in unexpected)", flush=True)

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out_obj = model(input_ids=ids, use_cache=False)
        logits = out_obj.logits if hasattr(out_obj, "logits") else out_obj[0]
    logits = logits[0].float().cpu()  # [L, vocab]

    last = logits[-1]
    rec = {
        "repo": REPO,
        "tokens": TOKENS,
        "vocab": int(logits.shape[-1]),
        "seq_len": int(logits.shape[0]),
        "last_argmax": int(last.argmax().item()),
        "last_top10_idx": [int(i) for i in last.topk(10).indices.tolist()],
        "last_top10_val": [float(v) for v in last.topk(10).values.tolist()],
        # full last-row logits for a strict numeric compare on the Krill side
        "last_logits": [float(v) for v in last.tolist()],
    }
    with open(out, "w") as f:
        json.dump(rec, f)
    print(f"seq_len={rec['seq_len']} vocab={rec['vocab']} "
          f"argmax={rec['last_argmax']} top10={rec['last_top10_idx']}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
