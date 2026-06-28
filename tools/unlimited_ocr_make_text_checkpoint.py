#!/usr/bin/env python3
"""Build a Krill-loadable, quantized TEXT-ONLY checkpoint of Unlimited-OCR's
DeepSeek-MoE language backbone.

The native DeepSeek MoE runtime is quantized-only (born-quantized switched
experts) and wants STACKED 3-D `switch_mlp` tensors, while the source ships
bf16 per-expert tensors. This script bridges both in one pass:

  1. keep only the language weights (drop sam_model / vision_model / projector / glue)
  2. stack per-expert `mlp.experts.{e}.{proj}` -> `mlp.switch_mlp.{proj}` (3-D)
  3. affine-quantize every Linear/Embedding weight (and the 3-D experts) to
     `bits` @ group 64, emitting `<m>.weight`(packed) + `.scales` + `.biases`,
     exactly matching what Krill's load-time quantize pass expects.
  4. write a `deepseek_v2`-typed config.json (use_mla:false routes to the new
     standard-attention path) with a matching `quantization` block.

loadModel(out) then routes to loadDeepSeek and the parity gate can run.

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_make_text_checkpoint.py \\
      <out_dir> [--bits 8]
"""
import argparse
import glob
import json
import os

import mlx.core as mx
from huggingface_hub import snapshot_download

REPO = "baidu/Unlimited-OCR"
GROUP = 64
DROP_PREFIXES = ("model.sam_model.", "model.vision_model.", "model.projector.")
DROP_EXACT = {"model.image_newline", "model.view_seperator"}
# Linear/Embedding weights to quantize (router `mlp.gate.weight` and 1-D norms
# are NOT quantized by Krill's quantize pass, so they stay original dtype).
QUANT_SUFFIXES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                  "gate_proj.weight", "up_proj.weight", "down_proj.weight")
QUANT_EXACT = ("model.embed_tokens.weight", "lm_head.weight")


def should_quant(key: str, arr) -> bool:
    if key.endswith("mlp.gate.weight"):   # MoE router: raw param, never quantized
        return False
    if arr.ndim < 2:
        return False
    return key.endswith(QUANT_SUFFIXES) or key in QUANT_EXACT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--bits", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    snap = snapshot_download(REPO)
    with open(os.path.join(snap, "config.json")) as f:
        top = json.load(f)
    lang = dict(top["language_config"])
    n_exp = lang["n_routed_experts"]
    n_layers = lang["num_hidden_layers"]

    w = {}
    for sf in sorted(glob.glob(os.path.join(snap, "*.safetensors"))):
        w.update(mx.load(sf))
    print(f"loaded {len(w)} source tensors", flush=True)

    # 1. drop vision/glue
    w = {k: v for k, v in w.items()
         if not (k.startswith(DROP_PREFIXES) or k in DROP_EXACT)}

    # 2. stack per-expert -> switch_mlp (3-D), drop the per-expert keys
    for li in range(n_layers):
        pref = f"model.layers.{li}.mlp"
        e0 = f"{pref}.experts.0.gate_proj.weight"
        if e0 not in w:
            continue  # dense (first_k_dense_replace) layer — no experts
        for proj in ("gate_proj", "up_proj", "down_proj"):
            stack = mx.stack([w.pop(f"{pref}.experts.{e}.{proj}.weight")
                              for e in range(n_exp)], axis=0)
            w[f"{pref}.switch_mlp.{proj}.weight"] = stack
    print(f"after stack/drop: {len(w)} tensors", flush=True)

    # 3. quantize
    out = {}
    nq = 0
    for k, v in w.items():
        if should_quant(k, v):
            wq, scales, biases = mx.quantize(v, group_size=GROUP, bits=args.bits)
            base = k[:-len(".weight")]
            out[f"{base}.weight"] = wq
            out[f"{base}.scales"] = scales
            out[f"{base}.biases"] = biases
            nq += 1
        else:
            out[k] = v
    print(f"quantized {nq} tensors @ {args.bits}b/group{GROUP}", flush=True)

    mx.save_safetensors(os.path.join(args.out, "model.safetensors"), out)

    # 4. config: deepseek_v2-typed, matching quantization block
    lang["model_type"] = "deepseek_v2"
    lang["architectures"] = ["DeepseekV2ForCausalLM"]
    lang["tie_word_embeddings"] = False
    lang["quantization"] = {"group_size": GROUP, "bits": args.bits}
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(lang, f, indent=2)
    print(f"wrote {args.out} (model.safetensors + config.json)", flush=True)


if __name__ == "__main__":
    main()
