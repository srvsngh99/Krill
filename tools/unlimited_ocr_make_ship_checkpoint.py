#!/usr/bin/env python3
"""Build the Krill-loadable, mixed-precision **nvfp4** ship checkpoint of
Unlimited-OCR (DeepSeek-OCR) — language backbone + DeepEncoder vision tower.

Quant policy (the locked 2026-06-28 decision):
  * MoE experts (the residency-dominant bulk: 64 experts x 3 projections,
    stacked into 3-D `switch_mlp`)            -> **nvfp4** group_size 16, bits 4
  * everything else quantizable (attention q/k/v/o_proj, the dense-layer +
    shared-expert FFN, embed_tokens, lm_head, the SAM/CLIP vision Linears, the
    projector)                                -> **8-bit affine** group_size 64
  * Conv2d weights (SAM patch-embed / neck / downsample), norms, learned
    position embeddings, `image_newline` / `view_seperator`, the MoE router
    `mlp.gate` -> kept unquantized (Conv weights transposed to MLX layout).

This mirrors `requant_gemma4_nvfp4.py`'s mixed scheme (top-level nvfp4 + a
per-module 8-bit affine override emitted for EACH protected module as it is
quantized — so config.json is, by construction, an exact description of the
tensors in the blob and `QuantizationConfig.effective(for:)` resolves every
module to the precision it was actually written at).

The conversion also bridges the two layout gaps the native runtime needs:
  1. stack per-expert `mlp.experts.{e}.{proj}` -> 3-D `mlp.switch_mlp.{proj}`
  2. transpose 4-D Conv2d weights PyTorch [O,I,kH,kW] -> MLX [O,kH,kW,I]
     (the `*pos_embed` params are NOT conv weights and stay as-is), and drop the
     bypassed `vision_model.embeddings.patch_embedding` (SAM supplies patch
     embeds).

Usage:
  ~/.krill/venv/bin/python tools/unlimited_ocr_make_ship_checkpoint.py <out_dir>
      [--text-only]   # drop the vision tower (text-parity validation checkpoint)
"""
import argparse
import glob
import json
import os

import mlx.core as mx
from huggingface_hub import snapshot_download

REPO = "baidu/Unlimited-OCR"

# expert (nvfp4) and protected (8-bit affine) quant params
EXPERT_GS, EXPERT_BITS, EXPERT_MODE = 16, 4, "nvfp4"
PROT_GS, PROT_BITS, PROT_MODE = 64, 8, "affine"

VISION_PREFIXES = ("model.sam_model.", "model.vision_model.", "model.projector.")
VISION_EXACT = {"model.image_newline", "model.view_seperator"}
# the CLIP patch_embedding Conv2d is bypassed (SAM provides patch_embeds)
DROP_VISION_SUBSTR = ("vision_model.embeddings.patch_embedding",)


def is_expert(key: str) -> bool:
    return ".mlp.switch_mlp." in key and key.endswith(".weight")


# Embedding tables read RAW (`.weight` accessed directly, not via an Embedding
# lookup) must stay unquantized or the native module's reshape breaks. The CLIP
# `position_embedding` is added to the patch grid as a raw table.
RAW_EMBED_SUBSTR = ("embeddings.position_embedding", "embeddings.class_embedding")


def should_protect_quant(key: str, arr, vision_bf16: bool) -> bool:
    """8-bit affine for every 2-D Linear weight that is not an expert, not the
    MoE router gate (raw param, like mlx-lm's MoEGate), and not a raw-accessed
    embedding table (CLIP position_embedding). With `vision_bf16` the entire
    DeepEncoder (SAM/CLIP/projector) is LEFT at bf16 — OCR character recognition
    is pixel-detail sensitive and the vision tower is small, so keeping it
    full-precision avoids the garbled-text degradation 8-bit vision causes."""
    if not key.endswith(".weight"):
        return False
    if key.endswith("mlp.gate.weight"):   # MoE router: never quantized
        return False
    if any(s in key for s in RAW_EMBED_SUBSTR):
        return False
    if vision_bf16 and key.startswith(VISION_PREFIXES):
        return False
    if is_expert(key):
        return False
    return arr.ndim == 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--text-only", action="store_true",
                    help="drop the vision tower (text-parity validation checkpoint)")
    ap.add_argument("--experts-8bit", action="store_true",
                    help="quantize experts at 8-bit affine instead of nvfp4 (isolation/quality test)")
    ap.add_argument("--vision-bf16", action="store_true",
                    help="leave the DeepEncoder vision tower at bf16 (recommended for OCR fidelity)")
    args = ap.parse_args()
    global EXPERT_GS, EXPERT_BITS, EXPERT_MODE
    if args.experts_8bit:
        EXPERT_GS, EXPERT_BITS, EXPERT_MODE = PROT_GS, PROT_BITS, PROT_MODE
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

    # 1. vision handling: drop (text-only) or sanitize (transpose conv, drop patch_embed)
    if args.text_only:
        w = {k: v for k, v in w.items()
             if not (k.startswith(VISION_PREFIXES) or k in VISION_EXACT)}
        print(f"text-only: dropped vision -> {len(w)} tensors", flush=True)
    else:
        san = {}
        for k, v in w.items():
            if any(s in k for s in DROP_VISION_SUBSTR):
                continue
            # transpose 4-D Conv2d weights [O,I,kH,kW] -> MLX [O,kH,kW,I];
            # *pos_embed are learned tables, not conv kernels (leave as-is).
            if v.ndim == 4 and not k.endswith("pos_embed"):
                v = v.transpose(0, 2, 3, 1)
            san[k] = v
        w = san
        print(f"sanitized vision (conv transpose, drop patch_embed) -> {len(w)} tensors",
              flush=True)

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

    # 3. quantize: experts -> nvfp4, everything else 2-D -> 8-bit affine.
    out = {}
    overrides = {}
    n_nvfp4 = n_prot = 0
    for k, v in w.items():
        if is_expert(k):  # experts -> nvfp4 (no biases) or 8-bit affine (--experts-8bit)
            q = mx.quantize(v, group_size=EXPERT_GS, bits=EXPERT_BITS, mode=EXPERT_MODE)
            base = k[:-len(".weight")]
            out[f"{base}.weight"] = q[0]
            out[f"{base}.scales"] = q[1]
            if len(q) > 2:                       # affine ships biases
                out[f"{base}.biases"] = q[2]
            n_nvfp4 += 1
        elif should_protect_quant(k, v, args.vision_bf16):  # 8-bit affine + per-module override
            wq, scales, biases = mx.quantize(v, group_size=PROT_GS, bits=PROT_BITS, mode=PROT_MODE)
            base = k[:-len(".weight")]
            out[f"{base}.weight"] = wq
            out[f"{base}.scales"] = scales
            out[f"{base}.biases"] = biases
            overrides[base] = {"group_size": PROT_GS, "bits": PROT_BITS, "mode": PROT_MODE}
            n_prot += 1
        else:
            out[k] = v
    print(f"quantized: experts nvfp4={n_nvfp4} | protected 8b-affine={n_prot}", flush=True)

    mx.save_safetensors(os.path.join(args.out, "model.safetensors"), out,
                        metadata={"format": "mlx"})

    # 4. config: keep the unlimited-ocr top-level so the loader routes to the
    #    multimodal path; nest the deepseek language_config; attach the mixed
    #    quantization block (top-level nvfp4 + per-module 8-bit overrides).
    qblock = {"group_size": EXPERT_GS, "bits": EXPERT_BITS, "mode": EXPERT_MODE}
    qblock.update(overrides)
    lang["quantization"] = qblock
    if args.text_only:
        # deepseek_v2-typed standalone config so loadModel -> loadDeepSeek.
        lang["model_type"] = "deepseek_v2"
        lang["architectures"] = ["DeepseekV2ForCausalLM"]
        lang["tie_word_embeddings"] = False
        cfg = lang
    else:
        # unlimited-ocr top-level: loader parses language_config + builds the
        # DeepEncoder. The same mixed quant block lives at top level (covers the
        # vision module overrides too) AND in language_config (experts).
        cfg = dict(top)
        cfg["language_config"] = lang
        cfg["quantization"] = qblock
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Copy the tokenizer / processor sidecar files (needed to serve + to ship a
    # self-contained HF repo). Skipped for the text-only validation checkpoint.
    if not args.text_only:
        import shutil
        for fn in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "processor_config.json", "conversation.py", "deepencoder.py",
                   "modeling_unlimitedocr.py", "configuration_deepseek_v2.py", "LICENSE"):
            src = os.path.join(snap, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(args.out, fn))
    print(f"wrote {args.out} (model.safetensors + config.json + sidecars)", flush=True)


if __name__ == "__main__":
    main()
