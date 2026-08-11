#!/usr/bin/env python3
"""Build a tiny random Muse Glimmer (`MuseGlimmerForConditionalGeneration`,
model_type "muse_glimmer") checkpoint, run it through the transformers
reference, and record the resulting logits so the native Krill
`MuseGlimmerForConditionalGeneration` runtime can be gated for logit parity on
the exact same weights.

OFFLINE TEST ORACLE ONLY -- never runs at inference time and is not part of the
Krill engine. The runtime is pure Swift + MLX
(Sources/KrillCore/MuseGlimmerModel.swift + MuseGlimmerVision.swift); this
fixture exists only to gate it, like tools/verify_nanbeige_parity.py.

GATE STATUS: GREEN as of 2026-08-11, against transformers 5.16.0.dev0 (the
first version carrying `muse_glimmer`). Text prefill, cached decode, the vision
tower, and the full image-feature chain all match at argmax + cosine > 0.9999 +
max-abs < 2e-3. This is a SYNTHETIC gate: the real 30B checkpoint has never been
loaded (the smallest published MLX build is 19.4 GB and does not fit a 24 GB
box), so there is no real-weight or throughput number behind it.

It caught two real bugs on first run, both invisible without it, both in the
learned position-embedding resample:

  1. The port used `align_corners=True` (the convention the SAME-NAMED helper in
     `transformers/vision_utils.py` documents). The model file OVERRIDES that
     helper with its own, which is half-pixel (`align_corners=False`). ~6% error
     on the patch embeddings.
  2. The port kept the weight of a clamped out-of-range edge tap. The model's
     resample is `F.grid_sample(..., padding="zeros")`, so an out-of-range tap
     contributes NOTHING and the corner weights do not sum to 1 at the border.
     ~2.5% error, and only visible once the patch grid is large relative to the
     position table — which is why the fixture below uses a deliberately tiny
     `pos_emb_height`.

Why tiny + synthetic: a real 30B does not need to load to prove the port is
correct. A tiny random model exercises exactly what is easy to get wrong here:

  - **The attention output gate.** `attn_out * sigmoid(gate_proj(x))` applied
    BEFORE `o_proj`, where `x` is the layer's NORMED input, not the attention
    output. Dropping it, or feeding it the wrong tensor, changes every logit.
  - **NoPE on the full-attention layers.** `layer_rope_theta[i] == 0` means that
    layer gets no positional encoding at all. The fixture puts a NoPE layer in
    the middle of the stack and uses a multi-token prompt, so a runtime that
    rotates it diverges.
  - **The sliding window.** `sliding_window` is set SMALLER than the prompt so
    the window actually bites; a runtime that gives the sliding layers the full
    context is out of distribution and diverges.
  - **Two RMSNorm formulas.** The four per-layer norms are `(1 + w)`-scaled and
    the final `norm` is `w`-scaled. The fixture initialises the layer-norm
    weights AWAY from zero so mixing the two up cannot cancel out.
  - **Split eps.** `post_norm_eps` (1e-8) differs from `rms_norm_eps` (1e-5).
  - **`qk_scale_factor` on top of the softmax scale.** `q` is normed, then
    multiplied by 3.87, and `scaling` is STILL `head_dim**-0.5`.
  - **Logit softcapping + output multiplier**, in that order.
  - **The vision tower**, whose window permutation, interleaved 2-axis RoPE and
    channel-major `pixel_shuffle` all have to line up before the projector.

Usage:
    python3 tools/verify_muse_glimmer_parity.py /tmp/krill-muse-glimmer-parity

Writes into that dir: config.json, model.safetensors, reference_logits.json.
Then run the gated Swift test against it:
    KRILL_MUSE_GLIMMER_PARITY_DIR=/tmp/krill-muse-glimmer-parity \\
        swift test --filter MuseGlimmerParityTests
"""
import argparse
import json
import os
import sys

import torch

# Fixed token sequence the Swift side replays. Longer than `sliding_window`
# below, so the windowed layers genuinely drop context.
TOKENS = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8]
SLIDING_WINDOW = 6
SEED = 20260811


GEOMETRIES = {
    # Fast default: small in every dimension. Catches semantic bugs (the gate,
    # NoPE, the norm formulas, the resample) in milliseconds.
    "tiny": dict(
        hidden_size=256, intermediate_size=512, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=512,
        layer_types=["sliding_attention", "sliding_attention",
                     "sliding_attention", "full_attention"],
        layer_rope_theta=[500000.0, 500000.0, 500000.0, 0],
    ),
    # The REAL per-tensor geometry, only shallow: every attention matrix has the
    # shape the 30B uses (hidden 6656, 32 q-heads / 2 kv-heads => GQA ratio 16,
    # head_dim 128, q_proj out 4096 != hidden 6656). Only depth, MLP width and
    # vocab are shrunk, so it stays ~800MB and runs in seconds.
    #
    # This exists because the `tiny` geometry uses GQA ratio 2 and head_dim 32,
    # and the real model produced garbage while `tiny` stayed green — any bug
    # that depends on the head geometry is invisible at `tiny`.
    # Depth in isolation: the real model's 52 layers with the 3:1 sliding/NoPE
    # pattern, but narrow. `tiny` and `real` are both shallow (2-4 layers) and
    # both stay green while the real 52-layer model emits garbage, so depth is
    # the last untested axis.
    "deep": dict(
        hidden_size=256, intermediate_size=512, num_hidden_layers=52,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, vocab_size=512,
        layer_types=["sliding_attention", "sliding_attention",
                     "sliding_attention", "full_attention"] * 13,
        layer_rope_theta=[500000.0, 500000.0, 500000.0, 0] * 13,
    ),
    "real": dict(
        hidden_size=6656, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=32, num_key_value_heads=2, head_dim=128, vocab_size=512,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[500000.0, 0],
    ),
}


def build_config(geometry="tiny"):
    """Text stack (sliding layers + a NoPE full layer) plus a 2-layer tower."""
    from transformers.models.muse_glimmer import MuseGlimmerConfig

    geo = GEOMETRIES[geometry]
    return MuseGlimmerConfig(
        image_token_id=61,
        video_token_id=60,
        out_hidden_size=64,          # vision hidden (16) * merge (2) ** 2
        projector_hidden_size=64,   # multiple of 64 so the projector quantizes like the real build
        projector_hidden_act="gelu",
        text_config=dict(
            model_type="muse_glimmer_text",
            # Sizes chosen so `num_attention_heads * head_dim` (128) differs
            # from `hidden_size` (256) — the real model is 32*128=4096 vs 6656.
            # An earlier revision of this fixture had them EQUAL (4*8 == 32),
            # which meant a hidden/qOut mix-up in the attention gate or o_proj
            # could not be detected at all. Every dim is also a multiple of 64
            # so the same fixture can be MLX-quantized at group_size 64.
            **geo,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            sliding_window=SLIDING_WINDOW,
            rope_parameters={"rope_theta": 500000.0, "rope_type": "default"},
            qk_scale_factor=3.87,
            output_multiplier=0.19611613513818404,
            final_logit_softcapping=20.0,
            attention_bias=False,
            tie_word_embeddings=False,
            max_position_embeddings=131072,
            hidden_activation="silu",
        ),
        vision_config=dict(
            model_type="muse_glimmer_vision",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            patch_size=14,
            patch_temporal=2,
            merge_size=2,
            # Deliberately TINY so `window_size = pos_emb_height * patch_size`
            # makes windows of 2 patches: the 4x6 grid below then spans
            # multiple windows and genuinely exercises the window
            # permutation, the empty-window dedup, and the un-permute.
            pos_emb_height=2,
            pos_emb_width=2,
            layer_norm_eps=1e-5,
            hidden_act="gelu",
            rope_parameters={"rope_theta": 10000.0},
            layer_types=["window_attention", "full_attention"],
        ),
    )


def randomize(model):
    """Give every parameter a distinctive non-degenerate value.

    The layer norms matter most: `MuseGlimmerTextCenteredRMSNorm` initialises
    `weight` to ZEROS, and `(1 + 0) == 1`, so a runtime that wrongly used the
    plain `norm(x) * w` formula would still agree with the reference on a
    freshly-initialised model. Pushing the weights away from zero makes the two
    formulas disagree, which is the whole point of the gate.
    """
    g = torch.Generator().manual_seed(SEED)
    for name, p in model.named_parameters():
        if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
            p.data = torch.rand(p.shape, generator=g) * 0.4 - 0.2
        elif p.dim() == 1:
            p.data = torch.rand(p.shape, generator=g) * 0.2 - 0.1
        else:
            p.data = torch.randn(p.shape, generator=g) * 0.02
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", help="directory to write the fixture into")
    ap.add_argument("--geometry", choices=sorted(GEOMETRIES), default="tiny",
                    help="tensor shapes: 'tiny' (fast) or 'real' "
                         "(the 30B's actual head geometry, shallow)")
    args = ap.parse_args()

    try:
        from transformers.models.muse_glimmer import MuseGlimmerForConditionalGeneration
    except Exception as exc:  # pragma: no cover - environment guard
        sys.exit(
            "transformers has no `muse_glimmer` module ({}).\n"
            "Muse Glimmer landed in transformers 5.15.0.dev0; install a build "
            "that has it (e.g. `pip install -U "
            "'git+https://github.com/huggingface/transformers'`) and re-run."
            .format(exc))

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(SEED)

    config = build_config(args.geometry)
    model = randomize(MuseGlimmerForConditionalGeneration(config)).eval()

    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
        logits = out.logits.float()

        # Incremental decode: prefill all but the last token, then step it.
        # This is what catches a mis-sized or mis-kinded KV cache — the sliding
        # layers need a windowed cache, the NoPE layer a full one.
        prefix = torch.tensor([TOKENS[:-1]], dtype=torch.long)
        pre = model(input_ids=prefix, use_cache=True)
        step = model(
            input_ids=torch.tensor([[TOKENS[-1]]], dtype=torch.long),
            past_key_values=pre.past_key_values,
            use_cache=True,
        )
        step_logits = step.logits.float()

    # Vision: a 4x4 patch grid (16 patches -> 4 merged tokens). This gates the
    # window permutation, the interleaved [w,h,w,h] 2-axis RoPE, the
    # channel-major pixel_shuffle, and the adapter -> projection -> weightless
    # norm chain, independently of the text stack.
    vision = config.vision_config
    # Non-square on purpose: a 4x4 grid cannot catch an h/w swap.
    grid_t, grid_h, grid_w = 1, 4, 6
    patch_dim = vision.patch_temporal * 3 * vision.patch_size ** 2
    gv = torch.Generator().manual_seed(SEED + 1)
    pixel_values = torch.randn(
        grid_t * grid_h * grid_w, patch_dim, generator=gv) * 0.5
    grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
    with torch.no_grad():
        tower_out = model.model.vision_tower(
            pixel_values=pixel_values, grid_thw=grid_thw).last_hidden_state.float()
        image_features = model.model.get_image_features(
            pixel_values=pixel_values, image_grid_thw=grid_thw).pooler_output[0].float()

    last = logits[0, -1].tolist()
    reference = {
        "vision_grid": [grid_t, grid_h, grid_w],
        "vision_patch_dim": patch_dim,
        "vision_pixel_values": pixel_values.flatten().tolist(),
        "vision_tower_out": tower_out.flatten().tolist(),
        "vision_tower_shape": list(tower_out.shape),
        "vision_image_features": image_features.flatten().tolist(),
        "vision_image_features_shape": list(image_features.shape),
        "tokens": TOKENS,
        "vocab_size": config.text_config.vocab_size,
        "num_hidden_layers": config.text_config.num_hidden_layers,
        "sliding_window": SLIDING_WINDOW,
        "layer_types": list(config.text_config.layer_types),
        "layer_rope_theta": list(config.text_config.layer_rope_theta),
        "last_token_logits": last,
        "argmax": int(max(range(len(last)), key=last.__getitem__)),
        "step_logits": step_logits[0, -1].tolist(),
        "softcap": config.text_config.final_logit_softcapping,
        "max_abs_tolerance": 2e-3,
    }

    model.config.save_pretrained(args.outdir)
    model.save_pretrained(args.outdir, safe_serialization=True)
    with open(os.path.join(args.outdir, "reference_logits.json"), "w") as fh:
        json.dump(reference, fh, indent=2)

    print("wrote fixture to", args.outdir)
    print("  tokens        :", TOKENS)
    print("  argmax        :", reference["argmax"])
    print("  |logit| max   :", max(abs(v) for v in last),
          "(softcap {})".format(reference["softcap"]))
    print()
    print("layer-0 weight keys (this dump IS the @ModuleInfo(key:) contract):")
    for name, _ in model.named_parameters():
        if ".layers.0." in name or "." not in name.split(".")[-2:][0]:
            print("   ", name)


if __name__ == "__main__":
    main()
