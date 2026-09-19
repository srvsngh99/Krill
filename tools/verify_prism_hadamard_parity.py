#!/usr/bin/env python3
"""Reference dumper for Krill's `prism_hadamard_qwen35` port (Prism ML's
Ternary-Bonsai-2-27B and any other checkpoint in the same pack format),
built from mlx-lm's `qwen3_5.TextModel` + the pack's own `runtime/runtime.py`
(`Packed`, `fwht`).

WHY THIS SCRIPT EXISTS, NOT THE PACK'S OWN `runtime/artifact.py`:
`artifact.py` is the entry point `PACK-RUNTIME.md` documents, but it
hard-requires `schema_version == 1` while Ternary-Bonsai-2-27B ships
`schema_version: 2`, so `load_model()` raises "Unsupported packed model
schema" on it. Schema 2 also moved the tensor namespace to mlx-vlm's
`language_model.` prefix, which `artifact.py` does not thread through.
The pack's OTHER bundled loader, `runtime/vision_artifact.py`'s
`load_vl_model`, handles both and opens this pack fine (verified against
the mlx-vlm 0.6.3 pinned in `runtime/requirements.txt`); it is simply not
the loader the docs point at. We do not reuse it here either, because it
builds an mlx-vlm VL model while the parity target is Krill's text-only
port. So this script drives `mlx_lm.models.qwen3_5.TextModel` directly and
swaps in `Packed` at the 402 manifest paths with the `language_model.`
prefix threaded through, then loads everything else with `strict=False`.

KNOWN WRINKLE - READ BEFORE TRUSTING `layer3_out`/`layer7_out`: the per-layer
intermediate dump below calls each decoder layer directly as
`layer(h, mask=None, cache=None)`, bypassing `Qwen3_5TextModel.__call__`'s
own `create_attention_mask` / `create_ssm_mask`. For the GatedDeltaNet
(linear-attention) layers this is harmless - the delta-rule scan is causal
by construction, `mask` only zeros invalid positions. For the FULL-ATTENTION
layers (3, 7, ...) it is NOT harmless: `mask=None` means unmasked softmax
attention, every position attends to every other position including future
ones. Compare `layer3_out`/`layer7_out` against a candidate runtime's own
UNMASKED layer output (or just don't assert on them - see
`Tests/KrillCoreTests/PrismHadamardReferenceParityTests.swift`, which
prints them for visibility but only asserts on the GDN layers and the final
logits). This is exactly the trap that cost an hour of bisection on the
Krill side before the cause was identified: Krill's own attention layer
always builds a real causal mask, so the "divergence" at those two layers
was a fixture artifact, not a runtime bug. Assert on `logits` (the full
forward pass, which DOES use masking end to end) and `embed_out` /
`layer{0,1,2}_out` instead.

Usage:
    python3 tools/verify_prism_hadamard_parity.py <checkpoint_dir> <output.safetensors> \\
        [--prompt "The capital of France is"]

    # then, to run Krill's gated parity suite against the dump:
    KRILL_PRISM_BONSAI2_DIR=<checkpoint_dir> \\
    KRILL_PRISM_PARITY_FIXTURE=<output.safetensors> \\
        swift test --filter PrismHadamardReferenceParityTests

Requires the pack's own `runtime/` on `sys.path` (ships alongside
`config.json` in the checkpoint directory) and `mlx_lm` installed
(`~/.krill/venv` in this repo's convention).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx


def build_reference(checkpoint_dir: Path, prompt: str):
    """Load the checkpoint through mlx-lm's `qwen3_5.TextModel`, schema-2
    aware, and return `(model, tokenizer, ids, ref_signs)`. Caller must have
    already put `<checkpoint_dir>/runtime` on `sys.path` (see `main`)."""
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
    from runtime import Packed

    cfg = json.loads((checkpoint_dir / "config.json").read_text())
    weights = mx.load(str(checkpoint_dir / "model.safetensors"))

    # Exhaustive sign-vector check: every packed module's `.signs` tensor
    # must equal the corresponding width-keyed slice of hadamard.json's flat
    # `sign_values`. Genuinely useful as an independent cross-check of the
    # pack's own internal consistency (not just of this dumper) - keep it.
    hadamard = json.loads((checkpoint_dir / "hadamard.json").read_text())
    widths = hadamard["prism.hadamard.sign_widths"]
    values = hadamard["prism.hadamard.sign_values"]
    ref_signs, offset = {}, 0
    for width in widths:
        ref_signs[width] = mx.array(values[offset:offset + width], dtype=mx.float32)
        offset += width
    mismatches = [
        key for key in weights
        if key.endswith(".signs")
        and not bool(mx.all(weights[key].astype(mx.float32) == ref_signs[weights[key].shape[0]]).item())
    ]
    print(f"signs match hadamard.json for all {sum(1 for k in weights if k.endswith('.signs'))} "
          f"packed modules: {not mismatches} (mismatches: {len(mismatches)})")

    # Schema-2-aware load: tensor namespace is mlx-vlm (`language_model.`
    # prefix on every checkpoint key), which `artifact.py`'s schema_version==1
    # check refuses outright.
    model = TextModel(TextModelArgs.from_dict(cfg["text_config"]))
    prefix = "language_model."
    for record in cfg["modules"]:
        path = record["path"]
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        arrays = [weights[prefix + path + "." + suffix] for suffix in ("weight", "scales", "biases")]
        signs = weights.get(prefix + path + ".signs")
        setattr(parent, parts[-1], Packed(
            arrays, block=record["block"], signs=signs,
            embedding=record["embedding"], dtype=mx.float16))
    rest = {
        key[len(prefix):]: value for key, value in weights.items()
        if key.startswith(prefix)
        and not any(key.startswith(prefix + m["path"] + ".") for m in cfg["modules"])
    }
    model.load_weights(list(rest.items()), strict=False)
    mx.eval(model.parameters())

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(checkpoint_dir / "tokenizer.json"))
    ids = tok.encode(prompt).ids
    return model, tok, ids, ref_signs


def dump(checkpoint_dir: Path, output: Path, prompt: str) -> None:
    sys.path.insert(0, str(checkpoint_dir / "runtime"))
    from runtime import fwht  # the pack's own reference transform

    model, tok, ids, ref_signs = build_reference(checkpoint_dir, prompt)
    logits = model(mx.array([ids]))
    last = logits[0, -1].astype(mx.float32)
    top = mx.argsort(-last)[:5].tolist()
    print("prompt ids:", ids)
    print("top5:", [(t, repr(tok.decode([t])), round(float(last[t]), 3)) for t in top])

    # Which layers to capture: the first 3 GatedDeltaNet (linear-attention)
    # layers and the first 2 full-attention layers, derived from THIS
    # checkpoint's own `full_attention_interval` rather than hardcoded to
    # (0, 1, 2, 3, 7) - that tuple is only correct for an interval of 4. On a
    # pack with a different interval, capturing the wrong indices as "full
    # attention" would make the Swift side assert tight tolerances against an
    # UNMASKED linear-attention layer (see the module docstring) and fail for
    # a reason that has nothing to do with the port - precisely the hour of
    # bisection this script exists to prevent a repeat of.
    full_attention_interval = model.args.full_attention_interval
    num_layers = len(model.model.layers)

    def is_linear_layer(i: int) -> bool:
        return (i + 1) % full_attention_interval != 0

    linear_layers = [i for i in range(num_layers) if is_linear_layer(i)][:3]
    full_attention_layers = [i for i in range(num_layers) if not is_linear_layer(i)][:2]
    capture_layers = sorted(set(linear_layers + full_attention_layers))
    layer_ceiling = max(capture_layers) + 1 if capture_layers else 0
    print(f"capturing layers {capture_layers} (full-attention: {full_attention_layers}, "
          f"interval {full_attention_interval})")

    out = {"prompt_ids": mx.array(ids), "logits": last}
    h = model.model.embed_tokens(mx.array([ids]))
    out["embed_out"] = h.astype(mx.float32)
    for i, layer in enumerate(model.model.layers):
        # See the module docstring: mask=None means every full-attention
        # layer captured below is UNMASKED. Do not assert on them downstream.
        h = layer(h, mask=None, cache=None)
        if i in capture_layers:
            out[f"layer{i}_out"] = h.astype(mx.float32)
        if i >= layer_ceiling - 1:
            break

    # fwht fixture: fixed input, both directions, at the checkpoint's own
    # block size and its own hidden_size's sign vector (derived, not
    # hardcoded - a pack whose hidden_size is not itself a declared sign
    # width is a malformed contract, not a case to silently mis-key into).
    block = int(json.loads((checkpoint_dir / "hadamard.json").read_text())["prism.hadamard.block_size"])
    hidden = model.args.hidden_size
    if hidden not in ref_signs:
        raise SystemExit(
            f"hadamard.json has no sign vector for width {hidden} (hidden_size); "
            f"declared sign_widths are {sorted(ref_signs)}")
    hidden_signs = ref_signs[hidden]
    mx.random.seed(0)
    t = mx.random.normal([2, hidden]).astype(mx.float16)
    out["fwht_in"] = t.astype(mx.float32)
    out["fwht_fwd"] = fwht(t, block, hidden_signs).astype(mx.float32)
    out["fwht_inv"] = fwht(t, block, hidden_signs, inverse=True).astype(mx.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(output), out)
    print("dumped:", {k: list(v.shape) for k, v in out.items()}, "->", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_dir", type=Path, help="Prism Hadamard pack directory (schema_version 2)")
    parser.add_argument("output", type=Path, help="output .safetensors path for the reference dump")
    parser.add_argument("--prompt", default="The capital of France is",
                         help='prompt to encode and run (default: "The capital of France is")')
    args = parser.parse_args()
    dump(args.checkpoint_dir, args.output, args.prompt)


if __name__ == "__main__":
    main()
