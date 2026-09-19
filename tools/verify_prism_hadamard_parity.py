#!/usr/bin/env python3
"""Reference dumper for Krill's `prism_hadamard_qwen35` port (Prism ML's
Ternary-Bonsai-2-27B and any other checkpoint in the same pack format),
built from mlx-lm's `qwen3_5.TextModel` + the pack's own `runtime/runtime.py`
(`Packed`, `fwht`).

WHY THIS SCRIPT EXISTS, NOT THE PACK'S OWN `runtime/artifact.py`: the
bundled loader hard-requires `schema_version == 1` and refuses any pack
whose `config.json` says otherwise - but every real pack shipped so far
(Ternary-Bonsai-2-27B included) is `schema_version: 2`, which moved the
tensor namespace to mlx-vlm's `language_model.` prefix. Their own loader
cannot open their own pack. This script is schema-2-aware: it drives
`mlx_lm.models.qwen3_5.TextModel` directly and swaps in `Packed` at the 402
manifest paths with the `language_model.` prefix threaded through, then
loads everything else with `strict=False`.

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

    out = {"prompt_ids": mx.array(ids), "logits": last}
    h = model.model.embed_tokens(mx.array([ids]))
    out["embed_out"] = h.astype(mx.float32)
    for i, layer in enumerate(model.model.layers):
        # See the module docstring: mask=None means layers 3/7 (full
        # attention) are captured UNMASKED. Do not assert on them downstream.
        h = layer(h, mask=None, cache=None)
        if i in (0, 1, 2, 3, 7):
            out[f"layer{i}_out"] = h.astype(mx.float32)
        if i >= 7:
            break

    # fwht fixture: fixed input, both directions, at the checkpoint's own
    # block size and the width-5120 sign vector (hidden_size for this pack;
    # adjust if verifying a differently-shaped checkpoint).
    block = int(json.loads((checkpoint_dir / "hadamard.json").read_text())["prism.hadamard.block_size"])
    hidden = model.args.hidden_size
    mx.random.seed(0)
    t = mx.random.normal([2, hidden]).astype(mx.float16)
    out["fwht_in"] = t.astype(mx.float32)
    out["fwht_fwd"] = fwht(t, block, ref_signs[hidden]).astype(mx.float32)
    out["fwht_inv"] = fwht(t, block, ref_signs[hidden], inverse=True).astype(mx.float32)

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
