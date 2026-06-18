#!/usr/bin/env python3
"""Byte-compare a native `krill quantize` output against a reference synthesized
from the SAME bf16 source via the canonical MLX op (`mx.quantize`). A match proves
the Swift path == the canonical op, tensor for tensor.

The layer selection AND the per-module precision are read from the produced
`<mine_dir>/config.json` quantization block (top-level `group_size`/`bits`/`mode`
plus per-module override entries), so this works for every mode the quantizer
emits: dense affine, nvfp4/mxfp4/mxfp8, and reference-set mixed precision
(protected projectors at 8-bit, 3-D experts forced affine).

Streams tensor-by-tensor (build reference -> compare -> discard) so peak memory
stays near one tensor, not source + reference + output at once.

Usage: native_quantize_compare.py <bf16_source_dir> <mine_dir> [ignored...]
Exits 0 on byte-identical, 1 otherwise.
"""
import glob
import json
import os
import sys

import mlx.core as mx

FLOAT = (mx.float16, mx.bfloat16, mx.float32)


def main() -> int:
    src_dir, mine_dir = sys.argv[1], sys.argv[2]

    cfg = json.load(open(os.path.join(mine_dir, "config.json")))
    q = cfg.get("quantization") or cfg.get("quantization_config") or {}
    top_gs = int(q["group_size"])
    top_bits = int(q["bits"])
    top_mode = q.get("mode", "affine")
    # Any nested object in the quant block is a per-module override.
    overrides = {k: v for k, v in q.items()
                 if isinstance(v, dict) and "group_size" in v and "bits" in v}

    def effective(module):
        o = overrides.get(module)
        if o:
            return int(o["group_size"]), int(o["bits"]), o.get("mode", top_mode)
        return top_gs, top_bits, top_mode

    # Lazy/mmap loads: only the tensors actually touched get realized.
    sw = {}
    for f in sorted(glob.glob(src_dir + "/*.safetensors")):
        sw.update(mx.load(f))
    mine = {}
    for f in sorted(glob.glob(mine_dir + "/*.safetensors")):
        mine.update(mx.load(f))

    ok = diff = miss = 0
    examples = []
    expected = set()

    def check(name, ref):
        nonlocal ok, diff, miss
        expected.add(name)
        a = mine.get(name)
        if a is None:
            miss += 1
            examples.append(("MISSING", name, str(ref.shape), "-", str(ref.dtype), "-")) if len(examples) < 8 else None
            return
        if a.shape == ref.shape and a.dtype == ref.dtype:
            mx.eval(a, ref)
            if bool(mx.all(a == ref).item()):
                ok += 1
                return
        diff += 1
        if len(examples) < 8:
            examples.append(("DIFF", name, str(ref.shape), str(a.shape), str(ref.dtype), str(a.dtype)))

    for k in sorted(sw):
        w = sw[k]
        # A `.weight` is quantized iff the output carries its `.scales`. Resolve the
        # module's effective precision from the config and recompute via mx.quantize.
        stem = k[: -len(".weight")] if k.endswith(".weight") else None
        if stem is not None and (stem + ".scales") in mine:
            gs, bits, mode = effective(stem)
            # Stacked 3-D experts are born-quantized affine at the TOP-LEVEL group
            # (the MoE runtime ignores per-module overrides for them); mirror the
            # loader/quantizer so the recomputed reference matches.
            if w.ndim == 3:
                gs, bits, mode = top_gs, top_bits, "affine"
            win = w.astype(mx.float16) if mode == "affine" else w
            r = mx.quantize(win, group_size=gs, bits=bits, mode=mode)
            check(k, r[0])
            check(stem + ".scales", r[1])
            if len(r) == 3 and r[2] is not None:
                check(stem + ".biases", r[2])
        else:
            # Pass-through: float tensors stored at fp16, others verbatim.
            check(k, w.astype(mx.float16) if w.dtype in FLOAT else w)
        sw[k] = None  # release the source tensor

    extra = [k for k in mine if k not in expected]
    print(f"byte-identical {ok}/{len(expected)}  diff {diff}  missing {miss}  extra {len(extra)}  "
          f"(top {top_mode} {top_bits}b/g{top_gs}, {len(overrides)} overrides)")
    for e in examples:
        print("  ", *e)
    for k in extra[:8]:
        print("  EXTRA", k)
    return 0 if diff == 0 and miss == 0 and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
