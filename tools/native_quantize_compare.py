#!/usr/bin/env python3
"""Byte-compare a native `krillm quantize` output against a reference synthesized
from the SAME bf16 source via the canonical MLX op (`mx.quantize`), applying the
same layer selection the Swift quantizer uses. A match proves the Swift path ==
the canonical op for that mode.

Streams tensor-by-tensor (build reference -> compare -> discard) so peak memory
stays near one tensor, not source + reference + output at once.

Usage: native_quantize_compare.py <bf16_source_dir> <mine_dir> <mode> <group> <bits>
Exits 0 on byte-identical, 1 otherwise.
"""
import glob
import sys

import mlx.core as mx

FLOAT = (mx.float16, mx.bfloat16, mx.float32)


def main() -> int:
    src, mine_dir, mode, gs, bits = (
        sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))

    # Lazy/mmap loads: only the tensors actually touched get realized.
    sw = {}
    for f in sorted(glob.glob(src + "/*.safetensors")):
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
            return
        if a.shape == ref.shape and a.dtype == ref.dtype:
            mx.eval(a, ref)
            if bool(mx.all(a == ref).item()):
                ok += 1
                return
        diff += 1
        if len(examples) < 6:
            examples.append((name, str(ref.shape), str(a.shape), str(ref.dtype), str(a.dtype)))

    for k in sorted(sw):
        w = sw[k]
        if k.endswith(".weight") and w.ndim == 2 and w.shape[1] % gs == 0:
            win = w.astype(mx.float16) if mode == "affine" else w
            r = mx.quantize(win, group_size=gs, bits=bits, mode=mode)
            stem = k[: -len("weight")]
            check(k, r[0])
            check(stem + "scales", r[1])
            if len(r) == 3 and r[2] is not None:
                check(stem + "biases", r[2])
        else:
            check(k, w.astype(mx.float16) if w.dtype in FLOAT else w)
        sw[k] = None  # release the source tensor

    extra = [k for k in mine if k not in expected]
    print(f"byte-identical {ok}/{len(expected)}  diff {diff}  missing {miss}  extra {len(extra)}")
    for e in examples:
        print("  DIFF", e)
    for k in extra[:6]:
        print("  EXTRA", k)
    return 0 if diff == 0 and miss == 0 and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
