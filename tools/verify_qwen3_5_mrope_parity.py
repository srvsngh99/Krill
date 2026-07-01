#!/usr/bin/env python3
"""Qwen3.5 interleaved-mRoPE parity oracle.

Dumps a random q/k + a 3D (t,h,w) position grid and the result of mlx_vlm's
real `Qwen3_5RotaryEmbedding.apply_rotary` (interleaved, partial rotary). The
Swift `Qwen35VLMRoPEParityTests` rebuilds the cos/sin via `Qwen35VLMRoPE` and
applies `applyPartialMRoPE`, asserting a match — validating the frequency
selector + half-split pairing in isolation, with distinct t/h/w so the axis
selection actually matters.

    ~/.krill/venv-ornith/bin/python tools/verify_qwen3_5_mrope_parity.py <outdir>
"""
import json
import sys
from pathlib import Path

import mlx.core as mx

from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding

HEAD_DIM = 256
PARTIAL = 0.25
ROTARY = int(HEAD_DIM * PARTIAL)  # 64
SECTION = [11, 11, 10]
THETA = 10_000_000


def main():
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/qwen3_5_mrope")
    outdir.mkdir(parents=True, exist_ok=True)

    mx.random.seed(1)
    emb = Qwen3_5RotaryEmbedding(
        ROTARY, max_position_embeddings=128, base=THETA, mrope_section=SECTION)

    L, heads, kv = 7, 4, 2
    q = mx.random.normal((1, heads, L, HEAD_DIM)).astype(mx.float32)
    k = mx.random.normal((1, kv, L, HEAD_DIM)).astype(mx.float32)
    # Distinct t/h/w so the interleaved axis selector is exercised.
    t = mx.arange(L)
    h = mx.arange(L) + 3
    w = mx.arange(L) * 2
    position_ids = mx.stack([t, h, w])[:, None, :]  # [3, 1, L]

    q2, k2 = emb.apply_rotary(q, k, position_ids, unsqueeze_dim=1)
    mx.eval(q2, k2)

    mx.save_safetensors(str(outdir / "mrope.safetensors"), {
        "q": q, "k": k,
        "positions": position_ids[:, 0, :].astype(mx.float32),  # [3, L]
        "q_out": q2.astype(mx.float32), "k_out": k2.astype(mx.float32),
    })
    (outdir / "meta.json").write_text(json.dumps(dict(
        head_dim=HEAD_DIM, partial_rotary_factor=PARTIAL, rotary_dim=ROTARY,
        mrope_section=SECTION, theta=THETA, L=L, heads=heads, kv=kv), indent=2))
    print(f"wrote mrope fixture -> {outdir}  q{tuple(q.shape)} -> q_out{tuple(q2.shape)}")


if __name__ == "__main__":
    main()
