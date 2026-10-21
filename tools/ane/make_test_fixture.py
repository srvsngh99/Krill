#!/usr/bin/env python3
"""Regenerate the tiny random-weight Core ML fixture used by the CI parity test
(Tests/KrillEngineTests/Fixtures/ane_tiny, ANECoreMLParityTests).

    python3 tools/ane/make_test_fixture.py [--out Tests/KrillEngineTests/Fixtures/ane_tiny]

A 3-layer, hidden-64 model with seeded random weights, one S=16 and one S=32 fp16 bucket, plus
`reference.json`: token ids and the PyTorch fp32 output for each, which the Swift test compares to
what Core ML returns through the real runtime path (host gather -> bucket -> prediction).
"""
import argparse, json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import eg2_ane
from build_ane_embedder import convert_bucket, compile_model

TINY = dict(
    hidden_size=64, num_attention_heads=2, num_key_value_heads=1, head_dim=32, intermediate_size=128,
    num_hidden_layers=3, hidden_size_per_layer_input=16, embedding_dim=48, vocab_size=100, rms_norm_eps=1e-6,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"],
    rope_parameters={"sliding_attention": {"rope_theta": 10000.0}, "full_attention": {"rope_theta": 1000000.0}},
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../../Tests/KrillEngineTests/Fixtures/ane_tiny"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    torch.manual_seed(7)
    tc = TINY
    work = tempfile.mkdtemp(prefix="ane-fixture-")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    table = (torch.randn(tc["vocab_size"], tc["hidden_size"]) * 0.2).numpy().astype(np.float16)
    scale = float(tc["hidden_size"]) ** 0.5
    buckets, reference = [], []
    rng = np.random.default_rng(3)
    for seq in (16, 32):
        # The same module the converter traces: re-create it with the converter's seeded init.
        torch.manual_seed(7)
        pkg = convert_bucket(tc, None, seq, "fp16", work)
        shutil.copytree(compile_model(pkg, work), os.path.join(out, f"s{seq}.mlmodelc"))
        buckets.append({"seq": seq, "batch": 1, "path": f"s{seq}.mlmodelc"})
        torch.manual_seed(7)
        ref = eg2_ane.EG2(tc, seq, B=1).eval()
        for n in ((3, 9, 16) if seq == 16 else (17, 25, 32)):   # lengths whose smallest bucket is `seq`
            ids = rng.integers(3, tc["vocab_size"], size=n).tolist()
            e = torch.zeros(1, seq, tc["hidden_size"])
            e[0, :n] = torch.from_numpy(table[ids].astype(np.float32)) * scale
            mask = torch.zeros(1, seq)
            mask[0, :n] = 1
            with torch.no_grad():
                v = ref(e, mask)[0].tolist()
            reference.append({"ids": ids, "bucket": seq, "vector": v})
    table.tofile(os.path.join(out, "embed_tokens.f16"))
    manifest = {"backend": "coreml", "format": 1, "variant": "fp16", "output_dim": tc["embedding_dim"],
                "embed_scale": scale, "max_tokens": 32, "buckets": buckets,
                "table": {"file": "embed_tokens.f16", "vocab": tc["vocab_size"], "dim": tc["hidden_size"], "dtype": "float16"},
                "fallback_model": "embeddinggemma-2-mlx-4bit-dyn-text"}
    json.dump(manifest, open(os.path.join(out, "krill_ane.json"), "w"), indent=2)
    json.dump(reference, open(os.path.join(out, "reference.json"), "w"))
    shutil.rmtree(work, ignore_errors=True)
    print("fixture:", out)


if __name__ == "__main__":
    main()
