#!/usr/bin/env python3
"""Regenerate the tiny random-weight format-2 Core ML fixtures used by the CI tests:
Tests/KrillEngineTests/Fixtures/ane_tiny_bert (mean-pooled embedder) and ane_tiny_rerank (cross-encoder).

    python3 tools/ane/make_generic_fixtures.py

Each is a 2-layer hidden-32 BERT with seeded random weights, one S=16 fp16 bucket, and `reference.json`
(token ids and the PyTorch fp32 output) which ANECoreMLParityTests compares with the real Core ML path."""
import json, os, shutil, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import encoders
from build_ane_package import convert_bucket, compile_model

CFG = dict(model_type="bert", hidden_size=32, num_attention_heads=2, intermediate_size=64, num_hidden_layers=2,
           max_position_embeddings=64, type_vocab_size=2, layer_norm_eps=1e-12, vocab_size=64)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../Tests/KrillEngineTests/Fixtures")


def make(name, cross, pooling):
    torch.manual_seed(11 if cross else 5)
    seq = 16
    core = encoders.build("bert", CFG, seq, 1.0, cross).eval()
    for p in core.parameters():                       # non-trivial LayerNorm scales / biases
        torch.nn.init.normal_(p, 0, 0.3) if p.dim() == 1 else None
    sd = {k: v.clone() for k, v in core.state_dict().items()}
    table = (torch.randn(CFG["vocab_size"], CFG["hidden_size"]) * 0.5)
    sd["embeddings.word_embeddings.weight"] = table
    spec = dict(kind="bert", cfg=CFG, cross=cross, pooling=pooling, prescale=1.0, hidden=CFG["hidden_size"])
    out = os.path.abspath(os.path.join(ROOT, name)); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    work = tempfile.mkdtemp(prefix="ane-fix-")
    shutil.copytree(compile_model(convert_bucket(spec, sd, seq, "fp16", work), work), os.path.join(out, f"s{seq}.mlmodelc"))
    t16 = table.numpy().astype(np.float16); t16.tofile(os.path.join(out, "embed_tokens.f16"))
    m = encoders.Pooled(core, pooling).eval(); rng = np.random.default_rng(3); ref = []
    for n in (3, 9, 16):
        ids = rng.integers(1, CFG["vocab_size"], size=n).tolist()
        e = torch.zeros(1, seq, CFG["hidden_size"]); e[0, :n] = torch.from_numpy(t16[ids].astype(np.float32))
        mk = torch.zeros(1, seq); mk[0, :n] = 1
        with torch.no_grad():
            ref.append({"ids": ids, "bucket": seq, "vector": m(e, mk)[0].tolist()})
    manifest = {"backend": "coreml", "format": 2, "variant": "fp16", "kind": "reranker" if cross else "embedder", "model": name,
                "pooling": pooling, "gather": "host", "tokenizer": "krill", "output_name": "score" if cross else "emb",
                "output_dim": 1 if cross else CFG["hidden_size"], "embed_scale": 1.0, "max_tokens": seq, "context_tokens": 64,
                "append_eos": False, "mrl_dims": None, "buckets": [{"seq": seq, "batch": 1, "path": f"s{seq}.mlmodelc"}],
                "table": {"file": "embed_tokens.f16", "vocab": CFG["vocab_size"], "dim": CFG["hidden_size"], "dtype": "float16"},
                "fallback_model": name}
    json.dump(manifest, open(os.path.join(out, "krill_ane.json"), "w"), indent=2)
    json.dump(ref, open(os.path.join(out, "reference.json"), "w"))
    shutil.rmtree(work, ignore_errors=True)
    print("fixture:", out, sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs), "bytes")


if __name__ == "__main__":
    make("ane_tiny_bert", False, "mean")
    make("ane_tiny_rerank", True, "classifier")
