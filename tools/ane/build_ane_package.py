#!/usr/bin/env python3
"""Build a Neural Engine package for any Krill embedder or reranker, parity-gate it, and install it.

    python3 tools/ane/build_ane_package.py --model bge-base-en [--variant lin8|fp16] [--corpus FILE] [--krill BIN]

Reads the weights `krill pull <model>` already put in the model store (KRILL_MODELS_DIR or ~/.krill/models),
converts the encoder to Core ML (one fixed-shape batch-1 model per sequence-length bucket, 64/128/256/512 capped
at the model's context), compiles it, and writes `<store>/blobs/<model>-ane-<variant>` with a `krill_ane.json`
(format 2). Before anything is installed the package is served by `krill serve` and compared with the GPU build of
the same model through the same server: embedders need cosine >= 0.999 (min over 200 queries and 200 passages),
rerankers Spearman >= 0.999 over the pooled scores. A model that fails writes nothing and prints why.
EmbeddingGemma 2 keeps its own hand port (eg2_ane.py) and the format-1 package of PR 343.
Needs the pins in tools/ane/requirements.txt and Xcode's `xcrun coremlcompiler`. See docs/ANE_EMBEDDER.md.
"""
import argparse, datetime, glob, hashlib, json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = (64, 128, 256, 512)
KEEP_SUFFIX = (".json", ".txt", ".model", ".tiktoken", ".jinja")


def models_dir():
    return os.environ.get("KRILL_MODELS_DIR") or os.path.expanduser("~/.krill/models")


def detect(src, name):
    """Krill's own per-model semantics (EmbeddingEngine.loadResolved): pooling, context, tokenizer behaviour."""
    cfg = json.load(open(os.path.join(src, "config.json")))
    mt = cfg.get("model_type", "")
    st_pool = None
    p = os.path.join(src, "1_Pooling", "config.json")
    if os.path.exists(p):
        j = json.load(open(p))
        st_pool = "cls" if j.get("pooling_mode_cls_token") else "last_token" if j.get("pooling_mode_lasttoken") else "mean" if j.get("pooling_mode_mean_tokens") else None
    archs = " ".join(cfg.get("architectures") or [])
    spec = dict(model=name, cfg=cfg, kind=None, cross=False, append_eos=False, prescale=1.0, mrl=None)
    if "SequenceClassification" in archs:
        spec.update(kind="bert", cross=True, pooling="classifier", ctx=cfg["max_position_embeddings"])
        spec["ctx"] = min(512, cfg["max_position_embeddings"] - (2 if mt in ("roberta", "xlm-roberta") else 0))
    elif mt == "mpnet":
        spec.update(kind="mpnet", pooling="mean", ctx=cfg["max_position_embeddings"] - cfg.get("pad_token_id", 1) - 1)
    elif mt == "nomic_bert":
        if cfg.get("moe_every_n_layers") or cfg.get("num_experts"):
            spec.update(kind="nomic_moe", pooling="mean", ctx=min(cfg.get("max_trained_positions", 2048), 2048))
        else:
            spec.update(kind="nomic_bert", pooling="mean", ctx=cfg.get("max_position_embeddings", 2048))
    elif mt == "new":
        spec.update(kind="new", pooling=st_pool or "cls", ctx=min(cfg.get("max_position_embeddings", 8192), 8192))
    elif mt == "modernbert":
        spec.update(kind="modernbert", pooling=st_pool or "cls", ctx=min(cfg.get("max_position_embeddings", 8192), 8192), prescale=1 / 8.)
    elif mt in ("qwen2",):
        spec.update(kind="qwen2", pooling=st_pool or "last_token", ctx=min(cfg.get("max_position_embeddings", 8192), 8192))
        spec["append_eos"] = spec["pooling"] == "last_token"
    elif mt == "bert" and cfg.get("position_embedding_type") == "alibi" or "JinaBert" in archs:
        spec.update(kind="jina", pooling=st_pool or "mean", ctx=min(cfg.get("max_position_embeddings", 8192), 8192))
    elif mt in ("bert", "roberta", "xlm-roberta"):
        spec.update(kind="bert", pooling="mean", ctx=cfg["max_position_embeddings"] - (2 if mt != "bert" else 0))
    else:
        raise SystemExit(f"no Neural Engine port for model_type '{mt}'")
    spec["hidden"] = cfg.get("hidden_size", cfg.get("n_embd"))
    spec["out_dim"] = 1 if spec["cross"] else spec["hidden"]
    return spec


def convert_bucket(spec, sd, seq, variant, work):
    import numpy as np, torch, coremltools as ct
    import coremltools.optimize.coreml as cto
    import encoders
    core = encoders.build(spec["kind"], spec["cfg"], seq, spec["prescale"], spec["cross"]).eval()
    encoders.load_core(core, sd)
    m = encoders.Pooled(core, spec["pooling"]).eval()
    H = spec["hidden"]
    ex = (torch.randn(1, seq, H) * 0.1, torch.ones(1, seq))
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
    ins = [ct.TensorType("embeds", shape=(1, seq, H), dtype=np.float16), ct.TensorType("mask", shape=(1, seq), dtype=np.float16)]
    oname = "score" if spec["cross"] else "emb"
    ml = ct.convert(traced, inputs=ins, outputs=[ct.TensorType(oname, dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.iOS18, convert_to="mlprogram")
    if variant == "lin8":
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        ml = cto.linear_quantize_weights(ml, cfg)
    pkg = os.path.join(work, f"s{seq}.mlpackage")
    ml.save(pkg)
    return pkg


def compile_model(pkg, out_dir):
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir], check=True, capture_output=True)
    return os.path.join(out_dir, os.path.basename(pkg).replace(".mlpackage", ".mlmodelc"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def register(store, out, name, variant, family, params):
    files, total = [], 0
    for dp, _, fns in os.walk(out):
        for fn in sorted(fns):
            full = os.path.join(dp, fn); size = os.path.getsize(full); total += size
            files.append({"path": os.path.relpath(full, out), "sha256": sha256(full), "sizeBytes": size})
    mf = {"name": name, "family": family, "params": params, "quant": f"ane-{variant}",
          "source": "local build (tools/ane/build_ane_package.py)", "context": 512, "chatTemplate": "bert",
          "files": files, "sizeBytes": total, "pulledAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    d = os.path.join(store, "manifests"); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(mf, f, indent=2, sort_keys=True)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="registry name already pulled, e.g. bge-base-en")
    ap.add_argument("--variant", choices=("lin8", "fp16", "auto"), default="lin8", help="auto = lin8, then fp16 if lin8 fails the parity gate")
    ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
    ap.add_argument("--work", help="scratch directory for intermediates (default: a temp dir)")
    ap.add_argument("--corpus", default=os.environ.get("ANE_CORPUS"), help='JSON {"queries":[...],"passages":[...]} for the parity gate')
    ap.add_argument("--krill", default=os.environ.get("KRILL_BIN") or ".build/release/krill", help="krill binary used by the parity gate")
    ap.add_argument("--port", type=int, default=57499)
    ap.add_argument("--reference", choices=("gpu", "hf"), default="gpu",
                    help="gpu = the GPU build through krill serve (default); hf = HF transformers fp32 (needs transformers; for a checkpoint the GPU path cannot load)")
    ap.add_argument("--no-gate", action="store_true", help="skip the parity gate (never for a package you ship)")
    ap.add_argument("--no-install", action="store_true", help="build and gate only; leave the package in --work")
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--min-spearman", type=float, default=0.999)
    ap.add_argument("--fallback", help="(EmbeddingGemma 2 only) MLX model for long text and media")
    a = ap.parse_args()
    if a.variant == "auto":
        for v in ("lin8", "fp16"):
            if subprocess.run([sys.executable, os.path.abspath(__file__)] + [v if x == "auto" else x for x in sys.argv[1:]]).returncode == 0:
                return
        raise SystemExit(2)

    store = models_dir()
    src = os.path.join(store, "blobs", a.model)
    if not os.path.exists(os.path.join(src, "config.json")):
        raise SystemExit(f"{a.model} is not installed in {store}; run: krill pull {a.model}")
    if json.load(open(os.path.join(src, "config.json"))).get("model_type") == "embedding_gemma2":
        import build_ane_embedder
        sys.argv = [sys.argv[0], "--variant", a.variant, "--buckets", a.buckets, "--register", "--source", src] + (
            ["--fallback", a.fallback] if a.fallback else [])
        return build_ane_embedder.main()

    import numpy as np, encoders
    spec = detect(src, a.model)
    buckets = [b for b in map(int, a.buckets.split(",")) if b <= spec["ctx"]] or [min(64, spec["ctx"])]
    name = f"{a.model}-ane-{a.variant}"
    work = a.work or tempfile.mkdtemp(prefix="ane-build-")
    pkgdir = os.path.join(work, name)
    shutil.rmtree(pkgdir, ignore_errors=True); os.makedirs(pkgdir)
    sd = encoders.read_weights(src)
    table = encoders.word_table(spec["kind"], sd).float().numpy().astype(np.float16)
    entries = []
    for seq in buckets:
        t0 = time.time()
        pkg = convert_bucket(spec, sd, seq, a.variant, work)
        mlc = compile_model(pkg, work)
        dst = os.path.join(pkgdir, f"s{seq}.mlmodelc")
        shutil.copytree(mlc, dst)
        shutil.rmtree(pkg, ignore_errors=True); shutil.rmtree(mlc, ignore_errors=True)
        entries.append({"seq": seq, "batch": 1, "path": f"s{seq}.mlmodelc"})
        print(f"bucket {seq}: {time.time() - t0:.0f}s", flush=True)
    table.tofile(os.path.join(pkgdir, "embed_tokens.f16"))
    for dp, _, fns in os.walk(src):
        for fn in fns:
            full = os.path.join(dp, fn); rel = os.path.relpath(full, src)
            if fn.endswith(KEEP_SUFFIX) and os.path.getsize(full) < 40 << 20:
                os.makedirs(os.path.dirname(os.path.join(pkgdir, rel)), exist_ok=True)
                shutil.copy(full, os.path.join(pkgdir, rel))
    manifest = {"backend": "coreml", "format": 2, "variant": a.variant, "kind": "reranker" if spec["cross"] else "embedder",
                "model": a.model, "pooling": spec["pooling"], "gather": "host", "tokenizer": "krill",
                "output_name": "score" if spec["cross"] else "emb", "output_dim": spec["out_dim"], "embed_scale": 1.0,
                "max_tokens": max(buckets), "context_tokens": spec["ctx"], "append_eos": spec["append_eos"],
                "mrl_dims": spec["mrl"], "buckets": entries,
                "table": {"file": "embed_tokens.f16", "vocab": int(table.shape[0]), "dim": int(table.shape[1]), "dtype": "float16"},
                "fallback_model": a.model}
    with open(os.path.join(pkgdir, "krill_ane.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    size = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(pkgdir) for fn in fns)
    print(f"built {name}: {size / 1e6:.0f} MB")

    family = "reranker" if spec["cross"] else "bert"
    if not a.no_gate:
        import parity_gate
        ok = parity_gate.run(a, store, name, pkgdir, spec, family)
        if not ok:
            print(f"PARITY FAILED: {name} was not installed (intermediates kept in {work})")
            raise SystemExit(2)
    if a.no_install:
        print("not installed (--no-install):", pkgdir); return
    dest = os.path.join(store, "blobs", name)
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(pkgdir, dest)
    total = register(store, dest, name, a.variant, family, "")
    print(f"installed {name} ({total / 1e6:.0f} MB) in {store}")


if __name__ == "__main__":
    main()
