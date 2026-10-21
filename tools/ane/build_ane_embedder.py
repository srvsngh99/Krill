#!/usr/bin/env python3
"""Build the Neural Engine EmbeddingGemma 2 package that `embed_device = ane` serves.

    python3 tools/ane/build_ane_embedder.py [--source DIR] [--variant lin8|fp16] [--out DIR] [--register]

Reads the full-precision (bf16) EmbeddingGemma 2 checkpoint that `krill pull embeddinggemma-2`
installs, converts its TEXT tower to Core ML (one fixed-shape model per sequence-length bucket,
batch 1), compiles each to `.mlmodelc`, and writes the package Krill's `EmbeddingEngine` loads:

    <out>/krill_ane.json            manifest (backend, buckets, table, MLX fallback model)
    <out>/embed_tokens.f16          fp16 [vocab, 512] token-embedding table (Krill mmaps it)
    <out>/s64.mlmodelc ... s512     compiled buckets (lin8 = int8 per-channel weights)
    <out>/tokenizer.json, tokenizer_config.json, config_sentence_transformers.json

With --register it also writes the registry manifest so `krill list` and the server see it.
Needs the pins in tools/ane/requirements.txt and Xcode's `xcrun coremlcompiler`. Nothing here is
imported by Krill at runtime. See docs/ANE_EMBEDDER.md.
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BUCKETS = (64, 128, 256, 512)
DEFAULT_FALLBACK = "embeddinggemma-2-mlx-4bit-dyn-text"


def models_dir():
    return os.environ.get("KRILL_MODELS_DIR") or os.path.expanduser("~/.krill/models")


def convert_bucket(tc, sd, seq, variant, work):
    import numpy as np, torch, coremltools as ct
    import coremltools.optimize.coreml as cto
    import eg2_ane
    m = eg2_ane.EG2(tc, seq, B=1, embed_in_graph=False).eval()
    if sd is not None:
        eg2_ane.load_weights(m, sd)
    ex = (torch.randn(1, seq, tc["hidden_size"]), torch.ones(1, seq))
    with torch.no_grad():
        traced = torch.jit.trace(m, ex)
    ins = [ct.TensorType("embeds", shape=(1, seq, tc["hidden_size"]), dtype=np.float16),
           ct.TensorType("mask", shape=(1, seq), dtype=np.float16)]
    ml = ct.convert(traced, inputs=ins, outputs=[ct.TensorType("emb", dtype=np.float16)],
                    compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE,
                    minimum_deployment_target=ct.target.iOS18, convert_to="mlprogram")
    if variant == "lin8":
        cfg = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel"))
        ml = cto.linear_quantize_weights(ml, cfg)
    pkg = os.path.join(work, f"s{seq}.mlpackage")
    ml.save(pkg)
    return pkg


def compile_model(pkg, out_dir):
    subprocess.run(["xcrun", "coremlcompiler", "compile", pkg, out_dir], check=True, capture_output=True)
    return os.path.join(out_dir, os.path.basename(pkg).replace(".mlpackage", ".mlmodelc"))


def tree_files(root):
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            full = os.path.join(dp, fn)
            yield os.path.relpath(full, root), full


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def register(out, name, variant):
    files, total = [], 0
    for rel, full in tree_files(out):
        size = os.path.getsize(full)
        total += size
        files.append({"path": rel, "sha256": sha256(full), "sizeBytes": size})
    mf = {"name": name, "family": "bert", "params": "740M", "quant": f"ane-{variant}",
          "source": "local build (tools/ane/build_ane_embedder.py)", "context": 8192,
          "chatTemplate": "bert", "files": files, "sizeBytes": total,
          "pulledAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    d = os.path.join(models_dir(), "manifests")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(mf, f, indent=2, sort_keys=True)
    print("registered", name, f"({total / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=os.path.join(models_dir(), "blobs", "embeddinggemma-2"),
                    help="full-precision EmbeddingGemma 2 directory (model.safetensors, config.json, tokenizer.json)")
    ap.add_argument("--variant", choices=("lin8", "fp16"), default="lin8")
    ap.add_argument("--buckets", default=",".join(map(str, BUCKETS)))
    ap.add_argument("--out", help="package directory (default: <models>/blobs/embeddinggemma-2-ane-<variant>)")
    ap.add_argument("--fallback", default=DEFAULT_FALLBACK, help="MLX model that serves long text and media")
    ap.add_argument("--work", help="scratch directory for the intermediate .mlpackage files")
    ap.add_argument("--register", action="store_true", help="write the registry manifest too")
    a = ap.parse_args()

    import numpy as np, torch
    import eg2_ane
    from safetensors import safe_open
    name = f"embeddinggemma-2-ane-{a.variant}"
    out = a.out or os.path.join(models_dir(), "blobs", name)
    buckets = [int(x) for x in a.buckets.split(",")]
    tc = eg2_ane.text_cfg(os.path.join(a.source, "config.json"))
    sd, table = {}, None
    with safe_open(os.path.join(a.source, "model.safetensors"), "pt") as f:
        for k in f.keys():
            if not k.startswith("language_model."):
                continue
            if k.endswith("embed_tokens.weight"):
                table = f.get_tensor(k).float().numpy().astype(np.float16)   # host-side table
                continue
            sd[k] = f.get_tensor(k)
    os.makedirs(out, exist_ok=True)
    work = a.work or tempfile.mkdtemp(prefix="ane-build-", dir=os.environ.get("TMPDIR"))
    os.makedirs(work, exist_ok=True)

    entries = []
    for seq in buckets:
        t0 = time.time()
        pkg = convert_bucket(tc, sd, seq, a.variant, work)
        mlc = compile_model(pkg, work)
        dst = os.path.join(out, f"s{seq}.mlmodelc")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(mlc, dst)
        shutil.rmtree(pkg, ignore_errors=True); shutil.rmtree(mlc, ignore_errors=True)
        entries.append({"seq": seq, "batch": 1, "path": f"s{seq}.mlmodelc"})
        print(f"bucket {seq}: {time.time() - t0:.0f}s", flush=True)

    table.tofile(os.path.join(out, "embed_tokens.f16"))
    for fn in ("tokenizer.json", "tokenizer_config.json", "config_sentence_transformers.json"):
        src = os.path.join(a.source, fn)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, fn))
    manifest = {"backend": "coreml", "format": 1, "variant": a.variant,
                "output_dim": tc["embedding_dim"], "embed_scale": float(tc["hidden_size"]) ** 0.5,
                "max_tokens": max(buckets), "buckets": entries,
                "table": {"file": "embed_tokens.f16", "vocab": int(table.shape[0]), "dim": int(table.shape[1]), "dtype": "float16"},
                "fallback_model": a.fallback}
    with open(os.path.join(out, "krill_ane.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if not a.work:
        shutil.rmtree(work, ignore_errors=True)
    print("wrote", out)
    if a.register:
        register(out, name, a.variant)


if __name__ == "__main__":
    main()
