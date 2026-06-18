#!/usr/bin/env python3
"""Requantize Gemma-4-12B from the ORIGINAL bf16 weights into a (mixed) nvfp4
checkpoint - single-quant, no double-quantization of attention.

Source  : mlx-community/gemma-4-12B-it-bf16   (ungated; full-precision weights)
Oracle  : mlx-community/gemma-4-12B-it-4bit    (used ONLY to learn which modules
          are quantized - the set of prefixes that own a `.scales` tensor; this
          matches the proven mlx-community coverage exactly).

Default: every quantized module -> nvfp4 (group_size 16, bits 4).  "Protected"
modules (any whose path contains one of --protect's substrings) instead use a
higher-precision override (default: 8-bit affine, group_size 64), which is what
lets a mixed checkpoint EXCEED Ollama's uniform nvfp4 on quality while the bulk
stays nvfp4 for speed.

The vision/audio projectors (`vision_embedder.patch_dense`,
`embed_vision.embedding_projection`, `embed_audio.embedding_projection`) are
auto-protected by default: nvfp4 on the patch-embedding attenuates the red
input channel enough to misread red-heavy colors (red->brown, yellow->olive,
magenta->purple) while text MMLU is unaffected. They are tiny, so 8-bit costs
almost nothing. Pass --no-protect-vision for a pure-uniform research baseline.

config.json gets a top-level nvfp4 block plus per-module override entries (keyed
by full module path, e.g. "language_model.model.layers.0.mlp.down_proj") for the
protected set - the exact format Krill's loader resolves via q.effective(path).

Usage:
  requant_gemma4_nvfp4.py --out <dir> [--protect down_proj --protect o_proj]
                         [--protect-bits 8] [--protect-gs 64] [--protect-mode affine]
                         [--protect-layers first4,last4]   # protect whole layers
  (no --protect / --protect-layers  =>  uniform nvfp4 baseline = P0)
"""
import argparse, glob, json, os, shutil
import mlx.core as mx

HUB = os.path.expanduser("~/.cache/huggingface/hub")

def snap(repo_dirname):
    cands = glob.glob(os.path.join(HUB, repo_dirname, "snapshots", "*"))
    cands = [c for c in cands if os.path.isfile(os.path.join(c, "config.json"))]
    if not cands:
        raise SystemExit(f"snapshot not found for {repo_dirname} (download it first)")
    return sorted(cands)[-1]

def module_of(t):
    for suf in (".weight", ".scales", ".biases"):
        if t.endswith(suf):
            return t[: -len(suf)]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-bf16", default="models--mlx-community--gemma-4-12B-it-bf16")
    ap.add_argument("--src-bf16-dir", default=None,
                    help="direct path to a bf16 checkpoint dir (bypasses the HF-cache "
                         "snapshot lookup; use for converted fine-tunes, e.g. the output "
                         "of tools/convert_gemma4_compressed_nvfp4_to_bf16.py)")
    ap.add_argument("--oracle-4bit", default="models--mlx-community--gemma-4-12B-it-4bit")
    ap.add_argument("--protect", action="append", default=[],
                    help="substring of module path to protect (repeatable), e.g. down_proj")
    ap.add_argument("--no-protect-vision", dest="protect_vision", action="store_false",
                    help="do NOT auto-protect the vision/audio projectors "
                         "(pure-uniform research baseline)")
    ap.set_defaults(protect_vision=True)
    ap.add_argument("--protect-layers", default="",
                    help="comma list like first4,last4 or explicit layer indices 0,1,47")
    ap.add_argument("--protect-bits", type=int, default=8)
    ap.add_argument("--protect-gs", type=int, default=64)
    ap.add_argument("--protect-mode", default="affine", choices=["affine", "mxfp8"])
    ap.add_argument("--n-layers", type=int, default=48, help="transformer layer count (for first/last)")
    args = ap.parse_args()

    # Auto-protect the vision/audio projectors unless explicitly disabled. nvfp4
    # on the patch-embedding attenuates the red channel (red->brown etc.); these
    # are tiny tensors so 8-bit is effectively free. Mirrors attn o_proj.
    VISION_PROTECT = ["vision_embedder.patch_dense",
                      "embed_vision.embedding_projection",
                      "embed_audio.embedding_projection"]
    if args.protect_vision:
        for s in VISION_PROTECT:
            if s not in args.protect:
                args.protect.append(s)
        print(f"[requant] auto-protecting vision/audio projectors @ "
              f"{args.protect_bits}b {args.protect_mode} (--no-protect-vision to disable)")

    SRC = os.path.expanduser(args.src_bf16_dir) if args.src_bf16_dir else snap(args.src_bf16)
    ORC = snap(args.oracle_4bit)
    DST = os.path.expanduser(args.out)
    os.makedirs(DST, exist_ok=True)

    # MLX requires mxfp8 to use exactly group_size 32 / bits 8; clamp so the
    # documented --protect-mode mxfp8 never crashes on the default --protect-gs.
    if args.protect_mode == "mxfp8" and (args.protect_gs != 32 or args.protect_bits != 8):
        print(f"[requant] mxfp8 requires gs=32 bits=8; overriding "
              f"gs={args.protect_gs}->32 bits={args.protect_bits}->8")
        args.protect_gs, args.protect_bits = 32, 8

    # ---- learn the quantized-module set from the 4-bit oracle's index ----
    orc_index = json.load(open(os.path.join(ORC, "model.safetensors.index.json")))
    quant_modules = set()
    for tname in orc_index["weight_map"]:
        if tname.endswith(".scales"):
            quant_modules.add(tname[: -len(".scales")])
    print(f"[requant] oracle quantized modules: {len(quant_modules)}")

    # ---- resolve protected layer indices ----
    protect_layer_idx = set()
    toks = [t for t in args.protect_layers.split(",") if t]
    for t in toks:
        if t.startswith("first"):
            protect_layer_idx |= set(range(int(t[5:])))
        elif t.startswith("last"):
            k = int(t[4:]); protect_layer_idx |= set(range(args.n_layers - k, args.n_layers))
        else:
            protect_layer_idx.add(int(t))

    def is_protected(mod):
        if any(s in mod for s in args.protect):
            return True
        if protect_layer_idx:
            # match ".layers.<idx>." exactly
            for i in protect_layer_idx:
                if f".layers.{i}." in mod:
                    return True
        return False

    # ---- stream bf16 shards; quantize the oracle set, protect the chosen subset ----
    bf16_index = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))
    shards = sorted(set(bf16_index["weight_map"].values()))
    out_weight_map = {}
    overrides = {}
    total_bytes = 0
    n_nvfp4 = n_prot = 0
    for shard in shards:
        w = mx.load(os.path.join(SRC, shard))
        out = {}
        for k in list(w):
            mod = module_of(k)
            if mod in quant_modules and k.endswith(".weight"):
                if is_protected(mod):
                    if args.protect_mode == "mxfp8":
                        nq, nsc = mx.quantize(w[k], group_size=args.protect_gs,
                                              bits=args.protect_bits, mode="mxfp8")
                        out[mod + ".weight"] = nq; out[mod + ".scales"] = nsc
                        overrides[mod] = {"group_size": args.protect_gs,
                                          "bits": args.protect_bits, "mode": "mxfp8"}
                    else:
                        nq, nsc, nb = mx.quantize(w[k], group_size=args.protect_gs,
                                                  bits=args.protect_bits, mode="affine")
                        out[mod + ".weight"] = nq; out[mod + ".scales"] = nsc
                        out[mod + ".biases"] = nb
                        overrides[mod] = {"group_size": args.protect_gs,
                                          "bits": args.protect_bits, "mode": "affine"}
                    n_prot += 1
                else:
                    nq, nsc = mx.quantize(w[k], group_size=16, bits=4, mode="nvfp4")
                    out[mod + ".weight"] = nq; out[mod + ".scales"] = nsc
                    n_nvfp4 += 1
            else:
                out[k] = w[k]
        mx.eval(out)
        for name, t in out.items():
            total_bytes += t.nbytes
            out_weight_map[name] = shard
        mx.save_safetensors(os.path.join(DST, shard), out, metadata={"format": "mlx"})
        print(f"[requant] wrote {shard} ({len(out)} tensors)")

    # ---- config.json: top-level nvfp4 + per-module overrides for protected ----
    cfg = json.load(open(os.path.join(SRC, "config.json")))
    qblock = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    qblock.update(overrides)
    cfg["quantization"] = qblock
    if isinstance(cfg.get("quantization_config"), dict):
        cfg["quantization_config"] = qblock
    json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2)

    out_index = {"metadata": {"total_size": total_bytes}, "weight_map": out_weight_map}
    json.dump(out_index, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=2)

    for fn in os.listdir(SRC):
        if fn.endswith(".safetensors") or fn in ("config.json", "model.safetensors.index.json"):
            continue
        src = os.path.join(SRC, fn)
        if os.path.isfile(src) or os.path.islink(src):
            shutil.copy(os.path.realpath(src), os.path.join(DST, fn))

    print(f"[requant] DONE. nvfp4={n_nvfp4} protected={n_prot} "
          f"({args.protect or args.protect_layers or 'none'} @ {args.protect_bits}b {args.protect_mode}). "
          f"{total_bytes/1e9:.2f} GB -> {DST}")

if __name__ == "__main__":
    main()
