#!/usr/bin/env python3
"""Convert an OpenAI / mlx-community Whisper checkpoint to a KrillLM model dir.

KrillLM loads weights from safetensors only (mlx-swift cannot read .npz), so
this tool re-packs the published `weights.npz` (or an HF safetensors export)
into `<out>/model.safetensors` with the OpenAI key layout preserved
(`encoder.*`, `decoder.*`), alongside the original `config.json`. The native
`WhisperEncoder`/`WhisperDecoder` module keys mirror that layout, so the
result loads with no remapping.

Usage:
    convert_whisper.py <src-npz-or-dir> <out-dir>

`<src>` may be a `weights.npz`, a directory containing one (plus config.json),
or an HF model id snapshot dir.
"""
import sys, os, json, glob
import numpy as np


def find_npz(src):
    if os.path.isfile(src) and src.endswith(".npz"):
        return src, os.path.join(os.path.dirname(src), "config.json")
    if os.path.isdir(src):
        npz = glob.glob(os.path.join(src, "**", "weights.npz"), recursive=True)
        if npz:
            cfg = glob.glob(os.path.join(src, "**", "config.json"), recursive=True)
            return npz[0], (cfg[0] if cfg else None)
    raise SystemExit(f"no weights.npz under {src}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    npz_path, cfg_path = find_npz(src)
    os.makedirs(out, exist_ok=True)

    from safetensors.numpy import save_file
    z = np.load(npz_path)
    tensors = {}
    for k in z.files:
        a = np.array(z[k])
        # alignment_heads is metadata, not a model parameter.
        if k == "alignment_heads":
            continue
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        tensors[k] = np.ascontiguousarray(a)
    save_file(tensors, os.path.join(out, "model.safetensors"))
    print(f"wrote {len(tensors)} tensors -> {out}/model.safetensors")

    if cfg_path and os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"wrote config.json (model_type={cfg.get('model_type')})")


if __name__ == "__main__":
    main()
