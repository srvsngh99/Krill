#!/usr/bin/env python3
"""Decompress a compressed-tensors **NVFP4** Gemma-4-12B checkpoint into plain
bf16 MLX-layout safetensors - WITHOUT going through GGUF.

Why this exists
---------------
The community coder fine-tune `gemma-4-12B-coder-fable5-composer2.5` is published
two ways:

  * `*-GGUF`        - llama.cpp k-quants (Q2_K .. Q8_0). Lossy, and not MLX-native.
  * `*-MTP-NVFP4`   - compressed-tensors `nvfp4-pack-quantized` safetensors
                      (e.g. `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4`).

We deliberately do NOT convert from GGUF (its k-quants bake in quality loss the
fine-tuner's original bf16 never had). Instead we take the NVFP4 safetensors -
already a clean 4-bit-FLOAT checkpoint in the same `gemma4_unified` architecture
KrillLM serves natively - and turn it back into bf16. From there
`tools/requant_gemma4_nvfp4.py` re-quantizes it with the proven KrillLM recipe
(uniform nvfp4 + 8-bit-protected attn `o_proj` + 8-bit vision/audio projectors =
the both-axes win), emitting the exact MLX QuantizedLinear layout the Swift
loader expects.

What this tool does
-------------------
1. Reads the single `model.safetensors` at the byte level (the dtypes here -
   F8_E4M3, BF16, U8 - have no numpy equivalent, so we parse the safetensors
   header + mmap the data region and decode each dtype ourselves; numpy only).
2. For every quantized Linear (a `*.weight_packed` U8 tensor with sibling
   `*.weight_scale` F8_E4M3 and `*.weight_global_scale` F32), reconstructs the
   bf16 `.weight`:
       value  = E2M1[code & 0x7] * (-1 if code & 0x8 else +1)     # FP4 E2M1
       weight = value * weight_scale / weight_global_scale         # NVFP4 two-level
   (FP4 LUT, nibble order and the two-level scale formula are taken verbatim
   from vllm-project/compressed-tensors `compressors/nvfp4/helpers.py` and
   `quantization/utils/helpers.py` - see the module docstrings there.)
3. Passes through the genuinely-bf16 tensors (norms, embeddings, layer_scalar,
   biases, positional tables, the un-quantized projectors).
4. Rewrites HF compressed-tensors keys into the MLX/KrillLM key scheme:
       model.language_model.<x>   -> language_model.model.<x>
       model.embed_vision.<x>     -> embed_vision.<x>
       model.embed_audio.<x>      -> embed_audio.<x>
       model.vision_embedder.<x>  -> vision_embedder.<x>
   (matches `mlx-community/gemma-4-12B-it-4bit`, the requant oracle.)
5. Writes sharded bf16 safetensors + `model.safetensors.index.json`, a cleaned
   `config.json` (compressed-tensors `quantization_config` stripped so requant
   sees plain bf16), and copies the tokenizer / chat-template / generation files.

Usage
-----
  # 0. download the NVFP4 source dir (model.safetensors + config + tokenizer)
  #    e.g. via `huggingface-cli download <repo> --local-dir <src>`
  python3 tools/convert_gemma4_compressed_nvfp4_to_bf16.py \
      --src   <downloaded-nvfp4-dir> \
      --out   ~/models/gemma-4-12b-coder-bf16 \
      --self-check                      # validate the decoder before the full run

  # then re-quantize with the proven KrillLM recipe:
  python3 tools/requant_gemma4_nvfp4.py \
      --src-bf16-dir ~/models/gemma-4-12b-coder-bf16 \
      --out          ~/models/gemma-4-12b-coder-nvfp4 \
      --protect o_proj

Pure numpy - no torch, no compressed-tensors, no GGUF. Conversion-time only;
the KrillLM runtime stays pure Swift + MLX.
"""
import argparse
import json
import mmap
import os
import shutil
import struct
import sys

import numpy as np

# ---- FP4 E2M1 magnitude lookup (index 0..7); bit 3 is the sign --------------
# verbatim from compressed-tensors compressors/nvfp4/helpers.py FLOAT_TO_E2M1
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
FP8_E4M3_MAX = 448.0  # finfo(float8_e4m3fn).max; the global-amax block hits this


def build_e4m3fn_lut():
    """256-entry float32 LUT decoding every float8_e4m3fn byte (1-4-3, bias 7,
    finite variant: no inf, NaN only at 0x7F/0xFF)."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b >> 7) & 1 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            val = (m / 8.0) * (2.0 ** (1 - 7))  # subnormal
        elif e == 0xF and m == 0x7:
            val = float("nan")  # the single NaN code
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))  # normal (incl. e==0xF, m<7)
        lut[b] = sign * val
    return lut


E4M3_LUT = build_e4m3fn_lut()


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 raw uint16 bits, round-to-nearest-even."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u32 = x.view(np.uint32)
    # round-to-nearest-even on the truncated low 16 bits
    rounding_bias = ((u32 >> 16) & 1) + np.uint32(0x7FFF)
    u16 = ((u32 + rounding_bias) >> 16).astype(np.uint16)
    # keep NaNs as NaN (preserve a set mantissa bit) - weights shouldn't NaN, but be safe
    nan_mask = np.isnan(x)
    if nan_mask.any():
        u16[nan_mask] = 0x7FC0
    return u16


def bf16_bits_to_f32(u16: np.ndarray) -> np.ndarray:
    """bf16 raw uint16 bits -> float32."""
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


# ---- minimal safetensors reader (byte level; handles F8_E4M3/BF16/U8/F32) ---
class SafeTensorsFile:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        (hlen,) = struct.unpack("<Q", self.mm[:8])
        self.header = json.loads(self.mm[8 : 8 + hlen].decode("utf-8"))
        self.data_start = 8 + hlen
        self.metadata = self.header.pop("__metadata__", None)

    def keys(self):
        return list(self.header.keys())

    def info(self, name):
        return self.header[name]

    def raw(self, name):
        """Raw little-endian bytes of a tensor as a flat numpy view (per dtype)."""
        h = self.header[name]
        s, e = h["data_offsets"]
        buf = self.mm[self.data_start + s : self.data_start + e]
        dt = h["dtype"]
        if dt in ("U8", "I8"):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif dt == "F8_E4M3":
            arr = np.frombuffer(buf, dtype=np.uint8)  # decoded later via LUT
        elif dt == "BF16":
            arr = np.frombuffer(buf, dtype=np.uint16)
        elif dt == "F16":
            arr = np.frombuffer(buf, dtype=np.float16)
        elif dt == "F32":
            arr = np.frombuffer(buf, dtype=np.float32)
        elif dt in ("I32", "U32"):
            arr = np.frombuffer(buf, dtype=np.uint32)
        elif dt in ("I64", "U64"):
            arr = np.frombuffer(buf, dtype=np.uint64)
        else:
            raise ValueError(f"unsupported dtype {dt} for {name}")
        return arr, h["dtype"], tuple(h["shape"])

    def close(self):
        self.mm.close()
        self.f.close()


# ---- minimal sharded safetensors writer (we emit BF16 + passthrough dtypes) -
_NUMPY_TO_ST = {
    np.dtype(np.float32): ("F32", 4),
    np.dtype(np.float16): ("F16", 2),
    np.dtype(np.uint8): ("U8", 1),
    np.dtype(np.uint16): ("BF16", 2),  # we only ever store uint16 as bf16 bits
}


class ShardWriter:
    """Accumulates tensors and flushes ~shard_bytes-sized safetensors shards."""

    def __init__(self, out_dir, shard_bytes):
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        self.buf = {}  # name -> (st_dtype, shape, raw_bytes)
        self.cur = 0
        self.shard_idx = 0
        self.weight_map = {}
        self.total = 0
        self.shards = []

    def add(self, name, st_dtype, shape, raw_bytes):
        self.buf[name] = (st_dtype, shape, raw_bytes)
        self.cur += len(raw_bytes)
        self.total += len(raw_bytes)
        if self.cur >= self.shard_bytes:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        self.shard_idx += 1
        fname = f"model-{self.shard_idx:05d}.safetensors"  # renamed after we know N
        path = os.path.join(self.out_dir, fname)
        header = {}
        offset = 0
        for name, (dt, shape, rb) in self.buf.items():
            header[name] = {"dtype": dt, "shape": list(shape),
                            "data_offsets": [offset, offset + len(rb)]}
            offset += len(rb)
            self.weight_map[name] = fname
        hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(hjson)))
            f.write(hjson)
            for _name, (_dt, _shape, rb) in self.buf.items():
                f.write(rb)
        self.shards.append((fname, path))
        print(f"[convert] wrote {fname} ({len(self.buf)} tensors, "
              f"{self.cur / 1e9:.2f} GB)")
        self.buf = {}
        self.cur = 0

    def finalize(self):
        self.flush()
        n = len(self.shards)
        # rename shards to of-N form and fix the weight_map
        rename = {}
        for i, (fname, path) in enumerate(self.shards, start=1):
            newname = f"model-{i:05d}-of-{n:05d}.safetensors"
            os.rename(path, os.path.join(self.out_dir, newname))
            rename[fname] = newname
        wmap = {k: rename[v] for k, v in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total}, "weight_map": wmap}
        with open(os.path.join(self.out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        return n


# ---- key remap: HF compressed-tensors -> MLX/KrillLM (oracle) scheme --------
def remap_key(k: str) -> str:
    if k.startswith("model."):
        k = k[len("model."):]
    if k.startswith("language_model."):
        k = "language_model.model." + k[len("language_model."):]
    return k


# ---- NVFP4 dequant of one Linear weight -------------------------------------
def dequant_nvfp4(packed_u8, packed_shape, scale_u8, scale_shape, global_scale,
                  group_size=16):
    """Reconstruct an [out, in] float32 weight from compressed-tensors NVFP4.

    packed_u8     : flat uint8, len = out * (in//2)   (two 4-bit codes per byte)
    packed_shape  : (out, in//2)
    scale_u8      : flat uint8 (float8_e4m3 bits), len = out * (in//group_size)
    scale_shape   : (out, in//group_size)
    global_scale  : float32 scalar
    """
    out, half = packed_shape
    n = half * 2
    codes = packed_u8.reshape(out, half)
    low = codes & 0x0F          # even/first column of each pair  -> low nibble
    high = (codes & 0xF0) >> 4  # odd/second column               -> high nibble
    # interleave back to [out, n]: col 2k = low[k], col 2k+1 = high[k]
    idx = np.empty((out, n), dtype=np.uint8)
    idx[:, 0::2] = low
    idx[:, 1::2] = high
    sign = np.where((idx & 0x08) != 0, -1.0, 1.0).astype(np.float32)
    mag = E2M1[(idx & 0x07).astype(np.int64)]
    fp4 = sign * mag  # [out, n] float32, exact grid values

    # per-group fp8 scale -> float32, broadcast over the group
    grp = E4M3_LUT[scale_u8].reshape(scale_shape)  # [out, in//gs]
    grp = np.repeat(grp, group_size, axis=1)       # [out, in]
    # stored scale = global_scale * local_scale  =>  weight = fp4 * scale / global
    w = fp4 * grp / np.float32(global_scale)
    return w  # float32 [out, in]


def self_check(st: SafeTensorsFile):
    """Validate the decoder against the actual stored bytes on one module:
      (a) max decoded fp8 group-scale of a tensor must equal FP8_E4M3_MAX (the
          global-amax block saturates fp8 by construction) -> e4m3 decode ok;
      (b) re-quantizing the dequantized weight reproduces the stored
          weight_packed bytes exactly -> LUT + nibble order + scale formula ok.
    """
    # pick the first quantized module
    mod = None
    for k in st.keys():
        if k.endswith(".weight_packed"):
            mod = k[: -len(".weight_packed")]
            break
    if mod is None:
        print("[self-check] no quantized module found - skipping")
        return True
    print(f"[self-check] module: {mod}")
    packed, _, pshape = st.raw(mod + ".weight_packed")
    scale_u8, _, sshape = st.raw(mod + ".weight_scale")
    gscale_arr, _, _ = st.raw(mod + ".weight_global_scale")
    gscale = float(gscale_arr.reshape(-1)[0])

    grp = E4M3_LUT[scale_u8]
    max_scale = float(np.nanmax(grp))
    ok_a = abs(max_scale - FP8_E4M3_MAX) < 1e-3
    print(f"[self-check] (a) max group-scale = {max_scale:.4f} "
          f"(expect {FP8_E4M3_MAX})  -> {'PASS' if ok_a else 'FAIL'}")

    w = dequant_nvfp4(packed, pshape, scale_u8, sshape, gscale)
    # re-quantize: divide by per-group real scale, snap to nearest E2M1, pack
    out, n = w.shape
    real = (grp.reshape(sshape).astype(np.float32) / np.float32(gscale))
    real = np.repeat(real, 16, axis=1)
    real[real == 0] = 1.0
    ratio = w / real
    diffs = np.abs(np.abs(ratio)[..., None] - E2M1)  # [out, n, 8]
    mag_idx = np.argmin(diffs, axis=-1).astype(np.uint8)
    sign_bit = (ratio < 0).astype(np.uint8) << 3
    code = mag_idx | sign_bit
    # Compare per-nibble codes. The ONLY tolerated difference is signed zero:
    # FP4 0x00 (+0) and 0x08 (-0) both dequantize to 0.0, and numpy encodes
    # -0.0 as +0 (-0.0 < 0 is False). Any mismatch on a non-zero magnitude would
    # mean a wrong LUT / nibble order / scale formula -> real bug.
    o_low = packed & 0x0F
    o_high = (packed & 0xF0) >> 4
    orig = np.empty((out, n), dtype=np.uint8)
    orig[:, 0::2] = o_low.reshape(pshape)
    orig[:, 1::2] = o_high.reshape(pshape)
    mism = orig != code
    both_zero = ((orig & 0x07) == 0) & ((code & 0x07) == 0)
    bad = int((mism & ~both_zero).sum())
    n_signed_zero = int((mism & both_zero).sum())
    ok_b = bad == 0
    print(f"[self-check] (b) re-pack matches stored codes -> "
          f"{'PASS' if ok_b else 'FAIL'} "
          f"({bad} real mismatches, {n_signed_zero} signed-zero-only "
          f"of {orig.size} nibbles)")
    return ok_a and ok_b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="local dir with the NVFP4 model.safetensors + config + tokenizer")
    ap.add_argument("--out", required=True, help="output bf16 dir")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    ap.add_argument("--self-check", action="store_true",
                    help="validate the decoder on one module, then exit")
    args = ap.parse_args()

    src = os.path.expanduser(args.src)
    st_path = os.path.join(src, "model.safetensors")
    if not os.path.isfile(st_path):
        sys.exit(f"no model.safetensors in {src}")
    st = SafeTensorsFile(st_path)

    if args.self_check:
        ok = self_check(st)
        st.close()
        sys.exit(0 if ok else 1)

    # sanity-gate the full run too (cheap)
    if not self_check(st):
        st.close()
        sys.exit("[convert] self-check FAILED - refusing to write garbage weights")

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    writer = ShardWriter(out, int(args.shard_gb * 1e9))

    keys = st.keys()
    quant_mods = {k[: -len(".weight_packed")] for k in keys if k.endswith(".weight_packed")}
    handled = set()
    n_q = n_p = 0
    for mod in sorted(quant_mods):
        packed, _, pshape = st.raw(mod + ".weight_packed")
        scale_u8, _, sshape = st.raw(mod + ".weight_scale")
        gscale_arr, _, _ = st.raw(mod + ".weight_global_scale")
        gscale = float(gscale_arr.reshape(-1)[0])
        w = dequant_nvfp4(packed, pshape, scale_u8, sshape, gscale)  # f32 [out,in]
        writer.add(remap_key(mod + ".weight"), "BF16", w.shape, f32_to_bf16_bits(w).tobytes())
        handled.update({mod + ".weight_packed", mod + ".weight_scale",
                        mod + ".weight_global_scale"})
        n_q += 1
        del w, packed, scale_u8

    # passthrough everything else (norms, embeds, biases, layer_scalar, ...)
    for k in keys:
        if k in handled:
            continue
        arr, dt, shape = st.raw(k)
        if dt == "BF16":
            writer.add(remap_key(k), "BF16", shape, arr.tobytes())
        elif dt == "F32":
            # keep precision tables as bf16 to match an MLX bf16 checkpoint
            writer.add(remap_key(k), "BF16", shape, f32_to_bf16_bits(arr).tobytes())
        elif dt == "F16":
            writer.add(remap_key(k), "F16", shape, arr.tobytes())
        else:
            # leftover quant sidecars already handled; anything else is unexpected
            print(f"[convert] WARNING skipping unexpected tensor {k} ({dt})")
            continue
        n_p += 1
    n_shards = writer.finalize()
    print(f"[convert] dequantized {n_q} Linear weights, passed through {n_p} tensors, "
          f"{n_shards} shard(s)")

    # cleaned config.json (strip the compressed-tensors quant block)
    cfg_path = os.path.join(src, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.pop("quantization_config", None)
    cfg["torch_dtype"] = "bfloat16"
    cfg["dtype"] = "bfloat16"
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)

    # copy tokenizer / chat-template / generation files
    for fn in os.listdir(src):
        if fn.endswith(".safetensors") or fn in ("config.json",
                                                 "model.safetensors.index.json"):
            continue
        srcf = os.path.join(src, fn)
        if os.path.isfile(srcf):
            shutil.copy2(srcf, os.path.join(out, fn))
    st.close()
    print(f"[convert] DONE -> {out}\n"
          f"  next: python3 tools/requant_gemma4_nvfp4.py --src-bf16-dir {out} "
          f"--out <nvfp4-dir> --protect o_proj")


if __name__ == "__main__":
    main()
