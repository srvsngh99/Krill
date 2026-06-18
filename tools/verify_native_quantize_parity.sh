#!/bin/bash
# Byte-exact parity gate for the native Swift+MLX quantizer (krill quantize) vs
# mlx_lm.convert. Quantizes a local bf16 source with `krill quantize` and compares
# every produced tensor against an mlx-community (mlx_lm.convert) 4-bit reference.
#
# Usage:
#   tools/verify_native_quantize_parity.sh <bf16_source_dir> <reference_4bit_dir> [bits] [group]
#
# Example (both already in the HF cache):
#   BF=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--GLM-4-9B-0414-bf16/snapshots/*/ | head -1)
#   REF=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--GLM-4-9B-0414-4bit/snapshots/*/ | head -1)
#   tools/verify_native_quantize_parity.sh "$BF" "$REF"
set -euo pipefail
SRC="$1"; REF="$2"; BITS="${3:-4}"; GROUP="${4:-64}"
PY="${KRILL_PY:-$HOME/.krill/venv/bin/python}"
OUT="$HOME/.krill/models/blobs/native-quant-paritycheck"

rm -rf "$OUT"
HF_HUB_OFFLINE=1 .build/release/krill quantize "$SRC" \
  --bits "$BITS" --group-size "$GROUP" --dtype fp16 --name native-quant-paritycheck

"$PY" - "$REF" "$OUT" <<'PYEOF'
import mlx.core as mx, glob, sys
ref_dir, mine_dir = sys.argv[1], sys.argv[2]
ref, out = {}, {}
for f in sorted(glob.glob(ref_dir + "/*.safetensors")): ref.update(mx.load(f))
for f in sorted(glob.glob(mine_dir + "/*.safetensors")): out.update(mx.load(f))
ok = miss = extra = diff = 0
for k, v in ref.items():
    a = out.get(k)
    if a is None: miss += 1; continue
    if a.shape == v.shape and a.dtype == v.dtype and bool(mx.all(a == v).item()): ok += 1
    else: diff += 1
extra = sum(1 for k in out if k not in ref)
print(f"byte-identical {ok}/{len(ref)}  diff {diff}  missing {miss}  extra {extra}")
sys.exit(0 if ok == len(ref) and diff == 0 and miss == 0 and extra == 0 else 1)
PYEOF
RC=$?
rm -rf "$OUT"
if [ $RC -eq 0 ]; then echo "GATE: PASS (native quantize byte-identical to mlx_lm.convert)"; else echo "GATE: FAIL"; fi
exit $RC
