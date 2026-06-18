#!/bin/bash
# Byte-parity gate for the native quantizer's FLOAT formats (nvfp4 / mxfp4 / mxfp8).
# There is no external dense float-format reference, so this synthesizes the
# reference from the SAME bf16 source via Python `mx.quantize` - the canonical MLX
# op mlx_lm uses - applying the SAME layer selection the Swift quantizer does, then
# byte-compares. A match proves the Swift float-format path == the canonical op.
#
# Usage:
#   tools/verify_native_quantize_nvfp4.sh <bf16_source_dir> [mode] [group] [bits]
#   mode defaults to nvfp4 (group 16, bits 4); mxfp4 -> group 32; mxfp8 -> group 32 bits 8.
set -uo pipefail
SRC="$1"; MODE="${2:-nvfp4}"; GS="${3:-16}"; BITS="${4:-4}"
PY="${KRILL_PY:-$HOME/.krillm/venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/.krillm/models/blobs/native-quant-nvfp4-paritycheck"

rm -rf "$OUT"
HF_HUB_OFFLINE=1 .build/release/krillm quantize "$SRC" \
  --mode "$MODE" --group-size "$GS" --bits "$BITS" --dtype fp16 \
  --name native-quant-nvfp4-paritycheck || { echo "GATE: FAIL (quantize errored)"; exit 1; }

"$PY" "$HERE/native_quantize_compare.py" "$SRC" "$OUT" "$MODE" "$GS" "$BITS"
RC=$?
rm -rf "$OUT"
if [ $RC -eq 0 ]; then echo "GATE: PASS (native $MODE byte-identical to mx.quantize)"; else echo "GATE: FAIL"; fi
exit $RC
