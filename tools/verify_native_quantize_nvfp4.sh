#!/bin/bash
# Byte-parity gate for the native quantizer's FLOAT formats (nvfp4 / mxfp4 / mxfp8).
# There is no external dense float-format reference, so this synthesizes the
# reference from the SAME bf16 source via Python `mx.quantize` - the canonical MLX
# op mlx_lm uses - applying the SAME layer selection the Swift quantizer does, then
# byte-compares. A match proves the Swift float-format path == the canonical op.
#
# Usage:
#   tools/verify_native_quantize_nvfp4.sh <bf16_source_dir> [mode]
#   mode defaults to nvfp4. The float formats have exactly one valid shape each, so
#   group/bits are derived from the mode (mirrors CheckpointQuantizer.effectiveParams)
#   and passed identically to the Swift quantizer and the Python comparer.
set -uo pipefail
SRC="$1"; MODE="${2:-nvfp4}"
case "$MODE" in
  nvfp4) GS=16; BITS=4;;
  mxfp4) GS=32; BITS=4;;
  mxfp8) GS=32; BITS=8;;
  *) echo "this gate is for the float formats (nvfp4/mxfp4/mxfp8); use verify_native_quantize_parity.sh for affine"; exit 2;;
esac
PY="${KRILL_PY:-$HOME/.krill/venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/.krill/models/blobs/native-quant-nvfp4-paritycheck"

rm -rf "$OUT"
HF_HUB_OFFLINE=1 .build/release/krill quantize "$SRC" \
  --mode "$MODE" --group-size "$GS" --bits "$BITS" --dtype fp16 \
  --name native-quant-nvfp4-paritycheck || { echo "GATE: FAIL (quantize errored)"; exit 1; }

"$PY" "$HERE/native_quantize_compare.py" "$SRC" "$OUT" "$MODE" "$GS" "$BITS"
RC=$?
rm -rf "$OUT"
if [ $RC -eq 0 ]; then echo "GATE: PASS (native $MODE byte-identical to mx.quantize)"; else echo "GATE: FAIL"; fi
exit $RC
