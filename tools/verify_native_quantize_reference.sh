#!/bin/bash
# Byte-parity gate for the native quantizer's REFERENCE-SET path (MoE / vision /
# Gemma). The quantized-module set is learned from a 4-bit reference build; the
# reference itself is used ONLY for the module set, so parity is still against a
# fresh `mx.quantize` recomputed from the SAME bf16 source at each module's
# effective precision (read from the produced config). A match proves the Swift
# reference-set path == the canonical op for every module, including the protected
# projectors (8-bit affine) and any 3-D experts (forced affine).
#
# Usage:
#   tools/verify_native_quantize_reference.sh <bf16_source_dir> <reference_4bit_dir> [mode] [extra krill flags...]
#   mode defaults to nvfp4. Extra flags pass straight to `krill quantize`
#   (e.g. --protect o_proj --no-protect-vision).
#
# Example (Gemma 4 12B, both already in the HF cache):
#   BF=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-bf16/snapshots/*/ | head -1)
#   REF=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--gemma-4-12B-it-4bit/snapshots/*/ | head -1)
#   tools/verify_native_quantize_reference.sh "$BF" "$REF" nvfp4
set -uo pipefail
SRC="$1"; REF="$2"; MODE="${3:-nvfp4}"; shift $(( $# >= 3 ? 3 : $# ))
EXTRA=("$@")
case "$MODE" in
  nvfp4) GS=16; BITS=4;;
  mxfp4) GS=32; BITS=4;;
  mxfp8) GS=32; BITS=8;;
  affine) GS=64; BITS=4;;
  *) echo "unknown mode $MODE"; exit 2;;
esac
PY="${KRILL_PY:-$HOME/.krill/venv/bin/python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/.krill/models/blobs/native-quant-ref-paritycheck"

rm -rf "$OUT"
HF_HUB_OFFLINE=1 .build/release/krill quantize "$SRC" \
  --reference "$REF" --mode "$MODE" --group-size "$GS" --bits "$BITS" --dtype fp16 \
  ${EXTRA[@]+"${EXTRA[@]}"} --name native-quant-ref-paritycheck \
  || { echo "GATE: FAIL (quantize errored)"; exit 1; }

"$PY" "$HERE/native_quantize_compare.py" "$SRC" "$OUT"
RC=$?
rm -rf "$OUT"
if [ $RC -eq 0 ]; then echo "GATE: PASS (native reference-set $MODE byte-identical to mx.quantize)"; else echo "GATE: FAIL"; fi
exit $RC
