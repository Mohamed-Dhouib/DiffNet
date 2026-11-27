#!/usr/bin/env bash
# Evaluation on the Syn2Real-TDoc test sets (full images, sliding window, global pixel metrics).
#   TDOC_EVAL=<dir with rtm/ finditagain/ findit/, each images/ + masks/> WEIGHTS=<checkpoint> LOG_DIR=<dir> [GPU=0] \
#     bash scripts/eval_tdoc_protocol.sh
set -euo pipefail
: "${TDOC_EVAL:?}"; : "${WEIGHTS:?}"; : "${LOG_DIR:?}"
TDOC_EVAL=$(realpath "${TDOC_EVAL}"); WEIGHTS=$(realpath "${WEIGHTS}"); LOG_DIR=$(realpath -m "${LOG_DIR}")
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p "${LOG_DIR}"
cd "$(dirname "$0")/../training"
for name in rtm finditagain findit; do
  echo "== ${name}"
  python eval_tdoc_protocol.py --config-path configs/tdoc_protocol/diffnet.yaml --weights-path "${WEIGHTS}" \
    --gt-images-path "${TDOC_EVAL}/${name}/images" --gt-masks-path "${TDOC_EVAL}/${name}/masks" > "${LOG_DIR}/${name}.log"
  sed -n '/=== Pixel-level (global) ===/,/IoU/p; /=== Image-level === for 0.5$/,/F1-score/p' "${LOG_DIR}/${name}.log"
done
