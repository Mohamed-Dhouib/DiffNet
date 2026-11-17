#!/usr/bin/env bash
# Evaluate one checkpoint on the 7 Doc Protocol test sets, then print the table.
#   DOC_DATA=<doc_protocol data root> WEIGHTS=<epoch30.pth | exported .safetensors> LOG_DIR=<dir> [GPU=0] \
#     bash scripts/eval_doc_protocol.sh
set -euo pipefail
: "${DOC_DATA:?}"; : "${WEIGHTS:?}"; : "${LOG_DIR:?}"
DOC_DATA=$(realpath "${DOC_DATA}"); WEIGHTS=$(realpath "${WEIGHTS}"); LOG_DIR=$(realpath -m "${LOG_DIR}")
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p "${LOG_DIR}"
cd "$(dirname "$0")/../training"
for root in \
  "${DOC_DATA}/DocTamperV1/DocTamperV1-TestingSet" \
  "${DOC_DATA}/DocTamperV1/DocTamperV1-FCD" \
  "${DOC_DATA}/DocTamperV1/DocTamperV1-SCD" \
  "${DOC_DATA}/cutted_datasets_fakes/T-SROIE_test" \
  "${DOC_DATA}/cutted_datasets_fakes/OSTF_test" \
  "${DOC_DATA}/cutted_datasets_fakes/Tampered-IC13_test" \
  "${DOC_DATA}/cutted_datasets_fakes/RealTextManipulation_test"; do
  name=$(basename "${root}")
  echo "== ${name}"
  python eval_doc_protocol.py --config-path configs/doc_protocol/diffnet.yaml --weights-path "${WEIGHTS}" \
    --gt-images-path "${root}/images" --gt-masks-path "${root}/masks" > "${LOG_DIR}/${name}.log"
  grep "F1-score" "${LOG_DIR}/${name}.log"
done
python doc_protocol_table.py --logs "${LOG_DIR}"
