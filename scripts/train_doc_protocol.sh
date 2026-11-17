#!/usr/bin/env bash
# Train DiffNet under the Doc Protocol on one GPU.
#   DOC_DATA=<doc_protocol data root> [OUTPUT_DIR=./result] [SEED=2027] [GPU=0] [NUM_WORKERS=N] [RESUME=<last.ckpt>] \
#     bash scripts/train_doc_protocol.sh
set -euo pipefail
: "${DOC_DATA:?set DOC_DATA to the Doc Protocol data root (contains DocTamperV1-TrainingSet_extracted/)}"
DOC_DATA=$(realpath "${DOC_DATA}")
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-./result}")
export SEED="${SEED:-2027}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"   # the recipe is single-GPU (batch 32)
export DATASET_NAME_OR_PATH="[\"${DOC_DATA}/DocTamperV1-TrainingSet_extracted/images\", \"${DOC_DATA}/DocTamperV1-TrainingSet_extracted/masks\"]"
[ -n "${RESUME:-}" ] && RESUME=$(realpath "${RESUME}")
cd "$(dirname "$0")/../training"
python train.py --config configs/doc_protocol/diffnet.yaml --exp_version diffnet_doc_protocol \
  --result_path "${OUTPUT_DIR}" ${NUM_WORKERS:+--num_workers "${NUM_WORKERS}"} \
  ${RESUME:+--resume_from_checkpoint_path "${RESUME}"}
