#!/usr/bin/env bash
# Train DiffNet under the Syn2Real-TDoc protocol: 8 GPUs (one node, or NUM_NODES=2 x 4) x 8 images = global batch 64 (DDP), starting
# from the Doc-Protocol (DocTamper-trained) DiffNet weights.
#   TDOC_DATA=<dir with images/ and masks/ of TDoc-2.8M> INIT=<Doc-Protocol weights, e.g. exported .safetensors> \
#     [OUTPUT_DIR=./result] [SEED=2025] [NUM_WORKERS=N] [NUM_NODES=1] [RESUME=<last.ckpt>] bash scripts/train_tdoc_protocol.sh
set -euo pipefail
: "${TDOC_DATA:?set TDOC_DATA to the TDoc-2.8M folder (contains images/ and masks/)}"
TDOC_DATA=$(realpath "${TDOC_DATA}")
OUTPUT_DIR=$(realpath -m "${OUTPUT_DIR:-./result}")
export SEED="${SEED:-2025}"
export DATASET_NAME_OR_PATH="[\"${TDOC_DATA}/images\", \"${TDOC_DATA}/masks\"]"
if [ -z "${RESUME:-}" ]; then
  : "${INIT:?set INIT to the Doc-Protocol DiffNet weights (training starts from them)}"
  export PRETRAINED_PATH=$(realpath "${INIT}")
fi
n=$(python -c "import torch; print(torch.cuda.device_count())")
[ $((n * ${NUM_NODES:-1})) = 8 ] || echo "WARNING: ${n} GPUs x ${NUM_NODES:-1} node(s); the protocol's global batch 64 assumes 8 GPUs (train_batch_size 8 per GPU)" >&2
[ -n "${RESUME:-}" ] && RESUME=$(realpath "${RESUME}")
cd "$(dirname "$0")/../training"
python train.py --config configs/tdoc_protocol/diffnet.yaml --exp_version diffnet_tdoc_protocol \
  --result_path "${OUTPUT_DIR}" ${NUM_WORKERS:+--num_workers "${NUM_WORKERS}"} \
  ${RESUME:+--resume_from_checkpoint_path "${RESUME}"}
