# Reproducing the paper protocols

Installation and weights: see [README.md](README.md). Commands here run from the repository root.

## Doc Protocol

Train on DocTamper and evaluate on 512×512 patches. Training uses one NVIDIA H100 80GB GPU.

### Data preparation

We used the Doc Protocol data shared through the Google Drive link in
[ForensicHub issue #26](https://github.com/scu-zjz/ForensicHub/issues/26). That link is no longer available; the
steps below rebuild the same data from the official DocTamper release and the ForensicHub cross-domain test sets.

Download the four DocTamper datasets (`DocTamperV1-TrainingSet`, `DocTamperV1-TestingSet`, `DocTamperV1-FCD`,
and `DocTamperV1-SCD`) from the [official release](https://github.com/qcf-568/DocTamper).
Put them in `/data/DocTamper/`. Download the three test compression files
`pks/DocTamperV1-{TestingSet,FCD,SCD}_75.pk` from the same repository into `/data/DocTamper/pks/`.

```bash
python tools/build_doctamper.py TrainingSet --lmdb-root /data/DocTamper --out /data/doc_protocol
for split in TestingSet FCD SCD; do
  python tools/build_doctamper.py "$split" --lmdb-root /data/DocTamper \
    --pks /data/DocTamper/pks --out /data/doc_protocol
done
```

The script prepares the training images and the compressed test images using the supplied compression files.
Each split has `images/` and `masks/` folders.

Download `cutted_datasets_fakes.zip` from
[Jason37437/Doc-Protocol-Data](https://huggingface.co/datasets/Jason37437/Doc-Protocol-Data)
and extract it into `/data/doc_protocol/`. These are the cross-domain test patches from ForensicHub;
please cite ForensicHub when using them.

The required layout is:

```text
/data/doc_protocol/
  DocTamperV1-TrainingSet_extracted/{images,masks}
  DocTamperV1/
    DocTamperV1-TestingSet/{images,masks}
    DocTamperV1-FCD/{images,masks}
    DocTamperV1-SCD/{images,masks}
  cutted_datasets_fakes/
    T-SROIE_test/{images,masks}
    OSTF_test/{images,masks}
    Tampered-IC13_test/{images,masks}
    RealTextManipulation_test/{images,masks}
```

### Training

Download the ImageNet initialisation weights, then launch training:

```bash
python training/models/diffnet/download_weights.py
DOC_DATA=/data/doc_protocol OUTPUT_DIR=./result NUM_WORKERS=22 bash scripts/train_doc_protocol.sh
```

The script uses `training/configs/doc_protocol/diffnet.yaml`: 512×512 inputs, batch size 32, and 30 epochs.
The launcher seed is 2027. Checkpoints are saved in
`result/diffnet/diffnet_doc_protocol_seed2027/checkpoints/`.
The evaluated checkpoint is `epoch30.pth`, exported as `doc_protocol/diffnet_doc_protocol.safetensors` on
Hugging Face. To resume training, set `RESUME` to the run's `last.ckpt`.

### Evaluation

```bash
DOC_DATA=/data/doc_protocol WEIGHTS=weights/diffnet_doc_protocol.safetensors \
  LOG_DIR=./eval_doc bash scripts/eval_doc_protocol.sh
```

To evaluate your own training run, set `WEIGHTS` to
`result/diffnet/diffnet_doc_protocol_seed2027/checkpoints/epoch30.pth`.
The script uses `training/eval_doc_protocol.py` and writes one log per set in `LOG_DIR`.
It then runs `training/doc_protocol_table.py` to summarise the logs.

The in-domain sets are DT-Test, DT-FCD, and DT-SCD. The cross-domain sets are T-SROIE, OSTF, Tampered-IC13,
and RTM. All inputs are 512×512 patches.

The reported metric is pixel F1 at threshold 0.5, computed separately for each patch and then averaged.
Patches with no tampered pixels in the ground-truth mask are excluded.
Use the `F1-score` line in each log or the `F1@0.5` summary row, which reports percentages and the means over
the in-domain and cross-domain sets. The threshold sweep is only for reference.

## Syn2Real-TDoc

Train on TDoc-2.8M and evaluate on full images from RTM, FindItAgain, and FindIt.
Training uses 8 GPUs with 8 images per GPU, for a total batch size of 64.

### Data preparation

Download [TDoc-2.8M](https://huggingface.co/datasets/MohamedDhouib1/TDoc-2.8M) and follow its reconstruction
instructions to obtain `images/` and `masks/`. Set `TDOC_DATA` to their parent folder.

Download the RTM, FindItAgain, and FindIt test sets from the evaluation-data links in the
[TDoc repository](https://github.com/Mohamed-Dhouib/TDOC). Keep the full images and arrange the folders as:

```text
/data/TDoc-2.8M/{images,masks}
/data/tdoc_eval/
  rtm/{images,masks}
  finditagain/{images,masks}
  findit/{images,masks}
```

### Training

Start from the released Doc Protocol weights:

```bash
TDOC_DATA=/data/TDoc-2.8M INIT=weights/diffnet_doc_protocol.safetensors OUTPUT_DIR=./result \
  bash scripts/train_tdoc_protocol.sh
```

The script uses `training/configs/tdoc_protocol/diffnet.yaml`: 768×768 crops and 2 epochs.
The launcher seed is 2025. Use one node with 8 visible GPUs, or two nodes with 4 GPUs each and `NUM_NODES=2`
with a multi-node launcher.

Checkpoints are saved in `result/diffnet/diffnet_tdoc_protocol_seed2025/checkpoints/`.
The evaluated checkpoint is `epoch2.pth`, exported as `tdoc_protocol/diffnet_tdoc_protocol.safetensors` on
Hugging Face. To resume training, set `RESUME` to the run's `last.ckpt`; `INIT` is not required when resuming.

### Evaluation

```bash
TDOC_EVAL=/data/tdoc_eval WEIGHTS=weights/diffnet_tdoc_protocol.safetensors \
  LOG_DIR=./eval_tdoc bash scripts/eval_tdoc_protocol.sh
```

To evaluate your own training run, set `WEIGHTS` to
`result/diffnet/diffnet_tdoc_protocol_seed2025/checkpoints/epoch2.pth`.
The script uses `training/eval_tdoc_protocol.py` and writes one log per test set in `LOG_DIR`.
It handles full images with 768×768 windows and pads smaller images automatically.

Pixel precision, recall, and F1 use counts summed over all pixels in each test set, at threshold 0.5.
Image precision, recall, and F1 use the image score at threshold 0.5. The score is the highest image-level
tampering probability across windows. A ground-truth image is tampered if its mask contains any tampered pixel.
Report pixel and image metrics for each of the three sets, and average F1 across the sets at each level.

## Launcher settings

The scripts in `scripts/` read these environment variables:

- `train_doc_protocol.sh`: `DOC_DATA` is required and must contain `DocTamperV1-TrainingSet_extracted/` with
  `images/` and `masks/`. `GPU` defaults to 0 and `SEED` to 2027.
- `train_tdoc_protocol.sh`: `TDOC_DATA` must contain `images/` and `masks/`. `INIT` sets the starting DiffNet
  weights and is required for a new run. `SEED` defaults to 2025. `NUM_NODES` defaults to 1.
- Both training scripts accept `OUTPUT_DIR` (default `./result`), `NUM_WORKERS` (default 20 from the configs),
  and `RESUME` (path to `last.ckpt`).
- `eval_doc_protocol.sh` requires `DOC_DATA`, `WEIGHTS`, and `LOG_DIR`.
  `eval_tdoc_protocol.sh` requires `TDOC_EVAL`, `WEIGHTS`, and `LOG_DIR`. Both accept `GPU` (default 0).
