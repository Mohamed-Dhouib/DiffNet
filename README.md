# DiffNet

[![arXiv](https://img.shields.io/badge/arXiv-2606.22285-b31b1b.svg)](https://arxiv.org/abs/2606.22285)
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-0073CF.svg)](https://doi.org/10.1007/978-3-032-37196-6_7)

**Efficient Document Tampering Localization with Multi-Level Discrepancy Features and Unified DCT–Quantization Embedding**

This repository includes the official code for training, evaluation, and inference for the DiffNet model.

To reproduce the paper results, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

The installation commands run from the repository root; all other commands run from the `training/` folder. Replace the example data paths with your own.

## Installation

```bash
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -U openmim
MMCV_WITH_OPS=1 pip install mmcv==2.1.0 --no-binary mmcv --no-build-isolation
pip install -r requirements.txt
```

## Pretrained weights

Download the weights from [Hugging Face](https://huggingface.co/MohamedDhouib1/DiffNet).
Save the files in a `weights/` folder at the repository root:

- `doc_protocol/diffnet_doc_protocol.safetensors` → `weights/diffnet_doc_protocol.safetensors` (512×512 input).
- `tdoc_protocol/diffnet_tdoc_protocol.safetensors` → `weights/diffnet_tdoc_protocol.safetensors` (1024×1024 input).

To train without a DiffNet checkpoint, first download the ImageNet initialisation weights:

```bash
python models/diffnet/download_weights.py
```

## Data format

Put images and masks in separate folders:

```text
/data/train/
  images/
    document.jpg
  masks/
    document.png
```

A mask has the same name and size as its image (a `.png` mask for a `.jpg` image is accepted).
Masks are binary: 0 = unchanged, 255 (or 1) = tampered.

## Training

To train on your own data:

```bash
DATASET_NAME_OR_PATH='["/data/train/images", "/data/train/masks"]' \
  python train.py --config configs/doc_protocol/diffnet.yaml --exp_version custom \
  --result_path ../result --train_batch_size 8
```

This command uses the ImageNet weights downloaded above. Set `PRETRAINED_PATH` to a DiffNet checkpoint to start
from those weights instead. Checkpoints are saved in `../result/diffnet/custom_seed2025/checkpoints/`.
To resume, add `--resume_from_checkpoint_path ../result/diffnet/custom_seed2025/checkpoints/last.ckpt`.

## Configuration

The YAML files in `training/configs/` set the data paths, input size, batch size, learning rate, augmentation,
and other training settings. Edit the YAML or pass `--<key> <value>` to `train.py`, such as
`--train_batch_size 8` or `--lr 0.00005`. Paths in the config are relative to the working directory.
The resolved config is saved as `config.yaml` beside the run's `checkpoints/` folder.

For direct training calls, `DATASET_NAME_OR_PATH`, `PRETRAINED_PATH`, and `SEED` override the corresponding
config values. `NUM_NODES` defaults to 1; training uses all visible GPUs.

Ready-made launchers for the paper setups are in `scripts/` (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

## Evaluation

For a folder of 512×512 test images and matching masks:

```bash
python eval_doc_protocol.py \
  --config-path configs/doc_protocol/diffnet.yaml \
  --weights-path ../weights/diffnet_doc_protocol.safetensors \
  --gt-images-path /data/test/images --gt-masks-path /data/test/masks
```

For full images with the 1024×1024 weights:

```bash
python eval_tdoc_protocol.py \
  --config-path configs/tdoc_protocol/diffnet.yaml \
  --weights-path ../weights/diffnet_tdoc_protocol.safetensors \
  --gt-images-path /data/test/images --gt-masks-path /data/test/masks
```

Add `--save-images --pred-save-dir ../predictions` to save predictions.
`--weights-path` accepts a training checkpoint or a released weights file.

## Inference

Run on one image or pass a folder to `--input`:

```bash
python infer.py --weights ../weights/diffnet_doc_protocol.safetensors \
  --input /data/images/document.jpg --output-dir ../predictions
```

`--input-size` defaults to 512. Use `--input-size 1024` with `diffnet_tdoc_protocol.safetensors`.
`--threshold` defaults to 0.5.
Each image produces `<name>_prob.png` and `<name>_mask.png`. The command also prints an image-level tampering
probability.

## Citation

```bibtex
@inproceedings{dhouib2026efficient,
  title={Efficient Document Tampering Localization with Multi-level Discrepancy Features and Unified DCT--Quantization Embedding},
  author={Dhouib, Mohamed and Zhu, Ye and Vanier, Sonia and Shabou, Aymen},
  booktitle={European Conference on Computer Vision},
  pages={110--127},
  year={2026},
  organization={Springer}
}
```
