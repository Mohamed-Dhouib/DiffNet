---
license: cc-by-nc-4.0
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - document-forensics
  - tampering-localization
  - image-forensics
---

# DiffNet

DiffNet localises tampered regions in document images. Given a document image, it predicts a per-pixel tampering
probability map and an image-level tampering probability.

Paper: [arXiv:2606.22285](https://arxiv.org/abs/2606.22285) · Code: <https://github.com/Mohamed-Dhouib/DiffNet>

## Weights

One folder per training protocol of the paper. Each folder contains the weights and the training config.

| Folder | Protocol | Inference input size |
|---|---|---|
| `doc_protocol/` | Doc Protocol: trained on the DocTamper training set (512x512); tested on the DocTamper test sets and on T-SROIE, OSTF, Tampered-IC13, RealTextManipulation | 512 |
| `tdoc_protocol/` | Syn2Real-TDoc protocol: trained on TDoc-2.8M (768x768) from the `doc_protocol/` weights; tested on RTM, FindItAgain, FindIt | 768 |


## Inference

The model code is in the GitHub repository (installation details in its README):

```bash
git clone https://github.com/Mohamed-Dhouib/DiffNet DiffNet
cd DiffNet
pip install -r requirements.txt      # after torch and mmcv, see the README
cd training
python infer.py --weights <path>/diffnet_doc_protocol.safetensors --input <image-or-folder> \
    --output-dir <output-folder> --input-size 512
```

Use `--input-size 768` with the `tdoc_protocol/` weights. Inference also runs on CPU (`--device cpu`). Larger
images are processed with non-overlapping windows of the input size, smaller ones are padded with white. For each
image, `infer.py` writes `<name>_prob.png` (probability x 255) and `<name>_mask.png` (probability > `--threshold`,
default 0.5, 0/255) and prints the image-level probability.

Training and evaluation instructions: `README.md` and `REPRODUCIBILITY.md` of the GitHub repository. Results are
reported in the paper.

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

## Licence

The weights are released under CC BY-NC 4.0 (they are initialised from the ConvNeXt-V2 ImageNet checkpoints,
which are CC BY-NC 4.0). The code on GitHub is MIT.
