"""Run DiffNet on an image or a folder of images.

The image is processed as in the evaluation scripts: an image no larger than the model input is padded with white
(centered) to input_size; a larger one is covered by non-overlapping input_size windows (the last window of a
row/column is shifted inside the image) and the per-pixel maximum is kept. JPEG inputs use their own quantization
table. Writes <name>_prob.png (tampering probability x 255) and <name>_mask.png (prob > --threshold, 0/255), and
prints the image-level tampering probability (max over windows).

  python infer.py --weights diffnet_doc_protocol.safetensors --input <image-or-folder> --output-dir <out>
      [--input-size 512] [--threshold 0.5] [--device cpu]
"""
import argparse
import contextlib
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import DocMDetector

EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def autocast_dtype(device):
    """Autocast dtype: bf16 on Ampere and newer GPUs, fp16 on older GPUs, None (no autocast) on CPU."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


@torch.inference_mode()
def predict(model, img_bgr, image_path, size, device):
    """(H, W) float32 tampering probability map (on device) and the image-level probability of one BGR uint8 image."""
    prepare_input = model.encoder.prepare_input
    amp_dtype = autocast_dtype(device)
    H, W = img_bgr.shape[:2]
    pad_top = pad_left = 0
    if H < size or W < size:
        pad_top, pad_left = max(size - H, 0) // 2, max(size - W, 0) // 2
        img_bgr = cv2.copyMakeBorder(img_bgr, pad_top, max(size - H, 0) - pad_top, pad_left,
                                     max(size - W, 0) - pad_left, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    Hp, Wp = img_bgr.shape[:2]
    prob = torch.zeros((Hp, Wp), dtype=torch.float32, device=device)
    cls_p = None
    for top0 in range(0, Hp, size):
        bottom = min(top0 + size, Hp)
        top = bottom - size
        for left0 in range(0, Wp, size):
            right = min(left0 + size, Wp)
            left = right - size
            crop = np.ascontiguousarray(img_bgr[top:bottom, left:right])
            mask = Image.fromarray(np.zeros((size, size), dtype=np.uint8))
            im, _, dct, qt = prepare_input(crop, mask, aug=False, qualitys=[], crop_pad=False,
                                           image_path_to_compute_qt=image_path)
            im, dct, qt = im[None].to(device), dct[None].to(device), qt[None].to(device)
            with torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else contextlib.nullcontext():
                logits, cls_logits = model.encoder(im, dct, qt)
            window = prob[top:bottom, left:right]
            torch.maximum(window, F.softmax(logits, dim=1)[0, 1], out=window)
            p = F.softmax(cls_logits, dim=1)[:, 1]
            cls_p = p if cls_p is None else torch.maximum(cls_p, p)
    return prob[pad_top:pad_top + H, pad_left:pad_left + W], float(cls_p.item())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="exported .safetensors / .pth, or a training checkpoint")
    ap.add_argument("--input", required=True, help="image file or folder")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--input-size", type=int, default=512, help="model input (512: Doc Protocol, 768: TDoc)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = DocMDetector(input_size=[args.input_size, args.input_size], pretrained_model_name_or_path=args.weights)
    model.to(device).eval()

    if os.path.isdir(args.input):
        paths = sorted(os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(EXTENSIONS))
    else:
        paths = [args.input]
    os.makedirs(args.output_dir, exist_ok=True)
    for path in paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip {path}: unreadable")
            continue
        prob, cls_p = predict(model, img, path, args.input_size, device)
        prob = prob.cpu().numpy()
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_prob.png"), (prob * 255.0).round().astype(np.uint8))
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_mask.png"),
                    (prob > args.threshold).astype(np.uint8) * 255)
        print(f"{path}\timage_prob={cls_p:.4f}\ttampered_pixels={(prob > args.threshold).mean():.4%}")


if __name__ == "__main__":
    main()
