"""Syn2Real-TDoc protocol evaluation: full images, non-overlapping sliding window of input_size (the last
window of a row/column is shifted inside the image), smaller images padded with white; pixel metrics from
global TP/FP/FN over the test set, image-level metrics from the max image-class probability over the windows."""
import argparse
import os
import numpy as np
import torch
import cv2
from PIL import Image
from tqdm import tqdm

from infer import predict
from model import DocMDetector
from sconf import Config
from torch.utils.data import Dataset


class TamperDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.elements = [
            (os.path.join(images_path, fn), os.path.join(masks_path, fn))
            for fn in os.listdir(images_path)
            if fn.lower().endswith((".png", ".jpg"))
        ]

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        img_path, mask_path = self.elements[idx]
        img = cv2.imread(img_path)
        try:
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            mask = Image.open(mask_path.replace(".jpg", ".png")).convert("L")
        return {"image_path": img_path, "image": img, "label": mask}


def _prep_sweep_thresholds(thr_min=0.3, thr_max=0.99, thr_step=0.01):
    thr = np.arange(thr_min, thr_max, thr_step)
    # uint8 map: pred > 255 * thr  <=>  pred >= floor(255 * thr) + 1
    bounds = (np.floor(255.0 * thr).astype(np.int32) + 1).clip(0, 256)
    return thr, bounds


def evaluate_and_sweep(
    cfg_file,
    gt_images_path,
    gt_masks_path,
    pred_save_dir="",
    save_imgs=False,
    weights_path=None,
    sweep_thr_min=0.01,
    sweep_thr_max=1.0,
    sweep_thr_step=0.01,
):
    if save_imgs:
        os.makedirs(pred_save_dir, exist_ok=True)
        msk_out = os.path.join(pred_save_dir, "masks")
        os.makedirs(msk_out, exist_ok=True)
    else:
        msk_out = None

    max_resolution = 0

    pretrained_path = weights_path if weights_path else cfg_file.pretrained_model_name_or_path
    if not pretrained_path:
        raise ValueError('Evaluation requires a checkpoint (--weights-path)')
    model = DocMDetector(input_size=cfg_file.input_size, pretrained_model_name_or_path=pretrained_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    threshold_seg = 0.5
    range_threshold_cls = [round(x, 2) for x in np.arange(0.2, 0.9, 0.05)]
    IN = cfg_file.input_size[0]

    test_data = TamperDataset(images_path=gt_images_path, masks_path=gt_masks_path)

    pixel_tp = pixel_fp = pixel_tn = pixel_fn = 0

    # Image-level cls metrics
    img_tp = {t: 0 for t in range_threshold_cls}
    img_fp = {t: 0 for t in range_threshold_cls}
    img_tn = {t: 0 for t in range_threshold_cls}
    img_fn = {t: 0 for t in range_threshold_cls}

    sweep_thr, sweep_bounds = _prep_sweep_thresholds(sweep_thr_min, sweep_thr_max, sweep_thr_step)
    TP = np.zeros(len(sweep_thr), dtype=np.int64)
    FP = np.zeros(len(sweep_thr), dtype=np.int64)
    FN = np.zeros(len(sweep_thr), dtype=np.int64)
    TN = np.zeros(len(sweep_thr), dtype=np.int64)

    with torch.inference_mode():
        for idx, el in enumerate(tqdm(test_data, desc="Evaluating")):
            im_path = el["image_path"]
            orig_img = el["image"]
            orig_mask = el["label"]

            H, W = orig_img.shape[:2]
            max_resolution = max(max_resolution, max(H, W))

            gt_np_u8 = (np.array(orig_mask) > 0).astype(np.uint8)
            gt_tensor = torch.from_numpy(gt_np_u8).to(device)
            gt_bool = gt_tensor.bool()

            seg_prob_to_save, cls_p_val = predict(model, orig_img, im_path, IN, device)
            full_mask = seg_prob_to_save > threshold_seg
            img_preds = {t: int(cls_p_val > t) for t in range_threshold_cls}

            if save_imgs:
                fn = os.path.basename(im_path)
                mask_vis = (seg_prob_to_save.cpu().numpy() * 255.0).round().astype(np.uint8)
                cv2.imwrite(os.path.join(msk_out, fn), mask_vis)
            else:
                mask_vis = (seg_prob_to_save.cpu().numpy() * 255.0).round().astype(np.uint8)

            pred_bool = full_mask.bool()
            tp = (pred_bool & gt_bool).sum()
            fp = (pred_bool & ~gt_bool).sum()
            tn = (~pred_bool & ~gt_bool).sum()
            fn = (~pred_bool & gt_bool).sum()
            tp_i, fp_i, tn_i, fn_i = torch.stack([tp, fp, tn, fn]).cpu().tolist()

            pixel_tp += tp_i
            pixel_fp += fp_i
            pixel_tn += tn_i
            pixel_fn += fn_i

            # Image-level cls counts
            img_gt = int(gt_bool.any().item())
            for key in range_threshold_cls:
                pred_img = img_preds[key]
                if img_gt == 1 and pred_img == 1:
                    img_tp[key] += 1
                elif img_gt == 1 and pred_img == 0:
                    img_fn[key] += 1
                elif img_gt == 0 and pred_img == 1:
                    img_fp[key] += 1
                else:
                    img_tn[key] += 1

            gt_bool_cpu = gt_np_u8.astype(bool)  # (H,W)
            pred_u8 = mask_vis  # uint8

            fg_vals = pred_u8[gt_bool_cpu]
            bg_vals = pred_u8[~gt_bool_cpu]

            fg_hist = np.bincount(fg_vals, minlength=256)
            bg_hist = np.bincount(bg_vals, minlength=256)

            fg_ge = np.cumsum(fg_hist[::-1])[::-1]  # fg_ge[i] = sum_{v>=i} fg_hist[v]
            bg_ge = np.cumsum(bg_hist[::-1])[::-1]

            gt_sum = int(gt_bool_cpu.sum())
            bg_sum = int((~gt_bool_cpu).sum())

            for j, b in enumerate(sweep_bounds):
                if b >= 256:
                    tp_s = 0
                    fp_s = 0
                else:
                    tp_s = int(fg_ge[b])
                    fp_s = int(bg_ge[b])
                fn_s = gt_sum - tp_s
                tn_s = bg_sum - fp_s

                TP[j] += tp_s
                FP[j] += fp_s
                FN[j] += fn_s
                TN[j] += tn_s

    p_precision = pixel_tp / (pixel_tp + pixel_fp + 1e-8)
    p_recall = pixel_tp / (pixel_tp + pixel_fn + 1e-8)
    p_f1 = 2 * pixel_tp / (2 * pixel_tp + pixel_fp + pixel_fn + 1e-8)
    p_iou = pixel_tp / (pixel_tp + pixel_fp + pixel_fn + 1e-8)

    print("\n=== Pixel-level (global) ===")
    print(f" Precision : {p_precision:.4f}")
    print(f" Recall    : {p_recall:.4f}")
    print(f" F1-score  : {p_f1:.4f}")
    print(f" IoU       : {p_iou:.4f}")

    img_f1_f = None
    for key in img_tp.keys():
        img_precision = img_tp[key] / (img_tp[key] + img_fp[key] + 1e-8)
        img_recall = img_tp[key] / (img_tp[key] + img_fn[key] + 1e-8)
        img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall + 1e-8)

        print(f"\n=== Image-level === for {key}")
        print(f"Precision (image)   : {img_precision:.4f}")
        print(f"Recall    (image)   : {img_recall:.4f}")
        print(f"F1-score  (image)   : {img_f1:.4f}")
        if key == 0.5:
            img_f1_f = img_f1

    print(f"max resolution is {max_resolution}")

    for j, threshold_mask in enumerate(sweep_thr):
        tp = TP[j]; fp = FP[j]; fn = FN[j]; tn = TN[j]

        iou = tp / (tp + fp + fn + 1e-12)
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)

        print(f"\n=== Pixel-level Metrics @ Threshold {threshold_mask:.4f} ===")
        print(f" TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f" IoU      : {iou*100:5.2f}%")
        print(f" Precision: {precision*100:5.2f}%")
        print(f" Recall   : {recall*100:5.2f}%")
        print(f" F1       : {f1*100:5.2f}%")

    return p_f1, img_f1_f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--gt-images-path", required=True, help="Path to gt images")
    parser.add_argument("--gt-masks-path", required=True, help="Path to gt masks")

    parser.add_argument("--config-path", required=True, help="Training config (gives input_size).")

    parser.add_argument("--pred-save-dir", default="", help="Where to save masks if --save-images is set.")
    parser.add_argument("--weights-path", default=None, help="DiffNet weights (training checkpoint or exported file).")

    parser.add_argument("--save-images", action="store_true", help="Enable saving prediction masks to disk.")

    args = parser.parse_args()
    cfg_file = Config(args.config_path)

    evaluate_and_sweep(
        cfg_file,
        gt_images_path=args.gt_images_path,
        gt_masks_path=args.gt_masks_path,
        pred_save_dir=args.pred_save_dir,
        save_imgs=args.save_images,
        weights_path=args.weights_path,
    )
