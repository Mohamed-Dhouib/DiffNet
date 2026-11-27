"""Doc Protocol evaluation on 512x512 patches: per-patch binary P/R/F1/IoU at 0.5, averaged over the patches whose
ground truth contains tampered pixels (F1-score line), plus a threshold sweep."""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from tqdm import tqdm

from infer import autocast_dtype
from model import DocMDetector
from sconf import Config
from torch.utils.data import Dataset
from torch.cuda.amp import autocast
import time

class TamperDataset(Dataset):
    def __init__(self, images_path, masks_path):
        fns = [
            fn for fn in os.listdir(images_path)
            if fn.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        fns.sort()  # deterministic order
        self.elements = [
            (os.path.join(images_path, fn), os.path.join(masks_path, fn))
            for fn in fns
        ]

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        img_path, mask_path = self.elements[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Try same filename; if not found, fallback jpg<->png swap
        try:
            mask = Image.open(mask_path).convert("L")
        except FileNotFoundError:
            if mask_path.lower().endswith(".jpg") or mask_path.lower().endswith(".jpeg"):
                alt = os.path.splitext(mask_path)[0] + ".png"
            else:
                alt = os.path.splitext(mask_path)[0] + ".jpg"
            mask = Image.open(alt).convert("L")

        return {"image_path": img_path, "image": img, "label": mask}


def _prep_sweep_thresholds(thr_min=0.3, thr_max=0.99, thr_step=0.01):
    thr = np.arange(thr_min, thr_max, thr_step)
    bounds = (np.floor(255.0 * thr).astype(np.int32) + 1).clip(0, 256)
    return thr, bounds


def _safe_precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return (tp / denom) if denom > 0 else 0.0


def _safe_recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return (tp / denom) if denom > 0 else 0.0


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _safe_iou(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return (tp / denom) if denom > 0 else 0.0


def _logits_to_prob(pred_logits: torch.Tensor) -> torch.Tensor:
    """
    pred_logits: (B, C, H, W)
    returns prob of foreground: (H, W) for B=1
    """
    if pred_logits.shape[1] == 1:
        # 1-channel logits
        return pred_logits.sigmoid()[0, 0]
    else:
        # 2+ classes
        return F.softmax(pred_logits, dim=1)[0, 1]


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
    max_images=None,
):
    if save_imgs:
        os.makedirs(pred_save_dir, exist_ok=True)
        msk_out = os.path.join(pred_save_dir, "masks")
        os.makedirs(msk_out, exist_ok=True)
    else:
        msk_out = None

    pretrained_path = weights_path if weights_path else cfg_file.pretrained_model_name_or_path
    if not pretrained_path:
        raise ValueError('Evaluation requires a checkpoint (--weights-path)')

    model = DocMDetector(input_size=cfg_file.input_size, pretrained_model_name_or_path=pretrained_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    input_size = cfg_file.input_size
    assert len(input_size) >= 2, f"input_size should be like [512,512], got {input_size}"
    assert input_size[0] == input_size[1], f"Model input must be square, got {input_size}"
    IN = int(input_size[0])

    head = model.encoder.model.offset_head
    threshold_seg = head.external_threshold.item()
    print(f'Learned offset applied: b={head.offset().item():.8f}, '
          f'raw_threshold={head.offset().sigmoid().item():.8f}, '
          f'external_threshold={threshold_seg:.4f} (fixed before training)')
    test_data = TamperDataset(images_path=gt_images_path, masks_path=gt_masks_path)
    if max_images is not None:
        test_data.elements = test_data.elements[:max_images]

    # Macro over patches (exclude GT-empty patches)
    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_iou = 0.0
    n_used = 0
    n_skipped = 0

    # Sweep (exclude GT-empty patches)
    sweep_thr, sweep_bounds = _prep_sweep_thresholds(sweep_thr_min, sweep_thr_max, sweep_thr_step)
    sumP = np.zeros(len(sweep_thr), dtype=np.float64)
    sumR = np.zeros(len(sweep_thr), dtype=np.float64)
    sumF1 = np.zeros(len(sweep_thr), dtype=np.float64)
    sumIoU = np.zeros(len(sweep_thr), dtype=np.float64)
    n_used_sweep = 0

    prepare_input = model.encoder.prepare_input
    encoder_fwd = model.encoder

    use_cuda = (device == "cuda")
    amp_dtype = autocast_dtype(device)
    
    warmup_steps = 10  # discard first N forward timings

    measured_forward_time_s = 0.0
    measured_forward_calls = 0

    n_forward_calls = 0  # total forward calls (warmup + measured)
    with torch.inference_mode():
        for el in tqdm(test_data, desc="Evaluating"):
            im_path = el["image_path"]
            img = el["image"]
            mask_pil = el["label"]

            H, W = img.shape[:2]
            if H != W:
                raise AssertionError(f"Image must be square. Got {H}x{W} for {im_path}")
            if H != IN or W != IN:
                raise AssertionError(f"Expected {IN}x{IN} crops. Got {H}x{W} for {im_path}")

            gt_np_u8 = (np.array(mask_pil) > 0).astype(np.uint8)
            gt_sum = int(gt_np_u8.sum())

            # GT-empty patches are excluded from the metrics
            if gt_sum == 0:
                n_skipped += 1
                if not save_imgs:
                    continue  # skip inference too (doesn't affect metrics)
                # if saving is enabled, we still run inference to write masks

            # Prepare model inputs (no resize/pad; already INxIN)
            im, tg, dct, qt = prepare_input(
                img, mask_pil,
                aug=False, qualitys=[], crop_pad=False,
                image_path_to_compute_qt=im_path
            )

            im = im.unsqueeze(0).to(device)
            dct = dct.unsqueeze(0).to(device)
            qt = qt.unsqueeze(0).to(device)

            # --- Time model forward ONLY (no data prep) ---
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if use_cuda:
                with autocast(dtype=amp_dtype):
                    pred_logits, *extra = encoder_fwd(im, dct, qt)
            else:
                pred_logits, *extra = encoder_fwd(im, dct, qt)

            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            dt = (t1 - t0)

            if n_forward_calls >= warmup_steps:
                measured_forward_time_s += dt
                measured_forward_calls += 1

            n_forward_calls += 1

            prob = _logits_to_prob(pred_logits)  # (H,W)
            full_mask = (prob > threshold_seg).to(torch.bool)  # (H,W) bool on device

            # Optional saving: save prob map as uint8 [0..255]
            mask_vis = (prob.detach().cpu().numpy() * 255.0).round().astype(np.uint8)
            if save_imgs:
                fn = os.path.basename(im_path)
                cv2.imwrite(os.path.join(msk_out, fn), mask_vis)

            # Metrics only if GT has positives
            if gt_sum > 0:
                gt_bool = torch.from_numpy(gt_np_u8.astype(bool)).to(device)

                tp = (full_mask & gt_bool).sum()
                fp = (full_mask & ~gt_bool).sum()
                fn = (~full_mask & gt_bool).sum()

                tp_i, fp_i, fn_i = torch.stack([tp, fp, fn]).cpu().tolist()

                prec_i = _safe_precision(tp_i, fp_i)
                rec_i = _safe_recall(tp_i, fn_i)
                f1_i = _safe_f1(tp_i, fp_i, fn_i)
                iou_i = _safe_iou(tp_i, fp_i, fn_i)

                sum_precision += prec_i
                sum_recall += rec_i
                sum_f1 += f1_i
                sum_iou += iou_i
                n_used += 1

                # Sweep (per-patch then average; exclude GT-empty)
                gt_bool_cpu = gt_np_u8.astype(bool)
                pred_u8 = mask_vis

                fg_vals = pred_u8[gt_bool_cpu]
                bg_vals = pred_u8[~gt_bool_cpu]

                fg_hist = np.bincount(fg_vals, minlength=256)
                bg_hist = np.bincount(bg_vals, minlength=256)

                fg_ge = np.cumsum(fg_hist[::-1])[::-1]
                bg_ge = np.cumsum(bg_hist[::-1])[::-1]

                for j, b in enumerate(sweep_bounds):
                    if b >= 256:
                        tp_s = 0
                        fp_s = 0
                    else:
                        tp_s = int(fg_ge[b])
                        fp_s = int(bg_ge[b])
                    fn_s = gt_sum - tp_s

                    sumP[j] += _safe_precision(tp_s, fp_s)
                    sumR[j] += _safe_recall(tp_s, fn_s)
                    sumF1[j] += _safe_f1(tp_s, fp_s, fn_s)
                    sumIoU[j] += _safe_iou(tp_s, fp_s, fn_s)

                n_used_sweep += 1

    print(f"\n=== Pixel-level (per-patch average; excluding GT-empty) @ threshold {threshold_seg:g} ===")
    print(f" Used patches  : {n_used}")
    print(f" Skipped patches (GT empty): {n_skipped}")

    if n_used == 0:
        print("No patches with positive GT pixels were found. Nothing to average.")
        macro_f1 = 0.0
    else:
        macro_precision = sum_precision / n_used
        macro_recall = sum_recall / n_used
        macro_f1 = sum_f1 / n_used
        macro_iou = sum_iou / n_used

        print(f" Precision : {macro_precision:.4f}")
        print(f" Recall    : {macro_recall:.4f}")
        print(f" F1-score  : {macro_f1:.4f}")
        print(f" IoU       : {macro_iou:.4f}")

    print("\n=== Pixel-level sweep (per-patch average; excluding GT-empty) ===")
    if n_used_sweep == 0:
        print("No patches with positive GT pixels were found. Nothing to sweep-average.")
    else:
        for j, thr in enumerate(sweep_thr):
            precision = sumP[j] / n_used_sweep
            recall = sumR[j] / n_used_sweep
            f1 = sumF1[j] / n_used_sweep
            iou = sumIoU[j] / n_used_sweep

            print(f"\n=== Pixel-level Metrics @ Threshold {thr:.4f} (per-patch avg) ===")
            print(f" IoU      : {iou*100:5.2f}%")
            print(f" Precision: {precision*100:5.2f}%")
            print(f" Recall   : {recall*100:5.2f}%")
            print(f" F1       : {f1*100:5.2f}%")


    print("\n=== Forward timing (model only) ===")
    print(f" Total forward calls : {n_forward_calls}")
    print(f" Warmup discarded    : {min(warmup_steps, n_forward_calls)}")

    if measured_forward_calls == 0:
        print("No measured forward calls (dataset too small / warmup too large).")
    else:
        avg_ms = (measured_forward_time_s / measured_forward_calls) * 1000.0
        print(f" Measured calls      : {measured_forward_calls}")
        print(f" Avg forward time    : {avg_ms:.3f} ms")

    return macro_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--gt-images-path", required=True, help="Path to gt images (512x512 crops)")
    parser.add_argument("--gt-masks-path", required=True, help="Path to gt masks (512x512 crops)")

    parser.add_argument("--config-path", required=True, help="Training config (gives input_size).")

    parser.add_argument("--pred-save-dir", default="", help="Where to save masks if --save-images is set.")
    parser.add_argument("--weights-path", default=None, help="DiffNet weights (training checkpoint or exported file).")
    parser.add_argument("--save-images", action="store_true", help="Enable saving prediction masks to disk.")
    parser.add_argument('--max-images', type=int, default=None, help='Evaluate only the first N images.')

    args = parser.parse_args()
    cfg_file = Config(args.config_path)

    evaluate_and_sweep(
        cfg_file,
        gt_images_path=args.gt_images_path,
        gt_masks_path=args.gt_masks_path,
        pred_save_dir=args.pred_save_dir,
        save_imgs=args.save_images,
        weights_path=args.weights_path,
        max_images=args.max_images,
    )
