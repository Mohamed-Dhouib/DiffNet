import os
import random
from typing import Tuple
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
import cv2
from copy import deepcopy                
from tqdm import tqdm



def load_and_check_image_and_mask(
    img_path: str,
    mask_path: str,
    image_resolution: int,
):
    """Load image and mask, validate dimensions, crop if larger than resolution."""
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not load image at '{img_path}'")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask at '{mask_path}'")
    
    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 255}):
        if set(unique_vals).issubset({0, 1}):
            print(f"[mask-fix] {mask_path}: converting {{0,1}} -> {{0,255}}")
            mask = mask * 255
        else:
            raise AssertionError(f"Mask contains values outside {{0,255}}: {unique_vals}  for {img_path} {mask_path}")

    ih, iw = image.shape[:2]
    mh, mw = mask.shape[:2]
    if (ih, iw) != (mh, mw):
        raise AssertionError(f"Image and mask size differ: img={iw}x{ih} vs mask={mw}x{mh} for {img_path} {mask_path}")

    H, W = mh, mw


    if H > image_resolution or W > image_resolution:
        res_h = min(image_resolution,H)
        res_w= min(image_resolution,W)
        total_pos = int((mask == 255).sum())
        if random.random() > 0.75:
            y0 = random.randint(0, H - res_h)
            x0 = random.randint(0, W - res_w)
        else:
            k = min(total_pos, 32)
            if k == 0:
                y0 = random.randint(0, H - res_h)
                x0 = random.randint(0, W - res_w)
            else:
                for i in range(100):
                    y0 = random.randint(0, H - res_h)
                    x0 = random.randint(0, W - res_w)
                    patch = mask[y0 : y0 + res_h, x0 : x0 + res_w]
                    if int((patch == 255).sum()) >= k:
                        break

        image = image[y0 : y0 + res_h, x0 : x0 + res_w]
        mask  = mask [y0 : y0 + res_h, x0 : x0 + res_w]

    ih, iw = image.shape[:2]
    mh, mw = mask.shape[:2]
    if (ih, iw) != (mh, mw):
        raise AssertionError(f"Resulting image and mask size differ: img={iw}x{ih} vs mask={mw}x{mh} for {img_path} {mask_path}")


    return image, mask


class DocMDetectorDataset(Dataset):
    """Dataset for document manipulation detection training."""

    def __init__(
        self,
        dataset_name_or_path: list,
        seg_model: nn.Module,
        split: str = "train",
        config=None,
    ):
        super().__init__()

        self.seg_model = deepcopy(seg_model)
        
        self.seg_model.encoder.model=None
        
        self.split = split

        self.dataset_length = 0
        self.csvs=[]
        self.paths=[]
        self.image_resolution=config.input_size[0]
        
        print(f"model input size is {self.image_resolution}x{self.image_resolution}")   
        
        if type(dataset_name_or_path[0])==str:
            dataset_name_or_paths=[dataset_name_or_path]
        else:
            dataset_name_or_paths=dataset_name_or_path

        

        for dataset_name_or_path in dataset_name_or_paths:
            dirpath=dataset_name_or_path[0]
            maskdirpath=dataset_name_or_path[1]
            
            if split != "train":
                dirpath=dirpath.replace("train","test")
                maskdirpath=maskdirpath.replace("train","test")

            need_to_change_mask_extension=None
            print(split)
            for filename in tqdm(os.listdir(dirpath), desc=f"[{split}] scanning {os.path.basename(dirpath)}"):
                if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                    filepath = os.path.join(dirpath, filename)
                    maskpath = os.path.join(maskdirpath, filename)
                    if need_to_change_mask_extension is None:
                        need_to_change_mask_extension = not os.path.exists(maskpath)
                    if need_to_change_mask_extension:
                        maskpath = os.path.join(maskdirpath, filename.replace(".jpg", ".png"))
           
                    self.paths.append([filepath, maskpath])
                    self.dataset_length += 1
            print(f"for split {split} got {self.dataset_length} images")
            
                    
    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        img_path,mask_path=self.paths[idx]
        img, mask = load_and_check_image_and_mask(img_path,mask_path,image_resolution=self.image_resolution)
        img,mask,dct,qt=self.seg_model.encoder.prepare_input(img,mask,image_path_to_compute_qt=img_path)

        return img, mask, dct, qt
