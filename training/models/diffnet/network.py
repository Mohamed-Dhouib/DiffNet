"""DiffNet network: RGB-DCT early-fusion backbone, multi-level discrepancy neck, FPN decoder, image-level
classifier and a learned threshold offset on the output logits."""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmseg.models.segmentors import EncoderDecoder
from mmseg.registry import MODELS

from threshold import LearnedOffset
from .backbone import DIMS, RGBDCTBackbone
from .neck import DiscrepancyFPN

NECK_CHANNELS = DIMS[1]     # 256
NUM_SCALES = 4
DROP_PATH_RGB = 0.4         # stochastic depth, linear 0 -> max over the 36 RGB ConvNeXt blocks
DROP_PATH_DCT = 0.2         # ... over the 18 DCT-branch blocks
DROP_PATH_NECK = 0.1        # ... over the 6 refinement blocks
DECODER_DROPOUT = 0.2       # Dropout2d before the decoder's classification conv
EXTERNAL_THRESHOLD = 0.5    # constant probability cut applied to the offset-adjusted logits
OFFSET_BOUND = 6.0          # |learned logit offset| < 6


class Last3DecoderClassifier(nn.Module):
    """Image-level logits from GAP+GMP of the last 3 feature maps and (mean, max, top-5% mean) of the tampered
    probability map."""
    def __init__(
        self,
        in_channels_each=256,
        num_classes=2,
        dropout=0.0,
        topk_ratio=0.05,
        hidden_dim=512,
        num_feature_maps=3,
    ):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        dec_feat_dim = num_feature_maps * (2 * in_channels_each)
        logit_feat_dim = 3
        total_in = dec_feat_dim + logit_feat_dim

        self.topk_ratio = float(topk_ratio)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.GELU(),
            self.dropout,
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, aligned_feats, seg_logits):
        n = min(self.num_feature_maps, len(aligned_feats))
        feats_to_use = aligned_feats[-n:]

        pooled = []
        for f in feats_to_use:
            avg = F.adaptive_avg_pool2d(f, 1)
            mx = F.adaptive_max_pool2d(f, 1)
            v = torch.cat([avg, mx], dim=1).flatten(1)
            pooled.append(v)

        dec_vec = torch.cat(pooled, dim=1)

        prob = seg_logits.softmax(dim=1)
        p_tamp = prob[:, 1]

        p_mean = p_tamp.mean(dim=(1, 2))
        p_max = p_tamp.amax(dim=(1, 2))

        B, H, W = p_tamp.shape
        N = H * W
        k = max(1, int(min(self.topk_ratio, 1.0) * N))
        flat = p_tamp.flatten(1)
        topk_vals, _ = flat.topk(k, dim=1)
        p_topk_mean = topk_vals.mean(dim=1)

        logit_stats = torch.stack([p_mean, p_max, p_topk_mean], dim=1)

        x = torch.cat([dec_vec, logit_stats], dim=1)
        x = self.mlp(x)
        return x


@MODELS.register_module(name='DiffNetBackbone')
class DiffNetBackboneModule(RGBDCTBackbone, BaseModule):
    def __init__(self, init_cfg=None, **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)
        RGBDCTBackbone.__init__(self, **kwargs)


@MODELS.register_module(name='DiffNetNeck')
class DiffNetNeckModule(DiscrepancyFPN, BaseModule):
    def __init__(self, init_cfg=None, **kwargs):
        BaseModule.__init__(self, init_cfg=init_cfg)
        DiscrepancyFPN.__init__(self, **kwargs)


class DiffNet(EncoderDecoder):
    def __init__(self, num_classes: int = 2, load_imagenet_weights: bool = True) -> None:
        backbone_cfg = dict(
            type='DiffNetBackbone',
            pretrained=load_imagenet_weights,
            rgb_drop_path_rate=DROP_PATH_RGB,
            dct_drop_path_rate=DROP_PATH_DCT,
        )
        neck_cfg = dict(
            type='DiffNetNeck',
            in_channels=DIMS,
            out_channels=NECK_CHANNELS,
            drop_path_rate=DROP_PATH_NECK,
            load_pretrained=load_imagenet_weights,
        )
        decode_head_cfg = dict(
            type='FPNHead',
            in_channels=[NECK_CHANNELS] * NUM_SCALES,
            in_index=list(range(NUM_SCALES)),
            feature_strides=[4 * (2 ** i) for i in range(NUM_SCALES)],
            channels=NECK_CHANNELS,
            dropout_ratio=DECODER_DROPOUT,
            num_classes=num_classes,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            align_corners=False,
        )
        super().__init__(
            backbone=backbone_cfg,
            decode_head=decode_head_cfg,
            neck=neck_cfg,
            auxiliary_head=None,
            init_cfg=None,
        )

        self.classifier = Last3DecoderClassifier(
            in_channels_each=NECK_CHANNELS,
            num_classes=num_classes,
            hidden_dim=NECK_CHANNELS,
            num_feature_maps=min(3, NUM_SCALES),
        )
        self.learned_offset = True
        self.offset_head = LearnedOffset(EXTERNAL_THRESHOLD, OFFSET_BOUND)
        self.return_activation_maps = False

    def set_return_activation_maps(self, enabled: bool = True) -> None:
        """Training forwards also return the activation maps of the 4 discrepancy blocks as last element."""
        self.neck.neck.set_return_activation_maps(enabled)
        self.return_activation_maps = bool(enabled)

    def extract_feat(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        x = self.backbone(inputs)
        x = self.neck(x)
        return x

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """inputs: {'x': (B,3,H,W) normalised RGB, 'dct': (B,H,W) clipped |quantized DCT|, 'qtb': (B,1,8,8)
        quantization table, optional 'return_raw'}. Returns (thresholded logits, image logits), plus the raw logits
        with return_raw, plus the activation maps in training once enabled. softmax(thresholded logits)[:, 1] > 0.5
        is the predicted mask."""
        input_size = inputs['x'].shape[2:]
        features = self.extract_feat(inputs)
        maps = self.return_activation_maps and self.training
        if maps:
            features, act_maps = features
        seg_logits = self.decode_head.forward(features)

        seg_logits = F.interpolate(
                seg_logits,
                size=input_size,
                mode='bilinear',
                align_corners=False
            )

        cls_logit = self.classifier(features, seg_logits)

        output_logits = self.offset_head(seg_logits)
        if inputs.get('return_raw', False):
            outputs = (output_logits, cls_logit, seg_logits)
        else:
            outputs = (output_logits, cls_logit)
        if maps:
            return outputs + (act_maps,)
        return outputs


def build_diffnet(num_classes: int = 2, load_imagenet_weights: bool = True) -> DiffNet:
    """load_imagenet_weights: initialise the ConvNeXt parts from ImageNet-1K checkpoints (training from scratch);
    not needed when a full DiffNet checkpoint is loaded afterwards."""
    return DiffNet(num_classes=num_classes, load_imagenet_weights=load_imagenet_weights)
