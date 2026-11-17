"""RGB ConvNeXt-V2-Base backbone; the DCT branch is fused into the RGB stream after stage 1."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .convnextv2 import convnextv2_base
from .dct_branch import DCTBranch, SCSEModule, WEIGHTS_DIR

DIMS = [128, 256, 512, 1024]  # ConvNeXt-V2-Base stage widths


class RGBDCTBackbone(nn.Module):
    def __init__(self, pretrained: bool = True, rgb_drop_path_rate: float = 0.0, dct_drop_path_rate: float = 0.0,
                 init_cfg=None) -> None:
        super().__init__()
        self.dims = list(DIMS)
        self.init_cfg = init_cfg

        self.timm_model = convnextv2_base(in_chans=3, num_classes=0, drop_path_rate=rgb_drop_path_rate)
        self.timm_model.head = None
        self.timm_model.norm = None

        if pretrained:
            checkpoint_path = os.path.join(WEIGHTS_DIR, 'convnextv2_base_1k_224_ema.pt')
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"{checkpoint_path} not found (run models/diffnet/download_weights.py)")
            state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.timm_model.load_state_dict(state_dict, strict=False)
            self._is_init = True
        else:
            self._is_init = False

        self.fph = DCTBranch(drop_path_rate=dct_drop_path_rate, load_pretrained=pretrained)

        stage1_ch = self.dims[1]
        fph_out_ch = 256
        fusion_in_ch = stage1_ch + fph_out_ch
        fusion_units = [
            nn.Sequential(
                SCSEModule(fusion_in_ch),
                nn.Conv2d(fusion_in_ch, stage1_ch, 3, 1, 1),
                nn.BatchNorm2d(stage1_ch),
                nn.ReLU(inplace=True),
            )
        ]
        proj = nn.Conv2d(stage1_ch, stage1_ch, 1, 1, 0)
        nn.init.zeros_(proj.weight)  # the fusion starts as the identity on the RGB stream
        nn.init.zeros_(proj.bias)
        fusion_units.append(proj)
        self.FU = nn.ModuleList(fusion_units)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        x = inputs['x']
        f_dct = self.fph(inputs['dct'], inputs['qtb'])

        outs = []
        x = self.timm_model.downsample_layers[0](x)
        x = self.timm_model.stages[0](x)
        outs.append(x)
        for idx in range(1, 4):
            x = self.timm_model.downsample_layers[idx](x)
            x = self.timm_model.stages[idx](x)
            if idx == 1:
                ext = self.FU[0](torch.cat((x, f_dct), dim=1))
                x = self.FU[1](ext) + x
            outs.append(x)
        return tuple(outs)
