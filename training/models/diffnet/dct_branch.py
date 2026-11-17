"""DCT branch: frequency-index-aware joint embedding of the quantized DCT coefficients and the quantization table,
followed by ConvNeXt-V2 blocks at 1/8 resolution."""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .convnextv2 import Block as ConvNeXtBlock

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'pretrained_weights')
NUM_DCT_CLASSES = 21     # clipped |quantized DCT| in [0, 20]
EMBED_DIM = 4            # d_dct
INTERNAL_DIM = 256
NUM_BLOCKS = 18


class DCTBranch(nn.Module):
    """e_{p,k} = (1 + gamma_k) * v_{p,k} + beta_k + f_k + t, concatenated with v and the 64 frequencies of each 8x8
    block (8x8 stride-8 conv), then NUM_BLOCKS ConvNeXt-V2 blocks. gamma_k, beta_k come from the embedded
    quantization entry Q_k, f_k is a frequency-index embedding and t a bias computed from the whole table."""

    def __init__(self, drop_path_rate: float = 0.0, load_pretrained: bool = True) -> None:
        super().__init__()
        self.neck_dim = EMBED_DIM
        self.obembed = nn.Embedding(NUM_DCT_CLASSES, EMBED_DIM)
        self.qtembed = nn.Embedding(64, self.neck_dim)
        self.qt_to_gamma_beta = nn.Linear(self.neck_dim, 2 * self.neck_dim)
        nn.init.zeros_(self.qt_to_gamma_beta.weight)  # identity start: gamma = beta = 0
        nn.init.zeros_(self.qt_to_gamma_beta.bias)
        self.kembed = nn.Embedding(64, self.neck_dim)
        self.qt_table_encoder = nn.Sequential(
            nn.Linear(64 * self.neck_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.neck_dim),
        )
        nn.init.zeros_(self.qt_table_encoder[-1].weight)  # t = 0 at the start
        nn.init.zeros_(self.qt_table_encoder[-1].bias)

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=2 * self.neck_dim, out_channels=INTERNAL_DIM, kernel_size=8, stride=8, padding=0,
                      bias=False),
            nn.BatchNorm2d(INTERNAL_DIM, momentum=0.01),
            nn.ReLU(inplace=True),
            *[ConvNeXtBlock(dim=INTERNAL_DIM, drop_path=rate)
              for rate in [x.item() for x in torch.linspace(0, drop_path_rate, NUM_BLOCKS)]]
        )
        if load_pretrained:
            self._load_convnext_pretrained_weights()

    def forward(self, x, qtable):
        """x: (B, H, W) clipped |quantized DCT|; qtable: (B, 1, 8, 8) quantization table."""
        x = self.obembed(x).permute(0, 3, 1, 2).contiguous()

        batch_size, channels, height, width = x.shape
        qt = qtable.squeeze(1).long() if qtable.dim() == 4 else qtable.long()
        qt = torch.clamp(qt, 0, 63)
        qt_embed = self.qtembed(qt)
        v = x.reshape(batch_size, channels, height // 8, 8, width // 8, 8).permute(0, 1, 3, 5, 2, 4)
        gamma_beta = self.qt_to_gamma_beta(qt_embed)  # (B,8,8,2C)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

        gamma64 = gamma.permute(0, 3, 1, 2).unsqueeze(-1).unsqueeze(-1)
        fused64 = v * gamma64  # v * (1 + gamma) = v * gamma + v
        fused64 += v

        beta64 = beta.permute(0, 3, 1, 2).unsqueeze(-1).unsqueeze(-1)
        f = self.kembed.weight.reshape(8, 8, self.neck_dim).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        t = self.qt_table_encoder(qt_embed.reshape(batch_size, -1)).view(batch_size, self.neck_dim, 1, 1, 1, 1)
        bias64 = beta64 + f
        bias64 = bias64 + t
        fused64 += bias64
        fused = fused64.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, channels, height, width)
        out = torch.cat((fused, x), dim=1)

        return self.conv0(out)

    def _load_convnext_pretrained_weights(self):
        """Initialise the ConvNeXt blocks from ConvNeXt-V2-pico stage 2 (dim 256, 6 blocks, cycled)."""
        variant, stage_idx, stage_depth = 'pico', 2, 6
        weights_path = os.path.join(WEIGHTS_DIR, f'convnextv2_{variant}_1k_224_ema.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"{weights_path} not found (run models/diffnet/download_weights.py)")
        state_dict = torch.load(weights_path, map_location='cpu')['model']
        for i in range(NUM_BLOCKS):
            prefix = f'stages.{stage_idx}.{i % stage_depth}.'
            block_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            self.conv0[3 + i].load_state_dict(block_state, strict=False)


class SCSEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.cSE(x) + x * self.sSE(x)
