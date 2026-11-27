"""Multi-level discrepancy transformation and multi-scale fusion.

Each backbone stage output is projected to 256 channels (1x1 conv) and replaced by magnitude-only responses of
learned 7x7 zero-sum filters: per channel one center-anchored filter (non-negative neighbours, center = -sum) and
one free zero-sum filter, abs, then a per-channel projection back (BasisFusionLayer). The four discrepancy maps are
fused bottom-up to stride 32 (BidirectionalFPN), refined there by 6 ConvNeXt-V2 blocks (RefinementBlocks), and
passed through an FPN.
"""
from __future__ import annotations

import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.necks.fpn import FPN

from .dct_branch import WEIGHTS_DIR

KERNEL_SIZE = 7
NUM_FILTERS = 2          # M zero-sum responses per channel: [center-anchored, free]
NUM_SCALES = 4
DEEP_BLOCKS = 6
PRE_BN_SCALE = 0.1       # identity skip: the input feature, scaled, is added to the discrepancy response


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=0, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SimpleProjection(nn.Module):
    """1x1 conv + BN + ReLU channel projection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = BasicConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class BasisFusionLayer(nn.Module):
    """Zero-sum discrepancy filters + abs + per-channel projection, then a SiLU / 1x1-conv / BN refinement.

    expand_weights (C, K^2 - 1, M) hold the neighbour weights a of the M filters of each channel (abs for the
    center-anchored half); the center weight is -sum(a), so every kernel sums to zero. contract_weights (C, M) map
    the M magnitudes back to one channel. Training on CUDA uses the fused Triton op (fused_basis.py); other forwards
    build the dense depthwise kernels (cached in eval mode).
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.num_basis = KERNEL_SIZE * KERNEL_SIZE - 1
        self.in_channels = in_channels
        self.intermediate_dim = NUM_FILTERS

        self.expand_weights = nn.Parameter(torch.empty(in_channels, self.num_basis, self.intermediate_dim))
        nn.init.xavier_uniform_(self.expand_weights)
        self.contract_dim = self.intermediate_dim
        self.contract_weights = nn.Parameter(torch.empty(in_channels, self.contract_dim))
        nn.init.xavier_uniform_(self.contract_weights)

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU(inplace=False)
        self.pre_bn_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.fix_bn_conv1_with_bias = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.fix_bn_conv2_with_bias = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.fix_bn_conv3_with_bias = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        self.post_bn_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.dirac_(self.fix_bn_conv1_with_bias.weight)
        nn.init.dirac_(self.fix_bn_conv2_with_bias.weight)
        nn.init.dirac_(self.fix_bn_conv3_with_bias.weight)
        nn.init.zeros_(self.fix_bn_conv1_with_bias.bias)
        nn.init.zeros_(self.fix_bn_conv2_with_bias.bias)
        nn.init.zeros_(self.fix_bn_conv3_with_bias.bias)

        self._fused_basis = None
        if torch.cuda.is_available():
            from .fused_basis import fused_basis
            self._fused_basis = fused_basis

        # When True (DiscrepancyNeck.set_return_activation_maps), training forwards also return the activation map
        # A(p) = mean over channels and filters of |u(p)| (after abs, before the projection).
        self.return_activation_map = False

        self.kernel_size = KERNEL_SIZE
        self.kernel_radius = self.kernel_size // 2
        self.fast_groups = in_channels
        self.fast_pad = self.kernel_radius
        r = self.kernel_radius
        self.center_index = r * self.kernel_size + r
        self._fast_weight_cache = {}
        self._fast_kernel_cache = {}
        self.padding_mode = 'reflect'
        # Eval-mode precomputed kernels.
        self.register_buffer('_fast_infer_kernels', torch.empty(0), persistent=False)
        self.register_buffer('_fast_infer_contract', torch.empty(0), persistent=False)
        self._fast_infer_expand_ver = -1
        self._fast_infer_contract_ver = -1
        self._fast_infer_ready = False

    def _get_expand_weights(self) -> torch.Tensor:
        weights = self.expand_weights  # (C, K^2 - 1, M)
        half = weights.shape[2] // 2
        w_pos = weights[:, :, :half].abs()   # center-anchored filters: non-negative neighbours
        w_free = weights[:, :, half:]
        return torch.cat([w_pos, w_free], dim=2)

    def _get_contract_weights(self) -> torch.Tensor:
        return self.contract_weights

    @staticmethod
    def _is_compiling() -> bool:
        compiler = getattr(torch, 'compiler', None)
        if compiler is not None and hasattr(compiler, 'is_compiling'):
            return bool(compiler.is_compiling())
        dynamo = getattr(torch, '_dynamo', None)
        if dynamo is not None and hasattr(dynamo, 'is_compiling'):
            return bool(dynamo.is_compiling())
        return False

    @torch.no_grad()
    def prepare_fast_inference_cache(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Precompute the dense kernels once for inference."""
        if device is None:
            target_device = self.expand_weights.device
        else:
            target_device = torch.device(device)
        target_dtype = self.expand_weights.dtype if dtype is None else dtype

        expand_w = self._get_expand_weights().to(device=target_device, dtype=target_dtype)
        contract_w = self._get_contract_weights().to(device=target_device, dtype=target_dtype)
        kernels = self._build_fast_kernels_no_cache(expand_w)

        self._fast_infer_kernels = kernels
        self._fast_infer_contract = contract_w.view(self.in_channels, self.contract_dim, 1, 1).contiguous()
        self._fast_infer_expand_ver = int(self.expand_weights._version)
        self._fast_infer_contract_ver = int(self.contract_weights._version)
        self._fast_infer_ready = True

    def _can_use_fast_inference_cache(self, x: torch.Tensor) -> bool:
        if self.training or torch.is_grad_enabled():
            return False
        if not self._fast_infer_ready:
            return False

        if not self._is_compiling():
            if self._fast_infer_expand_ver != int(self.expand_weights._version):
                return False
            if self._fast_infer_contract_ver != int(self.contract_weights._version):
                return False
        
        if self._fast_infer_kernels.numel() == 0 or self._fast_infer_contract.numel() == 0:
            return False
        if self._fast_infer_kernels.device != x.device or self._fast_infer_contract.device != x.device:
            return False
        if self._fast_infer_kernels.dtype != x.dtype or self._fast_infer_contract.dtype != x.dtype:
            return False
        return True

    def _build_fast_kernels_no_cache(self, expand_w: torch.Tensor) -> torch.Tensor:
        """(C*M, 1, K, K) depthwise kernels: neighbours a, center -sum(a)."""
        c, _, d = expand_w.shape
        ks = self.kernel_size
        w_basis = expand_w.transpose(1, 2)  # (C, M, K^2 - 1)
        center_w = -w_basis.sum(dim=2, keepdim=True)

        mid = self.center_index
        kernels_flat = torch.cat([
            w_basis[:, :, :mid],
            center_w,
            w_basis[:, :, mid:]
        ], dim=2)
        
        return kernels_flat.reshape(c * d, 1, ks, ks)

    def _get_fast_weights(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Processed expand/contract weights; cached by (device, dtype, parameter version) in eval + no_grad."""
        use_cache = (not self.training) and (not torch.is_grad_enabled()) and (not self._is_compiling())
        if not use_cache:
            expand_w = self._get_expand_weights()
            contract_w = self._get_contract_weights()
            return expand_w, contract_w

        w_dev = self.expand_weights.device
        key = (w_dev.type, w_dev.index, self.expand_weights.dtype)
        expand_ver = self.expand_weights._version
        contract_ver = self.contract_weights._version
        cached = self._fast_weight_cache.get(key, None)
        if cached is not None:
            if cached['expand_ver'] == expand_ver and cached['contract_ver'] == contract_ver:
                return cached['expand_w'], cached['contract_w']

        expand_w = self._get_expand_weights()
        contract_w = self._get_contract_weights()
        self._fast_weight_cache[key] = {
            'expand_ver': expand_ver,
            'contract_ver': contract_ver,
            'expand_w': expand_w,
            'contract_w': contract_w,
        }
        return expand_w, contract_w

    def _build_fast_kernels(self, expand_w: torch.Tensor) -> torch.Tensor:
        use_cache = (not self.training) and (not torch.is_grad_enabled()) and (not self._is_compiling())
        key = None
        expand_ver = None
        if use_cache:
            key = (expand_w.device.type, expand_w.device.index, expand_w.dtype)
            expand_ver = self.expand_weights._version
            cached = self._fast_kernel_cache.get(key, None)
            if cached is not None and cached['expand_ver'] == expand_ver:
                return cached['kernels']

        kernels = self._build_fast_kernels_no_cache(expand_w)

        if use_cache:
            self._fast_kernel_cache[key] = {
                'expand_ver': expand_ver,
                'kernels': kernels,
            }
        return kernels

    def _forward_fast_from_weights(
        self,
        x: torch.Tensor,
        kernels: torch.Tensor,
        w_contract: torch.Tensor,
    ) -> torch.Tensor:
        use_inplace_nonlin = (not torch.is_grad_enabled()) and (not self._is_compiling())

        x = F.pad(x, (self.fast_pad, self.fast_pad, self.fast_pad, self.fast_pad), mode=self.padding_mode)

        out = F.conv2d(x, kernels, bias=None, stride=1, padding=0, groups=self.fast_groups)

        out = out.abs_() if use_inplace_nonlin else out.abs()

        act_map = out.mean(dim=1, dtype=torch.float32) if self.return_activation_map and self.training else None
        out = F.conv2d(out, w_contract, bias=None, groups=self.fast_groups)
             
        return out if act_map is None else (out, act_map)

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        expand_w, contract_w = self._get_fast_weights(dtype=x.dtype, device=x.device)
        kernels = self._build_fast_kernels(expand_w)
        w_contract = contract_w.view(self.in_channels, self.contract_dim, 1, 1)
        return self._forward_fast_from_weights(x, kernels, w_contract)

    def _forward_fast_cached(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_fast_from_weights(x, self._fast_infer_kernels, self._fast_infer_contract)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._fast_infer_ready = False
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        if self._can_use_fast_inference_cache(x):
            x = self._forward_fast_cached(x)

        elif self.training and torch.is_grad_enabled() and x.is_cuda:
            kernels = self._build_fast_kernels_no_cache(self._get_expand_weights())
            contract = self._get_contract_weights()
            out = self._fused_basis(x, kernels, contract)
            if self.return_activation_map:
                # The fused op never materializes |u|: a second call with unit contraction weights returns
                # sum_m |u_m| per channel; its channel mean / M is A(p).
                ones = contract.new_ones(contract.shape)
                act_map = self._fused_basis(x, kernels, ones).mean(dim=1, dtype=torch.float32) / self.intermediate_dim
                out = (out, act_map)
            x = out

        else:
            if (not self.training) and (not torch.is_grad_enabled()) and (not self._is_compiling()):
                self.prepare_fast_inference_cache(dtype=x.dtype, device=x.device)
                x = self._forward_fast_cached(x)
            else:
                x = self._forward_fast(x)
        
        act_map = None
        if self.return_activation_map and self.training:
            x, act_map = x

        x = x + self.pre_bn_bias + PRE_BN_SCALE * x0
        x = self.act(x)
        x = self.fix_bn_conv1_with_bias(x)
        x = self.act(x)
        x = self.fix_bn_conv2_with_bias(x)
        x = self.act(x)
        x = self.fix_bn_conv3_with_bias(x)
        x = self.act(x)
        x = self.bn(x)
        x = x + self.post_bn_bias
        
        return x if act_map is None else (x, act_map)


class BidirectionalFPN(nn.Module):
    """Bottom-up fusion of the four discrepancy maps to stride 32: f1 -> f2 -> f3 -> f4."""

    def __init__(self, channels: int):
        super().__init__()

        def make_fuse():
            conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
            bn = nn.BatchNorm2d(channels)
            act = nn.SiLU(inplace=False)
            return nn.Sequential(conv, act, bn)

        self.down1 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.fuse_down1 = make_fuse()
        self.down2 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.fuse_down2 = make_fuse()
        self.down3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.fuse_down3 = make_fuse()

    def forward(self, wf1, wf2, wf3, wf4):
        fused = wf1
        down1 = self.down1(fused)
        fused = self.fuse_down1(wf2 + down1)
        down2 = self.down2(fused)
        fused = self.fuse_down2(wf3 + down2)
        down3 = self.down3(fused)
        fused = self.fuse_down3(wf4 + down3)
        return fused


class DropPath(nn.Module):
    """Stochastic depth per sample (timm)."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: DWConv 7x7 -> LayerNorm -> Linear -> GELU -> GRN -> Linear."""
    
    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return residual + self.drop_path(x)


class GRN(nn.Module):
    """Global Response Normalization layer from ConvNeXt V2."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class RefinementBlocks(nn.Module):
    """DEEP_BLOCKS ConvNeXt-V2 blocks at stride 32, initialised from ConvNeXt-V2-pico stage 2 (dim 256)."""

    def __init__(self, dim: int, drop_path_rate: float = 0.0, load_pretrained: bool = True):
        super().__init__()
        self.dim = dim
        self.num_blocks = DEEP_BLOCKS
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(dim, drop_path=rate)
            for rate in [x.item() for x in torch.linspace(0, drop_path_rate, self.num_blocks)]
        ])
        if load_pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        variant, stage_idx, stage_depth = 'pico', 2, 6
        weights_path = os.path.join(WEIGHTS_DIR, f'convnextv2_{variant}_1k_224_ema.pt')
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"{weights_path} not found (run models/diffnet/download_weights.py)")
        state = torch.load(weights_path, map_location='cpu', weights_only=False)['model']
        for i in range(self.num_blocks):
            prefix = f'stages.{stage_idx}.{i % stage_depth}.'
            block_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            self.blocks[i].load_state_dict(block_state, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DiscrepancyNeck(nn.Module):
    """Projection -> discrepancy transformation at the 4 scales -> fusion/refinement at stride 32."""

    def __init__(self, in_channels, out_channels, drop_path_rate=0.0, load_pretrained=True):
        super().__init__()
        self.output_channels = out_channels
        self.num_output_scales = NUM_SCALES
        self.target_scale = NUM_SCALES - 1

        self.T1 = SimpleProjection(in_channels[0], out_channels)
        self.T2 = SimpleProjection(in_channels[1], out_channels)
        self.T3 = SimpleProjection(in_channels[2], out_channels)
        self.T4 = SimpleProjection(in_channels[3], out_channels)

        self.filter_fusion1 = BasisFusionLayer(out_channels)
        self.filter_fusion2 = BasisFusionLayer(out_channels)
        self.filter_fusion3 = BasisFusionLayer(out_channels)
        self.filter_fusion4 = BasisFusionLayer(out_channels)

        self.bidirectional_fpn = BidirectionalFPN(out_channels)
        self.deep_processor = RefinementBlocks(dim=out_channels, drop_path_rate=drop_path_rate,
                                              load_pretrained=load_pretrained)

        self.return_activation_maps = False

    def set_return_activation_maps(self, enabled: bool = True) -> None:
        """Training forwards also return the activation maps (A1..A4), each (B, H_s, W_s) float32 with gradients,
        taken in the 4 zero-sum filter blocks after abs and before the projection (eval forwards never do)."""
        self.return_activation_maps = bool(enabled)
        for idx in range(1, 5):
            getattr(self, f"filter_fusion{idx}").return_activation_map = bool(enabled)

    def forward(self, inputs):
        assert len(inputs) == 4
        f1 = self.T1(inputs[0])
        f2 = self.T2(inputs[1])
        f3 = self.T3(inputs[2])
        f4 = self.T4(inputs[3])

        hh1 = self.filter_fusion1(f1)
        hh2 = self.filter_fusion2(f2)
        hh3 = self.filter_fusion3(f3)
        hh4 = self.filter_fusion4(f4)
        if self.return_activation_maps and self.training:
            (hh1, a1), (hh2, a2), (hh3, a3), (hh4, a4) = hh1, hh2, hh3, hh4
            act_maps = (a1, a2, a3, a4)

        fused_target = self.bidirectional_fpn(hh1, hh2, hh3, hh4)
        deep_target = self.deep_processor(fused_target)

        outputs = (hh1, hh2, hh3, deep_target)
        if self.return_activation_maps and self.training:
            return outputs, act_maps
        return outputs


class DiscrepancyFPN(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_rate=0.0, load_pretrained=True):
        super().__init__()
        self.neck = DiscrepancyNeck(in_channels=in_channels, out_channels=out_channels, drop_path_rate=drop_path_rate,
                            load_pretrained=load_pretrained)
        num_scales = self.neck.num_output_scales
        self.fpn = FPN(in_channels=[out_channels] * num_scales, out_channels=out_channels, num_outs=num_scales)

    def forward(self, inputs):
        if self.neck.return_activation_maps and self.training:
            inputs, act_maps = self.neck(inputs)
            return self.fpn(inputs), act_maps
        inputs = self.neck(inputs)
        return self.fpn(inputs)
