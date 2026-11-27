"""Learned scalar operating point (threshold offset) and positive-image macro soft-F1."""

import math

import torch
import torch.distributed as dist
from torch import nn


class LearnedOffset(nn.Module):
    def __init__(self, external_threshold=0.5, offset_bound=6.0):
        super().__init__()
        if not 0 < external_threshold < 1 or offset_bound <= 0:
            raise ValueError("external_threshold must be in (0, 1), offset_bound must be positive")
        # Center on the literal uint8 sweep boundary, including non-0.5 cutoffs.
        boundary = (math.floor(255 * external_threshold) + 0.5) / 255
        if not 0 < boundary < 1:
            raise ValueError("external_threshold must have an interior uint8 boundary")
        self.beta = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.register_buffer("anchor", torch.tensor(math.log(boundary / (1 - boundary)), dtype=torch.float32))
        self.register_buffer("bound", torch.tensor(offset_bound, dtype=torch.float32))
        self.register_buffer("external_threshold", torch.tensor(external_threshold, dtype=torch.float32))

    def offset(self):
        return self.bound.float() * self.beta.float().tanh()

    def forward(self, raw_logits):
        # Cast BEFORE subtraction: bf16 subtraction would lose small shifts.
        raw = raw_logits.float()
        margin = raw[:, 1] - raw[:, 0] - self.offset() + self.anchor.float()
        return torch.stack((torch.zeros_like(margin), margin), dim=1)


def soft_f1_sum_count(output_logits, masks, anchor=0.0, bandwidth=1.0):
    """Spatial ratio for each image, then sum losses over positive-GT images.

    Masking rather than variable-size indexing keeps all-empty batches connected
    to both the network and offset, and works with torch.compile.
    """
    margin = output_logits[:, 1].float() - output_logits[:, 0].float() - anchor
    r = torch.sigmoid(margin / bandwidth)
    y = masks.float()
    gt_area = y.sum(dim=(1, 2))
    positive = gt_area > 0
    tp = (r * y).sum(dim=(1, 2))
    denominator = r.sum(dim=(1, 2)) + gt_area
    f1 = 2 * tp / denominator.clamp_min(1e-6)
    return ((1 - f1) * positive).sum(), positive.sum()


def macro_soft_f1_loss(output_logits, masks, anchor=0.0, bandwidth=1.0):
    local_sum, count = soft_f1_sum_count(output_logits, masks, anchor, bandwidth)
    stats = torch.stack((local_sum.detach(), count.float()))
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(stats)
    denominator = stats[1].clamp_min(1)
    local_loss = world_size * local_sum / denominator
    # Same global value for logging, with the local gradient scaled for DDP's
    # gradient average. In particular, ranks with no positives still participate.
    loss = local_loss + (stats[0] / denominator - local_loss.detach())
    return loss, stats[1]


def soft_f1_schedule(step, total_steps, weight=0.0, ramp_fraction=0.05,
                     bandwidth_final=0.1, anneal_fraction=0.8):
    """step is one-based, so the first batch already trains the offset."""
    coefficient = weight * min(1.0, step / max(1.0, ramp_fraction * total_steps))
    bandwidth = bandwidth_final ** min(1.0, step / max(1.0, anneal_fraction * total_steps))
    return coefficient, bandwidth
