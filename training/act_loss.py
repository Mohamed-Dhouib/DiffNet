"""act_loss: auxiliary loss that sharpens the zero-sum-filter activation maps around the edits.

For every pyramid stage s, A_s(p) is the mean over channels and the M responses of |u_{s,c,m}(p)|, taken in the
zero-sum filter block after abs and before the contraction (models/diffnet/neck.py, BasisFusionLayer).
R_s is the GT mask max-pooled to the stage resolution and dilated by one cell (3x3 max). Per image and stage:
    a_s = mean of A_s inside R_s,  b_s = mean of the top 10% of A_s outside R_s (at least one cell),
    d_s = (a_s - b_s) / max(a_s + b_s, 1e-6),  loss_s = max(0, d_target_s - d_s)^2.
A stage is valid when R_s is non-empty and covers at most half of the map. The loss averages the valid stages of
an image, then the eligible images (at least one valid stage, so authentic images never contribute); it is zero
when no image is eligible. d_s is invariant to a positive rescaling of A_s. Optional stage weights multiply
the stages' terms, with the same division by the number of valid stages.
"""
import torch
import torch.nn.functional as F


def activation_contrast(act_maps, masks, top_fraction=0.1, eps=1e-6):
    """Return d, r = a/b and valid, each (B, S), for S maps (B, H_s, W_s) and GT masks (B, H, W)."""
    edited = (masks > 0).unsqueeze(1).float()
    d_all, r_all, valid_all = [], [], []
    for act in act_maps:
        _, h, w = act.shape
        act = act.float().flatten(1)
        region = F.adaptive_max_pool2d(edited, (h, w))           # cell touches an edited pixel
        region = F.max_pool2d(region, 3, stride=1, padding=1)    # dilate by one cell
        region = region.flatten(1) > 0
        n_in = region.sum(1)
        n_out = h * w - n_in
        a = (act * region).sum(1) / n_in.clamp(min=1)
        # Top 10% outside R: A >= 0, so cells inside R (set to -1) sort after every outside cell; the -1 sentinel
        # keeps everything finite for invalid images, which are masked out below.
        k = torch.ceil(top_fraction * n_out).clamp(min=1)
        ranked = act.masked_fill(region, -1.0).sort(dim=1, descending=True).values
        top = torch.arange(h * w, device=act.device).unsqueeze(0) < k.unsqueeze(1)
        b = (ranked * top).sum(1) / k
        d_all.append((a - b) / (a + b).clamp(min=eps))
        r_all.append(a / b.clamp(min=eps))
        valid_all.append((n_in > 0) & (2 * n_in <= h * w))
    return torch.stack(d_all, 1), torch.stack(r_all, 1), torch.stack(valid_all, 1)


def activation_contrast_loss(act_maps, masks, d_target, top_fraction=0.1, stage_weights=None):
    """Return (loss, d, r, valid); d_target holds one target per stage, stage_weights (optional) one weight each."""
    d, r, valid = activation_contrast(act_maps, masks, top_fraction)
    target = d.new_tensor(d_target)
    per_stage = torch.where(valid, F.relu(target - d) ** 2, torch.zeros_like(d))
    if stage_weights is not None:
        per_stage = per_stage * d.new_tensor(stage_weights)
    n_valid = valid.sum(1)
    per_image = per_stage.sum(1) / n_valid.clamp(min=1)
    loss = per_image.sum() / (n_valid > 0).sum().clamp(min=1)
    return loss, d, r, valid


def act_loss_weight(step, total_steps, lambda_max, delay_fraction, ramp_fraction):
    """0 for the first delay_fraction of the steps, then linear to lambda_max over the next ramp_fraction."""
    progress = (step / total_steps - delay_fraction) / ramp_fraction
    return lambda_max * min(max(progress, 0.0), 1.0)
