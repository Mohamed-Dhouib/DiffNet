"""bf16 backward of the fused zero-sum filter block for the shapes not handled by the packed kernels: scalar or
row-tiled input adjoint and a dense weight-gradient reduction accumulated across B*H*W.
"""
import torch
import triton
import triton.language as tl
from .fused_basis import _grad_u, _input_grad, _reflect
from .fused_basis_hybrid import _contract_partials, _contract_reduce
from .fused_basis_config import pinned_config, reduce_input_config


def _bench(kernel_call, quantiles):
    # Resolved only when tuning executes, so importing works on CPU-only hosts.
    from triton.testing import do_bench
    return do_bench(kernel_call, quantiles=quantiles)


@triton.autotune(configs=[
    triton.Config({'QB': 4, 'RB': 256}, num_warps=4),
    triton.Config({'QB': 8, 'RB': 128}, num_warps=4),
    triton.Config({'QB': 8, 'RB': 256}, num_warps=4),
    triton.Config({'QB': 8, 'RB': 256}, num_warps=8),
], key=['B', 'C', 'H', 'W', 'KS'], do_bench=_bench)
@triton.jit
def _weight_grad(X, V, G, S, DK, B: tl.constexpr, C: tl.constexpr,
                 H: tl.constexpr, W: tl.constexpr, KS: tl.constexpr,
                 QB: tl.constexpr, RB: tl.constexpr):
    nq: tl.constexpr = tl.cdiv(KS * KS, QB)
    c = tl.program_id(0) // nq
    q = (tl.program_id(0) % nq) * QB + tl.arange(0, QB)
    r = tl.arange(0, RB)
    acc0 = tl.full((QB, RB), 0, tl.float32)
    acc1 = tl.full((QB, RB), 0, tl.float32)
    for start in range(tl.cdiv(B * H * W, RB)):
        t = start * RB + r
        b, p = t // (H * W), t % (H * W)
        valid = t < B * H * W
        # bf16 contraction-backward rounding before the saved abs signs
        # (sign(0) = 0).
        g0, g1 = _grad_u(G, S, V, b * C + c, p, valid, C, H * W)
        ih = _reflect(p[None, :] // W + q[:, None] // KS - KS // 2, H)
        iw = _reflect(p[None, :] % W + q[:, None] % KS - KS // 2, W)
        x = tl.load(X + (b[None, :] * C + c) * H * W + ih * W + iw,
                    (q[:, None] < KS * KS) & valid[None, :], 0)
        # Reflection retains x.dtype; convolution autocast rounds afterwards.
        x = x.to(V.dtype.element_ty).to(tl.float32)
        acc0 = acc0 + x * g0[None, :]
        acc1 = acc1 + x * g1[None, :]
    # One final reduction/cast per dense coefficient, including the center.
    # Raw constraints and neighbor-minus-center gradients remain in PyTorch.
    tl.store(DK + (c * 2) * KS * KS + q, tl.sum(acc0, 1), q < KS * KS)
    tl.store(DK + (c * 2 + 1) * KS * KS + q, tl.sum(acc1, 1), q < KS * KS)


@triton.jit
def _input_grad_rows(K, V, G, S, DX, C: tl.constexpr, H: tl.constexpr,
                     W: tl.constexpr, KS: tl.constexpr, TH: tl.constexpr,
                     TW: tl.constexpr, GW: tl.constexpr):
    # Each rounded/signed grad-u row is reused across the horizontal taps;
    # same summation order as _input_grad.
    bc = tl.program_id(0)
    PH: tl.constexpr = H + KS - 1
    PW: tl.constexpr = W + KS - 1
    row = tl.program_id(1) * TH + tl.arange(0, TH)
    col = tl.program_id(2) * TW
    halo = col - (KS - 1) + tl.arange(0, GW)
    out_col = col + tl.arange(0, TW)
    acc0 = tl.full((TH, TW), 0, tl.float32)
    acc1 = tl.full((TH, TW), 0, tl.float32)
    for dy in tl.static_range(KS):
        oh = row - dy
        valid = ((row[:, None] < PH) & (oh[:, None] >= 0) & (oh[:, None] < H)
                 & (halo[None, :] >= 0) & (halo[None, :] < W)
                 & (tl.arange(0, GW)[None, :] < TW + KS - 1))
        g0, g1 = _grad_u(G, S, V, bc, oh[:, None] * W + halo[None, :],
                         valid, C, H * W)
        for dx in tl.static_range(KS):
            ix = tl.broadcast_to((tl.arange(0, TW) + KS - 1 - dx)[None, :], (TH, TW))
            a0 = tl.gather(g0, ix, axis=1)
            a1 = tl.gather(g1, ix, axis=1)
            k0 = tl.load(K + bc % C * 2 * KS * KS + dy * KS + dx).to(tl.float32)
            k1 = tl.load(K + (bc % C * 2 + 1) * KS * KS + dy * KS + dx).to(tl.float32)
            acc0 = acc0 + a0 * k0
            acc1 = acc1 + a1 * k1
    tl.store(DX + bc * PH * PW + row[:, None] * PW + out_col[None, :],
             acc0 + acc1, (row[:, None] < PH) & (out_col[None, :] < PW))


def backward(x, kernels, contract, signs, grad):
    b, c, h, w = x.shape
    ks = kernels.shape[2]
    padded_grad = torch.empty((b, c, h + ks - 1, w + ks - 1),
                              device=x.device, dtype=kernels.dtype)
    icfg = reduce_input_config(h, w)
    if icfg['kind'] == 'scalar':
        block = icfg['BLOCK']
        _input_grad[(b * c, triton.cdiv((h + ks - 1) * (w + ks - 1), block))](
            kernels, contract, grad, signs, padded_grad, c, h, w, ks, block,
            num_warps=icfg['num_warps'], enable_fp_fusion=True)
    else:
        th, tw = icfg['TH'], icfg['TW']
        _input_grad_rows[(b * c, triton.cdiv(h + ks - 1, th), triton.cdiv(w + ks - 1, tw))](
            kernels, contract, grad, signs, padded_grad, c, h, w, ks, th, tw,
            triton.next_power_of_2(tw + ks - 1), num_warps=icfg['num_warps'], enable_fp_fusion=True)
    dk = torch.empty(kernels.shape, device=kernels.device, dtype=kernels.dtype)
    cfg = pinned_config('reduce', h, w)
    _weight_grad.fn[(c * triton.cdiv(ks * ks, cfg.kwargs['QB']),)](
        x, contract, grad, signs, dk, b, c, h, w, ks, **cfg.all_kwargs(), enable_fp_fusion=True)
    # dV from the recomputed rounded responses.
    block = 1024
    nt = triton.cdiv(b * h * w, block)
    partials = torch.empty((c, 2, nt), device=x.device, dtype=torch.float32)
    dv = torch.empty(contract.shape, device=contract.device, dtype=contract.dtype)
    _contract_partials[(c, nt)](x, kernels, grad, partials, b, c, h, w, ks,
                              nt, block, num_warps=8, enable_fp_fusion=True)
    _contract_reduce[(c * 2,)](partials, dv, nt, triton.next_power_of_2(nt),
                             enable_fp_fusion=False)
    return padded_grad, dk, dv
