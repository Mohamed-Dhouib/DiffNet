"""Fused zero-sum filter block for training (PyTorch 2.7 / Triton 3.3): depthwise zero-sum filters, abs and the
per-channel contraction in one custom op with its own backward.

The dense kernels are synthesized by BasisFusionLayer (center tie, weight constraints and state_dict stay in
PyTorch). No response tensor of shape B,C,D,H,W is stored: only two ternary signs packed into one byte per pixel.
CUDA kernels use fp32 accumulators and explicit convolution dtype boundaries.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# bf16 kernels: the packed kernels (fused_basis_packed) at even H and W except 16x16, the reduce kernels
# (fused_basis_reduce) otherwise (fused_basis_config); fp32 kernels use the generic kernels below.


@triton.jit
def _reflect(p, size: tl.constexpr):
    return tl.minimum(tl.abs(p), 2 * size - 2 - p)


@triton.jit
def _responses(X, K, b, c, p, valid, C: tl.constexpr, H: tl.constexpr,
               W: tl.constexpr, KS: tl.constexpr):
    h, w = p // W, p % W
    u0 = tl.full(p.shape, 0, tl.float32)
    u1 = tl.full(p.shape, 0, tl.float32)
    for j in tl.static_range(KS * KS):
        ih = _reflect(h + j // KS - KS // 2, H)
        iw = _reflect(w + j % KS - KS // 2, W)
        # F.pad retains x's dtype; convolution autocast happens afterwards.
        v = tl.load(X + (b * C + c) * H * W + ih * W + iw,
                    valid, 0).to(K.dtype.element_ty).to(tl.float32)
        k0 = tl.load(K + c * 2 * KS * KS + j).to(tl.float32)
        k1 = tl.load(K + (c * 2 + 1) * KS * KS + j).to(tl.float32)
        u0 = u0 + v * k0
        u1 = u1 + v * k1
    return u0.to(K.dtype.element_ty).to(tl.float32), u1.to(K.dtype.element_ty).to(tl.float32)


@triton.jit
def _sign(x):
    return (x > 0).to(tl.int32) - (x < 0).to(tl.int32)


@triton.jit
def _forward(X, K, V, Y, S, C: tl.constexpr, H: tl.constexpr,
             W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    bc = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    u0, u1 = _responses(X, K, bc // C, bc % C, p, p < H * W, C, H, W, KS)
    v0 = tl.load(V + bc % C * 2).to(tl.float32)
    v1 = tl.load(V + bc % C * 2 + 1).to(tl.float32)
    y = tl.abs(u0) * v0 + tl.abs(u1) * v1
    tl.store(Y + bc * H * W + p, y, p < H * W)
    # 0,1,2 encode negative, zero, positive; no sign at zero is lost.
    signs = (_sign(u0) + 1) | ((_sign(u1) + 1) << 2)
    tl.store(S + bc * H * W + p, signs, p < H * W)


@triton.jit
def _grad_u(G, S, V, bc, p, valid, C: tl.constexpr, HW: tl.constexpr):
    g = tl.load(G + bc * HW + p, valid, 0).to(tl.float32)
    s = tl.load(S + bc * HW + p, valid, 5).to(tl.int32)
    v0 = tl.load(V + bc % C * 2).to(tl.float32)
    v1 = tl.load(V + bc % C * 2 + 1).to(tl.float32)
    # Grouped 1x1 backward rounds to its input dtype before abs backward.
    g0 = (g * v0).to(V.dtype.element_ty).to(tl.float32) * ((s & 3) - 1)
    g1 = (g * v1).to(V.dtype.element_ty).to(tl.float32) * (((s >> 2) & 3) - 1)
    return g0, g1


@triton.jit
def _input_grad(K, V, G, S, DX, C: tl.constexpr, H: tl.constexpr,
                W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    # Adjoint convolution on the padded domain; the reflection fold happens after
    # the bf16 rounding, as in the PyTorch path.
    bc = tl.program_id(0)
    PH: tl.constexpr = H + KS - 1
    PW: tl.constexpr = W + KS - 1
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    h, w = p // PW, p % PW
    acc0 = tl.full((BLOCK,), 0, tl.float32)
    acc1 = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(KS * KS):
        oh, ow = h - j // KS, w - j % KS
        valid = (p < PH * PW) & (oh >= 0) & (oh < H) & (ow >= 0) & (ow < W)
        g0, g1 = _grad_u(G, S, V, bc, oh * W + ow, valid, C, H * W)
        k0 = tl.load(K + bc % C * 2 * KS * KS + j).to(tl.float32)
        k1 = tl.load(K + (bc % C * 2 + 1) * KS * KS + j).to(tl.float32)
        acc0 = acc0 + g0 * k0
        acc1 = acc1 + g1 * k1
    tl.store(DX + bc * PH * PW + p, acc0 + acc1, p < PH * PW)


@triton.jit
def _parameter_partials(X, K, V, G, S, P, B: tl.constexpr, C: tl.constexpr,
                        H: tl.constexpr, W: tl.constexpr, KS: tl.constexpr,
                        NT: tl.constexpr, BLOCK: tl.constexpr):
    c, tile = tl.program_id(0), tl.program_id(1)
    t = tile * BLOCK + tl.arange(0, BLOCK)
    b, p = t // (H * W), t % (H * W)
    valid = t < B * H * W
    h, w = p // W, p % W
    g0, g1 = _grad_u(G, S, V, b * C + c, p, valid, C, H * W)
    u0 = tl.full((BLOCK,), 0, tl.float32)
    u1 = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(KS * KS):
        ih = _reflect(h + j // KS - KS // 2, H)
        iw = _reflect(w + j % KS - KS // 2, W)
        x = tl.load(X + (b * C + c) * H * W + ih * W + iw, valid, 0)
        x = x.to(K.dtype.element_ty).to(tl.float32)
        tl.store(P + (c * (2 * KS * KS + 2) + j) * NT + tile, tl.sum(x * g0, 0))
        tl.store(P + (c * (2 * KS * KS + 2) + KS * KS + j) * NT + tile, tl.sum(x * g1, 0))
        # Each patch load serves both dK and dV.
        k0 = tl.load(K + c * 2 * KS * KS + j).to(tl.float32)
        k1 = tl.load(K + (c * 2 + 1) * KS * KS + j).to(tl.float32)
        u0 = u0 + x * k0
        u1 = u1 + x * k1
    u0 = u0.to(K.dtype.element_ty).to(tl.float32)
    u1 = u1.to(K.dtype.element_ty).to(tl.float32)
    g = tl.load(G + (b * C + c) * H * W + p, valid, 0).to(tl.float32)
    base = (c * (2 * KS * KS + 2) + 2 * KS * KS) * NT + tile
    tl.store(P + base, tl.sum(g * tl.abs(u0), 0))
    tl.store(P + base + NT, tl.sum(g * tl.abs(u1), 0))


@triton.jit
def _parameter_reduce(P, DK, DV, KS: tl.constexpr, NT: tl.constexpr,
                      RT: tl.constexpr, RP: tl.constexpr):
    c = tl.program_id(0)
    q = tl.program_id(1) * RP + tl.arange(0, RP)
    t = tl.arange(0, RT)
    N: tl.constexpr = 2 * KS * KS + 2
    v = tl.load(P + (c * N + q[:, None]) * NT + t[None, :],
                (q[:, None] < N) & (t[None, :] < NT), 0)
    sums = tl.sum(v, 1)
    tl.store(DK + c * 2 * KS * KS + q, sums, q < 2 * KS * KS)
    tl.store(DV + c * 2 + q - 2 * KS * KS, sums,
             (q >= 2 * KS * KS) & (q < N))


def _validate(x, kernels, contract):
    torch._check(x.ndim == 4)
    torch._check(kernels.shape[0] == x.shape[1] * 2)
    torch._check(kernels.shape[1] == 1)
    torch._check(kernels.shape[2] == kernels.shape[3])
    torch._check(kernels.shape[2] % 2 == 1)
    torch._check(x.shape[2] > kernels.shape[2] // 2)
    torch._check(x.shape[3] > kernels.shape[2] // 2)
    torch._check(contract.shape == (x.shape[1], 2))
    torch._check(kernels.dtype == contract.dtype)
    torch._check(kernels.dtype in (torch.float32, torch.bfloat16))
    torch._check(x.dtype in (torch.float32, torch.bfloat16))
    torch._check(x.device == kernels.device == contract.device)


@torch.library.custom_op('diffnet::basis_forward', mutates_args=(), device_types='cuda')
def _basis_forward(x: torch.Tensor, kernels: torch.Tensor,
                   contract: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    _validate(x, kernels, contract)
    x, kernels, contract = x.contiguous(), kernels.contiguous(), contract.contiguous()
    from . import fused_basis_packed as packed
    if packed.applies(x, kernels):
        return packed.forward(x, kernels, contract)
    b, c, h, w = x.shape
    y = torch.empty(x.shape, device=x.device, dtype=kernels.dtype)
    signs = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
    ks = kernels.shape[2]
    # fp32 FMA contraction for bf16 inputs (bf16 products are exact in fp32);
    # the explicit bf16 rounding points stay in the kernel.
    _forward[(b * c, triton.cdiv(h * w, 256))](
        x, kernels, contract, y, signs, c, h, w, ks, 256,
        enable_fp_fusion=kernels.dtype == torch.bfloat16)
    return y, signs


@_basis_forward.register_fake
def _forward_fake(x, kernels, contract):
    _validate(x, kernels, contract)
    return (torch.empty(x.shape, device=x.device, dtype=kernels.dtype),
            torch.empty(x.shape, device=x.device, dtype=torch.uint8))


@torch.library.custom_op('diffnet::basis_backward', mutates_args=(), device_types='cuda')
def _basis_backward(x: torch.Tensor, kernels: torch.Tensor, contract: torch.Tensor,
                    signs: torch.Tensor, grad: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, kernels, contract = x.contiguous(), kernels.contiguous(), contract.contiguous()
    grad = grad.to(kernels.dtype).contiguous()
    if kernels.dtype == torch.bfloat16:
        from . import fused_basis_packed as packed
        if packed.applies(x, kernels):
            return packed.backward(x, kernels, contract, signs, grad)
        from .fused_basis_reduce import backward
        return backward(x, kernels, contract, signs, grad)
    b, c, h, w = x.shape
    ks = kernels.shape[2]
    padded_grad = torch.empty((b, c, h + ks - 1, w + ks - 1),
                              device=x.device, dtype=kernels.dtype)
    _input_grad[(b * c, triton.cdiv((h + ks - 1) * (w + ks - 1), 256))](
        kernels, contract, grad, signs, padded_grad, c, h, w, ks, 256,
        num_warps=8 if kernels.dtype == torch.bfloat16 else 4,
        enable_fp_fusion=kernels.dtype == torch.bfloat16)
    block = 1024
    nt = triton.cdiv(b * h * w, block)
    partials = torch.empty((c, 2 * ks * ks + 2, nt), device=x.device, dtype=torch.float32)
    dk, dv = torch.empty_like(kernels), torch.empty_like(contract)
    _parameter_partials[(c, nt)](x, kernels, contract, grad, signs, partials,
                                b, c, h, w, ks, nt, block, num_warps=8,
                                enable_fp_fusion=kernels.dtype == torch.bfloat16)
    _parameter_reduce[(c, triton.cdiv(2 * ks * ks + 2, 4))](
        partials, dk, dv, ks, nt, triton.next_power_of_2(nt), 4,
        enable_fp_fusion=False)
    return padded_grad, dk, dv


@_basis_backward.register_fake
def _backward_fake(x, kernels, contract, signs, grad):
    b, c, h, w = x.shape
    ks = kernels.shape[2]
    return (torch.empty((b, c, h + ks - 1, w + ks - 1),
                        device=x.device, dtype=kernels.dtype),
            torch.empty_like(kernels, memory_format=torch.contiguous_format),
            torch.empty_like(contract, memory_format=torch.contiguous_format))


def _setup_context(ctx, inputs, output):
    x, kernels, contract = inputs
    _, signs = output
    ctx.save_for_backward(x, kernels, contract, signs)
    ctx.mark_non_differentiable(signs)


def _backward(ctx, grad, _sign_grad):
    x, kernels, contract, signs = ctx.saved_tensors
    padded_grad, dk, dv = _basis_backward(x, kernels, contract, signs, grad)
    # Cast-back and reflection fold stay outside the custom op so that Inductor
    # traces them (fp32 intermediate accumulation of the fold).
    dx = torch.ops.aten.reflection_pad2d_backward.default(
        padded_grad.to(x.dtype), x, [kernels.shape[2] // 2] * 4)
    return dx, dk, dv


_basis_forward.register_autograd(_backward, setup_context=_setup_context)


def fused_basis(x: torch.Tensor, kernels: torch.Tensor, contract: torch.Tensor) -> torch.Tensor:
    """Same autocast behaviour as the PyTorch path (F.pad + two conv2d); the pad keeps x.dtype.

    The weight casts stay outside the custom op, so autograd casts the kernel
    gradients back to the fp32 parameters.
    """
    dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
    # contiguous() outside the op: Inductor sees the layout conversion and the
    # converted input is saved once for backward.
    y, _ = _basis_forward(x.contiguous(), kernels.to(dtype), contract.to(dtype))
    return y
