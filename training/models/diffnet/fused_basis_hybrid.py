"""Contraction-weight (dV) gradient kernels shared by the reduce and packed backward passes."""
import triton
import triton.language as tl
from .fused_basis import _responses


@triton.jit
def _contract_partials(X, K, G, P, B: tl.constexpr, C: tl.constexpr,
                       H: tl.constexpr, W: tl.constexpr, KS: tl.constexpr,
                       NT: tl.constexpr, BLOCK: tl.constexpr):
    c, tile = tl.program_id(0), tl.program_id(1)
    t = tile * BLOCK + tl.arange(0, BLOCK)
    b, p = t // (H * W), t % (H * W)
    valid = t < B * H * W
    # Recompute the forward's rounded responses (identical values and signs).
    u0, u1 = _responses(X, K, b, c, p, valid, C, H, W, KS)
    g = tl.load(G + (b * C + c) * H * W + p, valid, 0).to(tl.float32)
    tl.store(P + (c * 2) * NT + tile, tl.sum(g * tl.abs(u0), 0))
    tl.store(P + (c * 2 + 1) * NT + tile, tl.sum(g * tl.abs(u1), 0))


@triton.jit
def _contract_reduce(P, DV, NT: tl.constexpr, RT: tl.constexpr):
    cd = tl.program_id(0)
    t = tl.arange(0, RT)
    v = tl.load(P + cd * NT + t, t < NT, 0)
    # The store performs the final bf16 cast.
    tl.store(DV + cd, tl.sum(v, 0))
