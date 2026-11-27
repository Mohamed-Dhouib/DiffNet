"""bf16 word-pair kernels of the fused zero-sum filter block (even H and W, see fused_basis_config.uses_packed).

Each lane owns a 2x2 output block; a 32-bit word holds two adjacent bf16 values, so one aligned load feeds two
output columns and every loaded row feeds two output rows. Reflection padding is materialized once per pass (in
x.dtype, then cast to bf16), so every tap is an unmasked word at a constant offset.

Arithmetic (as in fused_basis): forward and the dV recomputation use one fp32 FMA chain per output and filter,
rounded to bf16 before abs/sign; grad-u = bf16(g * v) times the saved forward sign; padded dx keeps the tap order
of _input_grad, reflection folding stays in the traceable caller; dK and dV are fp32 sums.
"""
import torch
import triton
import triton.language as tl
from .fused_basis import _reflect, _sign
from .fused_basis_hybrid import _contract_reduce
from . import fused_basis_config as config
from .fused_basis_config import PACKED


@triton.jit
def _pair(w):
    # Little-endian word -> (even column, odd column); bf16 -> fp32 is exact.
    return (w << 16).to(tl.float32, bitcast=True), (w & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _pack(a, b):
    lo = a.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 65535
    hi = b.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) << 16
    return lo | hi


@triton.jit
def _row(P, valid, NW: tl.constexpr):
    # NW consecutive words -> 2*NW consecutive bf16 values as fp32.
    values = ()
    for m in tl.static_range(NW):
        lo, hi = _pair(tl.load(P + m, valid, 0))
        values = values + (lo, hi)
    return values


@triton.jit
def _taps(c0f0, c0f1, c1f0, c1f1, x, K0, K1, KS: tl.constexpr):
    # One kernel row for both output columns, dx ascending (as in fused_basis).
    for dx in tl.static_range(KS):
        k0 = tl.load(K0 + dx).to(tl.float32)
        k1 = tl.load(K1 + dx).to(tl.float32)
        c0f0 = c0f0 + x[dx] * k0
        c0f1 = c0f1 + x[dx] * k1
        c1f0 = c1f0 + x[dx + 1] * k0
        c1f1 = c1f1 + x[dx + 1] * k1
    return c0f0, c0f1, c1f0, c1f1


@triton.jit
def _lanes(B: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    # Lane -> (image, output row pair, output column pair) of one channel.
    W2: tl.constexpr = W // 2
    LP: tl.constexpr = (H // 2) * W2
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    q = t % LP
    return t < B * LP, t // LP, q // W2, q % W2


@triton.jit
def _responses(XP, K, b, c, hp, n, valid, C: tl.constexpr, H: tl.constexpr,
               W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    # Unrounded fp32 responses of the 2x2 block, both filters.
    PW2: tl.constexpr = (W + KS - 1) // 2
    src = XP + ((b * C + c) * (H + KS - 1) + 2 * hp) * PW2 + n
    K0 = K + c * 2 * KS * KS
    K1 = K0 + KS * KS
    z = tl.zeros((BLOCK,), tl.float32)
    a00, a01, a10, a11 = z, z, z, z
    b00, b01, b10, b11 = z, z, z, z
    for r in tl.static_range(KS + 1):
        x = _row(src + r * PW2, valid, (KS + 1) // 2)
        if r < KS:
            a00, a01, a10, a11 = _taps(a00, a01, a10, a11, x, K0 + r * KS, K1 + r * KS, KS)
        if r > 0:
            b00, b01, b10, b11 = _taps(b00, b01, b10, b11, x, K0 + (r - 1) * KS, K1 + (r - 1) * KS, KS)
    return a00, a01, a10, a11, b00, b01, b10, b11


@triton.jit
def _respond(u0, u1, v0, v1):
    # Exactly _forward's rounding, contraction and sign code.
    u0 = u0.to(tl.bfloat16).to(tl.float32)
    u1 = u1.to(tl.bfloat16).to(tl.float32)
    y = tl.abs(u0) * v0 + tl.abs(u1) * v1
    return y, (_sign(u0) + 1) | ((_sign(u1) + 1) << 2)


@triton.jit
def _forward_packed(XP, K, V, Y, S, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
                    W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.program_id(1)
    valid, b, hp, n = _lanes(B, H, W, BLOCK)
    a00, a01, a10, a11, b00, b01, b10, b11 = _responses(XP, K, b, c, hp, n, valid, C, H, W, KS, BLOCK)
    v0 = tl.load(V + c * 2).to(tl.float32)
    v1 = tl.load(V + c * 2 + 1).to(tl.float32)
    out = ((b * C + c) * H + 2 * hp) * (W // 2) + n
    y0, s0 = _respond(a00, a01, v0, v1)
    y1, s1 = _respond(a10, a11, v0, v1)
    tl.store(Y + out, _pack(y0, y1), valid)
    tl.store(S + out, (s0 | (s1 << 8)).to(tl.int16), valid)
    y0, s0 = _respond(b00, b01, v0, v1)
    y1, s1 = _respond(b10, b11, v0, v1)
    tl.store(Y + out + W // 2, _pack(y0, y1), valid)
    tl.store(S + out + W // 2, (s0 | (s1 << 8)).to(tl.int16), valid)


@triton.jit
def _magnitude(u):
    # Rounded response magnitude (as after abs in the forward).
    return tl.abs(u.to(tl.bfloat16).to(tl.float32))


@triton.jit
def _contract_packed(XP, K, G, P, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
                     W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    # dV partials from recomputed responses: identical rounded values to forward.
    c = tl.program_id(1)
    valid, b, hp, n = _lanes(B, H, W, BLOCK)
    a00, a01, a10, a11, b00, b01, b10, b11 = _responses(XP, K, b, c, hp, n, valid, C, H, W, KS, BLOCK)
    src = G + ((b * C + c) * H + 2 * hp) * (W // 2) + n
    g00, g01 = _pair(tl.load(src, valid, 0))
    g10, g11 = _pair(tl.load(src + W // 2, valid, 0))
    p0 = g00 * _magnitude(a00) + g01 * _magnitude(a10) + g10 * _magnitude(b00) + g11 * _magnitude(b10)
    p1 = g00 * _magnitude(a01) + g01 * _magnitude(a11) + g10 * _magnitude(b01) + g11 * _magnitude(b11)
    nt = tl.num_programs(0)
    tl.store(P + (c * 2) * nt + tl.program_id(0), tl.sum(p0, 0))
    tl.store(P + (c * 2 + 1) * nt + tl.program_id(0), tl.sum(p1, 0))


@triton.jit
def _grad_u_packed(G, S, V, GU, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
                   W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    # Zero-haloed bf16 grad-u planes, (H + 2KS - 2) x (W + 2KS - 2) per filter.
    c = tl.program_id(1)
    RG: tl.constexpr = H + 2 * (KS - 1)
    WG2: tl.constexpr = (W + 2 * (KS - 1)) // 2
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = t < B * RG * WG2
    b = t // (RG * WG2)
    q = t % (RG * WG2)
    row = q // WG2
    m = q % WG2
    h = row - (KS - 1)
    w2 = m - (KS - 1) // 2
    inside = valid & (h >= 0) & (h < H) & (w2 >= 0) & (w2 < W // 2)
    src = ((b * C + c) * H + h) * (W // 2) + w2
    g_lo, g_hi = _pair(tl.load(G + src, inside, 0))
    s = tl.load(S + src, inside, 1285).to(tl.int32)  # masked: both sign codes 1 (zero)
    v0 = tl.load(V + c * 2).to(tl.float32)
    v1 = tl.load(V + c * 2 + 1).to(tl.float32)
    s_lo = s & 255
    s_hi = (s >> 8) & 255
    # _grad_u: grouped 1x1 backward rounds to bf16 before abs backward.
    u0_lo = (g_lo * v0).to(tl.bfloat16).to(tl.float32) * ((s_lo & 3) - 1)
    u0_hi = (g_hi * v0).to(tl.bfloat16).to(tl.float32) * ((s_hi & 3) - 1)
    u1_lo = (g_lo * v1).to(tl.bfloat16).to(tl.float32) * (((s_lo >> 2) & 3) - 1)
    u1_hi = (g_hi * v1).to(tl.bfloat16).to(tl.float32) * (((s_hi >> 2) & 3) - 1)
    out = ((b * C + c) * 2 * RG + row) * WG2 + m
    tl.store(GU + out, _pack(u0_lo, u0_hi), valid)
    tl.store(GU + out + RG * WG2, _pack(u1_lo, u1_hi), valid)


@triton.jit
def _adjoint_taps(c0f0, c0f1, c1f0, c1f1, g0, g1, K0, K1, KS: tl.constexpr):
    # Output column b uses grad-u column b + KS - 1 - dx of the loaded window.
    for dx in tl.static_range(KS):
        k0 = tl.load(K0 + dx).to(tl.float32)
        k1 = tl.load(K1 + dx).to(tl.float32)
        c0f0 = c0f0 + g0[KS - 1 - dx] * k0
        c0f1 = c0f1 + g1[KS - 1 - dx] * k1
        c1f0 = c1f0 + g0[KS - dx] * k0
        c1f1 = c1f1 + g1[KS - dx] * k1
    return c0f0, c0f1, c1f0, c1f1


@triton.jit
def _input_grad_packed(GU, K, DX, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
                       W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr):
    # Padded-domain adjoint; lanes own 2x2 blocks of the (H+KS-1)x(W+KS-1) output.
    c = tl.program_id(1)
    valid, b, ip, n = _lanes(B, H + KS - 1, W + KS - 1, BLOCK)
    RG: tl.constexpr = H + 2 * (KS - 1)
    WG2: tl.constexpr = (W + 2 * (KS - 1)) // 2
    src = GU + ((b * C + c) * 2 * RG + 2 * ip) * WG2 + n
    K0 = K + c * 2 * KS * KS
    K1 = K0 + KS * KS
    z = tl.zeros((BLOCK,), tl.float32)
    a00, a01, a10, a11 = z, z, z, z
    b00, b01, b10, b11 = z, z, z, z
    # Output row a, tap dy reads grad-u row a + KS - 1 - dy: visit rows in
    # descending order so that dy ascends, as in _input_grad.
    for i in tl.static_range(KS + 1):
        r = KS - i
        g0 = _row(src + r * WG2, valid, (KS + 1) // 2)
        g1 = _row(src + (RG + r) * WG2, valid, (KS + 1) // 2)
        if r < KS:
            a00, a01, a10, a11 = _adjoint_taps(a00, a01, a10, a11, g0, g1,
                                               K0 + (KS - 1 - r) * KS, K1 + (KS - 1 - r) * KS, KS)
        if r > 0:
            b00, b01, b10, b11 = _adjoint_taps(b00, b01, b10, b11, g0, g1,
                                               K0 + (KS - r) * KS, K1 + (KS - r) * KS, KS)
    PW2: tl.constexpr = (W + KS - 1) // 2
    out = ((b * C + c) * (H + KS - 1) + 2 * ip) * PW2 + n
    tl.store(DX + out, _pack(a00 + a01, a10 + a11), valid)
    tl.store(DX + out + PW2, _pack(b00 + b01, b10 + b11), valid)


@triton.jit
def _weight_grad_packed(XP, GU, P, B: tl.constexpr, C: tl.constexpr, H: tl.constexpr,
                        W: tl.constexpr, KS: tl.constexpr, BLOCK: tl.constexpr,
                        NSPLIT: tl.constexpr):
    # Persistent over a contiguous lane range of one channel; 2*KS*KS fp32
    # register accumulators, one final reduction per coefficient.
    split, c = tl.program_id(0), tl.program_id(1)
    W2: tl.constexpr = W // 2
    LP: tl.constexpr = (H // 2) * W2
    PW2: tl.constexpr = (W + KS - 1) // 2
    RG: tl.constexpr = H + 2 * (KS - 1)
    WG2: tl.constexpr = (W + 2 * (KS - 1)) // 2
    NB: tl.constexpr = (B * LP + BLOCK - 1) // BLOCK
    acc = ()
    for i in tl.static_range(2 * KS * KS):
        acc = acc + (tl.zeros((BLOCK,), tl.float32),)
    for block in range(split * NB // NSPLIT, (split + 1) * NB // NSPLIT):
        t = block * BLOCK + tl.arange(0, BLOCK)
        valid = t < B * LP
        b = t // LP
        q = t % LP
        hp = q // W2
        n = q % W2
        gsrc = GU + ((b * C + c) * 2 * RG + 2 * hp + KS - 1) * WG2 + n + (KS - 1) // 2
        f0r0c0, f0r0c1 = _pair(tl.load(gsrc, valid, 0))
        f0r1c0, f0r1c1 = _pair(tl.load(gsrc + WG2, valid, 0))
        f1r0c0, f1r0c1 = _pair(tl.load(gsrc + RG * WG2, valid, 0))
        f1r1c0, f1r1c1 = _pair(tl.load(gsrc + (RG + 1) * WG2, valid, 0))
        gu = (f0r0c0, f0r0c1, f0r1c0, f0r1c1, f1r0c0, f1r0c1, f1r1c0, f1r1c1)
        src = XP + ((b * C + c) * (H + KS - 1) + 2 * hp) * PW2 + n
        for r in tl.static_range(KS + 1):
            x = _row(src + r * PW2, valid, (KS + 1) // 2)
            new = ()
            for f in tl.static_range(2):
                for dy in tl.static_range(KS):
                    for dx in tl.static_range(KS):
                        value = acc[(f * KS + dy) * KS + dx]
                        if dy == r:
                            value = value + gu[f * 4] * x[dx] + gu[f * 4 + 1] * x[dx + 1]
                        if dy == r - 1:
                            value = value + gu[f * 4 + 2] * x[dx] + gu[f * 4 + 3] * x[dx + 1]
                        new = new + (value,)
            acc = new
    for j in tl.static_range(2 * KS * KS):
        tl.store(P + (c * 2 * KS * KS + j) * NSPLIT + split, tl.sum(acc[j], 0))


@triton.jit
def _weight_reduce(P, DK, KS: tl.constexpr, NSPLIT: tl.constexpr, NP2: tl.constexpr,
                   SP2: tl.constexpr):
    c = tl.program_id(0)
    j = tl.arange(0, NP2)
    s = tl.arange(0, SP2)
    v = tl.load(P + (c * 2 * KS * KS + j[:, None]) * NSPLIT + s[None, :],
                (j[:, None] < 2 * KS * KS) & (s[None, :] < NSPLIT), 0)
    # Dense (C*2, 1, KS, KS) order; the store performs the final bf16 cast.
    tl.store(DK + c * 2 * KS * KS + j, tl.sum(v, 1), j < 2 * KS * KS)


@triton.jit
def _reflect_pad(X, XP, H: tl.constexpr, W: tl.constexpr, P: tl.constexpr, BLOCK: tl.constexpr):
    # F.pad(x, [P] * 4, mode='reflect') of one plane, stored as bf16.
    PW: tl.constexpr = W + 2 * P
    bc = tl.program_id(1)
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = t < (H + 2 * P) * PW
    i = _reflect(t // PW - P, H)
    j = _reflect(t % PW - P, W)
    v = tl.load(X + bc * H * W + i * W + j, valid, 0)
    tl.store(XP + bc * (H + 2 * P) * PW + t, v, valid)


def applies(x, kernels):
    """bf16 kernels at the packed shapes (fused_basis_config.uses_packed); everything else uses the reduce kernels."""
    return kernels.dtype == torch.bfloat16 and config.uses_packed(x.shape[2], x.shape[3])


def _padded(x, ks):
    # Reflect in x.dtype, then round to bf16 (store).
    b, c, h, w = x.shape
    p = ks // 2
    xp = torch.empty((b, c, h + 2 * p, w + 2 * p), device=x.device, dtype=torch.bfloat16)
    _reflect_pad[(triton.cdiv((h + 2 * p) * (w + 2 * p), 1024), b * c)](x, xp, h, w, p, 1024, num_warps=4)
    return xp


def forward(x, kernels, contract):
    b, c, h, w = x.shape
    ks = kernels.shape[2]
    y = torch.empty(x.shape, device=x.device, dtype=kernels.dtype)
    signs = torch.empty(x.shape, device=x.device, dtype=torch.uint8)
    cfg = PACKED['forward']
    _forward_packed[(triton.cdiv(b * (h // 2) * (w // 2), cfg['BLOCK']), c)](
        _padded(x, ks).view(torch.int32), kernels, contract, y.view(torch.int32), signs.view(torch.int16),
        b, c, h, w, ks, cfg['BLOCK'], num_warps=cfg['num_warps'], enable_fp_fusion=True)
    return y, signs


def backward(x, kernels, contract, signs, grad):
    b, c, h, w = x.shape
    ks = kernels.shape[2]
    if grad.storage_offset() % 2:
        grad = grad.clone()  # the word view needs an even element offset
    xp = _padded(x, ks).view(torch.int32)
    rg, wg = h + 2 * (ks - 1), w + 2 * (ks - 1)
    gu = torch.empty((b, c, 2, rg, wg), device=x.device, dtype=torch.bfloat16)
    cfg = PACKED['grad_u']
    _grad_u_packed[(triton.cdiv(b * rg * wg // 2, cfg['BLOCK']), c)](
        grad.view(torch.int32), signs.view(torch.int16), contract, gu.view(torch.int32),
        b, c, h, w, ks, cfg['BLOCK'], num_warps=cfg['num_warps'], enable_fp_fusion=True)
    padded_grad = torch.empty((b, c, h + ks - 1, w + ks - 1), device=x.device, dtype=kernels.dtype)
    cfg = PACKED['input_gradient']
    _input_grad_packed[(triton.cdiv(b * ((h + ks - 1) // 2) * ((w + ks - 1) // 2), cfg['BLOCK']), c)](
        gu.view(torch.int32), kernels, padded_grad.view(torch.int32),
        b, c, h, w, ks, cfg['BLOCK'], num_warps=cfg['num_warps'], enable_fp_fusion=True)
    cfg = PACKED['weight_gradient']
    partial_k = torch.empty((c, 2 * ks * ks, cfg['NSPLIT']), device=x.device, dtype=torch.float32)
    _weight_grad_packed[(cfg['NSPLIT'], c)](
        xp, gu.view(torch.int32), partial_k, b, c, h, w, ks, cfg['BLOCK'], cfg['NSPLIT'],
        num_warps=cfg['num_warps'], enable_fp_fusion=True)
    dk = torch.empty(kernels.shape, device=kernels.device, dtype=kernels.dtype)
    _weight_reduce[(c,)](partial_k, dk, ks, cfg['NSPLIT'], triton.next_power_of_2(2 * ks * ks),
                         triton.next_power_of_2(cfg['NSPLIT']), enable_fp_fusion=False)
    cfg = PACKED['contract']
    nt = triton.cdiv(b * (h // 2) * (w // 2), cfg['BLOCK'])
    partial_v = torch.empty((c, 2, nt), device=x.device, dtype=torch.float32)
    _contract_packed[(nt, c)](
        xp, kernels, grad.view(torch.int32), partial_v, b, c, h, w, ks, cfg['BLOCK'],
        num_warps=cfg['num_warps'], enable_fp_fusion=True)
    dv = torch.empty(contract.shape, device=contract.device, dtype=contract.dtype)
    _contract_reduce[(c * 2,)](partial_v, dv, nt, triton.next_power_of_2(nt), enable_fp_fusion=False)
    return padded_grad, dk, dv
