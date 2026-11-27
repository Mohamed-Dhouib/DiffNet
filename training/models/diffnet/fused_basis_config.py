"""Launch parameters of the fused zero-sum filter kernels (bf16 training).

Even H and W except 16x16: the packed word-pair kernels (fused_basis_packed, PACKED below); 16x16 and odd shapes:
the reduce kernels (fused_basis_reduce) with the launch configuration of reduce_config_index.
"""

PACKED = dict(forward=dict(BLOCK=512, num_warps=4), grad_u=dict(BLOCK=1024, num_warps=4),
              input_gradient=dict(BLOCK=256, num_warps=4),
              weight_gradient=dict(BLOCK=128, num_warps=4, NSPLIT=3),
              contract=dict(BLOCK=512, num_warps=4))


def uses_packed(h, w):
    if not h == w == 16:
        return h % 2 == 0 and w % 2 == 0
    return False


def reduce_config_index(h, w):
    if h is None or w is None:
        raise ValueError('the reduce kernels need the actual H and W')
    return 0 if h == w == 16 else 2


def pinned_config(backend, h=None, w=None):
    if backend == 'reduce':
        from .fused_basis_reduce import _weight_grad
        return _weight_grad.configs[reduce_config_index(h, w)]
    raise ValueError(backend)


def reduce_input_config(h, w):
    small = h == w == 16
    return (
        dict(kind='scalar', BLOCK=256, num_warps=8),
        dict(kind='scalar', BLOCK=128, num_warps=4),
        dict(kind='rows', TH=4 if small else 8, TW=32, num_warps=4),
        dict(kind='rows', TH=4, TW=64, num_warps=4),
    )[reduce_config_index(h, w)]
