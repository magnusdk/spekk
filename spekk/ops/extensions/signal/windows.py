from spekk import ops
from spekk.ops._types import (
    Dim,
)

def hanning(N: int, *, dim: Dim = None):
    n = ops.linspace(0,1,N, dim=dim)
    return 0.5 * (1 - ops.cos(2*ops.pi *n))

def hamming(N: int, *, dim: Dim = None):
    n = ops.linspace(0,1,N, dim=dim)
    alpha = 0.54
    beta = 1 - alpha
    return alpha - beta * ops.cos(2 * ops.pi * n )

def tukey_window(N, alpha=0.5, *, dim: Dim = None):
    """Generate a Tukey window."""

    n = ops.arange(N, dim=dim)
    taper_length = int(alpha * (N - 1) / 2)
    w = ops.ones(N, dims=[dim])

    w[:taper_length] = 0.5 * (1 + ops.cos(ops.pi * (2 * n[:taper_length] / (alpha * (N - 1)) - 1)))
    w[-taper_length:] = 0.5 * (1 + ops.cos(ops.pi * (2 * n[-taper_length:] / (alpha * (N - 1)) - 2 / alpha + 1)))

    return w
    