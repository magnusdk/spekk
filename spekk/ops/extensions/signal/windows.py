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

def tukey(N: int, *, alpha: float = 0.5, dim: Dim = None):
    """Tukey (tapered cosine) window.

    Parameters
    ----------
    N : int
        Number of points in the window.
    alpha : float
        Shape parameter. 0 = rectangular, 1 = Hann, 0 < alpha < 1 = tapered cosine.
    dim : str or None
        Named dimension for the output array.
    """
    if alpha <= 0:
        return ops.ones((N,), dims=[dim])
    if alpha >= 1:
        return hanning(N, dim=dim)
    n = ops.arange(N, dim=dim)
    width = alpha * (N - 1) / 2.0
    left = 0.5 * (1 + ops.cos(ops.pi * (n / width - 1)))
    right = 0.5 * (1 + ops.cos(ops.pi * (n / width - 2.0 / alpha + 1)))
    w = ops.where(n < width, left,
        ops.where(n <= (N - 1) * (1 - alpha / 2), 1.0, right))
    return w

