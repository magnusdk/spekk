from spekk import ops
from spekk.ops._types import (
    Dim,
    undefined_dim,
)
from spekk.ops.extensions.signal.special import i0


def kaiser(N: int, beta: float, *, dim: Dim = None):
    """Symmetric Kaiser window of length N, matching scipy.signal.windows.kaiser.

    ``beta`` controls sidelobe suppression. The result uses the active backend
    and named dimension. Large beta can overflow i0 at the backend's precision.
    """
    if dim is None:
        dim = undefined_dim
    if int(N) != N or N < 0:
        raise ValueError("Window length must be a non-negative integer")
    N = int(N)
    if N <= 1:
        return ops.ones((N,), dims=[dim])
    position = 2 * ops.arange(N, dim=dim) / (N - 1) - 1
    argument = beta * ops.sqrt(ops.maximum(0.0, 1 - position**2))
    return i0(argument) / i0(beta)

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

