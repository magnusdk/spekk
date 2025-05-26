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

