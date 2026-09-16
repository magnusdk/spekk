"""Special functions used by signal processing extensions."""

from spekk import ops
from spekk.ops._backend import backend


def i0(x: ops.array | float, /) -> ops.array:
    """Modified Bessel function of the first kind, order zero.

    Accepts real inputs, preserves named dimensions and floating-point dtype,
    and delegates to the active backend. Integral inputs use the backend's
    default floating-point dtype. Complex inputs are not supported.
    """
    values = ops.asarray(x)
    if ops.isdtype(values.dtype, "complex floating"):
        raise ValueError("i0 requires real input")
    if not ops.isdtype(values.dtype, "real floating"):
        values = ops.astype(values, ops.asarray(0.0).dtype)
    return ops.array(backend.i0(values.data), dims=values.dims)