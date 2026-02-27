__all__ = ["take"]

from spekk.ops._backend import backend
from spekk.ops._types import Dim
from spekk.ops.array_object import array


def take(
    x: array,
    indices: array,
    /,
    *,
    axis: int | str | None = None,
) -> array:
    """
    Returns elements of an array along an axis.

    Conceptually, ``take(x, indices, axis=3)`` is equivalent to
    ``x[:,:,:,indices,...]``.

    Parameters
    ----------
    x: array
        Input array.
    indices: array
        Array indices. Must be zero- or one-dimensional with an integer data
        type. Out-of-bounds behavior is backend-dependent.
    axis: int | str | None
        Axis over which to select values. Can be a dimension name or an integer
        position. If ``axis`` is negative, the axis is counted from the last
        dimension.

        If ``x`` is a one-dimensional array, providing an ``axis`` is optional;
        however, if ``x`` has more than one dimension, providing an ``axis`` is
        required.

    Returns
    -------
    out: array
        An array with the same data type and rank as ``x``. The shape is the
        same as ``x`` except along ``axis``, whose size equals the number of
        elements in ``indices``. If ``indices`` is zero-dimensional, the
        specified axis is removed.
    """


    # x, indices = array(x, dtype=x.dtype, device=x.device), array(indices, dtype=int, device=x.device)
    x, indices = array(x, dtype=x.dtype), array(indices, dtype=int)

    # array-api only allows indexing with 1D arrays. We differ because we also allow
    # 0D arrays (basically just int).
    if indices.ndim not in {0, 1}:
        raise ValueError("Indices must be a one-dimensional array of integers.")
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    indices = indices._data if isinstance(indices, array) else indices
    dims = list(x._dims)
    if indices.ndim == 0:
        del dims[axis]
    return array(backend.take(x._data, indices, axis=axis), dims)
    