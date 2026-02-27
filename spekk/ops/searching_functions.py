__all__ = ["argmax", "argmin", "nonzero", "searchsorted", "where"]

from typing import TYPE_CHECKING, Literal, TypeVar

from spekk.ops._backend import backend
from spekk.ops._types import Dim, undefined_dim
from spekk.ops._util import ensure_backend_compatible_data, ensure_broadcastable
from spekk.ops.array_object import array
from spekk.ops.exceptions import MismatchedDimensionsError


def argmax(x: array, /, *, axis: int | str | None = None, keepdims: bool = False) -> array:
    """
    Returns the indices of the maximum values along a specified axis.

    When the maximum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    x: array
        input array. Has a real-valued data type.
    axis: int | str | None
        axis along which to search. If ``None``, returns the index of the maximum value of the flattened array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) are included in the result as singleton dimensions, and the result is broadcastable with the input array. Otherwise, if ``False``, the reduced axes (dimensions) are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if ``axis`` is ``None``, a zero-dimensional array containing the index of the first occurrence of the maximum value; otherwise, a non-zero-dimensional array containing the indices of the maximum values. The returned array has the default array index data type.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if not keepdims:
        if axis is None:
            dims = []
        else:
            del dims[axis]
    return array(backend.argmax(x._data, axis=axis, keepdims=keepdims), dims)


def argmin(x: array, /, *, axis: int | str | None = None, keepdims: bool = False) -> array:
    """
    Returns the indices of the minimum values along a specified axis.

    When the minimum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

    Parameters
    ----------
    x: array
        input array. Has a real-valued data type.
    axis: int | str | None
        axis along which to search. If ``None``, returns the index of the minimum value of the flattened array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes (dimensions) are included in the result as singleton dimensions, and the result is broadcastable with the input array. Otherwise, if ``False``, the reduced axes (dimensions) are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if ``axis`` is ``None``, a zero-dimensional array containing the index of the first occurrence of the minimum value; otherwise, a non-zero-dimensional array containing the indices of the minimum values. The returned array has the default array index data type.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if not keepdims:
        if axis is None:
            dims = []
        else:
            del dims[axis]
    return array(backend.argmin(x._data, axis=axis, keepdims=keepdims), dims)


def nonzero(x: array, /, *, dim: str | None = None) -> tuple[array, ...]:
    """
    Returns the indices of the array elements which are non-zero.

    .. note::
       If ``x`` has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero.

    .. note::
       If ``x`` has a boolean data type, non-zero elements are those elements which are equal to ``True``.

    .. admonition:: Data-dependent output shape
       :class: important

       The output shape of this function depends on the data values in the input array. Array libraries that build computation graphs (e.g., JAX, Dask) may find this function difficult to implement without knowing array values and may choose to omit it.

    Parameters
    ----------
    x: array
        input array. Must have a positive rank. If ``x`` is zero-dimensional, the function raises an exception.
    dim: str | None
        name to assign to the output dimension of each returned index array. Default: ``None``.

    Returns
    -------
    out: tuple[array, ...]
        a tuple of ``k`` arrays, one for each dimension of ``x`` and each of size ``n`` (where ``n`` is the total number of non-zero elements), containing the indices of the non-zero elements in that dimension. The indices are in row-major, C-style order. The returned array has the default array index data type.
    """
    if dim is None:
        dim = undefined_dim
    return tuple(array(result, [dim]) for result in backend.nonzero(x._data))


def searchsorted(
    x1: array,
    x2: int | float | array,
    /,
    *,
    side: Literal["left", "right"] = "left",
    sorter: array | None = None,
) -> array:
    """
    Finds the indices into ``x1`` such that, if the corresponding elements in ``x2`` were inserted before the indices, the order of ``x1``, when sorted in ascending order, would be preserved.

    Parameters
    ----------
    x1: array
        input array. Must be a one-dimensional array. Has a real-valued data type. If ``sorter`` is ``None``, must be sorted in ascending order; otherwise, ``sorter`` must be an array of indices that sort ``x1`` in ascending order.
    x2: int | float | array
        array containing search values. Has a real-valued data type.
    side: Literal['left', 'right']
        argument controlling which index is returned if a value lands exactly on an edge.

        Let ``x`` be an array of rank ``N`` where ``v`` is an individual element given by ``v = x2[n,m,...,j]``.

        If ``side == 'left'``, then

        - each returned index ``i`` satisfies the index condition ``x1[i-1] < v <= x1[i]``.
        - if no index satisfies the index condition, then the returned index for that element is ``0``.

        Otherwise, if ``side == 'right'``, then

        - each returned index ``i`` satisfies the index condition ``x1[i-1] <= v < x1[i]``.
        - if no index satisfies the index condition, then the returned index for that element is ``N``, where ``N`` is the number of elements in ``x1``.

        Default: ``'left'``.
    sorter: array | None
        array of indices that sort ``x1`` in ascending order. The array must have the same shape as ``x1`` and have an integer data type. Default: ``None``.

    Returns
    -------
    out: array
        an array of indices with the same shape as ``x2``. The returned array has the default array index data type.

    Notes
    -----
    For real-valued floating-point arrays, the sort order of NaNs and signed zeros is backend-dependent. Accordingly, when a real-valued floating-point array contains NaNs and signed zeros, what constitutes ascending order may vary among backends.

    Results are consistent with ``sort`` and ``argsort``: if a value in ``x2`` is inserted into ``x1`` at the corresponding index in the output array and ``sort`` is invoked on the resultant array, the sorted result is in the same order.
    """
    if sorter is not None:
        sorter = sorter._data
        if sorter.shape != x1._data.shape:
            raise MismatchedDimensionsError(
                f"sorter must have the same shape as x1, but received sorter.shape={sorter.shape} and x1.shape={x1.shape}."
            )
    return array(
        backend.searchsorted(x1._data, x2._data, side=side, sorter=sorter),
        x2._dims,
    )


def where(
    condition: bool | array,
    x1: bool | int | float | complex | array,
    x2: bool | int | float | complex | array,
    /,
) -> array:
    """
    Returns elements chosen from ``x1`` or ``x2`` depending on ``condition``.

    Parameters
    ----------
    condition: bool | array
        when ``True``, yield ``x1_i``; otherwise, yield ``x2_i``. Broadcasted with ``x1`` and ``x2``.
    x1: bool | int | float | complex | array
        first input array. Broadcasted with ``condition`` and ``x2``.
    x2: bool | int | float | complex | array
        second input array. Broadcasted with ``condition`` and ``x1``.

    Returns
    -------
    out: array
        an array with elements from ``x1`` where ``condition`` is ``True``, and elements from ``x2`` elsewhere. The returned array has a data type determined by type-promotion rules with the arrays ``x1`` and ``x2``.
    """

    broadcasted_dims, (condition, x1, x2) = ensure_broadcastable(condition, x1, x2)
    condition, x1, x2 = ensure_backend_compatible_data(condition, x1, x2)
    return array(backend.where(condition, x1, x2), broadcasted_dims)
