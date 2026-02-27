__all__ = ["argsort", "sort"]


from spekk.ops._types import Dim
from spekk.ops.array_object import array
from spekk.ops._backend import backend


def argsort(
    x: array, /, *, axis: int | str = -1, descending: bool = False, stable: bool = True
) -> array:
    """
    Returns the indices that sort an array ``x`` along a specified axis.

    Parameters
    ----------
    x : array
        input array. Has a real-valued data type.
    axis: int | str
        axis along which to sort. May be an integer index or a named dimension name. If set to ``-1``, the function sorts along the last axis. Default: ``-1``.
    descending: bool
        sort order. If ``True``, the returned indices sort ``x`` in descending order (by value). If ``False``, the returned indices sort ``x`` in ascending order (by value). Default: ``False``.
    stable: bool
        sort stability. If ``True``, the returned indices maintain the relative order of ``x`` values which compare as equal. If ``False``, the relative order of ``x`` values which compare as equal is backend-dependent. Default: ``True``.

    Returns
    -------
    out : array
        an array of indices. The returned array has the same shape as ``x`` and the default array index data type.
    """
    axis = x._dims.index(axis) if isinstance(axis, Dim) else axis
    return array(
        backend.argsort(x._data, axis=axis, descending=descending, stable=stable),
        x._dims,
    )


def sort(
    x: array, /, *, axis: int | str = -1, descending: bool = False, stable: bool = True
) -> array:
    """
    Returns a sorted copy of an input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a real-valued data type.
    axis: int | str
        axis along which to sort. May be an integer index or a named dimension name. If set to ``-1``, the function sorts along the last axis. Default: ``-1``.
    descending: bool
        sort order. If ``True``, the array is sorted in descending order (by value). If ``False``, the array is sorted in ascending order (by value). Default: ``False``.
    stable: bool
        sort stability. If ``True``, the returned array maintains the relative order of ``x`` values which compare as equal. If ``False``, the relative order of ``x`` values which compare as equal is backend-dependent. Default: ``True``.

    Returns
    -------
    out : array
        a sorted array with the same data type and shape as ``x``.
    """
    axis = x._dims.index(axis) if isinstance(axis, Dim) else axis
    return array(
        backend.sort(x._data, axis=axis, descending=descending, stable=stable),
        x._dims,
    )
