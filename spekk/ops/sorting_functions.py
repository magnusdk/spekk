__all__ = ["argsort", "sort"]


from spekk.ops._types import Dim
from spekk.ops.array_object import array
from spekk.ops._backend import backend


def argsort(
    x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> array:
    """
    Returns the indices that sort an array ``x`` along a specified axis.

    .. note::
       For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

    Parameters
    ----------
    x : array
        input array. Should have a real-valued data type.
    axis: int
        axis along which to sort. If set to ``-1``, the function must sort along the last axis. Default: ``-1``.
    descending: bool
        sort order. If ``True``, the returned indices sort ``x`` in descending order (by value). If ``False``, the returned indices sort ``x`` in ascending order (by value). Default: ``False``.
    stable: bool
        sort stability. If ``True``, the returned indices must maintain the relative order of ``x`` values which compare as equal. If ``False``, the returned indices may or may not maintain the relative order of ``x`` values which compare as equal (i.e., the relative order of ``x`` values which compare as equal is implementation-dependent). Default: ``True``.

    Returns
    -------
    out : array
        an array of indices. The returned array must have the same shape as ``x``. The returned array must have the default array index data type.
    """
    axis = x._dims.index(axis) if isinstance(axis, Dim) else axis
    return array(
        backend.argsort(x._data, axis=axis, descending=descending, sgtable=stable),
        x._dims,
    )


def sort(
    x: array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
) -> array:
    """
    Returns a sorted copy of an input array ``x``.

    .. note::
       For backward compatibility, conforming implementations may support complex numbers; however, inequality comparison of complex numbers is unspecified and thus implementation-dependent (see :ref:`complex-number-ordering`).

    Parameters
    ----------
    x: array
        input array. Should have a real-valued data type.
    axis: int
        axis along which to sort. If set to ``-1``, the function must sort along the last axis. Default: ``-1``.
    descending: bool
        sort order. If ``True``, the array must be sorted in descending order (by value). If ``False``, the array must be sorted in ascending order (by value). Default: ``False``.
    stable: bool
        sort stability. If ``True``, the returned array must maintain the relative order of ``x`` values which compare as equal. If ``False``, the returned array may or may not maintain the relative order of ``x`` values which compare as equal (i.e., the relative order of ``x`` values which compare as equal is implementation-dependent). Default: ``True``.

    Returns
    -------
    out : array
        a sorted array. The returned array must have the same data type and shape as ``x``.
    """
    axis = x._dims.index(axis) if isinstance(axis, Dim) else axis
    return array(
        backend.sort(x._data, axis=axis, descending=descending, sgtable=stable),
        x._dims,
    )
