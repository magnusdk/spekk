__all__ = ["all", "any"]


from spekk.ops._util import get_reduction_axes_and_resulting_dims
from spekk.ops.array_object import array
from spekk.ops._backend import backend


def all(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    keepdims: bool = False,
) -> array:
    """
    Tests whether all input array elements evaluate to ``True`` along a specified axis.

    .. note::
       Positive infinity, negative infinity, and NaN evaluate to ``True``.

    .. note::
       If ``x`` has a complex floating-point data type, elements having a non-zero component (real or imaginary) evaluate to ``True``.

    .. note::
       If ``x`` is an empty array or the size of the axis along which to evaluate elements is zero, the test result is ``True``.

    Parameters
    ----------
    x: array
        Input array.
    axis: int | str | tuple[int | str, ...] | None
        Axis or axes along which to perform a logical AND reduction. If ``None``, reduces over the entire array. Default: ``None``.
    keepdims: bool
        If ``True``, reduced axes are kept as singleton dimensions. Default: ``False``.

    Returns
    -------
    out: array
        Array containing the test result(s) with data type ``bool``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.all(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)


def any(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    keepdims: bool = False,
) -> array:
    """
    Tests whether any input array element evaluates to ``True`` along a specified axis.

    .. note::
       Positive infinity, negative infinity, and NaN evaluate to ``True``.

    .. note::
       If ``x`` has a complex floating-point data type, elements having a non-zero component (real or imaginary) evaluate to ``True``.

    .. note::
       If ``x`` is an empty array or the size of the axis along which to evaluate elements is zero, the test result is ``False``.

    Parameters
    ----------
    x: array
        Input array.
    axis: int | str | tuple[int | str, ...] | None
        Axis or axes along which to perform a logical OR reduction. If ``None``, reduces over the entire array. Default: ``None``.
    keepdims: bool
        If ``True``, reduced axes are kept as singleton dimensions. Default: ``False``.

    Returns
    -------
    out: array
        Array containing the test result(s) with data type ``bool``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.any(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)
