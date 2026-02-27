__all__ = ["cumulative_sum", "max", "mean", "min", "prod", "std", "sum", "var"]


from spekk.ops._backend import backend
from spekk.ops._types import Dim, dtype
from spekk.ops._util import get_reduction_axes_and_resulting_dims
from spekk.ops.array_object import array
from spekk.ops.data_types import DType


def cumulative_sum(
    x: array,
    /,
    *,
    axis: int | str | None = None,
    dtype: dtype | None = None,
    include_initial: bool = False,
) -> array:
    """
    Calculates the cumulative sum of elements in the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a numeric data type.
    axis: int | str | None
        axis along which to compute the cumulative sum. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension.

        If ``x`` has more than one dimension, providing an ``axis`` is required.

    dtype: dtype | None
        data type of the returned array. If ``None``, the returned array has the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type, in which case the default integer data type (or its unsigned equivalent) is used. If specified and differs from the data type of ``x``, the input array is cast before computing the sum. Default: ``None``.

    include_initial: bool
        whether to include the initial value (zero) as the first value in the output. Default: ``False``.

    Returns
    -------
    out: array
        an array containing the cumulative sums.

        Let ``N`` be the size of the axis along which to compute the cumulative sum.

        -   if ``include_initial`` is ``True``, the returned array has the same shape as ``x``, except the size of the cumulated axis is ``N+1``.
        -   if ``include_initial`` is ``False``, the returned array has the same shape as ``x``.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if axis is None:
        if x.ndim != 1:
            raise ValueError("dim must be provided when x has more than one dimension.")
        axis = 0

    axis = x.dims.index(axis) if isinstance(axis, Dim) else axis
    data = backend.cumulative_sum(
        x.data, axis=axis, dtype=dtype, include_initial=include_initial
    )
    return array(data, list(x.dims))


def max(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    keepdims: bool = False,
) -> array:
    """
    Calculates the maximum value of the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a real-valued data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which maximum values are computed. By default, the maximum value is computed over the entire array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the maximum value was computed over the entire array, a zero-dimensional array containing the maximum value; otherwise, a non-zero-dimensional array containing the maximum values. The returned array has the same data type as ``x``.

    Notes
    -----

    - The behavior when the reduction is over zero elements is backend-dependent.
    - The order of signed zeros is backend-dependent.
    - If any element is ``NaN``, the maximum value is ``NaN``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.max(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)


def mean(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    keepdims: bool = False,
) -> array:
    """
    Calculates the arithmetic mean of the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a numeric data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which arithmetic means are computed. By default, the mean is computed over the entire array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the arithmetic mean was computed over the entire array, a zero-dimensional array containing the arithmetic mean; otherwise, a non-zero-dimensional array containing the arithmetic means. If ``x`` has an integer data type, the returned array has the default real-valued floating-point data type; otherwise, the returned array has the same data type as ``x``.

    Notes
    -----

    - If the number of elements is ``0``, the arithmetic mean is ``NaN``.
    - If any element is ``NaN``, the arithmetic mean is ``NaN``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.mean(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)


def min(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    keepdims: bool = False,
) -> array:
    """
    Calculates the minimum value of the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a real-valued data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which minimum values are computed. By default, the minimum value is computed over the entire array. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the minimum value was computed over the entire array, a zero-dimensional array containing the minimum value; otherwise, a non-zero-dimensional array containing the minimum values. The returned array has the same data type as ``x``.

    Notes
    -----

    - The behavior when the reduction is over zero elements is backend-dependent.
    - The order of signed zeros is backend-dependent.
    - If any element is ``NaN``, the minimum value is ``NaN``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.min(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)


def prod(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    dtype: dtype | None = None,
    keepdims: bool = False,
) -> array:
    """
    Calculates the product of input array ``x`` elements.

    Parameters
    ----------
    x: array
        input array. Has a numeric data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which products are computed. By default, the product is computed over the entire array. If a tuple, products are computed over multiple axes. Default: ``None``.
    dtype: dtype | None
        data type of the returned array. If ``None``, the returned array has the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type, in which case the default integer data type is used. If the resolved data type differs from the data type of ``x``, the input array is cast before computing the product. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the product was computed over the entire array, a zero-dimensional array containing the product; otherwise, an array containing the products. The returned array has a data type as described by the ``dtype`` parameter above.

    Notes
    -----

    -   If the number of elements over which to compute the product is ``0``, the product is ``1`` (i.e., the empty product).
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.prod(x._data, axis=axis, dtype=dtype, keepdims=keepdims)
    return array(data, dims)


def std(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> array:
    """
    Calculates the standard deviation of the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a numeric data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which standard deviations are computed. By default, the standard deviation is computed over the entire array. If a tuple, standard deviations are computed over multiple axes. Default: ``None``.
    correction: int | float
        degrees of freedom adjustment. The divisor used in the calculation is ``N - correction`` where ``N`` is the number of elements. Use ``0`` for population standard deviation and ``1`` for sample standard deviation (Bessel's correction). Default: ``0``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the standard deviation was computed over the entire array, a zero-dimensional array containing the standard deviation; otherwise, an array containing the standard deviations. If ``x`` has an integer data type, the returned array has the default real-valued floating-point data type; otherwise, the returned array has the same data type as ``x``.

    Notes
    -----

    -   If ``N - correction`` is less than or equal to ``0``, the standard deviation is ``NaN``.
    -   If any element is ``NaN``, the standard deviation is ``NaN``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.std(x._data, axis=axis, correction=correction, keepdims=keepdims)
    return array(data, dims)


def sum(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    dtype: dtype | None = None,
    keepdims: bool = False,
) -> array:
    """
    Calculates the sum of the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a numeric data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which sums are computed. By default, the sum is computed over the entire array. If a tuple, sums are computed over multiple axes. Default: ``None``.
    dtype: dtype | None
        data type of the returned array. If ``None``, the returned array has the same data type as ``x``, unless ``x`` has an integer data type supporting a smaller range of values than the default integer data type, in which case the default integer data type is used. If the resolved data type differs from the data type of ``x``, the input array is cast before computing the sum. Default: ``None``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the sum was computed over the entire array, a zero-dimensional array containing the sum; otherwise, an array containing the sums. The returned array has a data type as described by the ``dtype`` parameter above.

    Notes
    -----

    -   If the number of elements over which to compute the sum is ``0``, the sum is ``0`` (i.e., the empty sum).
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.sum(x._data, axis=axis, dtype=dtype, keepdims=keepdims)
    return array(data, dims)


def var(
    x: array,
    /,
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> array:
    """
    Calculates the variance of the input array ``x``.

    Parameters
    ----------
    x: array
        input array. Has a numeric data type.
    axis: int | str | tuple[int | str, ...] | None
        axis or axes along which variances are computed. By default, the variance is computed over the entire array. If a tuple, variances are computed over multiple axes. Default: ``None``.
    correction: int | float
        degrees of freedom adjustment. The divisor used in the calculation is ``N - correction`` where ``N`` is the number of elements. Use ``0`` for population variance and ``1`` for sample variance (Bessel's correction). Default: ``0``.
    keepdims: bool
        if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

    Returns
    -------
    out: array
        if the variance was computed over the entire array, a zero-dimensional array containing the variance; otherwise, an array containing the variances. If ``x`` has an integer data type, the returned array has the default real-valued floating-point data type; otherwise, the returned array has the same data type as ``x``.

    Notes
    -----

    -   If ``N - correction`` is less than or equal to ``0``, the variance is ``NaN``.
    -   If any element is ``NaN``, the variance is ``NaN``.
    """
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x._dims, keepdims)
    data = backend.var(x._data, axis=axis, correction=correction, keepdims=keepdims)
    return array(data, dims)
