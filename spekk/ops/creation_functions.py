__all__ = [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]
from collections.abc import Buffer

import numpy as np

from spekk import ops
from spekk.ops._backend import backend
from spekk.ops._types import (
    BackendArray,
    BackendDevice,
    BackendDtype,
    NestedSequence,
    _UndefinedDim,
)
from spekk.ops._util import ensure_backend_compatible_data, ensure_broadcastable
from spekk.ops.array_object import array
from spekk.ops.data_types import DType


def arange(
    start: int | float | array,
    /,
    stop: int | float | array | None = None,
    step: int | float | array = 1,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dim: str | None = None,
) -> array:
    """
    Returns evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array.

    Parameters
    ----------
    start: int | float | array
        if ``stop`` is specified, the start of the interval (inclusive); otherwise, the end of the interval (exclusive) and the starting value defaults to ``0``.
    stop: int | float | array | None
        the end of the interval (exclusive). Default: ``None``.
    step: int | float | array
        the distance between two adjacent elements (``out[i+1] - out[i]``). Must not be ``0``; may be negative, which results in an empty array if ``stop >= start``. Default: ``1``.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the type is inferred from ``start``, ``stop``, and ``step``: all-integer inputs produce the default integer dtype; any floating-point input produces the default floating-point dtype. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dim: str | None
        name of the dimension in the output array. If ``None``, the dimension is unnamed. Default: ``None``.

    Returns
    -------
    out: array
        a one-dimensional array of evenly spaced values. The length is ``ceil((stop-start)/step)`` when ``stop - start`` and ``step`` have the same sign, and ``0`` otherwise.
    """
    dims = None if dim is None else [dim]
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    start, stop, step = ensure_backend_compatible_data(start, stop, step)
    return array(
        backend.arange(start, stop, step, dtype=dtype, device=device),
        dims,
        device=device,
    )


def asarray(
    obj: (
        bool
        | int
        | float
        | complex
        | array
        | NestedSequence[bool | int | float | complex]
        | Buffer
        | BackendArray
    ),
    /,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    copy: bool | None = None,
    dims: list[str] | None = None,
) -> array:
    r"""
    Convert the input to an array.

    Parameters
    ----------
    obj: bool | int | float | complex | array | NestedSequence[bool | int | float | complex] | Buffer | BackendArray
        object to be converted to an array. May be a Python scalar, a (possibly nested) sequence of Python scalars, an object supporting the Python buffer protocol, or an existing ``array``.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the type is inferred from ``obj``. For Python scalars: ``bool`` inputs produce ``bool``; ``int`` (or mixed ``bool``/``int``) produces the default integer dtype; ``float`` produces the default floating-point dtype; ``complex`` produces the default complex dtype. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. If ``None`` and ``obj`` is an array, the device is inferred from ``obj``. Default: ``None``.
    copy: bool | None
        whether to copy the input. If ``True``, always copies. If ``False``, never copies and raises ``ValueError`` if a copy would be required. If ``None``, copies only when necessary. Default: ``None``.
    dims: list[str] | None
        names of the dimensions in the output array. If ``None``, all dimensions are unnamed. Default: ``None``.

    Returns
    -------
    out: array
        an array containing the data from ``obj``.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    if isinstance(obj, array):
        if dims is None:
            dims = obj.dims
        obj = obj.data

    data = backend.asarray(obj, dtype=dtype, device=device, copy=copy)
    return array(data, dims, device=device)


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dims: list[str] | None = None,
) -> array:
    """
    Returns an uninitialized array having a specified `shape`.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the default floating-point dtype is used. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[str] | None
        names of the dimensions in the output array. If ``None``, all dimensions are unnamed. Default: ``None``.

    Returns
    -------
    out: array
        an array containing uninitialized data.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(backend.empty(shape, dtype=dtype, device=device), dims, device=device)


def empty_like(
    x: array,
    /,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
) -> array:
    """
    Returns an uninitialized array with the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from ``x``. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. If ``None``, the device is inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array with the same shape and dimensions as ``x`` containing uninitialized data.
    """
    if device is None:
        device = ops.backend.device
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    return array(
        backend.empty_like(x.data, dtype=dtype, device=device),
        x.dims,
        device=device,
    )


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dims: list[str] | None = None,
) -> array:
    r"""
    Returns a two-dimensional array with ones on the ``k``\th diagonal and zeros elsewhere.

    Parameters
    ----------
    n_rows: int
        number of rows in the output array.
    n_cols: int | None
        number of columns in the output array. If ``None``, defaults to ``n_rows``. Default: ``None``.
    k: int
        index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal, and ``0`` to the main diagonal. Default: ``0``.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the default floating-point dtype is used. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[str] | None
        names of the dimensions in the output array. If ``None``, all dimensions are unnamed. Default: ``None``.

    Returns
    -------
    out: array
        an array where all elements are zero except for the ``k``\th diagonal, whose values are one.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(
        backend.eye(n_rows, n_cols, k=k, dtype=dtype, device=device),
        dims,
        device=device,
    )


def from_dlpack(
    x: object,
    /,
    *,
    device: BackendDevice | None = None,
    copy: bool | None = None,
    dims: list[str] | None = None,
) -> array:
    """
    Returns a new array containing the data from another (array) object with a ``__dlpack__`` method.

    Parameters
    ----------
    x: object
        input object exposing a ``__dlpack__`` method.
    device: BackendDevice | None
        device on which to place the created array. If ``None`` and ``x`` supports DLPack, the output array is placed on the same device as ``x``. Default: ``None``.
    copy: bool | None
        whether to copy the input. If ``True``, always copies. If ``False``, never copies and raises ``BufferError`` if a copy would be required (e.g., for a cross-device transfer). If ``None``, reuses the existing buffer if possible and copies otherwise. Default: ``None``.
    dims: list[str] | None
        names of the dimensions in the output array. If ``None``, all dimensions are unnamed. Default: ``None``.

    Returns
    -------
    out: array
        an array containing the data in ``x``.

    Raises
    ------
    BufferError
        Raised by the ``__dlpack__`` or ``__dlpack_device__`` methods on ``x`` when the data cannot be exported (e.g., incompatible dtype, strides, or device, or insufficient memory). ``from_dlpack`` propagates these exceptions.
    AttributeError
        If ``x`` does not have ``__dlpack__`` and ``__dlpack_device__`` methods.
    ValueError
        If data exchange requires a copy but ``copy`` is set to ``False``.
    """
    data = backend.from_dlpack(x, device=device, copy=copy)
    if device is None:
        device = ops.backend.device
    return array(data, dims, device=device)


def full(
    shape: int | tuple[int, ...],
    fill_value: bool | int | float | complex,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dims: list[str] | None = None,
) -> array:
    """
    Returns a new array of the given shape filled with ``fill_value``.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    fill_value: bool | int | float | complex
        fill value.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the data type is inferred from ``fill_value``: ``bool`` fill values produce a boolean array; ``int`` produces the default integer type; ``float`` produces the default floating-point type; ``complex`` produces the default complex type. If ``fill_value`` exceeds the precision of the resolved type, the value may be truncated or rounded. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[str] | None
        dimension names for the output array. If ``None``, all dimensions are undefined. Default: ``None``.

    Returns
    -------
    out: array
        an array where every element is equal to ``fill_value``.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(
        backend.full(shape, fill_value, dtype=dtype, device=device), dims, device=device
    )


def full_like(
    x: array,
    /,
    fill_value: bool | int | float | complex,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
) -> array:
    """
    Returns a new array filled with ``fill_value`` and having the same shape as ``x``.

    The output array inherits the dimension names of ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape and dimension names.
    fill_value: bool | int | float | complex
        fill value.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the data type is inferred from ``x``. If ``fill_value`` exceeds the precision of the resolved type, the value may be truncated or rounded. Values of a differing kind (boolean, integer, or floating-point) are converted following type-promotion rules. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. If ``None``, the device is inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and where every element is equal to ``fill_value``.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(
        backend.full_like(x.data, fill_value, dtype=dtype, device=device),
        x.dims,
        device=device,
    )


def linspace(
    start: int | float | complex,
    stop: int | float | complex,
    /,
    num: int,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    endpoint: bool = True,
    dim: str | None = None,
) -> array:
    """
    Returns evenly spaced numbers over a specified interval.

    Parameters
    ----------
    start: int | float | complex
        the start of the interval.
    stop: int | float | complex
        the end of the interval. When ``endpoint`` is ``True``, the interval is closed ``[start, stop]``; when ``False``, the interval is half-open ``[start, stop)`` and ``stop`` is excluded.
    num: int
        number of samples. Must be a nonnegative integer value.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. Should be a floating-point type. If ``None``, the type is inferred: complex if either ``start`` or ``stop`` is complex, otherwise the default real-valued floating-point type. When ``dtype`` is provided, ``start`` and ``stop`` are converted following type-promotion rules. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    endpoint: bool
        if ``True``, ``stop`` is the last sample. If ``False``, ``stop`` is excluded and the step size is adjusted accordingly. Default: ``True``.
    dim: str | None
        dimension name for the output array. If ``None``, the dimension is undefined. Default: ``None``.

    Returns
    -------
    out: array
        a one-dimensional array containing evenly spaced values.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device

    dims: list[str | _UndefinedDim]
    dims, (start, stop) = ensure_broadcastable(start, stop, ensure_same_ndim=True)
    dims = [dim if dim is not None else _UndefinedDim(), *dims]
    start, stop, num = ensure_backend_compatible_data(start, stop, num)
    return array(
        backend.linspace(
            start, stop, num, dtype=dtype, device=device, endpoint=endpoint
        ),
        dims,
        device=device,
    )


def meshgrid(*arrays: array, indexing: str = "xy") -> list[array]:
    """
    Returns coordinate matrices from coordinate vectors.

    Each input array must be one-dimensional. Dimension names are taken from the input arrays. When ``indexing='xy'``, the dimension names of the first two inputs are swapped in the output, matching the shape transposition that ``'xy'`` indexing applies.

    Parameters
    ----------
    arrays: array
        one or more one-dimensional arrays representing grid coordinates. All arrays must be one-dimensional.
    indexing: str
        ``'xy'`` for Cartesian indexing or ``'ij'`` for matrix indexing. With ``'xy'``, the first two output dimensions are transposed relative to ``'ij'``. Has no effect when zero or one array is provided. Default: ``'xy'``.

    Returns
    -------
    out: list[array]
        list of N arrays, one per input. Each output array has rank N and the same data type as its corresponding input. With ``'ij'`` indexing and input lengths ``N1, N2, ..., Nn``, each output has shape ``(N1, N2, ..., Nn)``; with ``'xy'`` indexing the first two dimensions are swapped.
    """
    if any(a.ndim != 1 for a in arrays):
        shapes = [a.shape for a in arrays]
        raise ValueError(f"All arrays must be one-dimensional, got shapes {shapes}.")
    dims = [a.dims[0] for a in arrays]
    if len(arrays) > 1 and indexing == "xy":
        # Transpose first two dimensions, as is "xy" convention.
        dims = [dims[1], dims[0], *dims[2:]]
    result = backend.meshgrid(*[a.data for a in arrays], indexing=indexing)
    return list(array(data, dims) for data in result)


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dims: list[str] | None = None,
) -> array:
    """
    Returns a new array of the given shape filled with ones.

    For complex dtypes the fill value is ``1 + 0j``.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the default real-valued floating-point type is used. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[str] | None
        dimension names for the output array. If ``None``, all dimensions are undefined. Default: ``None``.

    Returns
    -------
    out: array
        an array filled with ones.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(backend.ones(shape, dtype=dtype, device=device), dims, device=device)


def ones_like(
    x: array,
    /,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
) -> array:
    """
    Returns a new array filled with ones and having the same shape as ``x``.

    The output array inherits the dimension names of ``x``. For complex dtypes the fill value is ``1 + 0j``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape and dimension names.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``None``, the data type is inferred from ``x``. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. If ``None``, the device is inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and filled with ones.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(
        backend.ones_like(x.data, dtype=dtype, device=device),
        x.dims,
        device=device,
    )


def tril(x: array, /, *, k: int = 0) -> array:
    """
    Returns the lower triangular part of a matrix (or a stack of matrices) ``x``.

    The lower triangular part consists of elements on and below diagonal ``k``.

    Parameters
    ----------
    x: array
        Input array whose innermost two dimensions form ``MxN`` matrices.
    k: int
        Diagonal above which to zero elements. ``k = 0`` (default) is the main
        diagonal; ``k < 0`` is below it; ``k > 0`` is above it.

    Returns
    -------
    out: array
        Array containing the lower triangular part(s), with the same shape and
        data type as ``x``. All elements above diagonal ``k`` are zeroed.
    """
    return array(backend.tril(x.data, k=k), x.dims)


def triu(x: array, /, *, k: int = 0) -> array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    The upper triangular part consists of elements on and above diagonal ``k``.

    Parameters
    ----------
    x: array
        Input array whose innermost two dimensions form ``MxN`` matrices.
    k: int
        Diagonal below which to zero elements. ``k = 0`` (default) is the main
        diagonal; ``k < 0`` is below it; ``k > 0`` is above it.

    Returns
    -------
    out: array
        Array containing the upper triangular part(s), with the same shape and
        data type as ``x``. All elements below diagonal ``k`` are zeroed.
    """
    return array(backend.triu(x.data, k=k), x.dims)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dims: list[str] | None = None,
) -> array:
    """
    Returns a new array of the given shape filled with zeros.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        Shape of the output array.
    dtype: DType | BackendDtype | np.dtype | str | None
        Data type of the output array. If ``None``, the backend default
        floating-point type is used. Default: ``None``.
    device: BackendDevice | None
        Device on which to place the array. Default: ``None``.
    dims: list[str] | None
        Named dimensions for the output array. If ``None``, all dimensions
        are undefined. Default: ``None``.

    Returns
    -------
    out: array
        Array of zeros with the specified shape, dtype, and named dimensions.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(backend.zeros(shape, dtype=dtype, device=device), dims, device=device)


def zeros_like(
    x: array,
    /,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
) -> array:
    """
    Returns a new array filled with zeros having the same shape and dims as ``x``.

    Parameters
    ----------
    x: array
        Input array whose shape and dims are used for the output.
    dtype: DType | BackendDtype | np.dtype | str | None
        Data type of the output array. If ``None``, the dtype is inferred from
        ``x``. Default: ``None``.
    device: BackendDevice | None
        Device on which to place the array. If ``None``, the device is inferred
        from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        Array of zeros with the same shape and dims as ``x``.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(
        backend.zeros_like(x.data, dtype=dtype, device=device), x.dims, device=device
    )
