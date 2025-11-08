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
    PossiblyUndefinedDim,
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
    dim: PossiblyUndefinedDim | None = None,
) -> array:
    """
    Returns evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array.

    Parameters
    ----------
    start: int | float
        if ``stop`` is specified, the start of interval (inclusive); otherwise, the end of the interval (exclusive). If ``stop`` is not specified, the default starting value is ``0``.
    stop: int | float | None
        the end of the interval. Default: ``None``.
    step: int | float
        the distance between two adjacent elements (``out[i+1] - out[i]``). Must not be ``0``; may be negative, this results in an empty array if ``stop >= start``. Default: ``1``.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from ``start``, ``stop`` and ``step``. If those are all integers, the output array dtype is the default integer dtype; if one or more have type ``float``, then the output array dtype is the default real-valued floating-point data type. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dim: PossiblyUndefinedDim | None
        the name of the dimension in the output array. If ``dim`` is ``None``, the dimension will be undefined. Default: ``None``.


    .. note::
       This function cannot guarantee that the interval does not include the ``stop`` value in those cases where ``step`` is not an integer and floating-point rounding errors affect the length of the output array.

    Returns
    -------
    out: array
        a one-dimensional array containing evenly spaced values. The length of the output array is ``ceil((stop-start)/step)`` if ``stop - start`` and ``step`` have the same sign, and length ``0`` otherwise.
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
        array
        | bool
        | int
        | float
        | complex
        | NestedSequence[bool | int | float | complex]
        | Buffer
        | BackendArray
    ),
    /,
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    copy: bool | None = None,
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    r"""
    Convert the input to an array.

    Parameters
    ----------
    obj: array | bool | int | float | complex | NestedSequence[bool | int | float | complex] | Buffer | BackendArray
        object to be converted to an array. May be a Python scalar, a (possibly nested) sequence of Python scalars, or an object supporting the Python buffer protocol.

        .. admonition:: Tip
           :class: important

           An object supporting the buffer protocol can be turned into a memoryview through ``memoryview(obj)``.

    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from the data type(s) in ``obj``. If all input values are Python scalars, then, in order of precedence,

        -   if all values are of type ``bool``, the output data type is ``bool``.
        -   if all values are of type ``int`` or are a mixture of ``bool`` and ``int``, the output data type is the default integer data type.
        -   if one or more values are ``complex`` numbers, the output data type is the default complex floating-point data type.
        -   if one or more values are ``float``, the output data type is the default real-valued floating-point data type.

        Default: ``None``.

    device: BackendDevice | None
        device on which to place the created array. If ``device`` is ``None`` and ``obj`` is an array, the output array device is inferred from ``obj``. Default: ``None``.
    copy: bool | None
        boolean indicating whether or not to copy the input. If ``True``, the function always copies. If ``False``, the function never copies for input which supports the buffer protocol and raises a ``ValueError`` in case a copy would be necessary. If ``None``, the function reuses existing memory buffer if possible and copy otherwise. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all its dimensions will be undefined. Default: ``None``.

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
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    """
    Returns an uninitialized array having a specified `shape`.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is the default real-valued floating-point data type. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all its dimensions will be undefined. Default: ``None``.

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
        device on which to place the created array. If ``device`` is ``None``, the output array device is inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and containing uninitialized data.
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
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    r"""
    Returns a two-dimensional array with ones on the ``k``\th diagonal and zeros elsewhere.

    .. note::
       An output array having a complex floating-point data type has the value ``1 + 0j`` along the ``k``\th diagonal and ``0 + 0j`` elsewhere.

    Parameters
    ----------
    n_rows: int
        number of rows in the output array.
    n_cols: int | None
        number of columns in the output array. If ``None``, the default number of columns in the output array is equal to ``n_rows``. Default: ``None``.
    k: int
        index of the diagonal. A positive value refers to an upper diagonal, a negative value to a lower diagonal, and ``0`` to the main diagonal. Default: ``0``.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is the default real-valued floating-point data type. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all dimensions will be undefined. Default: ``None``.

    Returns
    -------
    out: array
        an array where all elements are equal to zero, except for the ``k``\th diagonal, whose values are equal to one.
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
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    """
    Returns a new array containing the data from another (array) object with a ``__dlpack__`` method.

    Parameters
    ----------
    x: object
        input (array) object.
    device: BackendDevice | None
        device on which to place the created array. If ``device`` is ``None`` and ``x`` supports DLPack, the output array is on the same device as ``x``. Default: ``None``.
    copy: bool | None
        boolean indicating whether or not to copy the input. If ``True``, the function always copies. If ``False``, the function never copies, and raises ``BufferError`` in case a copy is deemed necessary (e.g.  if a cross-device data movement is requested, and it is not possible without a copy). If ``None``, the function reuses the existing memory buffer if possible and copies otherwise. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all dimensions will be undefined. Default: ``None``.

    Returns
    -------
    out: array
        an array containing the data in ``x``.

        .. admonition:: Note
           :class: note

           The returned array may be either a copy or a view. See :ref:`data-interchange` for details.

    Raises
    ------
    BufferError
        The ``__dlpack__`` and ``__dlpack_device__`` methods on the input array
        may raise ``BufferError`` when the data cannot be exported as DLPack
        (e.g., incompatible dtype, strides, or device). It may also raise other errors
        when export fails for other reasons (e.g., not enough memory available
        to materialize the data). ``from_dlpack`` propagates such
        exceptions.
    AttributeError
        If the ``__dlpack__`` and ``__dlpack_device__`` methods are not present
        on the input array. This may happen for libraries that are never able
        to export their data with DLPack.
    ValueError
        If data exchange is possible via an explicit copy but ``copy`` is set to ``False``.



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
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    """
    Returns a new array having a specified ``shape`` and filled with ``fill_value``.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    fill_value: bool | int | float | complex
        fill value.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from ``fill_value`` according to the following rules:

        - If the fill value is an ``int``, the output array data type is the default integer data type.
        - If the fill value is a ``float``, the output array data type is the default real-valued floating-point data type.
        - If the fill value is a ``complex`` number, the output array data type is the default complex floating-point data type.
        - If the fill value is a ``bool``, the output array has a boolean data type. Default: ``None``.

        .. note::
           If the ``fill_value`` exceeds the precision of the resolved default output array data type, the value may be truncated or rounded.

    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all dimensions will be undefined. Default: ``None``.

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
    Returns a new array filled with ``fill_value`` and having the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    fill_value: bool | int | float | complex
        fill value.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from ``x``. Default: ``None``.

        .. note::
           If the ``fill_value`` exceeds the precision of the resolved output array data type, the value may be truncated or rounded.

        .. note::
           If the ``fill_value`` has a data type which is not of the same data type kind (boolean, integer, or floating-point) as the resolved output array data type, the value will be converted following type-promotion rules.

    device: BackendDevice | None
        device on which to place the created array. If ``device`` is ``None``, the output array device is inferred from ``x``. Default: ``None``.

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
    dim: PossiblyUndefinedDim | None = None,
) -> array:
    r"""
    Returns evenly spaced numbers over a specified interval.

    Let :math:`N` be the number of generated values (which is either ``num`` or ``num+1`` depending on whether ``endpoint`` is ``True`` or ``False``, respectively). For real-valued output arrays, the spacing between values is given by

    .. math::
       \Delta_{\textrm{real}} = \frac{\textrm{stop} - \textrm{start}}{N - 1}

    For complex output arrays, let ``a = real(start)``, ``b = imag(start)``, ``c = real(stop)``, and ``d = imag(stop)``. The spacing between complex values is given by

    .. math::
       \Delta_{\textrm{complex}} = \frac{c-a}{N-1} + \frac{d-b}{N-1} j

    Parameters
    ----------
    start: int | float | complex
        the start of the interval.
    stop: int | float | complex
        the end of the interval. If ``endpoint`` is ``False``, the function generates a sequence of ``num+1`` evenly spaced numbers starting with ``start`` and ending with ``stop`` and excludes the ``stop`` from the returned array such that the returned array consists of evenly spaced numbers over the half-open interval ``[start, stop)``. If ``endpoint`` is ``True``, the output array consists of evenly spaced numbers over the closed interval ``[start, stop]``. Default: ``True``.

        .. note::
           The step size changes when `endpoint` is `False`.

    num: int
        number of samples. Must be a nonnegative integer value.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. Should be a floating-point data type. If ``dtype`` is ``None``,

        -   if either ``start`` or ``stop`` is a ``complex`` number, the output data type is the default complex floating-point data type.
        -   if both ``start`` and ``stop`` are real-valued, the output data type is the default real-valued floating-point data type.

        Default: ``None``.

        .. admonition:: Note
           :class: note

           If ``dtype`` is not ``None``, conversion of ``start`` and ``stop`` follows type-promotion rules.

    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    endpoint: bool
        boolean indicating whether to include ``stop`` in the interval. Default: ``True``.
    dim: PossiblyUndefinedDim | None
        the name of the dimension in the output array. If ``dim`` is ``None``, the dimension will be undefined. Default: ``None``.

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

    Parameters
    ----------
    arrays: array
        an arbitrary number of one-dimensional arrays representing grid coordinates. Each array should have the same numeric data type.
    indexing: str
        Cartesian ``'xy'`` or matrix ``'ij'`` indexing of output. If provided zero or one one-dimensional vector(s) (i.e., the zero- and one-dimensional cases, respectively), the ``indexing`` keyword has no effect. Default: ``'xy'``.

    Returns
    -------
    out: list[array]
        list of N arrays, where ``N`` is the number of provided one-dimensional input arrays. Each returned array has rank ``N``. For ``N`` one-dimensional arrays having lengths ``Ni = len(xi)``,

        - if matrix indexing ``ij``, then each returned array has the shape ``(N1, N2, N3, ..., Nn)``.
        - if Cartesian indexing ``xy``, then each returned array has shape ``(N2, N1, N3, ..., Nn)``.

        Accordingly, for the two-dimensional case with input one-dimensional arrays of length ``M`` and ``N``, if matrix indexing ``ij``, then each returned array has shape ``(M, N)``, and, if Cartesian indexing ``xy``, then each returned array has shape ``(N, M)``.

        Similarly, for the three-dimensional case with input one-dimensional arrays of length ``M``, ``N``, and ``P``, if matrix indexing ``ij``, then each returned array has shape ``(M, N, P)``, and, if Cartesian indexing ``xy``, then each returned array has shape ``(N, M, P)``.

        Each returned array has the same data type as the input arrays.
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
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    """
    Returns a new array having a specified ``shape`` and filled with ones.

    .. note::
       An output array having a complex floating-point data type contains complex numbers having a real component equal to one and an imaginary component equal to zero (i.e., ``1 + 0j``).

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is the default real-valued floating-point data type. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all dimensions will be undefined. Default: ``None``.

    Returns
    -------
    out: array
        an array containing ones.
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
    Returns a new array filled with ones and having the same ``shape`` as an input array ``x``.

    .. note::
       An output array having a complex floating-point data type contains complex numbers having a real component equal to one and an imaginary component equal to zero (i.e., ``1 + 0j``).

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from ``x``. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. If ``device`` is ``None``, the output array device is inferred from ``x``. Default: ``None``.

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

    .. note::
       The lower triangular part of the matrix is defined as the elements on and below the specified diagonal ``k``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    k: int
        diagonal above which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

        .. note::
           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out: array
        an array containing the lower triangular part(s). The returned array has the same shape and data type as ``x``. All elements above the specified diagonal ``k`` are zeroed. The returned array is allocated on the same device as ``x``.
    """
    return array(backend.tril(x.data, k=k), x.dims)


def triu(x: array, /, *, k: int = 0) -> array:
    """
    Returns the upper triangular part of a matrix (or a stack of matrices) ``x``.

    .. note::
       The upper triangular part of the matrix is defined as the elements on and above the specified diagonal ``k``.

    Parameters
    ----------
    x: array
        input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
    k: int
        diagonal below which to zero elements. If ``k = 0``, the diagonal is the main diagonal. If ``k < 0``, the diagonal is below the main diagonal. If ``k > 0``, the diagonal is above the main diagonal. Default: ``0``.

        .. note::
           The main diagonal is defined as the set of indices ``{(i, i)}`` for ``i`` on the interval ``[0, min(M, N) - 1]``.

    Returns
    -------
    out: array
        an array containing the upper triangular part(s). The returned array has the same shape and data type as ``x``. All elements below the specified diagonal ``k`` are zeroed. The returned array is allocated on the same device as ``x``.
    """
    return array(backend.triu(x.data, k=k), x.dims)


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | BackendDtype | np.dtype | str | None = None,
    device: BackendDevice | None = None,
    dims: list[PossiblyUndefinedDim] | None = None,
) -> array:
    """
    Returns a new array having a specified ``shape`` and filled with zeros.

    Parameters
    ----------
    shape: int | tuple[int, ...]
        output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is the default real-valued floating-point data type. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. Default: ``None``.
    dims: list[PossiblyUndefinedDim] | None
        the name of the dimensions in the output array. If ``dims`` is ``None``, all dimensions will be undefined. Default: ``None``.

    Returns
    -------
    out: array
        an array containing zeros.
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
    Returns a new array filled with zeros and having the same ``shape`` as an input array ``x``.

    Parameters
    ----------
    x: array
        input array from which to derive the output array shape.
    dtype: DType | BackendDtype | np.dtype | str | None
        output array data type. If ``dtype`` is ``None``, the output array data type is inferred from ``x``. Default: ``None``.
    device: BackendDevice | None
        device on which to place the created array. If ``device`` is ``None``, the output array device is inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the same shape as ``x`` and filled with zeros.
    """
    if dtype is not None:
        dtype = DType._to_backend_dtype(dtype)
    if device is None:
        device = ops.backend.device
    return array(
        backend.zeros_like(x.data, dtype=dtype, device=device), x.dims, device=device
    )
