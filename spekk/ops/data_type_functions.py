__all__ = ["astype", "can_cast", "finfo", "iinfo", "isdtype", "result_type"]

from dataclasses import dataclass

from spekk.ops._backend import backend
from spekk.ops._types import (
    device,
    dtype,
    finfo_object,
    iinfo_object,
)
from spekk.ops.array_object import array
from spekk.ops.data_types import DType
from spekk import ops

def astype(
    x: array, dtype: dtype, /, *, copy: bool = True, device: device | None = None
) -> array:
    """
    Copies an array to a specified data type irrespective of type promotion rules.

    .. note::
       Casting floating-point ``NaN`` and ``infinity`` values to integral data types is backend-dependent.

    Parameters
    ----------
    x: array
        array to cast.
    dtype: dtype
        desired data type.
    copy: bool
        if ``True``, always returns a new array. If ``False`` and ``dtype`` matches the input array's data type, the input array is returned; otherwise a new array is returned. Default: ``True``.
    device: device | None
        device on which to place the returned array. If ``device`` is ``None``, the output array device is inferred from ``x``. Default: ``None``.

    Returns
    -------
    out: array
        an array having the specified data type. The returned array has the same shape as ``x``.
    """
    # NOTE: Hack to make it work with Numpy. Remove this (and just pass copy and device
    # directly to backend.astype) when it has been fixed.
    kwargs = dict(copy=copy)
    if device is None:
        device = ops.backend.device 
    if device is not None:
        kwargs["device"] = device
    dtype = DType._to_backend_dtype(dtype)
    data = backend.astype(x._data, dtype, **kwargs)
    return array(data, x._dims, device=device)


def can_cast(from_: dtype | array, to: dtype, /) -> bool:
    """
    Determines if one data type can be cast to another data type according to type promotion rules.

    Parameters
    ----------
    from_: dtype | array
        input data type or array from which to cast.
    to: dtype
        desired data type.

    Returns
    -------
    out: bool
        ``True`` if the cast can occur according to type promotion rules; otherwise, ``False``.
    """
    from_ = from_._data if isinstance(from_, array) else from_
    from_ = DType._to_backend_dtype(from_)
    to = DType._to_backend_dtype(to)
    return backend.can_cast(from_, to)


@dataclass
class finfo_object:
    bits: int
    # Note: The types of the float data here are float, whereas in NumPy they
    # are scalars of the corresponding float dtype.
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DType


@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int
    dtype: DType


def finfo(type: dtype | array, /) -> finfo_object:
    """
    Machine limits for floating-point data types.

    Parameters
    ----------
    type: dtype | array
        the kind of floating-point data-type about which to get information. If complex, the information is about its component data type.

    Returns
    -------
    out: finfo object
        an object with the following attributes:

        - **bits** (*int*): number of bits occupied by the floating-point data type.
        - **eps** (*float*): difference between 1.0 and the next representable floating-point number larger than 1.0.
        - **max** (*float*): largest representable finite number.
        - **min** (*float*): smallest representable finite number.
        - **smallest_normal** (*float*): smallest positive floating-point number with full precision.
        - **dtype** (*dtype*): the floating-point data type.
    """
    if isinstance(type, array):
        type = type._data
    if not backend._is_backend_array(type):
        type = DType._to_backend_dtype(type)
    backend_finfo = backend.finfo(type)
    return finfo_object(
        int(backend_finfo.bits),
        float(backend_finfo.eps),
        float(backend_finfo.max),
        float(backend_finfo.min),
        float(backend_finfo.smallest_normal),
        DType(backend_finfo.dtype),
    )


def iinfo(type: dtype | array, /) -> iinfo_object:
    """
    Machine limits for integer data types.

    Parameters
    ----------
    type: dtype | array
        the kind of integer data-type about which to get information.

    Returns
    -------
    out: iinfo object
        an object with the following attributes:

        - **bits** (*int*): number of bits occupied by the integer data type.
        - **max** (*int*): largest representable integer.
        - **min** (*int*): smallest representable integer.
        - **dtype** (*dtype*): the integer data type.
    """
    if isinstance(type, array):
        type = type._data
    elif not backend._is_backend_array(type):
        type = DType._to_backend_dtype(type)
    return backend.iinfo(type)


def isdtype(
    dtype: dtype, kind: dtype | str | tuple[dtype | str, ...]
) -> bool:
    """
    Returns a boolean indicating whether a provided dtype is of a specified data type ``kind``.

    Parameters
    ----------
    dtype: dtype
        the input dtype.
    kind: dtype | str | tuple[dtype | str, ...]
        data type kind.

        -   If ``kind`` is a dtype, returns whether the input ``dtype`` is equal to the dtype specified by ``kind``.
        -   If ``kind`` is a string, returns whether the input ``dtype`` is of a specified data type kind. The following dtype kinds are supported:

            -   ``'bool'``: boolean data types (e.g., ``bool``).
            -   ``'signed integer'``: signed integer data types (e.g., ``int8``, ``int16``, ``int32``, ``int64``).
            -   ``'unsigned integer'``: unsigned integer data types (e.g., ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            -   ``'integral'``: integer data types. Shorthand for ``('signed integer', 'unsigned integer')``.
            -   ``'real floating'``: real-valued floating-point data types (e.g., ``float32``, ``float64``).
            -   ``'complex floating'``: complex floating-point data types (e.g., ``complex64``, ``complex128``).
            -   ``'numeric'``: numeric data types. Shorthand for ``('integral', 'real floating', 'complex floating')``.

        -   If ``kind`` is a tuple, the tuple specifies a union of dtypes and/or kinds, and returns whether the input ``dtype`` is either equal to a specified dtype or belongs to at least one specified data type kind.

    Returns
    -------
    out: bool
        boolean indicating whether a provided dtype is of a specified data type kind.
    """
    dtype = DType._to_backend_dtype(dtype)
    if isinstance(kind, tuple):
        return any(isdtype(dtype, k) for k in kind)
    elif not isinstance(kind, str):
        kind = DType._to_backend_dtype(kind)
    return backend.isdtype(dtype, kind)


def result_type(*arrays_and_dtypes: dtype | array) -> dtype:
    """
    Returns the dtype that results from applying type promotion rules to the arguments.

    .. note::
       If provided mixed dtypes (e.g., integer and floating-point), the returned dtype is backend-dependent.

    Parameters
    ----------
    arrays_and_dtypes: dtype | array
        an arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    out: dtype
        the dtype resulting from an operation involving the input arrays and dtypes.
    """
    arrays_and_dtypes = [
        (
            x._data
            if isinstance(x, array)
            else DType._to_backend_dtype(x)
            if isinstance(x, DType)
            else x
        )
        for x in arrays_and_dtypes
    ]
    return DType(backend.result_type(*arrays_and_dtypes))
