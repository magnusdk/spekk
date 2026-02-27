from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import array_api_compat
import numpy as np

import spekk.ops.data_types as data_types
from spekk import ops
from spekk.ops._backend import backend
from spekk.ops._types import (
    ArrayLike,
    DeviceLike,
    DTypeLike,
    Enum,
    PossiblyUndefinedDim,
    PyCapsule,
    _UndefinedDim,
    ellipsis,
)
from spekk.ops._types import device as Device
from spekk.ops.data_types import DType

if TYPE_CHECKING:
    from spekk.ops._indexing import ArrayIndexUpdateHelper

__all__ = ["array"]

_sentinel = object()


class array:
    def __init__(
        self: array,
        data: ArrayLike,
        /,
        dims: list[str] | None = None,
        *,
        dtype: DTypeLike | None = None,
        device: Device | None = None,
    ):
        """
        Create a spekk array wrapping array-like data with optional named dimensions.

        Parameters
        ----------
        data: ArrayLike
            The array data. May be a Python scalar, nested list, NumPy array,
            backend array, or another spekk array. If a spekk array is given,
            its underlying data and dims are reused unless overridden.
        dims: list[str] | None
            Named dimensions for each axis. If ``None``, all axes receive
            anonymous (undefined) dimensions. The number of entries must equal
            the number of axes in ``data``. Dimension names must be unique.
        dtype: DTypeLike | None
            Target data type. If ``None``, the dtype is inferred from ``data``.
        device: Device | None
            Target device. If ``None``, the current backend's default device is
            used.

        Raises
        ------
        ValueError
            If the length of ``dims`` does not match the number of axes in
            ``data``, or if ``dims`` contains duplicate names.
        """
        if device is None:
            device = ops.backend.device

        if dtype is not None:
            dtype = DType._to_backend_dtype(dtype)
        # else:
        #     if ops.backend.backend_name=="torch":
        #         if isinstance(data, np.ndarray):
        #             if data.dtype == np.float64:
        #                 dtype = _DType._to_backend_dtype(ops.float32)
        #             if data.dtype == np.complex128:
        #                 dtype = _DType._to_backend_dtype(ops.complex64)

        if isinstance(data, array):
            if dims is None:
                dims = data._dims
            data = data._data

        if not backend._is_backend_array(data):
            data = backend.asarray(data, dtype=dtype, device=device)
        else:
            if dtype is None:
                dtype = data.dtype

            # if dtype != data.dtype or (hasattr(data, "device") and device!=data.device):
            # if dtype != data.dtype or device!=data.device:
            if dtype != data.dtype:
                data = backend.astype(data, dtype, device=device)

        if dims is None:
            dims = [_UndefinedDim() for _ in range(data.ndim)]
        elif data.ndim != len(dims):
            raise ValueError(
                "The number of dimensions must equal the number of axes in the data "
                f"(got ndim={data.ndim} and {len(dims)=})."
            )
        else:
            # Ensure that undefined dimensions are unique objects.
            dims = [
                _UndefinedDim() if isinstance(dim, _UndefinedDim) else dim
                for dim in dims
            ]
        if len(set(dims)) != len(dims):
            raise ValueError(f"The dimensions must be unique, but got {dims=}")

        self._data: ArrayLike = data
        self._dims = dims

    @property
    def dtype(self: array) -> DType:
        """
        Data type of the array elements.

        Returns
        -------
        out: DType
            The data type of the array elements.
        """
        return DType(self._data.dtype)

    @property
    def device(self: array) -> DeviceLike:
        """
        Hardware device the array data resides on.

        Returns
        -------
        out: DeviceLike
            The device on which the underlying backend array lives.
        """
        return self._data.device

    @property
    def mT(self: array) -> array:
        """
        Transpose of a matrix (or a stack of matrices).

        Delegates to ``spekk.ops.matrix_transpose``. Raises an error if
        the array has fewer than two dimensions.

        Returns
        -------
        out: array
            An array whose last two dimensions are permuted in reverse order.
            For shape ``(..., M, N)`` the result has shape ``(..., N, M)`` and
            the same data type as ``self``.
        """
        return ops.matrix_transpose(self)

    @property
    def ndim(self: array) -> int:
        """
        Number of array dimensions (axes).

        Returns
        -------
        out: int
            The number of dimensions (axes) in the array.
        """
        return self._data.ndim

    @property
    def shape(self: array) -> tuple[int, ...]:
        """
        Array dimensions.

        Returns
        -------
        out: tuple[int, ...]
            A tuple of integers representing the size of each axis.
        """
        return self._data.shape

    @property
    def size(self: array) -> int:
        """
        Total number of elements in the array.

        Returns
        -------
        out: int
            The product of all axis sizes (``math.prod(self.shape)``).
        """
        return math.prod(self._data.shape)

    @property
    def T(self: array) -> array:
        """
        Transpose of the array.

        The array must be two-dimensional; raises ``ValueError`` otherwise.
        To reverse all axes of a higher-dimensional array, use
        ``spekk.ops.permute_dims``.

        Returns
        -------
        out: array
            A two-dimensional array with its axes permuted in reverse order.
            Has the same data type as ``self``, with dimension names swapped
            accordingly.
        """
        if self.ndim != 2:
            raise ValueError(
                "Transpose is only defined for arrays with two dimensions. See permute_dims instead."
            )
        dims = self._dims[::-1]
        return array(backend.transpose(self._data), dims)

    def __abs__(self: array, /) -> array:
        """
        Computes the element-wise absolute value. Delegates to
        ``spekk.ops.abs``.

        For real-valued arrays the result has the same data type as ``self``.
        For complex floating-point arrays the result is real-valued with
        matching precision (e.g. ``complex64`` → ``float32``). For signed
        integer types, the absolute value of the minimum representable integer
        is backend-dependent.

        Parameters
        ----------
        self: array
            Array with a numeric data type.

        Returns
        -------
        out: array
            An array containing the element-wise absolute values, preserving
            named dimensions.
        """
        return ops.abs(self)

    def __add__(self: array, other: array | int | float | complex, /) -> array:
        """
        Computes the element-wise sum. Delegates to ``spekk.ops.add``.

        Parameters
        ----------
        self: array
            Augend array with a numeric data type.
        other: int | float | complex | array
            Addend. Must be broadcast-compatible with ``self`` and have a
            numeric data type.

        Returns
        -------
        out: array
            An array containing the element-wise sums. The data type is
            determined by type-promotion rules.
        """
        return ops.add(self, other)

    def __and__(self: array, other: array | int, /) -> array:
        """
        Computes the element-wise bitwise AND (``self_i & other_i``). Delegates
        to ``spekk.ops.bitwise_and``.

        Parameters
        ----------
        self: array
            Array with an integer or boolean data type.
        other: int | array
            Operand. Must be broadcast-compatible with ``self`` and have an
            integer or boolean data type.

        Returns
        -------
        out: array
            An array containing the element-wise bitwise AND results. The data
            type is determined by type-promotion rules.
        """
        return ops.bitwise_and(self, other)

    def __array_namespace__(self: array, /, *, api_version: str | None = None) -> Any:
        """
        Returns the ``spekk.ops`` namespace, which implements the Python Array
        API standard.

        Parameters
        ----------
        api_version: str | None
            Array API version string in ``'YYYY.MM'`` form. Currently only
            ``'2023.12'`` is supported. Pass ``None`` to get the latest
            supported version. Raises ``NotImplementedError`` for unsupported
            versions. Default: ``None``.

        Returns
        -------
        out: Any
            The ``spekk.ops`` module, providing all Array API top-level
            functions.
        """
        if api_version != "2023.12" and api_version is not None:
            raise NotImplementedError(
                f"API version '{api_version}' not supported. Only array API version '2023.12' is supported."
            )
        return ops

    def __bool__(self: array, /) -> bool:
        """
        Converts a zero-dimensional array to a Python ``bool``.

        Parameters
        ----------
        self: array
            A zero-dimensional array.

        Returns
        -------
        out: bool
            The single element of the array as a Python ``bool``.

        Notes
        -----
        **Special cases** (real-valued floating-point)

        - ``NaN`` → ``True``
        - ``+infinity`` or ``-infinity`` → ``True``
        - ``+0`` or ``-0`` → ``False``

        For complex operands, the result is
        ``bool(real(self)) or bool(imag(self))``.

        Backends that use lazy/graph-based evaluation may raise
        ``ValueError`` if the value cannot be materialized.
        """
        return self._data.__bool__()

    def __complex__(self: array, /) -> complex:
        """
        Converts a zero-dimensional array to a Python ``complex`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance.

        Returns
        -------
        out: complex
            a Python ``complex`` object representing the single element of the array.

        Notes
        -----

        **Special cases**

        For boolean operands,

        - If ``self`` is ``True``, the result is ``1+0j``.
        - If ``self`` is ``False``, the result is ``0+0j``.

        For real-valued floating-point operands,

        - If ``self`` is ``NaN``, the result is ``NaN + NaN j``.
        - If ``self`` is ``+infinity``, the result is ``+infinity + 0j``.
        - If ``self`` is ``-infinity``, the result is ``-infinity + 0j``.
        - If ``self`` is a finite number, the result is ``self + 0j``.
        """
        return self._data.__complex__()

    def __dlpack__(
        self: array,
        /,
        *,
        stream: int | Any | None = None,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[Enum, int] | None = None,
        copy: bool | None = None,
    ) -> PyCapsule:
        """
        Exports the array as a DLPack capsule for use with ``from_dlpack``.

        Parameters
        ----------
        self: array
            array instance.
        stream: int | Any | None
            for CUDA and ROCm, a Python integer representing a pointer to a stream.
            ``stream`` instructs the producer to ensure operations can safely be
            performed on the array. The pointer must be an integer >= ``-1``.
            If ``stream`` is ``-1``, no synchronization is performed by the producer.
            On CPU and other devices without streams, only ``None`` is accepted.
            Support for non-``None`` values is backend-dependent.

            Device-specific values of ``stream`` for CUDA:

            - ``None``: legacy default stream (default).
            - ``1``: the legacy default stream.
            - ``2``: the per-thread default stream.
            - ``> 2``: stream number as a Python integer.
            - ``0`` is disallowed (ambiguous).

            Device-specific values of ``stream`` for ROCm:

            - ``None``: legacy default stream (default).
            - ``0``: the default stream.
            - ``> 2``: stream number as a Python integer.
            - ``1`` and ``2`` are not supported.

            When ``dl_device`` is provided, ``stream`` must be valid for that
            device type. For ``kDLCPU``, ``stream`` must be ``None``.
        max_version: tuple[int, int] | None
            the maximum DLPack version the consumer supports, as ``(major, minor)``.
            The returned capsule may be of this version or a different one; the
            consumer must verify the version regardless.
        dl_device: tuple[Enum, int] | None
            the DLPack device type to export to, as ``(device_type, device_id)``
            (same format as ``__dlpack_device__``). ``None`` means export on the
            same device as ``self``. If the device type is not supported, raises
            ``BufferError``. If a copy is required to reach ``kDLCPU`` but
            ``copy=False``, raises ``ValueError``.
        copy: bool | None
            whether to copy the array. If ``True``, always copies. If ``False``,
            never copies and raises ``BufferError`` if a copy would be required.
            If ``None``, reuses existing memory if possible, otherwise copies.
            Default: ``None``.

        Returns
        -------
        capsule: PyCapsule
            a DLPack capsule for the array.

        Raises
        ------
        BufferError
            Raised when the data cannot be exported as DLPack (e.g., incompatible
            dtype or strides), or when ``copy=False`` but a copy is required.
        """
        return self._data.__dlpack__(
            stream=stream, max_version=max_version, dl_device=dl_device, copy=copy
        )

    def __dlpack_device__(self: array, /) -> tuple[Enum, int]:
        """
        Returns device type and device ID in DLPack format, for use with ``from_dlpack``.

        Parameters
        ----------
        self: array
            array instance.

        Returns
        -------
        device: tuple[Enum, int]
            a tuple ``(device_type, device_id)`` in DLPack format. Valid device type enum members are:

            ::

              CPU = 1
              CUDA = 2
              CPU_PINNED = 3
              OPENCL = 4
              VULKAN = 7
              METAL = 8
              VPI = 9
              ROCM = 10
              CUDA_MANAGED = 13
              ONE_API = 14
        """
        return self._data.__dlpack_device__()

    def __eq__(self: array, other: array | int | float | complex, /) -> array:  # type: ignore
        r"""
        Computes the truth value of ``self_i == other_i`` for each element of the array.

        Parameters
        ----------
        self: array
            array instance. May have any data type.
        other: array | int | float | complex
            other array. Must be broadcast-compatible with ``self``. May have any data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type of ``bool``.
        """
        return ops.equal(self, other)

    def __float__(self: array, /) -> float:
        """
        Converts a zero-dimensional array to a Python ``float`` object.

        Parameters
        ----------
        self: array
            zero-dimensional array instance. Has a real-valued or boolean data type.
            If ``self`` has a complex floating-point data type, the function must raise a ``TypeError``.

        Returns
        -------
        out: float
            a Python ``float`` object representing the single element of the array.

        Notes
        -----

        **Special cases**

        For boolean operands,

        - If ``self`` is ``True``, the result is ``1``.
        - If ``self`` is ``False``, the result is ``0``.
        """
        return self._data.__float__()

    def __floordiv__(self: array, other: array | int | float, /) -> array:
        """
        Evaluates ``self_i // other_i`` for each element of the array.

        Parameters
        ----------
        self: array
            array instance. Has a real-valued data type.
        other: int | float | array
            other array. Must be broadcast-compatible with ``self``. Has a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type determined by type promotion.
        """
        return ops.floor_divide(self, other)

    def __ge__(self: array, other: array | int | float, /) -> array:
        """
        Computes the truth value of ``self_i >= other_i`` for each element of the array.

        Parameters
        ----------
        self: array
            array instance. Has a real-valued data type.
        other: int | float | array
            other array. Must be broadcast-compatible with ``self``. Has a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type of ``bool``.
        """
        return ops.greater_equal(self, other)

    def __getitem__(
        self: array,
        key: int
        | slice
        | ellipsis
        | None
        | tuple[int | slice | ellipsis | None, ...]
        | array,
        /,
    ) -> array:
        """
        Returns ``self[key]``.

        Parameters
        ----------
        self: array
            array instance.
        key: int | slice | ellipsis | None | tuple[int | slice | ellipsis | None, ...] | array
            index key.

        Returns
        -------
        out: array
            an array containing the accessed value(s) with the same data type as ``self``.
        """
        from spekk.ops._indexing import getitem

        return getitem(self, key)

    def __gt__(self: array, other: array | int | float, /) -> array:
        """
        Computes the truth value of ``self_i > other_i`` for each element of the array.

        Parameters
        ----------
        self: array
            array instance. Has a real-valued data type.
        other: int | float | array
            other array. Must be broadcast-compatible with ``self``. Has a real-valued data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type of ``bool``.
        """
        return ops.greater(self, other)

    def __index__(self: array, /) -> int:
        """
        Converts a zero-dimensional integer array to a Python ``int`` object.

        Called to implement ``operator.index()``.

        Parameters
        ----------
        self: array
            Zero-dimensional array instance. Must have an integer data type. If ``self`` has a floating-point data type, raises ``TypeError``.

        Returns
        -------
        out: int
            A Python ``int`` representing the single element of the array.
        """
        return self._data.__index__()

    def __int__(self: array, /) -> int:
        """
        Converts a zero-dimensional array to a Python ``int`` object.

        Parameters
        ----------
        self: array
            Zero-dimensional array instance. Must have a boolean or real-valued data type.
            If ``self`` has a complex floating-point data type, raises ``TypeError``.

        Returns
        -------
        out: int
            A Python ``int`` representing the single element of the array.

        Notes
        -----

        **Special cases**

        For boolean operands,

        - If ``self`` is ``True``, the result is ``1``.
        - If ``self`` is ``False``, the result is ``0``.

        For floating-point operands,

        - If ``self`` is a finite number, the result is the integer part of ``self``.
        - If ``self`` is ``-0``, the result is ``0``.
        - If ``self`` is either ``+infinity`` or ``-infinity``, raises ``OverflowError``.
        - If ``self`` is ``NaN``, raises ``ValueError``.
        """
        return self._data.__int__()

    def __invert__(self: array, /) -> array:
        """
        Evaluates ``~self_i`` for each element of an array instance.

        Parameters
        ----------
        self: array
            Array instance. Has an integer or boolean data type.

        Returns
        -------
        out: array
            An array containing the element-wise results with the same data type as ``self``.
        """
        return ops.bitwise_invert(self)

    def __le__(self: array, other: int | float | array, /) -> array:
        """
        Computes the truth value of ``self_i <= other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            Array instance. Has a real-valued data type.
        other: int | float | array
            Other array. Must be broadcast-compatible with ``self``. Has a real-valued data type.

        Returns
        -------
        out: array
            An array containing the element-wise results with a boolean data type.
        """
        return ops.less_equal(self, other)

    def __lshift__(self: array, other: int | array, /) -> array:
        """
        Evaluates ``self_i << other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            Array instance. Has an integer data type.
        other: int | array
            Other array. Must be broadcast-compatible with ``self``. Has an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: array
            An array containing the element-wise results with the same data type as ``self``.
        """
        return ops.bitwise_left_shift(self, other)

    def __lt__(self: array, other: int | float | array, /) -> array:
        """
        Computes the truth value of ``self_i < other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            Array instance. Has a real-valued data type.
        other: int | float | array
            Other array. Must be broadcast-compatible with ``self``. Has a real-valued data type.

        Returns
        -------
        out: array
            An array containing the element-wise results with a boolean data type.
        """
        return ops.less(self, other)

    def __matmul__(self: array, other: array, /) -> array:
        """
        Computes the matrix product (implements the ``@`` operator).

        Parameters
        ----------
        self: array
            Array instance. Has a numeric data type. Must have at least one dimension.
            If ``self`` is one-dimensional with shape ``(M,)`` and ``other`` has more than one
            dimension, ``self`` is treated as shape ``(1, M)`` and the leading dimension is removed
            from the result. If ``self`` has shape ``(..., M, K)``, the innermost two dimensions
            form the matrices to multiply. ``shape(self)[:-2]`` must be broadcast-compatible with
            ``shape(other)[:-2]``.
        other: array
            Other array. Has a numeric data type. Must have at least one dimension.
            If ``other`` is one-dimensional with shape ``(N,)`` and ``self`` has more than one
            dimension, ``other`` is treated as shape ``(N, 1)`` and the trailing dimension is
            removed from the result. If ``other`` has shape ``(..., K, N)``, the innermost two
            dimensions form the matrices to multiply. ``shape(other)[:-2]`` must be
            broadcast-compatible with ``shape(self)[:-2]``.

        Returns
        -------
        out: array
            The matrix product result. Shape depends on the dimensionality of the inputs:

            - Both 1-D with shape ``(N,)``: a zero-dimensional array (inner product).
            - ``self`` is ``(M, K)`` and ``other`` is ``(K, N)``: shape ``(M, N)``.
            - ``self`` is ``(K,)`` and ``other`` is ``(..., K, N)``: shape ``(..., N)``.
            - ``self`` is ``(..., M, K)`` and ``other`` is ``(K,)``: shape ``(..., M)``.
            - Otherwise: batched result with shape from broadcasting ``shape(self)[:-2]``
              against ``shape(other)[:-2]``.

            The data type is determined by type promotion.

        Raises
        ------
        ValueError
            If either ``self`` or ``other`` is a zero-dimensional array, or if the inner
            dimensions are incompatible (``K != L``).
        """
        return ops.matmul(self, other)

    def __mod__(self: array, other: int | float | array, /) -> array:
        """
        Evaluates ``self_i % other_i`` for each element of an array instance with the respective element of the array ``other``.

        For integer inputs, the result of division by zero is backend-dependent.

        Parameters
        ----------
        self: array
            Array instance. Has a real-valued data type.
        other: int | float | array
            Other array. Must be broadcast-compatible with ``self``. Has a real-valued data type.

        Returns
        -------
        out: array
            An array containing the element-wise results. Each element-wise result has the same sign as the respective element ``other_i``. The data type is determined by type promotion.
        """
        return ops.remainder(self, other)

    def __mul__(self: array, other: int | float | complex | array, /) -> array:
        r"""
        Calculates the product for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            Array instance. Has a numeric data type.
        other: int | float | complex | array
            Other array. Must be broadcast-compatible with ``self``. Has a numeric data type.

        Returns
        -------
        out: array
            An array containing the element-wise products. The data type is determined by type promotion.
        """
        return ops.multiply(self, other)

    def __ne__(self: array, other: int | float | complex | array, /) -> array:  # type: ignore
        """
        Computes the truth value of ``self_i != other_i`` for each element of an array instance with the respective element of the array ``other``.

        Parameters
        ----------
        self: array
            Array instance. May have any data type.
        other: int | float | complex | array
            Other array. Must be broadcast-compatible with ``self``. May have any data type.

        Returns
        -------
        out: array
            An array containing the element-wise results with a boolean data type.
        """
        return ops.not_equal(self, other)

    def __neg__(self: array, /) -> array:
        """
        Evaluates ``-self_i`` for each element of an array instance.

        For signed integer data types, the numerical negative of the minimum representable integer
        is backend-dependent. For complex data types, both the real and imaginary components are
        negated.

        Parameters
        ----------
        self: array
            Array instance. Has a numeric data type.

        Returns
        -------
        out: array
            An array containing the element-wise negations. The data type is determined by type
            promotion.
        """
        return ops.negative(self)

    def __or__(self: array, other: bool | int | array, /) -> array:
        """
        Evaluates ``self_i | other_i`` for each element of an array instance with the respective element of ``other``.

        Parameters
        ----------
        self: array
            array instance. Has an integer or boolean data type.
        other: bool | int | array
            other array. Must be broadcast-compatible with ``self``. Has an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type determined by type-promotion.
        """
        return ops.bitwise_or(self, other)

    def __pos__(self: array, /) -> array:
        """
        Evaluates ``+self_i`` for each element of an array instance.

        Parameters
        ----------
        self: array
            array instance. Has a numeric data type.

        Returns
        -------
        out: array
            an array containing the evaluated result for each element. The returned array has the same data type as ``self``.
        """
        return ops.positive(self)

    def __pow__(self: array, other: int | float | complex | array, /) -> array:
        r"""
        Raises each element of an array instance to the power of the corresponding element of ``other``.

        .. note::
           If both ``self`` and ``other`` have integer data types, the result when ``other_i`` is negative (less than zero) is backend-dependent.

           If ``self`` has an integer data type and ``other`` has a floating-point data type, behavior is backend-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

        Parameters
        ----------
        self: array
            array instance whose elements are the exponentiation base. Has a numeric data type.
        other: int | float | complex | array
            exponent array. Must be broadcast-compatible with ``self``. Has a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type determined by type-promotion.
        """
        return ops.pow(self, other)

    def __rshift__(self: array, other: int | array, /) -> array:
        """
        Evaluates ``self_i >> other_i`` for each element of an array instance with the respective element of ``other``.

        Parameters
        ----------
        self: array
            array instance. Has an integer data type.
        other: int | array
            other array. Must be broadcast-compatible with ``self``. Has an integer data type. Each element must be greater than or equal to ``0``.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has the same data type as ``self``.
        """
        return ops.bitwise_right_shift(self, other)

    def __setitem__(
        self: array,
        key: int | slice | ellipsis | tuple[int | slice | ellipsis, ...] | array,
        value: bool | int | float | array,
        /,
    ) -> None:
        from spekk.ops._indexing import setitem

        new_array = setitem(self, key, value)
        self._data = new_array.data
        self._dims = new_array.dims

    def __sub__(self: array, other: int | float | complex | array, /) -> array:
        """
        Calculates the element-wise difference ``self_i - other_i``.

        The result of ``self_i - other_i`` is the same as ``self_i + (-other_i)`` and is governed by the same floating-point rules as addition (see ``__add__``).

        Parameters
        ----------
        self: array
            array instance (minuend). Has a numeric data type.
        other: int | float | complex | array
            subtrahend array. Must be broadcast-compatible with ``self``. Has a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise differences. The returned array has a data type determined by type-promotion.
        """
        return ops.subtract(self, other)

    def __truediv__(self: array, other: int | float | complex | array, /) -> array:
        r"""
        Evaluates ``self_i / other_i`` for each element of an array instance with the respective element of ``other``.

        .. note::
           If one or both of ``self`` and ``other`` have integer data types, the result is backend-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

        Parameters
        ----------
        self: array
            array instance. Has a numeric data type.
        other: int | float | complex | array
            other array. Must be broadcast-compatible with ``self``. Has a numeric data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a floating-point data type determined by type-promotion.
        """
        return ops.divide(self, other)

    def __xor__(self: array, other: bool | int | array, /) -> array:
        """
        Evaluates ``self_i ^ other_i`` for each element of an array instance with the respective element of ``other``.

        Parameters
        ----------
        self: array
            array instance. Has an integer or boolean data type.
        other: bool | int | array
            other array. Must be broadcast-compatible with ``self``. Has an integer or boolean data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a data type determined by type-promotion.
        """
        return ops.bitwise_xor(self, other)

    def __radd__(self: array, other: int | float | complex | array) -> array:

        return ops.add(other, self)

    def __rsub__(self: array, other: int | float | complex | array) -> array:

        return ops.subtract(other, self)

    def __rmul__(self: array, other: int | float | complex | array) -> array:

        return ops.multiply(other, self)

    def __rtruediv__(self: array, other: int | float | complex | array) -> array:

        return ops.divide(other, self)

    def __rfloordiv__(self: array, other: int | float | array) -> array:

        return ops.floor_divide(other, self)

    def __rmod__(self: array, other: int | float | array) -> array:

        return ops.remainder(other, self)

    def __rpow__(self: array, other: int | float | complex | array) -> array:

        return ops.pow(other, self)

    def __rmatmul__(self: array, other: array) -> array:

        return ops.matmul(other, self)

    def __rand__(self: array, other: bool | array) -> array:

        return ops.logical_and(other, self)

    def __ror__(self: array, other: bool | array) -> array:

        return ops.logical_or(other, self)

    def __rxor__(self: array, other: bool | array) -> array:

        return ops.logical_xor(other, self)

    def __rlshift__(self: array, other: int | array) -> array:

        return ops.bitwise_left_shift(other, self)

    def __rrshift__(self: array, other: int | array) -> array:

        return ops.bitwise_right_shift(other, self)

    def to_device(
        self: array, device: DeviceLike, /, *, stream: int | Any | None = None
    ) -> array:
        """
        Copy the array to the specified ``device``.

        Parameters
        ----------
        self: array
            array instance.
        device: DeviceLike
            target device.
        stream: int | Any | None
            stream object to use during copy. Default: ``None``.

        Returns
        -------
        out: array
            an array with the same data and data type as ``self`` and located on the specified ``device``.

        Notes
        -----

        -   When ``device`` corresponds to the current device, the backend may return ``self`` or an explicit copy.
        -   If ``stream`` is provided, the copy is enqueued on that stream; otherwise the default stream is used. Whether the copy is synchronous or asynchronous is backend-dependent.
        """
        return array(array_api_compat.to_device(self.data, device, stream=stream), self._dims)

    # We use the _sentinel as default values instead of None, because None has a
    # semantic meaning in Numpy's __array__ implementation.
    def __array__(self, dtype=_sentinel, copy=_sentinel) -> np.ndarray:
        kwargs = {}
        if dtype is not _sentinel:
            kwargs["dtype"] = dtype
        if copy is not _sentinel:
            kwargs["copy"] = copy
        return self._data.__array__(**kwargs)

    def __iter__(self):
        return iter(array(x, self.dims[1:]) for x in iter(self.data))

    def __len__(self):
        return self.shape[0]

    @property
    def data(self):
        return self._data

    @property
    def dims(self) -> list[PossiblyUndefinedDim]:
        return self._dims.copy()

    @property
    def dim_sizes(self) -> dict[str, int]:
        return {d: s for d, s in zip(self.dims, self.shape)}

    def dim_index(self, dim: str) -> int:
        """
        Return the positional index of a named dimension.

        Parameters
        ----------
        dim: str
            the name of the dimension to look up.

        Returns
        -------
        out: int
            the zero-based index of ``dim`` in ``self.dims``.
        """
        return self.dims.index(dim)

    def rename_dim(self, dim: str | int, new_dim: str) -> "array":
        """
        Return a new array with one dimension renamed.

        Parameters
        ----------
        dim: str | int
            the dimension to rename, identified by name or positional index.
        new_dim: str
            the new name for the dimension.

        Returns
        -------
        out: array
            an array with the same data and shape as ``self``, with the
            specified dimension renamed to ``new_dim``.
        """
        axis = self.dims.index(dim) if isinstance(dim, str) else dim
        dims = list(self.dims)
        dims[axis] = new_dim
        return array(self.data, dims)

    def clear_dims(self):
        """
        Clear all named dimensions on this array in-place.

        Each dimension is replaced with an undefined dimension marker.
        Returns ``self`` to allow chaining.

        Returns
        -------
        out: array
            ``self``, with all dimensions set to undefined.
        """
        self._dims = [_UndefinedDim() for din in self.dims]
        return self  # return self for now to prevent updating array ID

    def max(self):
        """
        Return the maximum value of all elements in the array.

        Returns
        -------
        out: array
            a zero-dimensional array containing the maximum value.
        """
        return ops.max(self)

    def min(self):
        """
        Return the minimum value of all elements in the array.

        Returns
        -------
        out: array
            a zero-dimensional array containing the minimum value.
        """
        return ops.min(self)

    # Methods for casting dtype
    def int8(self):
        """Cast the array to the ``int8`` data type."""
        return data_types.int8(self)

    def int16(self):
        """Cast the array to the ``int16`` data type."""
        return data_types.int16(self)

    def int32(self):
        """Cast the array to the ``int32`` data type."""
        return data_types.int32(self)

    def int64(self):
        """Cast the array to the ``int64`` data type."""
        return data_types.int64(self)

    def uint8(self):
        """Cast the array to the ``uint8`` data type."""
        return data_types.uint8(self)

    def uint16(self):
        """Cast the array to the ``uint16`` data type."""
        return data_types.uint16(self)

    def uint32(self):
        """Cast the array to the ``uint32`` data type."""
        return data_types.uint32(self)

    def uint64(self):
        """Cast the array to the ``uint64`` data type."""
        return data_types.uint64(self)

    def float32(self):
        """Cast the array to the ``float32`` data type."""
        return data_types.float32(self)

    def float64(self):
        """Cast the array to the ``float64`` data type."""
        return data_types.float64(self)

    def complex64(self):
        """Cast the array to the ``complex64`` data type."""
        return data_types.complex64(self)

    def complex128(self):
        """Cast the array to the ``complex128`` data type."""
        return data_types.complex128(self)

    def bool(self):
        """Cast the array to the ``bool`` data type."""
        return data_types.bool(self)

    @property
    def at(self) -> "ArrayIndexUpdateHelper":
        """
        Return an index-update helper for immutable-style element updates.

        Use this property to perform functional updates on slices of the
        array without modifying it in place. For example::

            x = x.at[{"dim": 0}].set(value)

        Returns
        -------
        out: ArrayIndexUpdateHelper
            a helper object that supports ``.set()``, ``.add()``, and
            similar update operations indexed by named dimensions.
        """
        from spekk.ops._indexing import ArrayIndexUpdateHelper

        return ArrayIndexUpdateHelper(self)

    def __repr__(self):
        """
        Return a string representation of the array.

        For zero-dimensional arrays, the representation omits the ``dims``
        field. For all other arrays, ``dims`` and ``dtype`` are shown
        alongside the underlying data.
        """
        if self.ndim == 0:
            return f"spekk.ops.array({self.data}, dtype={self.dtype.name})"
        return (
            f"spekk.ops.array(\n{self.data}, dims={self.dims}, dtype={self.dtype.name})"
        )

    def all(
        self: array,
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
           If the array has a complex floating-point data type, elements having a non-zero component (real or imaginary) evaluate to ``True``.

        .. note::
           If the array is an empty array or the size of the axis along which to evaluate elements is zero, the test result is ``True``.

        Parameters
        ----------
        self: array
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
        return ops.all(self, axis=axis, keepdims=keepdims)

    def any(
        self: array,
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
           If the array has a complex floating-point data type, elements having a non-zero component (real or imaginary) evaluate to ``True``.

        .. note::
           If the array is an empty array or the size of the axis along which to evaluate elements is zero, the test result is ``False``.

        Parameters
        ----------
        self: array
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
        return ops.any(self, axis=axis, keepdims=keepdims)

    def argmax(
        self: array, /, *, axis: int | str | None = None, keepdims: bool = False
    ) -> array:
        """
        Returns the indices of the maximum values along a specified axis.

        When the maximum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

        Parameters
        ----------
        self: array
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
        return ops.argmax(self, axis=axis, keepdims=keepdims)

    def argmin(
        self: array, /, *, axis: int | str | None = None, keepdims: bool = False
    ) -> array:
        """
        Returns the indices of the minimum values along a specified axis.

        When the minimum value occurs multiple times, only the indices corresponding to the first occurrence are returned.

        Parameters
        ----------
        self: array
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
        return ops.argmin(self, axis=axis, keepdims=keepdims)

    def argsort(
        self: array,
        /,
        *,
        axis: int | str = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> array:
        """
        Returns the indices that sort the array along a specified axis.

        Parameters
        ----------
        self : array
            input array. Has a real-valued data type.
        axis: int | str
            axis along which to sort. May be an integer index or a named dimension name. If set to ``-1``, the function sorts along the last axis. Default: ``-1``.
        descending: bool
            sort order. If ``True``, the returned indices sort the array in descending order (by value). If ``False``, the returned indices sort the array in ascending order (by value). Default: ``False``.
        stable: bool
            sort stability. If ``True``, the returned indices maintain the relative order of the array values which compare as equal. If ``False``, the relative order of the array values which compare as equal is backend-dependent. Default: ``True``.

        Returns
        -------
        out : array
            an array of indices. The returned array has the same shape as the input array and the default array index data type.
        """
        return ops.argsort(self, axis=axis, descending=descending, stable=stable)

    def clip(
        self: array,
        /,
        min: int | float | array | None = None,
        max: int | float | array | None = None,
    ) -> array:
        r"""
        Clamps each element ``x_i`` of the input array to the range ``[min, max]``.

        Parameters
        ----------
        self: array
            input array. Has a real-valued data type.
        min: int | float | array | None
            lower-bound of the range to which to clamp. If ``None``, no lower bound is applied. Broadcasted with the input array. Has a real-valued data type. Default: ``None``.
        max: int | float | array | None
            upper-bound of the range to which to clamp. If ``None``, no upper bound is applied. Broadcasted with the input array. Has a real-valued data type. Default: ``None``.

        Returns
        -------
        out: array
            an array containing element-wise results. The returned array has the same data type as the input array.

        Notes
        -----

        - If both ``min`` and ``max`` are ``None``, the elements of the returned array equal the respective elements in the array.
        - If a broadcasted element in ``min`` is greater than a corresponding broadcasted element in ``max``, behavior is backend-dependent.
        - If the array and either ``min`` or ``max`` have different data type kinds (e.g., integer versus floating-point), behavior is backend-dependent.
        """
        return ops.clip(self, min=min, max=max)

    def cumsum(
        self: array,
        /,
        *,
        axis: int | str | None = None,
        dtype: DType | None = None,
        include_initial: bool = False,
    ) -> array:
        """
        Calculates the cumulative sum of elements in the input array.

        Parameters
        ----------
        self: array
            input array. Has a numeric data type.
        axis: int | str | None
            axis along which to compute the cumulative sum. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension.

            If the array has more than one dimension, providing an ``axis`` is required.

        dtype: DType | None
            data type of the returned array. If ``None``, the returned array has the same data type as the input array, unless the array has an integer data type supporting a smaller range of values than the default integer data type, in which case the default integer data type (or its unsigned equivalent) is used. If specified and differs from the data type of the array, the input array is cast before computing the sum. Default: ``None``.

        include_initial: bool
            whether to include the initial value (zero) as the first value in the output. Default: ``False``.

        Returns
        -------
        out: array
            an array containing the cumulative sums.

            Let ``N`` be the size of the axis along which to compute the cumulative sum.

            -   if ``include_initial`` is ``True``, the returned array has the same shape as the input array, except the size of the cumulated axis is ``N+1``.
            -   if ``include_initial`` is ``False``, the returned array has the same shape as the input array.
        """
        return ops.cumulative_sum(
            self, axis=axis, dtype=dtype, include_initial=include_initial
        )

    def diagonal(
        self: array,
        /,
        *,
        offset: int = 0,
        new_dim: str | None = None,
    ) -> array:
        """
        Returns the specified diagonals of a matrix (or a stack of matrices).

        Parameters
        ----------
        self: array
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices.
        offset: int
            offset specifying the off-diagonal relative to the main diagonal.

            - ``offset = 0``: the main diagonal.
            - ``offset > 0``: off-diagonal above the main diagonal.
            - ``offset < 0``: off-diagonal below the main diagonal.

            Default: ``0``.
        new_dim: str | None
            name for the new dimension created by extracting the diagonal. If ``None``, the dimension name is inferred.

            Default: ``None``.

        Returns
        -------
        out: array
            an array containing the diagonals and whose shape is determined by removing the last two dimensions and appending a dimension equal to the size of the resulting diagonals. The returned array has the same data type as the input array.
        """
        return ops.diagonal(self, offset=offset, new_dim=new_dim)

    def imag(
        self: array,
        /,
    ) -> array:
        """
        Returns the imaginary component of each element ``x_i`` of the array.

        Parameters
        ----------
        self: array
            Input array. Has a complex floating-point data type.

        Returns
        -------
        out: array
            An array containing the element-wise results. The returned array has a
            floating-point data type with the same precision as the input array (e.g., if
            the input is ``complex64``, the returned array has data type ``float32``).
        """
        return ops.imag(self)

    def mean(
        self: array,
        /,
        *,
        axis: int | str | tuple[int | str, ...] | None = None,
        keepdims: bool = False,
    ) -> array:
        """
        Calculates the arithmetic mean of the input array.

        Parameters
        ----------
        self: array
            input array. Has a numeric data type.
        axis: int | str | tuple[int | str, ...] | None
            axis or axes along which arithmetic means are computed. By default, the mean is computed over the entire array. Default: ``None``.
        keepdims: bool
            if ``True``, the reduced axes are included in the result as singleton dimensions. Otherwise, the reduced axes are not included in the result. Default: ``False``.

        Returns
        -------
        out: array
            if the arithmetic mean was computed over the entire array, a zero-dimensional array containing the arithmetic mean; otherwise, a non-zero-dimensional array containing the arithmetic means. If the array has an integer data type, the returned array has the default real-valued floating-point data type; otherwise, the returned array has the same data type as the input array.

        Notes
        -----

        - If the number of elements is ``0``, the arithmetic mean is ``NaN``.
        - If any element is ``NaN``, the arithmetic mean is ``NaN``.
        """
        return ops.mean(self, axis=axis, keepdims=keepdims)

    def nonzero(self: array, /, *, dim: str | None = None) -> tuple[array, ...]:
        """
        Returns the indices of the array elements which are non-zero.

        .. note::
           If the input array has a complex floating-point data type, non-zero elements are those elements having at least one component (real or imaginary) which is non-zero.

        .. note::
           If the input array has a boolean data type, non-zero elements are those elements which are equal to ``True``.

        .. admonition:: Data-dependent output shape
           :class: important

           The output shape of this function depends on the data values in the input array. Array libraries that build computation graphs (e.g., JAX, Dask) may find this function difficult to implement without knowing array values and may choose to omit it.

        Parameters
        ----------
        self: array
            input array. Must have a positive rank. If the array is zero-dimensional, the function raises an exception.
        dim: str | None
            name to assign to the output dimension of each returned index array. Default: ``None``.

        Returns
        -------
        out: tuple[array, ...]
            a tuple of ``k`` arrays, one for each dimension of the array and each of size ``n`` (where ``n`` is the total number of non-zero elements), containing the indices of the non-zero elements in that dimension. The indices are in row-major, C-style order. The returned array has the default array index data type.
        """
        return ops.nonzero(self, dim=dim)

    def prod(
        self: array,
        /,
        *,
        axis: int | str | tuple[int | str, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> array:
        """
        Calculates the product of input array elements.

        Parameters
        ----------
        self: array
            input array. Has a numeric data type.
        axis: int | str | tuple[int | str, ...] | None
            axis or axes along which products are computed. By default, the product is computed over the entire array. If a tuple, products are computed over multiple axes. Default: ``None``.
        dtype: DType | None
            data type of the returned array. If ``None``, the returned array has the same data type as the input array, unless the input array has an integer data type supporting a smaller range of values than the default integer data type, in which case the default integer data type is used. If the resolved data type differs from the data type of the input array, the input array is cast before computing the product. Default: ``None``.
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
        return ops.prod(self, axis=axis, dtype=dtype, keepdims=keepdims)

    def real(self: array, /) -> array:
        """
        Returns the real component of a complex number for each element ``x_i`` of the input array.

        Parameters
        ----------
        self: array
            input array. Has a complex floating-point data type.

        Returns
        -------
        out: array
            an array containing the element-wise results. The returned array has a floating-point data type with the same floating-point precision as the input array (e.g., if the input array is ``complex64``, the returned array has the floating-point data type ``float32``).
        """
        return ops.real(self)

    def repeat(
        self: array,
        repeats: int,
        /,
        *,
        axis: int | str | None = None,
    ) -> array:
        """
        Repeats each element of an array a specified number of times.

        Parameters
        ----------
        self: array
            input array containing elements to repeat.
        repeats: int
            the number of repetitions for each element.
        axis: int | str | None
            the axis (dimension) along which to repeat elements. If ``axis`` is ``None``,
            the input array is flattened in row-major (C-style) order before repeating, and
            the result is a one-dimensional array. A string value is interpreted as a
            dimension name. Default: ``None``.

        Returns
        -------
        out: array
            an output array containing repeated elements. The returned array has the same
            data type as the input array. If ``axis`` is ``None``, the returned array is
            one-dimensional; otherwise, it has the same shape as the input array except along the
            repeated axis.
        """
        return ops.repeat(self, repeats, axis=axis)

    def reshape(
        self: array,
        /,
        shape: tuple[int, ...] | list[int],
        dims: Sequence[str] | None = None,
        *,
        copy: bool | None = None,
    ) -> array:
        """
        Reshapes an array without changing its data.

        Parameters
        ----------
        self: array
            input array to reshape.
        shape: tuple[int, ...] | list[int]
            a new shape compatible with the original shape. One shape dimension is allowed
            to be ``-1``. When a shape dimension is ``-1``, the corresponding output array
            shape dimension is inferred from the length of the array and the remaining
            dimensions.
        dims: Sequence[str] | None
            the new list of dimension names. Must have the same length as ``shape``.
            Default: ``None``.
        copy: bool | None
            whether or not to copy the input array. If ``True``, the function always
            copies. If ``False``, the function never copies. If ``None``, the function
            avoids copying if possible, and may copy otherwise. Default: ``None``.

        Returns
        -------
        out: array
            an output array having the same data type and elements as the input array.

        Raises
        ------
        ValueError
            If ``copy=False`` and a copy would be necessary. Also raised if ``shape`` and
            ``dims`` do not have the same length.
        """
        return ops.reshape(self, shape, dims, copy=copy)

    def round(self: array, /) -> array:
        """
        Rounds each element ``x_i`` of the input array to the nearest integer-valued number.

        For complex floating-point operands, real and imaginary components are independently rounded.

        Parameters
        ----------
        self: array
            Input array. Has a numeric data type.

        Returns
        -------
        out: array
            An array containing the rounded result for each element in the array. The returned array has the same data type as the input array.

        Notes
        -----

        **Special cases**

        - If ``x_i`` is already integer-valued, the result is ``x_i``.
        - If ``x_i`` is ``NaN``, the result is ``NaN``.
        - If two integers are equally close to ``x_i``, the result is the even integer closest to ``x_i`` (banker's rounding).
        """
        return ops.round(self)

    def searchsorted(
        self: array,
        x2: int | float | array,
        /,
        *,
        side: Literal["left", "right"] = "left",
        sorter: array | None = None,
    ) -> array:
        """
        Finds the indices into the array such that, if the corresponding elements in ``x2`` were inserted before the indices, the order of the array, when sorted in ascending order, would be preserved.

        Parameters
        ----------
        self: array
            input array. Must be a one-dimensional array. Has a real-valued data type. If ``sorter`` is ``None``, must be sorted in ascending order; otherwise, ``sorter`` must be an array of indices that sort the array in ascending order.
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
            - if no index satisfies the index condition, then the returned index for that element is ``N``, where ``N`` is the number of elements in the array.

            Default: ``'left'``.
        sorter: array | None
            array of indices that sort the array in ascending order. The array must have the same shape as ``self`` and have an integer data type. Default: ``None``.

        Returns
        -------
        out: array
            an array of indices with the same shape as ``x2``. The returned array has the default array index data type.

        Notes
        -----
        For real-valued floating-point arrays, the sort order of NaNs and signed zeros is backend-dependent. Accordingly, when a real-valued floating-point array contains NaNs and signed zeros, what constitutes ascending order may vary among backends.

        Results are consistent with ``sort`` and ``argsort``: if a value in ``x2`` is inserted into the array at the corresponding index in the output array and ``sort`` is invoked on the resultant array, the sorted result is in the same order.
        """
        return ops.searchsorted(self, x2, side=side, sorter=sorter)

    def sort(
        self: array,
        /,
        *,
        axis: int | str = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> array:
        """
        Returns a sorted copy of the input array.

        Parameters
        ----------
        self: array
            input array. Has a real-valued data type.
        axis: int | str
            axis along which to sort. May be an integer index or a named dimension name. If set to ``-1``, the function sorts along the last axis. Default: ``-1``.
        descending: bool
            sort order. If ``True``, the array is sorted in descending order (by value). If ``False``, the array is sorted in ascending order (by value). Default: ``False``.
        stable: bool
            sort stability. If ``True``, the returned array maintains the relative order of the input array values which compare as equal. If ``False``, the relative order of the input array values which compare as equal is backend-dependent. Default: ``True``.

        Returns
        -------
        out : array
            a sorted array with the same data type and shape as the input array.
        """
        return ops.sort(self, axis=axis, descending=descending, stable=stable)

    def squeeze(
        self: array,
        /,
        axis: int | str | Sequence[int] | Sequence[str],
    ) -> array:
        """
        Removes singleton dimensions (axes) from the input array.

        Parameters
        ----------
        self: array
            input array.
        axis: int | str | Sequence[int] | Sequence[str]
            axis (or axes) to squeeze. A string value is interpreted as a dimension name.

        Returns
        -------
        out: array
            an output array having the same data type and elements as the input array.

        Raises
        ------
        ValueError
            If a specified axis has a size greater than one (i.e., it is not a
            singleton dimension).
        """
        return ops.squeeze(self, axis=axis)

    def std(
        self: array,
        /,
        *,
        axis: int | str | tuple[int | str, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> array:
        """
        Calculates the standard deviation of the input array.

        Parameters
        ----------
        self: array
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
            if the standard deviation was computed over the entire array, a zero-dimensional array containing the standard deviation; otherwise, an array containing the standard deviations. If the input array has an integer data type, the returned array has the default real-valued floating-point data type; otherwise, the returned array has the same data type as the input array.

        Notes
        -----

        -   If ``N - correction`` is less than or equal to ``0``, the standard deviation is ``NaN``.
        -   If any element is ``NaN``, the standard deviation is ``NaN``.
        """
        return ops.std(self, axis=axis, correction=correction, keepdims=keepdims)

    def sum(
        self: array,
        /,
        *,
        axis: int | str | tuple[int | str, ...] | None = None,
        dtype: DType | None = None,
        keepdims: bool = False,
    ) -> array:
        """
        Calculates the sum of the input array.

        Parameters
        ----------
        self: array
            input array. Has a numeric data type.
        axis: int | str | tuple[int | str, ...] | None
            axis or axes along which sums are computed. By default, the sum is computed over the entire array. If a tuple, sums are computed over multiple axes. Default: ``None``.
        dtype: DType | None
            data type of the returned array. If ``None``, the returned array has the same data type as the input array, unless the input array has an integer data type supporting a smaller range of values than the default integer data type, in which case the default integer data type is used. If the resolved data type differs from the data type of the input array, the input array is cast before computing the sum. Default: ``None``.
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
        return ops.sum(self, axis=axis, dtype=dtype, keepdims=keepdims)

    def take(
        self: array,
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
        self: array
            Input array.
        indices: array
            Array indices. Must be zero- or one-dimensional with an integer data
            type. Out-of-bounds behavior is backend-dependent.
        axis: int | str | None
            Axis over which to select values. Can be a dimension name or an integer
            position. If ``axis`` is negative, the axis is counted from the last
            dimension.

            If the array is one-dimensional, providing an ``axis`` is optional;
            however, if the array has more than one dimension, providing an ``axis`` is
            required.

        Returns
        -------
        out: array
            An array with the same data type and rank as the input array. The shape is the
            same as the input array except along ``axis``, whose size equals the number of
            elements in ``indices``. If ``indices`` is zero-dimensional, the
            specified axis is removed.
        """
        return ops.take(self, indices, axis=axis)

    def trace(self: array, /, *, offset: int = 0, dtype: DType | None = None) -> array:
        """
        Returns the sum along the specified diagonals of a matrix (or a stack of matrices).

        Parameters
        ----------
        self: array
            input array having shape ``(..., M, N)`` and whose innermost two dimensions form ``MxN`` matrices. Has a numeric data type.
        offset: int
            offset specifying the off-diagonal relative to the main diagonal.

            -   ``offset = 0``: the main diagonal.
            -   ``offset > 0``: off-diagonal above the main diagonal.
            -   ``offset < 0``: off-diagonal below the main diagonal.

            Default: ``0``.
        dtype: DType | None
            data type of the returned array. If ``None``, the returned array has the same data type as the input array, unless the input array has an integer data type supporting a smaller range of values than the default integer data type (e.g., the input array has an ``int16`` or ``uint32`` data type and the default integer data type is ``int64``). In those latter cases:

            -   if the input array has a signed integer data type (e.g., ``int16``), the returned array has the default integer data type.
            -   if the input array has an unsigned integer data type (e.g., ``uint16``), the returned array has an unsigned integer data type having the same number of bits as the default integer data type (e.g., if the default integer data type is ``int32``, the returned array has a ``uint32`` data type).

            If the data type (either specified or resolved) differs from the data type of the input array, the input array should be cast to the specified data type before computing the sum (rationale: the ``dtype`` keyword argument is intended to help prevent overflows). Default: ``None``.

        Returns
        -------
        out: array
            an array containing the traces and whose shape is determined by removing the last two dimensions and storing the traces in the last array dimension. For example, if the input array has rank ``k`` and shape ``(I, J, K, ..., L, M, N)``, then an output array has rank ``k-2`` and shape ``(I, J, K, ..., L)`` where

            ::

              out[i, j, k, ..., l] = trace(a[i, j, k, ..., l, :, :])

            The returned array has a data type as described by the ``dtype`` parameter above.

        Notes
        -----

        **Special Cases**

        Let ``N`` equal the number of elements over which to compute the sum.

        -   If ``N`` is ``0``, the sum is ``0`` (i.e., the empty sum).

        For both real-valued and complex floating-point operands, special cases are handled as if the operation is implemented by successive application of addition.
        """
        return ops.trace(self, offset=offset, dtype=dtype)

    def transpose(
        self: array,
        /,
        axes: tuple[int | str, ...] | list[int | str],
    ) -> array:
        """
        Permutes the axes (dimensions) of an array.

        Parameters
        ----------
        self: array
            input array.
        axes: tuple[int | str, ...] | list[int | str]
            a permutation of axes specifying the desired order. Accepts dimension names (strings) or positional indices (integers).

        Returns
        -------
        out: array
            an array containing the axes permutation. The returned array has the same data
            type as the input array.
        """
        return ops.permute_dims(self, axes=axes)

    def var(
        self: array,
        /,
        *,
        axis: int | str | tuple[int | str, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> array:
        """
        Calculates the variance of the input array.

        Parameters
        ----------
        self: array
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
            if the variance was computed over the entire array, a zero-dimensional array containing the variance; otherwise, an array containing the variances. If the input array has an integer data type, the returned array has the default real-valued floating-point data type; otherwise, the returned array has the same data type as the input array.

        Notes
        -----

        -   If ``N - correction`` is less than or equal to ``0``, the variance is ``NaN``.
        -   If any element is ``NaN``, the variance is ``NaN``.
        """
        return ops.var(self, axis=axis, correction=correction, keepdims=keepdims)
