__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "clip",
    "conj",
    "copysign",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "hypot",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]


from spekk.ops._backend import backend
from spekk.ops._util import (
    ensure_backend_compatible_data,
    ensure_broadcastable,
    get_dims,
)
from spekk.ops.array_object import array


def abs(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the absolute value for each element ``x_i`` of the input array ``x``.

    For real-valued input arrays, the element-wise result has the same magnitude as the respective element in ``x`` but has positive sign.

    .. note::
       For signed integer data types, the absolute value of the minimum representable integer is backend-dependent.

    .. note::
       For complex floating-point operands, the complex absolute value is known as the norm, modulus, or magnitude and, for a complex number :math:`z = a + bj` is computed as

       .. math::
          \operatorname{abs}(z) = \sqrt{a^2 + b^2}

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a numeric data type.

    Returns
    -------
    out: array
        an array containing the absolute value of each element in ``x``. If ``x`` has a real-valued data type, the returned array has the same data type as ``x``. If ``x`` has a complex floating-point data type, the returned array has a real-valued floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``complex128``, then the returned array has a ``float64`` data type).

    Notes
    -----

    **Special Cases**

    For real-valued floating-point operands,

    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If ``x_i`` is ``-0``, the result is ``+0``.
    - If ``x_i`` is ``-infinity``, the result is ``+infinity``.

    For complex floating-point operands, let ``a = real(x_i)``, ``b = imag(x_i)``, and

    - If ``a`` is either ``+infinity`` or ``-infinity`` and ``b`` is any value (including ``NaN``), the result is ``+infinity``.
    - If ``a`` is any value (including ``NaN``) and ``b`` is either ``+infinity`` or ``-infinity``, the result is ``+infinity``.
    - If ``a`` is either ``+0`` or ``-0``, the result is equal to ``abs(b)``.
    - If ``b`` is either ``+0`` or ``-0``, the result is equal to ``abs(a)``.
    - If ``a`` is ``NaN`` and ``b`` is a finite number, the result is ``NaN``.
    - If ``a`` is a finite number and ``b`` is ``NaN``, the result is ``NaN``.
    - If ``a`` is ``NaN`` and ``b`` is ``NaN``, the result is ``NaN``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.abs(x), dims)


def acos(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the inverse cosine for each element ``x_i`` of the input array ``x``.

    Each element-wise result is expressed in radians.

    .. note::
       For complex floating-point operands, ``acos(conj(x))`` equals ``conj(acos(x))``.

    .. note::
       The inverse cosine (or arc cosine) is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty, -1)` and :math:`(1, \infty)` of the real axis.

       Accordingly, for complex arguments, the function returns the inverse cosine in the range of a strip unbounded along the imaginary axis and in the interval :math:`[0, \pi]` along the real axis.

       *Note: branch cuts follow C99 and have provisional status.*

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse cosine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is greater than ``1`` or less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.acos(x), dims)


def acosh(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the inverse hyperbolic cosine for each element ``x_i`` of the input array ``x``.

    .. note::
       For complex floating-point operands, ``acosh(conj(x))`` equals ``conj(acosh(x))``.

    .. note::
       The inverse hyperbolic cosine is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segment :math:`(-\infty, 1)` of the real axis.

       Accordingly, for complex arguments, the function returns the inverse hyperbolic cosine in the interval :math:`[0, \infty)` along the real axis and in the interval :math:`[-\pi j, +\pi j]` along the imaginary axis.

       *Note: branch cuts follow C99 and have provisional status.*

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse hyperbolic cosine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is less than ``1``, the result is ``NaN``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.acosh(x), dims)


def add(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    """
    Calculates the sum for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1: int | float | complex | array
        first input array. Has a numeric data type.
    x2: int | float | complex | array
        second input array. Broadcasted with ``x1``. Has a numeric data type.

    Returns
    -------
    out: array
        an array containing the element-wise sums. The returned array has a data type determined by type promotion.

    Notes
    -----
    Floating-point addition is a commutative operation, but not always associative.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.add(x1, x2), broadcasted_dims)


def asin(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the inverse sine for each element ``x_i`` of the input array ``x``.

    Each element-wise result is expressed in radians.

    .. note::
       For complex floating-point operands, ``asin(conj(x))`` equals ``conj(asin(x))``.

    .. note::
       The inverse sine (or arc sine) is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty, -1)` and :math:`(1, \infty)` of the real axis.

       Accordingly, for complex arguments, the function returns the inverse sine in the range of a strip unbounded along the imaginary axis and in the interval :math:`[-\pi/2, +\pi/2]` along the real axis.

       *Note: branch cuts follow C99 and have provisional status.*

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse sine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is greater than ``1`` or less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.asin(x), dims)


def asinh(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the inverse hyperbolic sine for each element ``x_i`` in the input array ``x``.

    .. note::
       For complex floating-point operands, ``asinh(conj(x))`` equals ``conj(asinh(x))`` and ``asinh(-z)`` equals ``-asinh(z)``.

    .. note::
       The inverse hyperbolic sine is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty j, -j)` and :math:`(j, \infty j)` of the imaginary axis.

       Accordingly, for complex arguments, the function returns the inverse hyperbolic sine in the range of a strip unbounded along the real axis and in the interval :math:`[-\pi j/2, +\pi j/2]` along the imaginary axis.

       *Note: branch cuts follow C99 and have provisional status.*

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse hyperbolic sine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.asinh(x), dims)


def atan(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the inverse tangent for each element ``x_i`` of the input array ``x``.

    Each element-wise result is expressed in radians.

    .. note::
       For complex floating-point operands, ``atan(conj(x))`` equals ``conj(atan(x))``.

    .. note::
       The inverse tangent (or arc tangent) is a multi-valued function and requires a branch on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty j, -j)` and :math:`(+j, \infty j)` of the imaginary axis.

       Accordingly, for complex arguments, the function returns the inverse tangent in the range of a strip unbounded along the imaginary axis and in the interval :math:`[-\pi/2, +\pi/2]` along the real axis.

       *Note: branch cuts follow C99 and have provisional status.*

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse tangent of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is ``+infinity``, the result is a backend-dependent approximation to ``+π/2``.
    - If ``x_i`` is ``-infinity``, the result is a backend-dependent approximation to ``-π/2``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.atan(x), dims)


def atan2(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    """
    Calculates the inverse tangent of the quotient ``x1/x2``, for each pair of elements ``(x1_i, x2_i)`` of the input arrays ``x1`` and ``x2``. Each element-wise result is expressed in radians in the range ``[-pi, +pi]``.

    The signs of ``x1_i`` and ``x2_i`` determine the quadrant of each element-wise result. The quadrant is chosen such that each result is the signed angle in radians between the ray ending at the origin and passing through ``(1, 0)`` and the ray ending at the origin and passing through ``(x2_i, x1_i)``.

    .. note::
       The "y-coordinate" is the first function parameter; the "x-coordinate" is the second. This parameter order is intentional and traditional for the two-argument inverse tangent function.

    Parameters
    ----------
    x1: int | float | complex | array
        input array corresponding to the y-coordinates. Has a real-valued floating-point data type.
    x2: int | float | complex | array
        input array corresponding to the x-coordinates. Must be compatible with ``x1`` for broadcasting. Has a real-valued floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse tangent of the quotient ``x1/x2``. The returned array has a real-valued floating-point data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.atan2(x1, x2), broadcasted_dims)


def atanh(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the inverse hyperbolic tangent for each element ``x_i`` of the input array ``x``.

    .. note::
       The principal value of the inverse hyperbolic tangent of a complex number :math:`z` is

       .. math::
          \operatorname{atanh}(z) = \frac{\ln(1+z)-\ln(z-1)}{2}

       For any :math:`z`,

       .. math::
          \operatorname{atanh}(z) = \frac{\operatorname{atan}(zj)}{j}

    .. note::
       For complex floating-point operands, ``atanh(conj(x))`` equals ``conj(atanh(x))`` and ``atanh(-x)`` equals ``-atanh(x)``.

    .. note::
       The inverse hyperbolic tangent is a multi-valued function and requires a branch cut on the complex plane. By convention, a branch cut is placed at the line segments :math:`(-\infty, 1]` and :math:`[1, \infty)` of the real axis.

       Accordingly, for complex arguments, the function returns the inverse hyperbolic tangent in the range of a half-strip unbounded along the real axis and in the interval :math:`[-\pi j/2, +\pi j/2]` along the imaginary axis.

       *Note: branch cuts follow C99 and have provisional status.*

    Parameters
    ----------
    x: int | float | complex | array
        input array whose elements each represent the area of a hyperbolic sector. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the inverse hyperbolic tangent of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is less than ``-1`` or greater than ``1``, the result is ``NaN``.
    - If ``x_i`` is ``-1``, the result is ``-infinity``.
    - If ``x_i`` is ``+1``, the result is ``+infinity``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.atanh(x), dims)


def bitwise_and(
    x1: int | array,
    x2: int | array,
    /,
) -> array:
    """
    Computes the element-wise bitwise AND of ``x1`` and ``x2``.

    Parameters
    ----------
    x1: int | array
        first input array. Has an integer or boolean data type.
    x2: int | array
        second input array. Broadcasted with ``x1``. Has an integer or boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.bitwise_and(x1, x2), broadcasted_dims)


def bitwise_left_shift(
    x1: int | array,
    x2: int | array,
    /,
) -> array:
    """
    Shifts the bits of each element ``x1_i`` of ``x1`` to the left by ``x2_i`` positions.

    Parameters
    ----------
    x1: int | array
        first input array. Has an integer data type.
    x2: int | array
        second input array. Broadcasted with ``x1``. Has an integer data type. Each element must be greater than or equal to ``0``.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.bitwise_left_shift(x1, x2), broadcasted_dims)


def bitwise_invert(x: int | array, /) -> array:
    """
    Inverts (flips) each bit for each element ``x_i`` of ``x``.

    Parameters
    ----------
    x: int | array
        input array. Has an integer or boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has the same data type as ``x``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.bitwise_invert(x), dims)


def bitwise_or(
    x1: int | array,
    x2: int | array,
    /,
) -> array:
    """
    Computes the element-wise bitwise OR of ``x1`` and ``x2``.

    Parameters
    ----------
    x1: int | array
        first input array. Has an integer or boolean data type.
    x2: int | array
        second input array. Broadcasted with ``x1``. Has an integer or boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.bitwise_or(x1, x2), broadcasted_dims)


def bitwise_right_shift(
    x1: int | array,
    x2: int | array,
    /,
) -> array:
    """
    Shifts the bits of each element ``x1_i`` of ``x1`` to the right by ``x2_i`` positions.

    .. note::
       This operation is an arithmetic shift (i.e., sign-propagating) and thus equivalent to floor division by a power of two.

    Parameters
    ----------
    x1: int | array
        first input array. Has an integer data type.
    x2: int | array
        second input array. Broadcasted with ``x1``. Has an integer data type. Each element must be greater than or equal to ``0``.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.bitwise_right_shift(x1, x2), broadcasted_dims)


def bitwise_xor(
    x1: int | array,
    x2: int | array,
    /,
) -> array:
    """
    Computes the element-wise bitwise XOR of ``x1`` and ``x2``.

    Parameters
    ----------
    x1: int | array
        first input array. Has an integer or boolean data type.
    x2: int | array
        second input array. Broadcasted with ``x1``. Has an integer or boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.bitwise_xor(x1, x2), broadcasted_dims)


def ceil(x: int | float | array, /) -> array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the smallest integer-valued number that is not less than ``x_i``.

    Parameters
    ----------
    x: int | float | array
        input array. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the rounded result for each element in ``x``. The returned array has the same data type as ``x``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.ceil(x), dims)


def clip(
    x: int | float | array,
    /,
    min: int | float | array | None = None,
    max: int | float | array | None = None,
) -> array:
    r"""
    Clamps each element ``x_i`` of the input array ``x`` to the range ``[min, max]``.

    Parameters
    ----------
    x: int | float | array
      input array. Has a real-valued data type.
    min: int | float | array | None
      lower-bound of the range to which to clamp. If ``None``, no lower bound is applied. Broadcasted with ``x``. Has a real-valued data type. Default: ``None``.
    max: int | float | array | None
      upper-bound of the range to which to clamp. If ``None``, no upper bound is applied. Broadcasted with ``x``. Has a real-valued data type. Default: ``None``.

    Returns
    -------
    out: array
      an array containing element-wise results. The returned array has the same data type as ``x``.

    Notes
    -----

    - If both ``min`` and ``max`` are ``None``, the elements of the returned array equal the respective elements in ``x``.
    - If a broadcasted element in ``min`` is greater than a corresponding broadcasted element in ``max``, behavior is backend-dependent.
    - If ``x`` and either ``min`` or ``max`` have different data type kinds (e.g., integer versus floating-point), behavior is backend-dependent.
    """
    if isinstance(min, array) and isinstance(max, array):
        dims, (x, min, max) = ensure_broadcastable(x, min, max)
    elif isinstance(min, array):
        dims, (x, min) = ensure_broadcastable(x, min)
    elif isinstance(max, array):
        dims, (x, max) = ensure_broadcastable(x, max)
    else:
        dims = get_dims(x)
    x, min, max = ensure_backend_compatible_data(x, min, max)

    from spekk import ops

    # block below is needed due to a bug in array compat lib. Need to cast min/max to
    # float when x.dtype is int
    if isinstance(x, array) and not ops.isdtype(x.dtype, "integral"):
        min = float(min) if isinstance(min, int) else min
        max = float(max) if isinstance(max, int) else max

    return array(backend.clip(x, min=min, max=max), dims)


def conj(x: int | float | complex | array, /) -> array:
    """
    Returns the complex conjugate for each element ``x_i`` of the input array ``x``.

    For complex numbers of the form

    .. math::
       a + bj

    the complex conjugate is defined as

    .. math::
       a - bj

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a complex floating-point data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has the same data type as ``x``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.conj(x), dims)


def copysign(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    r"""
    Composes a floating-point value with the magnitude of ``x1_i`` and the sign of ``x2_i`` for each element of the input array ``x1``.

    Parameters
    ----------
    x1: int | float | array
       input array containing magnitudes. Has a real-valued floating-point data type.
    x2: int | float | array
       input array whose sign bits are applied to the magnitudes of ``x1``. Broadcasted with ``x1``. Has a real-valued floating-point data type.

    Returns
    -------
    out: array
       an array containing the element-wise results. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    - If ``x2_i`` is ``NaN``, the sign bit of ``x2_i`` (not the ``NaN`` value) determines the sign of the result.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.copysign(x1, x2), broadcasted_dims)


def cos(x: int | float | complex | array, /) -> array:
    r"""
    Computes the cosine for each element ``x_i`` of the input array ``x``.

    Each element ``x_i`` is assumed to be expressed in radians.

    Parameters
    ----------
    x: int | float | complex | array
        Input array whose elements are each expressed in radians. Has a floating-point data type.

    Returns
    -------
    out: array
        An array containing the cosine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is ``+infinity`` or ``-infinity``, the result is ``NaN``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.cos(x), dims)


def cosh(x: int | float | complex | array, /) -> array:
    r"""
    Computes the hyperbolic cosine for each element ``x_i`` in the input array ``x``.

    The mathematical definition of the hyperbolic cosine is

    .. math::
       \operatorname{cosh}(x) = \frac{e^x + e^{-x}}{2}

    Parameters
    ----------
    x: int | float | complex | array
        Input array whose elements each represent a hyperbolic angle. Has a floating-point data type.

    Returns
    -------
    out: array
        An array containing the hyperbolic cosine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - ``cosh(x)`` equals ``cosh(-x)`` for all operands.
    - If ``x_i`` is ``+0`` or ``-0``, the result is ``1``.
    - If ``x_i`` is ``+infinity`` or ``-infinity``, the result is ``+infinity``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.cosh(x), dims)


def divide(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    r"""
    Divides each element ``x1_i`` of the input array ``x1`` by the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       If one or both of the input arrays have integer data types, the result is backend-dependent, as type promotion between data type "kinds" (e.g., integer versus floating-point) is unspecified.

    Parameters
    ----------
    x1: int | float | complex | array
        Dividend input array. Has a numeric data type.
    x2: int | float | complex | array
        Divisor input array. Broadcasted with ``x1``. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing the element-wise results. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is either ``+infinity`` or ``-infinity`` and ``x2_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - If ``x1_i`` is either ``+0`` or ``-0`` and ``x2_i`` is either ``+0`` or ``-0``, the result is ``NaN``.
    - Division by ``+0`` or ``-0`` for a nonzero finite ``x1_i`` produces an ``infinity`` with appropriate sign.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.divide(x1, x2), broadcasted_dims)


def equal(
    x1: bool | int | float | complex | array,
    x2: bool | int | float | complex | array,
    /,
) -> array:
    r"""
    Computes the truth value of ``x1_i == x2_i`` for each element ``x1_i`` of ``x1`` with the respective element ``x2_i`` of ``x2``.

    Parameters
    ----------
    x1: bool | int | float | complex | array
        First input array. May have any data type.
    x2: bool | int | float | complex | array
        Second input array. Broadcasted with ``x1``. May have any data type.

    Returns
    -------
    out: array
        An array containing the element-wise results. The returned array has a data type of ``bool``.

    Notes
    -----

    **Special Cases**

    - If ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``False``.
    - ``+0`` and ``-0`` are considered equal.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.equal(x1, x2), broadcasted_dims)


def exp(x: int | float | complex | array, /) -> array:
    """
    Computes the exponential function for each element ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``, where ``e`` is the base of the natural logarithm).

    .. note::
       The exponential function is an entire function in the complex plane and has no branch cuts.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a floating-point data type.

    Returns
    -------
    out: array
        An array containing the evaluated exponential function result for each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``+0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.exp(x), dims)


def expm1(x: int | float | complex | array, /) -> array:
    """
    Computes ``exp(x) - 1`` for each element ``x_i`` of the input array ``x``. More accurate than ``exp(x) - 1.0`` when ``x`` is close to zero.

    .. note::
       The exponential function is an entire function in the complex plane and has no branch cuts.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a floating-point data type.

    Returns
    -------
    out: array
        An array containing the evaluated result for each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    - If ``x_i`` is ``-infinity``, the result is ``-1``.
    - Signed zeros are preserved: ``expm1(+0)`` is ``+0`` and ``expm1(-0)`` is ``-0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.expm1(x), dims)


def floor(x: int | float | array, /) -> array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the greatest (i.e., closest to ``+infinity``) integer-valued number that is not greater than ``x_i``.

    Parameters
    ----------
    x: int | float | array
        input array. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the rounded result for each element in ``x``. The returned array has the same data type as ``x``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.floor(x), dims)


def floor_divide(x1: int | float | array, x2: int | float | array, /) -> array:
    """
    Rounds the result of dividing each element ``x1_i`` of the input array ``x1`` by the respective element ``x2_i`` of the input array ``x2`` to the greatest (i.e., closest to ``+infinity``) integer-valued number that is not greater than the division result.

    .. note::
       For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus backend-dependent.

    Parameters
    ----------
    x1: int | float | array
        dividend input array. Has a real-valued data type.
    x2: int | float | array
        divisor input array. Broadcasted with ``x1``. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - When one of the operands is ``infinity``, the result may vary across backends. Some backends follow ``floor(a/b)`` while others pair ``//`` with ``%``.
    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If both operands are ``infinity`` or both are zero, the result is ``NaN``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.floor_divide(x1, x2), broadcasted_dims)


def greater(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    """
    Computes the truth value of ``x1_i > x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       Inequality comparison of complex numbers is unspecified and thus backend-dependent.

    Parameters
    ----------
    x1: int | float | array
        first input array. Has a real-valued data type.
    x2: int | float | array
        second input array. Broadcasted with ``x1``. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.greater(x1, x2), broadcasted_dims)


def greater_equal(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    """
    Computes the truth value of ``x1_i >= x2_i`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    .. note::
       Inequality comparison of complex numbers is unspecified and thus backend-dependent.

    Parameters
    ----------
    x1: int | float | array
        first input array. Has a real-valued data type.
    x2: int | float | array
        second input array. Broadcasted with ``x1``. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.greater_equal(x1, x2), broadcasted_dims)


def hypot(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    r"""
    Computes the square root of the sum of squares for each element ``x1_i`` of
    ``x1`` with the respective element ``x2_i`` of ``x2``, avoiding underflow
    and overflow during intermediate stages of computation.

    Parameters
    ----------
    x1: int | float | array
       First input array. Has a real-valued floating-point data type.
    x2: int | float | array
       Second input array. Broadcasted with ``x1``. Has a real-valued
       floating-point data type.

    Returns
    -------
    out: array
       An array containing the element-wise results. The returned array has a
       real-valued floating-point data type determined by type promotion.

    Notes
    -----

    **Special Cases**

    - If either input element is ``+infinity`` or ``-infinity``, the result is
      ``+infinity``, even if the other value is ``NaN``.
    - ``hypot(x1, x2)`` is symmetric and sign-insensitive:
      ``hypot(x1, x2) == hypot(x2, x1) == hypot(-x1, -x2)``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.hypot(x1, x2), broadcasted_dims)


def imag(x: int | float | complex | array, /) -> array:
    """
    Returns the imaginary component of each element ``x_i`` of ``x``.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a complex floating-point data type.

    Returns
    -------
    out: array
        An array containing the element-wise results. The returned array has a
        floating-point data type with the same precision as ``x`` (e.g., if
        ``x`` is ``complex64``, the returned array has data type ``float32``).
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.imag(x), dims)


def isfinite(x: int | float | complex | array, /) -> array:
    """
    Tests each element ``x_i`` of ``x`` to determine if finite.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing test results. The returned array has a data type of
        ``bool``.

    Notes
    -----

    - Returns ``False`` for ``+infinity``, ``-infinity``, and ``NaN``.
    - For complex inputs, returns ``True`` only if both real and imaginary parts
      are finite.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.isfinite(x), dims)


def isinf(x: int | float | complex | array, /) -> array:
    """
    Tests each element ``x_i`` of ``x`` to determine if equal to positive or
    negative infinity.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing test results. The returned array has a data type of
        ``bool``.

    Notes
    -----

    - For complex inputs, returns ``True`` if either the real or imaginary part
      is infinite.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.isinf(x), dims)


def isnan(x: int | float | complex | array, /) -> array:
    """
    Tests each element ``x_i`` of ``x`` to determine whether the element is
    ``NaN``.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing test results. The returned array has a data type of
        ``bool``.

    Notes
    -----

    - For complex inputs, returns ``True`` if either the real or imaginary part
      is ``NaN``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.isnan(x), dims)


def less(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    """
    Computes the truth value of ``x1_i < x2_i`` element-wise.

    Parameters
    ----------
    x1: int | float | array
        first input array. Has a real-valued data type.
    x2: int | float | array
        second input array. Broadcasted with ``x1``. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.less(x1, x2), broadcasted_dims)


def less_equal(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    """
    Computes the truth value of ``x1_i <= x2_i`` element-wise.

    Parameters
    ----------
    x1: int | float | array
        first input array. Has a real-valued data type.
    x2: int | float | array
        second input array. Broadcasted with ``x1``. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.less_equal(x1, x2), broadcasted_dims)


def log(x: int | float | complex | array, /) -> array:
    r"""
    Calculates the natural (base ``e``) logarithm for each element ``x_i`` of the input array ``x``.

    .. note::
       The natural logarithm of a complex number :math:`z` with polar coordinates :math:`(r,\theta)` equals :math:`\ln r + (\theta + 2n\pi)j` with principal value :math:`\ln r + \theta j`.

    .. note::
       By convention, the branch cut of the natural logarithm is the negative real axis :math:`(-\infty, 0)`.

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the evaluated natural logarithm for each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is less than ``0``, the result is ``NaN``.
    - If ``x_i`` is either ``+0`` or ``-0``, the result is ``-infinity``.
    - If ``x_i`` is ``1``, the result is ``+0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.log(x), dims)


def log1p(x: int | float | complex | array, /) -> array:
    r"""
    Calculates ``log(1+x)`` for each element ``x_i`` of the input array ``x``, where ``log`` is the natural (base ``e``) logarithm.

    More accurate than computing ``log(1+x)`` directly when ``x`` is close to zero.

    .. note::
       By convention, the branch cut of the natural logarithm is the negative real axis :math:`(-\infty, 0)`.

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the evaluated result for each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    For real-valued floating-point operands,

    - If ``x_i`` is less than ``-1``, the result is ``NaN``.
    - If ``x_i`` is ``-1``, the result is ``-infinity``.
    - If ``x_i`` is ``-0``, the result is ``-0``.
    - If ``x_i`` is ``+0``, the result is ``+0``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.log1p(x), dims)


def log2(x: bool | int | float | complex | array, /) -> array:
    r"""
    Calculates a backend-dependent approximation to the base ``2`` logarithm for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x: bool | int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the evaluated base ``2`` logarithm for each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    For complex floating-point operands, ``log2(conj(x))`` equals ``conj(log2(x))``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.log2(x), dims)


def log10(x: bool | int | float | complex | array, /) -> array:
    r"""
    Calculates a backend-dependent approximation to the base ``10`` logarithm for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x: bool | int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the evaluated base ``10`` logarithm for each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    For complex floating-point operands, ``log10(conj(x))`` equals ``conj(log10(x))``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.log10(x), dims)


def logaddexp(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    """
    Calculates the logarithm of the sum of exponentiations ``log(exp(x1) + exp(x2))`` for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Parameters
    ----------
    x1: int | float | complex | array
        first input array. Has a real-valued floating-point data type.
    x2: int | float | complex | array
        second input array. Broadcasted with ``x1``. Has a real-valued floating-point data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a real-valued floating-point data type determined by type promotion.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.logaddexp(x1, x2), broadcasted_dims)


def logical_and(
    x1: bool | array,
    x2: bool | array,
    /,
) -> array:
    """
    Computes the logical AND for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Non-boolean inputs are supported: zeros are treated as ``False``, non-zeros as ``True``.

    Parameters
    ----------
    x1: bool | array
        first input array. Has a boolean data type.
    x2: bool | array
        second input array. Broadcasted with ``x1``. Has a boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.logical_and(x1, x2), broadcasted_dims)


def logical_not(x: bool | array, /) -> array:
    """
    Computes the logical NOT for each element ``x_i`` of the input array ``x``.

    Non-boolean inputs are supported: zeros are treated as ``False``, non-zeros as ``True``.

    Parameters
    ----------
    x: bool | array
        input array. Has a boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.logical_not(x), dims)


def logical_or(
    x1: bool | array,
    x2: bool | array,
    /,
) -> array:
    """
    Computes the logical OR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Non-boolean inputs are supported: zeros are treated as ``False``, non-zeros as ``True``.

    Parameters
    ----------
    x1: bool | array
        first input array. Has a boolean data type.
    x2: bool | array
        second input array. Broadcasted with ``x1``. Has a boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.logical_or(x1, x2), broadcasted_dims)


def logical_xor(
    x1: bool | array,
    x2: bool | array,
    /,
) -> array:
    """
    Computes the logical XOR for each element ``x1_i`` of the input array ``x1`` with the respective element ``x2_i`` of the input array ``x2``.

    Non-boolean inputs are supported: zeros are treated as ``False``, non-zeros as ``True``.

    Parameters
    ----------
    x1: bool | array
        first input array. Has a boolean data type.
    x2: bool | array
        second input array. Broadcasted with ``x1``. Has a boolean data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.logical_xor(x1, x2), broadcasted_dims)


def maximum(
    x1: bool | int | float | complex | array,
    x2: bool | int | float | complex | array,
    /,
) -> array:
    r"""
    Computes the element-wise maximum of ``x1`` and ``x2``.

    Parameters
    ----------
    x1: bool | int | float | complex | array
       First input array. Has a real-valued data type.
    x2: bool | int | float | complex | array
       Second input array. Broadcasted with ``x1``.

    Returns
    -------
    out: array
       An array containing the element-wise maximum values.

    Notes
    -----

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - The ordering of signed zeros is backend-dependent.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.maximum(x1, x2), broadcasted_dims)


def minimum(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    r"""
    Computes the element-wise minimum of ``x1`` and ``x2``.

    Parameters
    ----------
    x1: int | float | array
       First input array. Has a real-valued data type.
    x2: int | float | array
       Second input array. Broadcasted with ``x1``.

    Returns
    -------
    out: array
       An array containing the element-wise minimum values.

    Notes
    -----

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - The ordering of signed zeros is backend-dependent.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.minimum(x1, x2), broadcasted_dims)


def multiply(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    r"""
    Computes the element-wise product of ``x1`` and ``x2``.

    .. note::
       Floating-point multiplication is not always associative due to finite precision.

    Parameters
    ----------
    x1: int | float | complex | array
        First input array. Has a numeric data type.
    x2: int | float | complex | array
        Second input array. Broadcasted with ``x1``.

    Returns
    -------
    out: array
        An array containing the element-wise products.

    Notes
    -----

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - ``infinity * 0`` is ``NaN``.
    - For complex operands, special-case behavior involving ``NaN`` or ``infinity`` is backend-dependent.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.multiply(x1, x2), broadcasted_dims)


def negative(x: int | float | complex | array, /) -> array:
    """
    Computes the numerical negative of each element (i.e., ``y_i = -x_i``).

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing the negated value for each element in ``x``.

    Notes
    -----

    - For signed integer data types, the numerical negative of the minimum representable
      integer is backend-dependent.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.negative(x), dims)


def not_equal(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    """
    Computes the truth value of ``x1_i != x2_i`` for each element ``x1_i`` of ``x1`` with the respective element ``x2_i`` of ``x2``.

    Parameters
    ----------
    x1: int | float | complex | array
        first input array. May have any data type.
    x2: int | float | complex | array
        second input array. Broadcasted with ``x1``.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type of ``bool``.

    Notes
    -----

    **Special Cases**

    - If ``x1_i`` is ``NaN`` or ``x2_i`` is ``NaN``, the result is ``True``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.not_equal(x1, x2), broadcasted_dims)


def positive(x: int | float | complex | array, /) -> array:
    """
    Computes the numerical positive of each element ``x_i`` (i.e., ``y_i = +x_i``) of the input array ``x``.

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a numeric data type.

    Returns
    -------
    out: array
        an array containing the evaluated result for each element in ``x``. The returned array has the same data type as ``x``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.positive(x), dims)


def pow(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    r"""
    Calculates a backend-dependent approximation of exponentiation by raising each element ``x1_i`` (the base) of ``x1`` to the power of ``x2_i`` (the exponent), where ``x2_i`` is the corresponding element of ``x2``.

    .. note::
       If both ``x1`` and ``x2`` have integer data types, the result of ``pow`` when ``x2_i`` is negative (i.e., less than zero) is unspecified and thus backend-dependent.

       If ``x1`` has an integer data type and ``x2`` has a floating-point data type, behavior is backend-dependent (type promotion between data type "kinds" (integer versus floating-point) is unspecified).

    Parameters
    ----------
    x1: int | float | complex | array
        first input array whose elements correspond to the exponentiation base. Has a numeric data type.
    x2: int | float | complex | array
        second input array whose elements correspond to the exponentiation exponent. Broadcasted with ``x1``. Has a numeric data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - If ``x2_i`` is ``+0`` or ``-0``, the result is ``1``, even if ``x1_i`` is ``NaN``.
    - If ``x1_i`` is ``1``, the result is ``1``, even if ``x2_i`` is ``NaN``.
    - If ``x1_i`` is less than ``0``, ``x1_i`` is finite, ``x2_i`` is finite, and ``x2_i`` is not an integer value, the result is ``NaN``.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.pow(x1, x2), broadcasted_dims)


def real(x: int | float | complex | array, /) -> array:
    """
    Returns the real component of a complex number for each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a complex floating-point data type.

    Returns
    -------
    out: array
        an array containing the element-wise results. The returned array has a floating-point data type with the same floating-point precision as ``x`` (e.g., if ``x`` is ``complex64``, the returned array has the floating-point data type ``float32``).
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.real(x), dims)


def remainder(
    x1: int | float | array,
    x2: int | float | array,
    /,
) -> array:
    """
    Returns the remainder of division for each element ``x1_i`` of ``x1`` and the respective element ``x2_i`` of ``x2``.

    Equivalent to the Python modulus operator ``x1_i % x2_i``.

    .. note::
       For input arrays which promote to an integer data type, the result of division by zero is unspecified and thus backend-defined.

    .. note::
       This function is **not** recommended for floating-point operands as semantics do not follow IEEE 754.

    Parameters
    ----------
    x1: int | float | array
        Dividend input array. Has a real-valued data type.
    x2: int | float | array
        Divisor input array. Broadcasted with ``x1``. Has a real-valued data type.

    Returns
    -------
    out: array
        An array containing the element-wise results. Each element-wise result has the same sign as the respective element ``x2_i``. The returned array has a type determined by type promotion.

    Notes
    -----

    **Special cases**

    For floating-point operands,

    - If either ``x1_i`` or ``x2_i`` is ``NaN``, the result is ``NaN``.
    - If ``x1_i`` is infinite or ``x2_i`` is zero, the result is ``NaN``.
    - If ``x1_i`` is finite and ``x2_i`` is infinite, the result matches Python ``%`` behavior.
    - In the remaining cases, the result matches that of the Python ``%`` operator.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.remainder(x1, x2), broadcasted_dims)


def round(x: int | float | complex | array, /) -> array:
    """
    Rounds each element ``x_i`` of the input array ``x`` to the nearest integer-valued number.

    For complex floating-point operands, real and imaginary components are independently rounded.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing the rounded result for each element in ``x``. The returned array has the same data type as ``x``.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is already integer-valued, the result is ``x_i``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.
    - If two integers are equally close to ``x_i``, the result is the even integer closest to ``x_i`` (banker's rounding).
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.round(x), dims)


def sign(x: int | float | complex | array, /) -> array:
    r"""
    Returns an indication of the sign of a number for each element ``x_i`` of ``x``.

    The sign function (also known as the **signum function**) of a number :math:`x_i` is defined as

    .. math::
       \operatorname{sign}(x_i) = \begin{cases}
       0 & \textrm{if } x_i = 0 \\
       \frac{x_i}{|x_i|} & \textrm{otherwise}
       \end{cases}

    where :math:`|x_i|` is the absolute value of :math:`x_i`.

    Parameters
    ----------
    x: int | float | complex | array
        Input array. Has a numeric data type.

    Returns
    -------
    out: array
        An array containing the sign of each element in ``x``. The returned array has the same data type as ``x``.

    Notes
    -----

    **Special cases**

    For real-valued operands,

    - If ``x_i`` is less than ``0``, the result is ``-1``.
    - If ``x_i`` is either ``-0`` or ``+0``, the result is ``0``.
    - If ``x_i`` is greater than ``0``, the result is ``+1``.
    - If ``x_i`` is ``NaN``, the result is ``NaN``.

    For complex floating-point operands, ``sign(x_i)`` is defined as ``x_i / abs(x_i)``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.sign(x), dims)


def signbit(x: int | float | array, /) -> array:
    r"""
    Determines whether the sign bit is set for each element ``x_i`` of ``x``.

    The sign bit is set whenever ``x_i`` is ``-0``, less than zero, or a ``NaN`` whose sign bit is ``1``.

    Parameters
    ----------
    x: int | float | array
        Input array. Has a real-valued floating-point data type.

    Returns
    -------
    out: array
        An array containing the evaluated result for each element in ``x``. The returned array has a data type of ``bool``.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is ``-0``, the result is ``True``.
    - If ``x_i`` is ``NaN``, the result depends on the sign bit of the ``NaN``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.signbit(x), dims)


def sin(x: int | float | complex | array, /) -> array:
    r"""
    Computes the sine of each element ``x_i`` of the input array ``x``.

    Each element ``x_i`` is assumed to be expressed in radians.

    Parameters
    ----------
    x: int | float | complex | array
        input array whose elements are each expressed in radians. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the sine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is ``+infinity`` or ``-infinity``, the result is ``NaN``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.sin(x), dims)


def sinh(x: int | float | complex | array, /) -> array:
    r"""
    Computes the hyperbolic sine of each element ``x_i`` of the input array ``x``.

    The mathematical definition of the hyperbolic sine is

    .. math::
       \operatorname{sinh}(x) = \frac{e^x - e^{-x}}{2}

    Parameters
    ----------
    x: int | float | complex | array
        input array whose elements each represent a hyperbolic angle. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the hyperbolic sine of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    ``sinh(x)`` equals ``-sinh(-x)`` for all operands.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.sinh(x), dims)


def square(x: int | float | complex | array, /) -> array:
    r"""
    Squares each element ``x_i`` of the input array ``x``.

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a numeric data type.

    Returns
    -------
    out: array
        an array containing the square of each element in ``x``. The returned array has a data type determined by type promotion.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.square(x), dims)


def sqrt(x: int | float | complex | array, /) -> array:
    r"""
    Computes the principal square root of each element ``x_i`` of the input array ``x``.

    For complex arguments, the branch cut is the negative real axis :math:`(-\infty, 0)`. The function returns values in the right half-plane (including the imaginary axis).

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the square root of each element in ``x``. The returned array has a floating-point data type determined by type promotion.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is less than ``0`` (real-valued input), the result is ``NaN``.
    - If ``x_i`` is ``+infinity``, the result is ``+infinity``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.sqrt(x), dims)


def subtract(
    x1: int | float | complex | array,
    x2: int | float | complex | array,
    /,
) -> array:
    """
    Calculates the element-wise difference of two input arrays.

    Parameters
    ----------
    x1: int | float | complex | array
        first input array. Has a numeric data type.
    x2: int | float | complex | array
        second input array. Broadcasted with ``x1``. Has a numeric data type.

    Returns
    -------
    out: array
        an array containing the element-wise differences.
    """
    broadcasted_dims, (x1, x2) = ensure_broadcastable(x1, x2)
    x1, x2 = ensure_backend_compatible_data(x1, x2)
    return array(backend.subtract(x1, x2), broadcasted_dims)


def tan(x: int | float | complex | array, /) -> array:
    """
    Calculates the tangent for each element of the input array.

    Each element is assumed to be expressed in radians.

    Parameters
    ----------
    x: int | float | complex | array
        input array whose elements are expressed in radians. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the tangent of each element in ``x``.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is either ``+infinity`` or ``-infinity``, the result is ``NaN``.
    - For complex operands, special cases are handled as if computed via ``-1j * tanh(x*1j)``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.tan(x), dims)


def tanh(x: int | float | complex | array, /) -> array:
    """
    Calculates the hyperbolic tangent for each element of the input array.

    Parameters
    ----------
    x: int | float | complex | array
        input array. Has a floating-point data type.

    Returns
    -------
    out: array
        an array containing the hyperbolic tangent of each element in ``x``.

    Notes
    -----

    **Special cases**

    - If ``x_i`` is ``+infinity``, the result is ``+1``.
    - If ``x_i`` is ``-infinity``, the result is ``-1``.
    - ``tanh(-x)`` equals ``-tanh(x)``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.tanh(x), dims)


def trunc(x: int | float | array, /) -> array:
    """
    Rounds each element of the input array toward zero to the nearest integer.

    Parameters
    ----------
    x: int | float | array
        input array. Has a real-valued data type.

    Returns
    -------
    out: array
        an array containing the truncated result for each element in ``x``. Has the same data type as ``x``.
    """
    dims = get_dims(x)
    (x,) = ensure_backend_compatible_data(x)
    return array(backend.trunc(x), dims)
