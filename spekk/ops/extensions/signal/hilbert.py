from typing import Optional, Union

from spekk import ops
from spekk.ops._types import Dim


def hilbert(
    x: ops.array,
    N: Optional[int] = None,
    axis: Union[int, Dim] = -1,
) -> ops.array:
    """Compute the analytic signal, using the Hilbert transform.

    The transformation is done along the dimension specified by ``axis`` and is
    equivalent to :func:`scipy.signal.hilbert`, but operates on named ``spekk``
    arrays and is compatible with JIT compilation and the configured backend.

    The analytic signal ``x_a(t) = x(t) + i y(t)`` has ``x`` as its real part and
    the Hilbert transform of ``x`` as its imaginary part. The magnitude
    ``abs(x_a)`` is the envelope of the signal.

    Parameters
    ----------
    x : array
        Signal data. Must be real-valued.
    N : int, optional
        Number of Fourier components. If ``None`` (default), ``N`` is the size of
        ``x`` along ``axis``. If ``N`` is larger, ``x`` is zero-padded along
        ``axis``; if smaller, ``x`` is trimmed.
    axis : int or str, optional
        Axis along which to do the transformation. May be given either as an
        integer index or as a named dimension. Default: ``-1`` (last axis).

    Returns
    -------
    array
        Analytic signal of ``x`` along ``axis``. Has a complex floating-point
        data type and size ``N`` along ``axis``.

    Notes
    -----
    The analytic signal is computed in the frequency domain following the same
    construction as :func:`scipy.signal.hilbert`:

    1. ``Xf = fft(x, N, axis)``
    2. Multiply by a step filter ``h`` that zeroes the negative-frequency
       components and doubles the positive-frequency components (leaving the DC
       and, for even ``N``, the Nyquist component unchanged).
    3. ``x_a = ifft(Xf * h, axis)``

    References
    ----------
    .. [1] Wikipedia, "Analytic signal".
           https://en.wikipedia.org/wiki/Analytic_signal

    Examples
    --------
    >>> from spekk import ops
    >>> t = ops.linspace(0, 1, 512, dim="time")
    >>> x = ops.cos(2 * ops.pi * 5 * t)
    >>> analytic = ops.signal.hilbert(x, axis="time")
    >>> envelope = ops.abs(analytic)
    """
    if ops.isdtype(x.dtype, "complex floating"):
        raise ValueError("x must be real.")

    # Resolve ``axis`` to a named dimension so the doubling filter broadcasts
    # against ``x`` along the correct dimension regardless of how ``axis`` was
    # specified.
    if isinstance(axis, Dim):
        axis_dim = axis
    else:
        axis_int = axis if axis >= 0 else x.ndim + axis
        axis_dim = x.dims[axis_int]

    if N is None:
        N = x.dim_sizes[axis_dim]
    if N <= 0:
        raise ValueError("N must be positive.")

    Xf = ops.fft.fft(x, n=N, axis=axis_dim)

    # Build the frequency-domain step filter ``h`` (same construction as
    # scipy.signal.hilbert). ``h`` is a 1-D array carrying the ``axis_dim`` name,
    # so multiplication with ``Xf`` broadcasts across all other dimensions.
    k = ops.arange(N, dim=axis_dim)
    if N % 2 == 0:
        h = ops.where(
            k == 0,
            1.0,
            ops.where(
                k < N // 2,
                2.0,
                ops.where(k == N // 2, 1.0, 0.0),
            ),
        )
    else:
        h = ops.where(
            k == 0,
            1.0,
            ops.where(k <= (N - 1) // 2, 2.0, 0.0),
        )

    return ops.fft.ifft(Xf * h, axis=axis_dim)

