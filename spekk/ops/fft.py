__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]
from typing import Literal

from spekk.ops._backend import backend
from spekk.ops._types import (
    Dim,
    Sequence,
    device,
    undefined_dim,
)
from spekk.ops.array_object import array
from spekk import ops

def fft(
    x: array,
    /,
    *,
    n: int | None = None,
    axis: int | str = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dim: str | None = None,
) -> array:
    """
    Computes the one-dimensional discrete Fourier transform.

    .. note::
       Applying the one-dimensional inverse discrete Fourier transform to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``ifft(fft(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (number of elements, axis, and normalization mode).

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    n: int | None
        number of elements over which to compute the transform along the axis (dimension) specified by ``axis``. Let ``M`` be the size of the input array along the axis specified by ``axis``. When ``n`` is ``None``, ``n`` is set equal to ``M``.

        -   If ``n`` is greater than ``M``, the axis specified by ``axis`` is zero-padded to size ``n``.
        -   If ``n`` is less than ``M``, the axis specified by ``axis`` is trimmed to size ``n``.
        -   If ``n`` equals ``M``, all elements along the axis specified by ``axis`` are used when computing the transform.

        Default: ``None``.
    axis: int | str
        axis (dimension) of the input array over which to compute the transform. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension. Default: ``-1``.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: no normalization.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: normalize by ``1/n``.

        Default: ``'backward'``.
    rename_dim: str | None
        if provided, the transformed axis is renamed to this dimension name in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axis (dimension) specified by ``axis``. The returned array has the same data type as ``x`` and the same shape as ``x``, except for the axis specified by ``axis`` which has size ``n``.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if rename_dim is not None:
        dims[axis] = rename_dim
    return array(backend.fft.fft(x.data, n=n, axis=axis, norm=norm), dims)


def ifft(
    x: array,
    /,
    *,
    n: int | None = None,
    axis: int | str = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dim: str | None = None,
) -> array:
    """
    Computes the one-dimensional inverse discrete Fourier transform.

    .. note::
       Applying the one-dimensional discrete Fourier transform to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``fft(ifft(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (number of elements, axis, and normalization mode).

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    n: int | None
        number of elements over which to compute the transform along the axis (dimension) specified by ``axis``. Let ``M`` be the size of the input array along the axis specified by ``axis``. When ``n`` is ``None``, ``n`` is set equal to ``M``.

        -   If ``n`` is greater than ``M``, the axis specified by ``axis`` is zero-padded to size ``n``.
        -   If ``n`` is less than ``M``, the axis specified by ``axis`` is trimmed to size ``n``.
        -   If ``n`` equals ``M``, all elements along the axis specified by ``axis`` are used when computing the transform.

        Default: ``None``.
    axis: int | str
        axis (dimension) of the input array over which to compute the transform. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension. Default: ``-1``.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: normalize by ``1/n``.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: no normalization.

        Default: ``'backward'``.
    rename_dim: str | None
        if provided, the transformed axis is renamed to this dimension name in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axis (dimension) specified by ``axis``. The returned array has the same data type as ``x`` and the same shape as ``x``, except for the axis specified by ``axis`` which has size ``n``.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if rename_dim is not None:
        dims[axis] = rename_dim
    return array(backend.fft.ifft(x.data, n=n, axis=axis, norm=norm), dims)


def fftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int | str] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dims: Sequence[str] | None = None,
) -> array:
    """
    Computes the n-dimensional discrete Fourier transform.

    .. note::
       Applying the n-dimensional inverse discrete Fourier transform to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``ifftn(fftn(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (sizes, axes, and normalization mode).

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    s: Sequence[int] | None
        number of elements over which to compute the transform along the axes (dimensions) specified by ``axes``. Let ``i`` be the index of the ``n``-th axis specified by ``axes`` (i.e., ``i = axes[n]``) and ``M[i]`` be the size of the input array along axis ``i``. When ``s`` is ``None``, ``s`` is set equal to a sequence of integers such that ``s[i]`` equals ``M[i]`` for all ``i``.

        -   If ``s[i]`` is greater than ``M[i]``, axis ``i`` is zero-padded to size ``s[i]``.
        -   If ``s[i]`` is less than ``M[i]``, axis ``i`` is trimmed to size ``s[i]``.
        -   If ``s[i]`` equals ``M[i]`` or ``-1``, all elements along axis ``i`` are used when computing the transform.

        If ``s`` is not ``None``, ``axes`` must not be ``None``. Default: ``None``.
    axes: Sequence[int | str] | None
        axes (dimensions) over which to compute the transform. Integers refer to positional axes (negative counts from the end), strings refer to named dimensions.

        If ``s`` is provided, the corresponding ``axes`` to be transformed must also be provided. If ``axes`` is ``None``, the transform is computed over all axes. Default: ``None``.

        If ``axes`` contains two or more entries which resolve to the same axis (i.e., resolved axes are not unique), the behavior is unspecified and thus backend-dependent.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: no normalization.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: normalize by ``1/n``.

        where ``n = prod(s)`` is the logical FFT size.

        Default: ``'backward'``.
    rename_dims: Sequence[str] | None
        if provided, the transformed axes are renamed to these dimension names in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axes (dimensions) specified by ``axes``. The returned array has the same data type as ``x`` and the same shape as ``x``, except for the axes specified by ``axes`` which have size ``s[i]``.
    """
    if isinstance(axes, Sequence):
        axes = [x._dims.index(dim) if isinstance(dim, Dim) else dim for dim in axes]
    dims = list(x._dims)
    if rename_dims is not None:
        for axis, rename_dim in zip(axes, rename_dims):
            dims[axis] = rename_dim
    return array(backend.fft.fftn(x.data, s=s, axes=axes, norm=norm), dims)


def ifftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int | str] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dims: Sequence[str] | None = None,
) -> array:
    """
    Computes the n-dimensional inverse discrete Fourier transform.

    .. note::
       Applying the n-dimensional discrete Fourier transform to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``fftn(ifftn(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (sizes, axes, and normalization mode).

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    s: Sequence[int] | None
        number of elements over which to compute the transform along the axes (dimensions) specified by ``axes``. Let ``i`` be the index of the ``n``-th axis specified by ``axes`` (i.e., ``i = axes[n]``) and ``M[i]`` be the size of the input array along axis ``i``. When ``s`` is ``None``, ``s`` is set equal to a sequence of integers such that ``s[i]`` equals ``M[i]`` for all ``i``.

        -   If ``s[i]`` is greater than ``M[i]``, axis ``i`` is zero-padded to size ``s[i]``.
        -   If ``s[i]`` is less than ``M[i]``, axis ``i`` is trimmed to size ``s[i]``.
        -   If ``s[i]`` equals ``M[i]`` or ``-1``, all elements along axis ``i`` are used when computing the transform.

        If ``s`` is not ``None``, ``axes`` must not be ``None``. Default: ``None``.
    axes: Sequence[int | str] | None
        axes (dimensions) over which to compute the transform. Integers refer to positional axes (negative counts from the end), strings refer to named dimensions.

        If ``s`` is provided, the corresponding ``axes`` to be transformed must also be provided. If ``axes`` is ``None``, the transform is computed over all axes. Default: ``None``.

        If ``axes`` contains two or more entries which resolve to the same axis (i.e., resolved axes are not unique), the behavior is unspecified and thus backend-dependent.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: normalize by ``1/n``.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: no normalization.

        where ``n = prod(s)`` is the logical FFT size.

        Default: ``'backward'``.
    rename_dims: Sequence[str] | None
        if provided, the transformed axes are renamed to these dimension names in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axes (dimensions) specified by ``axes``. The returned array has the same data type as ``x`` and the same shape as ``x``, except for the axes specified by ``axes`` which have size ``s[i]``.
    """
    if isinstance(axes, Sequence):
        axes = [x._dims.index(dim) if isinstance(dim, Dim) else dim for dim in axes]
    dims = list(x._dims)
    if rename_dims is not None:
        for axis, rename_dim in zip(axes, rename_dims):
            dims[axis] = rename_dim
    return array(backend.fft.ifftn(x.data, s=s, axes=axes, norm=norm), dims)


def rfft(
    x: array,
    /,
    *,
    n: int | None = None,
    axis: int | str = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dim: str | None = None,
) -> array:
    """
    Computes the one-dimensional discrete Fourier transform for real-valued input.

    .. note::
       Applying the one-dimensional inverse discrete Fourier transform for real-valued input to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``irfft(rfft(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (axis and normalization mode) and consistent values for the number of elements over which to compute the transforms.

    Parameters
    ----------
    x: array
        input array. Has a real-valued floating-point data type.
    n: int | None
        number of elements over which to compute the transform along the axis (dimension) specified by ``axis``. Let ``M`` be the size of the input array along the axis specified by ``axis``. When ``n`` is ``None``, ``n`` is set equal to ``M``.

        -   If ``n`` is greater than ``M``, the axis specified by ``axis`` is zero-padded to size ``n``.
        -   If ``n`` is less than ``M``, the axis specified by ``axis`` is trimmed to size ``n``.
        -   If ``n`` equals ``M``, all elements along the axis specified by ``axis`` are used when computing the transform.

        Default: ``None``.
    axis: int | str
        axis (dimension) of the input array over which to compute the transform. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension. Default: ``-1``.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: no normalization.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: normalize by ``1/n``.

        Default: ``'backward'``.
    rename_dim: str | None
        if provided, the transformed axis is renamed to this dimension name in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axis (dimension) specified by ``axis``. The returned array has a complex floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``float64``, then the returned array has a ``complex128`` data type). The returned array has the same shape as ``x``, except for the axis specified by ``axis`` which has size ``n//2 + 1``.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if rename_dim is not None:
        dims[axis] = rename_dim
    return array(backend.fft.rfft(x.data, n=n, axis=axis, norm=norm), dims)


def irfft(
    x: array,
    /,
    *,
    n: int | None = None,
    axis: int | str = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dim: str | None = None,
) -> array:
    """
    Computes the one-dimensional inverse of ``rfft`` for complex-valued input.

    .. note::
       Applying ``rfft`` to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``rfft(irfft(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (axis and normalization mode) and consistent values for the number of elements over which to compute the transforms.

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    n: int | None
        number of elements along the transformed axis (dimension) specified by ``axis`` in the **output array**. Let ``M`` be the size of the input array along the axis specified by ``axis``. When ``n`` is ``None``, ``n`` is set equal to ``2*(M-1)``.

        -   If ``n//2+1`` is greater than ``M``, the axis of the input array specified by ``axis`` is zero-padded to size ``n//2+1``.
        -   If ``n//2+1`` is less than ``M``, the axis of the input array specified by ``axis`` is trimmed to size ``n//2+1``.
        -   If ``n//2+1`` equals ``M``, all elements along the axis of the input array specified by ``axis`` are used when computing the transform.

        Default: ``None``.
    axis: int | str
        axis (dimension) of the input array over which to compute the transform. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension. Default: ``-1``.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: normalize by ``1/n``.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: no normalization.

        Default: ``'backward'``.
    rename_dim: str | None
        if provided, the transformed axis is renamed to this dimension name in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axis (dimension) specified by ``axis``. The returned array has a real-valued floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``complex128``, then the returned array has a ``float64`` data type). The returned array has the same shape as ``x``, except for the axis specified by ``axis`` which has size ``n``.

    Notes
    -----

    -   In order to return an array having an odd number of elements along the transformed axis, an odd integer must be provided for ``n``.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if rename_dim is not None:
        dims[axis] = rename_dim
    return array(backend.fft.irfft(x.data, n=n, axis=axis, norm=norm), dims)


def rfftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int | str] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dims: Sequence[str] | None = None,
) -> array:
    """
    Computes the n-dimensional discrete Fourier transform for real-valued input.

    .. note::
       Applying the n-dimensional inverse discrete Fourier transform for real-valued input to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``irfftn(rfftn(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (axes and normalization mode) and consistent sizes.

    Parameters
    ----------
    x: array
        input array. Has a real-valued floating-point data type.
    s: Sequence[int] | None
        number of elements over which to compute the transform along axes (dimensions) specified by ``axes``. Let ``i`` be the index of the ``n``-th axis specified by ``axes`` (i.e., ``i = axes[n]``) and ``M[i]`` be the size of the input array along axis ``i``. When ``s`` is ``None``, ``s`` is set equal to a sequence of integers such that ``s[i]`` equals ``M[i]`` for all ``i``.

        -   If ``s[i]`` is greater than ``M[i]``, axis ``i`` is zero-padded to size ``s[i]``.
        -   If ``s[i]`` is less than ``M[i]``, axis ``i`` is trimmed to size ``s[i]``.
        -   If ``s[i]`` equals ``M[i]`` or ``-1``, all elements along axis ``i`` are used when computing the transform.

        If ``s`` is not ``None``, ``axes`` must not be ``None``. Default: ``None``.
    axes: Sequence[int | str] | None
        axes (dimensions) over which to compute the transform. Integers refer to positional axes (negative counts from the end), strings refer to named dimensions.

        If ``s`` is provided, the corresponding ``axes`` to be transformed must also be provided. If ``axes`` is ``None``, the transform is computed over all axes. Default: ``None``.

        If ``axes`` contains two or more entries which resolve to the same axis (i.e., resolved axes are not unique), the behavior is unspecified and thus backend-dependent.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: no normalization.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: normalize by ``1/n``.

        where ``n = prod(s)``, the logical FFT size.

        Default: ``'backward'``.
    rename_dims: Sequence[str] | None
        if provided, the transformed axes are renamed to these dimension names in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axes (dimension) specified by ``axes``. The returned array has a complex floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``float64``, then the returned array has a ``complex128`` data type). The returned array has the same shape as ``x``, except for the last transformed axis which has size ``s[-1]//2 + 1`` and the remaining transformed axes which have size ``s[i]``.
    """
    if isinstance(axes, Sequence):
        axes = [x._dims.index(dim) if isinstance(dim, Dim) else dim for dim in axes]
    dims = list(x._dims)
    if rename_dims is not None:
        for axis, rename_dim in zip(axes, rename_dims):
            dims[axis] = rename_dim
    return array(backend.fft.rfftn(x.data, s=s, axes=axes, norm=norm), dims)


def irfftn(
    x: array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int | str] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dims: Sequence[str] | None = None,
) -> array:
    """
    Computes the n-dimensional inverse of ``rfftn`` for complex-valued input.

    .. note::
       Applying ``rfftn`` to the output of this function returns the original (i.e., non-transformed) input array within numerical accuracy (i.e., ``rfftn(irfftn(x)) == x``), provided that the transform and inverse transform are performed with the same arguments (axes and normalization mode) and consistent sizes.

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    s: Sequence[int] | None
        number of elements along the transformed axes (dimensions) specified by ``axes`` in the **output array**. Let ``i`` be the index of the ``n``-th axis specified by ``axes`` (i.e., ``i = axes[n]``) and ``M[i]`` be the size of the input array along axis ``i``. When ``s`` is ``None``, ``s`` is set equal to a sequence of integers such that ``s[i]`` equals ``M[i]`` for all ``i``, except for the last transformed axis in which ``s[i]`` equals ``2*(M[i]-1)``. For each ``i``, let ``n`` equal ``s[i]``, except for the last transformed axis in which ``n`` equals ``s[i]//2+1``.

        -   If ``n`` is greater than ``M[i]``, axis ``i`` of the input array is zero-padded to size ``n``.
        -   If ``n`` is less than ``M[i]``, axis ``i`` of the input array is trimmed to size ``n``.
        -   If ``n`` equals ``M[i]`` or ``-1``, all elements along axis ``i`` of the input array are used when computing the transform.

        If ``s`` is not ``None``, ``axes`` must not be ``None``. Default: ``None``.
    axes: Sequence[int | str] | None
        axes (dimensions) over which to compute the transform. Integers refer to positional axes (negative counts from the end), strings refer to named dimensions.

        If ``s`` is provided, the corresponding ``axes`` to be transformed must also be provided. If ``axes`` is ``None``, the transform is computed over all axes. Default: ``None``.

        If ``axes`` contains two or more entries which resolve to the same axis (i.e., resolved axes are not unique), the behavior is unspecified and thus backend-dependent.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: normalize by ``1/n``.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: no normalization.

        where ``n = prod(s)`` is the logical FFT size.

        Default: ``'backward'``.
    rename_dims: Sequence[str] | None
        if provided, the transformed axes are renamed to these dimension names in the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axes (dimension) specified by ``axes``. The returned array has a real-valued floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``complex128``, then the returned array has a ``float64`` data type). The returned array has the same shape as ``x``, except for the transformed axes which have size ``s[i]``.

    Notes
    -----

    -   In order to return an array having an odd number of elements along the last transformed axis, an odd integer must be provided for ``s[-1]``.
    """
    if isinstance(axes, Sequence):
        axes = [x._dims.index(dim) if isinstance(dim, Dim) else dim for dim in axes]
    dims = list(x._dims)
    if rename_dims is not None:
        for axis, rename_dim in zip(axes, rename_dims):
            dims[axis] = rename_dim
    return array(backend.fft.irfftn(x.data, s=s, axes=axes, norm=norm), dims)


def hfft(
    x: array,
    /,
    *,
    n: int | None = None,
    axis: int | str = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dim: str | None = None,
) -> array:
    """
    Computes the one-dimensional discrete Fourier transform of a signal with Hermitian symmetry.

    Parameters
    ----------
    x: array
        input array. Has a complex floating-point data type.
    n: int | None
        number of elements along the transformed axis (dimension) specified by ``axis`` in the **output array**. Let ``M`` be the size of the input array along the axis specified by ``axis``. When ``n`` is ``None``, ``n`` is set to ``2*(M-1)``.

        -   If ``n//2+1`` is greater than ``M``, the axis of the input array specified by ``axis`` is zero-padded to length ``n//2+1``.
        -   If ``n//2+1`` is less than ``M``, the axis of the input array specified by ``axis`` is trimmed to size ``n//2+1``.
        -   If ``n//2+1`` equals ``M``, all elements along the axis of the input array specified by ``axis`` are used when computing the transform.

        Default: ``None``.
    axis: int | str
        axis (dimension) of the input array over which to compute the transform. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension. Default: ``-1``.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: no normalization.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: normalize by ``1/n``.

        Default: ``'backward'``.
    rename_dim: str | None
        new name for the transformed dimension. If ``None``, the dimension name is unchanged. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axis (dimension) specified by ``axis``. The returned array has a real-valued floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``complex128``, then the returned array has a ``float64`` data type). The returned array has the same shape as ``x``, except for the axis specified by ``axis`` which has size ``n``.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if rename_dim is not None:
        dims[axis] = rename_dim
    return array(backend.fft.hfft(x.data, n=n, axis=axis, norm=norm), dims)


def ihfft(
    x: array,
    /,
    *,
    n: int | None = None,
    axis: int | str = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
    rename_dim: str | None = None,
) -> array:
    """
    Computes the one-dimensional inverse discrete Fourier transform of a signal with Hermitian symmetry.

    Parameters
    ----------
    x: array
        input array. Has a real-valued floating-point data type.
    n: int | None
        number of elements over which to compute the transform along the axis (dimension) specified by ``axis``. Let ``M`` be the size of the input array along the axis specified by ``axis``. When ``n`` is ``None``, ``n`` is set to ``M``.

        -   If ``n`` is greater than ``M``, the axis specified by ``axis`` is zero-padded to size ``n``.
        -   If ``n`` is less than ``M``, the axis specified by ``axis`` is trimmed to size ``n``.
        -   If ``n`` equals ``M``, all elements along the axis specified by ``axis`` are used when computing the transform.

        Default: ``None``.
    axis: int | str
        axis (dimension) of the input array over which to compute the transform. An integer refers to a positional axis (negative counts from the end), a string refers to a named dimension. Default: ``-1``.
    norm: Literal['backward', 'ortho', 'forward']
        normalization mode:

        - ``'backward'``: normalize by ``1/n``.
        - ``'ortho'``: normalize by ``1/sqrt(n)`` (i.e., make the FFT orthonormal).
        - ``'forward'``: no normalization.

        Default: ``'backward'``.
    rename_dim: str | None
        new name for the transformed dimension. If ``None``, the dimension name is unchanged. Default: ``None``.

    Returns
    -------
    out: array
        an array transformed along the axis (dimension) specified by ``axis``. The returned array has a complex floating-point data type whose precision matches the precision of ``x`` (e.g., if ``x`` is ``float64``, then the returned array has a ``complex128`` data type). The returned array has the same shape as ``x``, except for the axis specified by ``axis`` which has size ``n//2 + 1``.
    """
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    dims = list(x._dims)
    if rename_dim is not None:
        dims[axis] = rename_dim
    return array(backend.fft.ihfft(x.data, n=n, axis=axis, norm=norm), dims)


def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    device: device | None = None,
    dim: str | None = None,
) -> array:
    """
    Computes the discrete Fourier transform sample frequencies.

    For a Fourier transform of length ``n`` and length unit of ``d``, the frequencies are described as:

    .. code-block::

      f = [0, 1, ..., n/2-1, -n/2, ..., -1] / (d*n)        # if n is even
      f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (d*n)  # if n is odd

    Parameters
    ----------
    n: int
        window length.
    d: float
        sample spacing between individual samples of the Fourier transform input. Default: ``1.0``.
    device: device | None
        device on which to place the created array. Default: ``None``.
    dim: str | None
        name for the dimension of the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array of shape ``(n,)`` containing the sample frequencies. The returned array has the default real-valued floating-point data type.
    """
    if dim is None:
        dim = undefined_dim
    if device is None:
        device = ops.backend.device           
    return array(backend.fft.fftfreq(n, d=d, device=device), [dim], device=device)


def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    device: device | None = None,
    dim: str | None = None,
) -> array:
    """
    Computes the discrete Fourier transform sample frequencies (for ``rfft`` and ``irfft``).

    For a Fourier transform of length ``n`` and length unit of ``d``, the frequencies are described as:

    .. code-block::

      f = [0, 1, ...,     n/2-1,     n/2] / (d*n)  # if n is even
      f = [0, 1, ..., (n-1)/2-1, (n-1)/2] / (d*n)  # if n is odd

    The Nyquist frequency component is considered to be positive.

    Parameters
    ----------
    n: int
        window length.
    d: float
        sample spacing between individual samples of the Fourier transform input. Default: ``1.0``.
    device: device | None
        device on which to place the created array. Default: ``None``.
    dim: str | None
        name for the dimension of the output array. Default: ``None``.

    Returns
    -------
    out: array
        an array of shape ``(n//2+1,)`` containing the sample frequencies. The returned array has the default real-valued floating-point data type.
    """
    if dim is None:
        dim = undefined_dim
    if device is None:
        device = ops.backend.device         
    return array(backend.fft.rfftfreq(n, d=d, device=device), [dim], device=device)


def fftshift(
    x: array,
    /,
    *,
    axes: int | str | Sequence[int | str] | None = None,
    rename_dims: Sequence[str] | None = None,
) -> array:
    """
    Shifts the zero-frequency component to the center of the spectrum.

    This function swaps half-spaces for all axes (dimensions) specified by ``axes``.

    .. note::
       ``out[0]`` is the Nyquist component only if the length of the input is even.

    Parameters
    ----------
    x: array
        input array. Has a floating-point data type.
    axes: int | str | Sequence[int | str] | None
        axes over which to shift. If ``None``, all axes are shifted. Default: ``None``.

        If ``axes`` contains two or more entries which resolve to the same axis (i.e., resolved axes are not unique), the behavior is backend-dependent.
    rename_dims: Sequence[str] | None
        new names for the shifted dimensions. Default: ``None``.

    Returns
    -------
    out: array
        the shifted array. The returned array has the same data type and shape as ``x``.
    """
    if isinstance(axes, Dim):
        axes = x._dims.index(axes)
    elif isinstance(axes, Sequence):
        axes = [x._dims.index(dim) if isinstance(dim, Dim) else dim for dim in axes]
    dims = list(x._dims)
    if rename_dims is not None:
        for axis, rename_dim in zip(axes, rename_dims):
            dims[axis] = rename_dim
    return array(backend.fft.fftshift(x.data, axes=axes), dims)


def ifftshift(
    x: array,
    /,
    *,
    axes: int | str | Sequence[int | str] | None = None,
    rename_dims: Sequence[str] | None = None,
) -> array:
    """
    Inverse of ``fftshift``.

    .. note::
       Although identical for even-length ``x``, ``fftshift`` and ``ifftshift`` differ by one sample for odd-length ``x``.

    Parameters
    ----------
    x: array
        input array. Has a floating-point data type.
    axes: int | str | Sequence[int | str] | None
        axes over which to perform the inverse shift. If ``None``, all axes are shifted. Default: ``None``.

        If ``axes`` contains two or more entries which resolve to the same axis (i.e., resolved axes are not unique), the behavior is backend-dependent.
    rename_dims: Sequence[str] | None
        new names for the shifted dimensions. Default: ``None``.

    Returns
    -------
    out: array
        the shifted array. The returned array has the same data type and shape as ``x``.
    """
    if isinstance(axes, Dim):
        axes = x._dims.index(axes)
    elif isinstance(axes, Sequence):
        axes = [x._dims.index(dim) if isinstance(dim, Dim) else dim for dim in axes]
    dims = list(x._dims)
    if rename_dims is not None:
        for axis, rename_dim in zip(axes, rename_dims):
            dims[axis] = rename_dim
    return array(backend.fft.ifftshift(x.data, axes=axes), dims)
