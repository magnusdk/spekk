from typing import (
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from spekk import ops, tree
from spekk.ops._backend import backend
from spekk.ops._types import Dim, Dims, undefined_dim
from spekk.ops._util import get_reduction_axes_and_resulting_dims
from spekk.ops.array_object import array
from spekk.ops.extensions.function_transformations import (
    argmax_over_dims,
    argmin_over_dims,
    checkpoint,
    grad,
    jit,
    map_over_dim,
    map_reduce_over_dim,
    reduce_over_dim,
    scan,
    jit_static_argnames,
    value_and_grad,
    vmap,
)


def lerp(a: array, b: array, p: array) -> array:
    "Linearly interpolate between a and b, using p which is a number between 0 and 1."
    return a + (b - a) * p


def deg2rad(x: array) -> array:
    return x / 180 * backend.pi


def rad2deg(x: array) -> array:
    return x * 180 / backend.pi


def angle(x: array) -> array:
    data = backend.angle(x.data)
    return array(data, dims=x.dims)

def sinc(x: ops.array) -> ops.array:
    """Compute normalized sinc function: sin(pi*x) / (pi*x). """
    # sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1
    pi_x = ops.pi * x
    return ops.where(x == 0, ops.ones_like(x), ops.sin(pi_x) / pi_x)

def flatten(x: array, dim: Optional[Dim] = None) -> array:
    data = backend.flatten(x.data)
    if dim is None:
        dim = undefined_dim
    return array(data, dims=[dim])

def allclose(a, b, atol=None, rtol=None):
    """Check if two arrays are element-wise equal within a tolerance.
    
    Default tolerances are dtype-dependent:
        float32: atol=1e-5, rtol=1e-4
        float64: atol=1e-8, rtol=1e-5
    """
    dtype = getattr(a, 'dtype', getattr(b, 'dtype', None))
    is_float32 = dtype is not None and 'float32' in str(dtype)
    if atol is None:
        atol = 1e-5 if is_float32 else 1e-8
    if rtol is None:
        rtol = 1e-4 if is_float32 else 1e-5
    return bool(ops.all(ops.abs(a - b) <= atol + rtol * ops.abs(b)))

def diag(v, k=0, dims: Dims = None):
    if v.ndim == 1 or v.ndim == 2:
        data = backend.diag(v.data, k=k)
        return ops.array(data, dims=dims)
    else:
        raise ValueError(f"diag only supported ndim 1 or 2, diag={v.ndim}")


def to_numpy(x: array):
    return backend.to_numpy(x.data)


def to_device(data, device):
    def _move(x):
        if not isinstance(x, ops.array):
            return x
        # Work around broken device-to-device transfers in JAX ≥0.9 by
        # routing through host memory (numpy) when the array is not already
        # on the target device.
        try:
            current_devices = x.data.devices()
            if device not in current_devices:
                import jax
                np_data = backend.to_numpy(x.data)
                return ops.array(jax.device_put(np_data, device), x._dims)
        except (AttributeError, TypeError):
            pass
        return x.to_device(device)
    return tree.map(_move, data)

    
def median(a, axis: Optional[Dim], out=None, keepdims: bool = False) -> array:
    if axis is None:
        arr = backend.median(a, axis, out=out, keepdims=keepdims)
        dims = None
    else:
        if isinstance(axis, int):
            arr = backend.median(a.data, axis, out=out, keepdims=keepdims)
            dims = a.dims.pop(axis)
        elif isinstance(axis[0], int):
            arr = backend.median(a.data, axis, out=out, keepdims=keepdims)
            dims = a.dims
            for ax in axis:
                dims = dims.pop(axis)
        else:
            axis_indices = []
            dims = []
            for idx, dim in enumerate(a.dims):
                if dim in axis:
                    axis_indices.append(idx)
                    if keepdims:
                        dims.append(dim)
                else:
                    dims.append(dim)
            arr = backend.median(
                a.data, tuple(axis_indices), out=out, keepdims=keepdims
            )
    return ops.array(arr, dims=dims)

def quantile(a, q, axis=None, keepdims=False):
    axis, dims = get_reduction_axes_and_resulting_dims(axis, a.dims, keepdims)
    data = backend.quantile(a.data, q, axis=axis, keepdims=keepdims)
    return array(data, dims=dims)

def nanmean(
    x: array,
    /,
    *,
    axis: Optional[Union[Dim, Tuple[Dim, ...], int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x.dims, keepdims)
    data = backend.nanmean(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)


def nansum(
    x: array,
    /,
    *,
    axis: Optional[Union[Dim, Tuple[Dim, ...], int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x.dims, keepdims)
    data = backend.nansum(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)

def nanmax(
    x: array,
    /,
    *,
    axis: Optional[Union[Dim, Tuple[Dim, ...], int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> array:
    axis, dims = get_reduction_axes_and_resulting_dims(axis, x.dims, keepdims)
    data = backend.nanmax(x._data, axis=axis, keepdims=keepdims)
    return array(data, dims)

def pad(
    x: array,
    pad_width: tuple,
    mode: str = "constant",
    reflect_type: Union[str, None] = None,
    dims: Dims = None,
) -> array:
    if dims is None:
        if reflect_type is None:
            x_pad = backend.pad(x.data, pad_width, mode)
        else:
            x_pad = backend.pad(x.data, pad_width, mode, reflect_type=reflect_type)
    else:
        # check that all dims are in x.dims
        if sum([dim in x.dims for dim in dims]) != len(dims):
            raise ValueError(
                f"not all pad dims {dims} are present in arrays dims {x.dims}"
            )

        # Standardise format to: ((before_1, after_1), (before_2, after_2), ... (before_N, after_N))
        if isinstance(pad_width, int):
            pad_width = [(pad_width, pad_width) for d in dims]
        elif isinstance(pad_width[0], int):
            if len(pad_width) == 1:
                pad_width = [(pad_width[0], pad_width[0]) for d in dims]
            elif len(pad_width) == 2:
                pad_width = [(pad_width[0], pad_width[1]) for d in dims]

        # pad_width is individual for each axis
        pad_width_out = [(0, 0) for i in range(len(x.dims))]
        sorted_indices = [x.dims.index(dim) for dim in dims]
        for ii, pad in enumerate(pad_width):
            idx = sorted_indices[ii]
            pad_width_out[idx] = pad

        if reflect_type is None:
            x_pad = backend.pad(x.data, pad_width_out, mode)
        else:
            x_pad = backend.pad(x.data, pad_width_out, mode, reflect_type=reflect_type)

    return array(x_pad, dims=x.dims)


def nan_to_num(
    x: array,
    nan: Optional[float] = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> array:
    from spekk import ops

    if nan is not None:
        x = ops.where(ops.isnan(x), nan, x)
    if posinf is not None:
        x = ops.where(x == ops.inf, posinf, x)
    if neginf is not None:
        x = ops.where(x == -ops.inf, neginf, x)
    return x


def _get_conv_mode_slice(
    x_size: Tuple[int],
    filter_size: Tuple[int],
    mode: Literal["full", "same", "valid"],
) -> slice:
    full_size = x_size + filter_size - 1

    # Calculate what samples to include in the convolved result based on the mode.
    if mode == "full":
        start = 0
        stop = full_size
    elif mode == "same":
        if filter_size > x_size:
            raise NotImplementedError(
                "mode='same' is not implemented when the a is larger than v"
            )
        start = (filter_size - 1) // 2
        stop = start + x_size
    elif mode == "valid":
        raise NotImplementedError("mode='valid' is not implemented.")
    else:
        raise ValueError(f"Invalid mode: '{mode}'")
    return slice(start, stop)


def convolve1d(
    x: array,
    filter: array,  # 1D array
    *,
    mode: Literal["full", "same", "valid"],
    axis: Dim,
):
    dims = x.dims
    filter = filter.rename_dim(filter.dims[0], axis)
    axis_idx = x.dims.index(axis)

    data_filtered = backend.convolve1d(x.data, filter.data, mode=mode, axis=axis_idx)

    return array(data_filtered, dims=dims)


def correlate2d(x1: array, x2: array):
    if x1.ndim != 2 or x2.ndim != 2:
        raise ValueError("Input arrays must be 2D arrays")
    if x1.dims != x2.dims:
        raise ValueError("Input arrays must have equal dims")
    return ops.array(backend.correlate2d(x1.data, x2.data), dims=x1.dims)


def dilation1d(
    x: array,
    dilation_factor: int,
    value: float,
    *,
    axis: Dim,
):
    # Number of zeros to interleave
    axis_idx = x.dims.index(axis)

    # Calculate the new shape after interleaving zeros
    new_shape = list(x.shape)
    new_shape[axis_idx] = x.shape[axis_idx] + (x.shape[axis_idx] - 1) * dilation_factor

    # Create an array of zeros with the new shape
    result = ops.ones(new_shape, dtype=x.dtype, dims=x.dims) * value

    # Create an index array to place the original values
    indices = [slice(None)] * x.ndim
    indices[axis_idx] = slice(0, new_shape[axis_idx], dilation_factor + 1)

    # Place the original values into the zeros array
    result[
        axis,
        ops.array(list(range(0, new_shape[axis_idx], dilation_factor + 1)), [axis]),
    ] = x

    return result


def fftconvolve(
    x: array,
    filter: array,
    *,
    mode: Literal["full", "same", "valid"],
    axis: Dim,
    filter_axis: Dim,
):
    from spekk import ops

    # Get data sizes along axis
    x_size = x.dim_sizes[axis]
    filter_size = filter.dim_sizes[filter_axis]
    full_size = x_size + filter_size - 1

    # Perform convolution
    x = ops.fft.fft(x, n=full_size, axis=axis)
    filter = ops.fft.fft(filter, n=full_size, axis=filter_axis, rename_dim=axis)
    convolved = x * filter
    convolved = ops.fft.ifft(convolved, axis=axis)

    # "Trim" the result according to mode.
    mode_slice = _get_conv_mode_slice(x_size, filter_size, mode)
    convolved = convolved[axis, mode_slice]

    return convolved


def reshape_at_dim(
    x: array,
    dim: Dim,
    new_dim_shape: Sequence[int],
    new_dim_names: Sequence[Dim],
):
    """
    >>> x = ops.ones((3,4,5), dims=["a", "b", "c"])
    >>> y = reshape_at_dim(x, "b", (2, 2), ("b1", "b2"))
    >>> y.dims
    ['a', 'b1', 'b2', 'c']
    >>> y.shape
    (3, 2, 2, 5)
    """
    from spekk import ops

    axis = x.dims.index(dim)
    new_shape = (*x.shape[:axis], *new_dim_shape, *x.shape[axis + 1 :])
    new_dims = [*x.dims[:axis], *new_dim_names, *x.dims[axis + 1 :]]
    return ops.reshape(x, new_shape, new_dims)


def merge_dims(
    x: array,
    merged_dims: Sequence[Dim],
    new_dim_name: Dim,
) -> array:
    from spekk import ops

    merged_set = set(merged_dims)

    # Find the position of the first dim to be merged.
    first_merged_idx = min(x.dims.index(d) for d in merged_dims)

    # Split dims into: before the merge point, the merged dims (in original
    # order), and after.
    before = list(x.dims[:first_merged_idx])
    merged_in_order = [d for d in x.dims if d in merged_set]
    after = [d for d in x.dims[first_merged_idx:] if d not in merged_set]

    # Permute to make merged dims contiguous at the merge point.
    # If the merged dims are already contiguous neighbors, this is a no-op.
    perm = [*before, *merged_in_order, *after]
    x = ops.permute_dims(x, perm)

    # Reshape to merge the contiguous dims into one.
    before_sizes = [x.dim_sizes[d] for d in before]
    after_sizes = [x.dim_sizes[d] for d in after]
    new_dims = [*before, new_dim_name, *after]
    new_shape = [*before_sizes, -1, *after_sizes]
    return ops.reshape(x, new_shape, new_dims)


def expand_slice_to_axis(s: Union[slice, int, array], axis: int):
    """Make a given slice s work along a given axis.

    Example:
    >>> from spekk import ops
    >>> arr = ops.reshape(ops.arange(8), (2, 2, 2))
    >>> s = expand_slice_to_axis(slice(1, 2), 1)
    >>> arr[s]
    array(shape=(2, 1, 2), dims=[?, ?, ?], dtype=int64, data=[[[2 3]]
    <BLANKLINE>
     [[6 7]]])
    >>> arr[s] = -1
    >>> arr
    array(shape=(2, 2, 2), dims=[?, ?, ?], dtype=int64, data=[[[ 0  1]
      [-1 -1]]
    <BLANKLINE>
     [[ 4  5]
      [-1 -1]]])
    """
    return (slice(None),) * axis + (s, ...)


if __name__ == "__main__":
    import doctest

    from spekk import ops

    ops.backend.set_backend("numpy")
    doctest.testmod()
