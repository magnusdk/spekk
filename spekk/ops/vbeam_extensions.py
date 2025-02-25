import functools
from typing import Callable, Literal, Optional, Sequence, Tuple, TypeVar, Union

from spekk import module, ops
from spekk.module.base import Module
from spekk.ops._backend import backend
from spekk.ops._types import Dim, undefined_dim
from spekk.ops._util import prepare_slicing_along_dim
from spekk.ops.array_object import array

TFunc = TypeVar("TFunc", bound=Callable)
TCarry = TypeVar("TCarry")
TInputData = TypeVar("TInputData", bound=Module)
TMappedInputData = TypeVar("TMappedInputData")
TOutputData = TypeVar("TOutputData")
TReducedOutputData = TypeVar("TReducedOutputData")


def deg2rad(x: array) -> array:
    return x / 180 * backend.pi


def rad2deg(x: array) -> array:
    return x * 180 / backend.pi


def angle(x: array) -> array:
    data = backend.angle(x.data)
    return array(data, dims=x.dims)


def flatten(x: array, dim: Optional[Dim] = None) -> array:
    data = backend.flatten(x.data)
    if dim is None:
        dim = undefined_dim
    return array(data, dims=[dim])

def to_numpy(x: array):
    return backend.to_numpy(x.data)

def nan_to_num(
    x: array,
    nan: Optional[float] = 0.0,
    posinf: Optional[float] = None,
    neginf: Optional[float] = None,
) -> array:
    from spekk import ops

    if nan is not None:
        x = ops.where(x == ops.nan, nan, x)
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


def dilation1d(
    x: array,
    dilation_factor: int,
    value: float,
    *,
    axis: Dim,
):
    # Number of zeros to interleave
    axis_idx = x.dim_index(axis)

    # Calculate the new shape after interleaving zeros
    new_shape = list(x.shape)
    new_shape[axis_idx] = x.shape[axis_idx] + (x.shape[axis_idx] - 1) * dilation_factor

    # Create an array of zeros with the new shape
    result = ops.ones(new_shape, dtype=x.dtype, dims=x.dims) * value

    # Create an index array to place the original values
    indices = [slice(None)] * x.ndim
    indices[axis_idx] = slice(0, new_shape[axis_idx], dilation_factor + 1)

    # Place the original values into the zeros array
    result[tuple(indices)] = x

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

    # Get the dim names and sizes of the dimensions that are not being merged.
    other_dim_names = [d for d in x.dims if d not in merged_dims]
    other_dim_sizes = [x.dim_sizes[d] for d in x.dims if d not in merged_dims]

    # Move the dimensions to the start and in the right order.
    x = ops.permute_dims(x, [*merged_dims, *other_dim_names])

    # Perform the actual reshape operation and return.
    new_dims = [new_dim_name, *other_dim_names]
    new_shape = [-1, *other_dim_sizes]
    return ops.reshape(x, new_shape, new_dims)


def take_along_dim(x: array, i: array, dim: Dim) -> array:
    x, slices, dims = prepare_slicing_along_dim(x, i, dim)
    data = x.data[slices]
    return array(data, dims)


def update_indices_along_dim(
    f: Callable[[array], array], x: array, i: array, x_dim: Dim, i_dim: Dim
) -> array:
    from spekk import ops

    x, slices, dims = prepare_slicing_along_dim(x, i, x_dim)
    updated_x_slices = f(array(x.data[slices], dims))
    updated_x_slices._dims = [x_dim if d == i_dim else d for d in updated_x_slices.dims]
    x, updated_x_slices = ops.broadcast_arrays(x, updated_x_slices)
    updated_data = backend._setitem_impl(x.data, slices, updated_x_slices.data)
    return array(updated_data, x.dims)


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


def jit(f: TFunc, *, cache_module_methods: bool = False) -> TFunc:
    if cache_module_methods:
        original_f = f

        @functools.wraps(original_f)
        def f(*args, **kwargs):
            with module.cache_module_methods():
                return original_f(*args, **kwargs)

    return backend.jit(f)


def scan_over_dim(
    f: Callable[[TCarry, TInputData], Tuple[TCarry, TOutputData]],
    data: TInputData,
    dim: Dim,
    *,
    init: TCarry,
    include_index: bool = False,
) -> Tuple[TCarry, TOutputData]:
    from spekk import ops

    def scan_fn(carry, i):
        args = [carry, data.slice_dim(dim)[i]]
        if include_index:
            args.append(i)
        return f(*args)

    init, y0 = scan_fn(init, 0)
    n = data.dim_sizes[dim]
    result, ys = backend.scan(scan_fn, init, backend.arange(1, n))
    ys = ops.concat([ops.array([y0]), ys])
    return result, ys


def reduce_over_dim(
    reduce_f: Callable[[TCarry, TInputData], TCarry],
    data: TInputData,
    *,
    init: TCarry,
    dim: Dim,
    include_index: bool = False,
) -> TCarry:
    from spekk.module import flatten as flatten_tree

    dim_sizes = data.dim_sizes
    if dim not in dim_sizes:
        raise ValueError(f"Dimension {dim} not found in the data.")

    # Optionally include an index array to the data.
    if include_index:
        data = (ops.arange(dim_sizes[dim], dim=dim), data)

    # Flatten the initial carry state.
    flat_carry = flatten_tree(init, flatten_spekk_arrays=True)

    # Flatten the object that we want to reduce over into a sequence of arrays that has
    # dim in its dimensions.
    flat_outer = flatten_tree(data).filter_dynamic(
        lambda x: isinstance(x, ops.array) and (dim in x.dims)
    )
    # Ensure that the dimension being reduced over is at the first axis.
    dynamic = [ops.moveaxis(x, dim, 0) for x in flat_outer.dynamic]

    # Extract underlying data and dims
    dynamic_data = tuple(x.data for x in dynamic)
    dynamic_dims = tuple(x.dims for x in dynamic)

    def _scan_f(carry, dyn_data):
        nonlocal flat_carry
        # Reconstruct the carry from its flattened version.
        carry = flat_carry.unflatten(carry)

        # Rebuild the current dynamic arrays for this time step without the leading
        # dimension. It does not have the leading dimension because that is what we are
        # "iterating" over.
        current_dynamic = [
            ops.array(data, dims[1:]) for data, dims in zip(dyn_data, dynamic_dims)
        ]
        # Unflatten the data, call reduce_f, and flatten the result
        flat_carry = flatten_tree(
            reduce_f(carry, flat_outer.unflatten(current_dynamic)),
            flatten_spekk_arrays=True,
        )
        return flat_carry.dynamic, None

    # Run the backend scan implementation on the dynamic backend data.
    new_dynamic_0, _ = _scan_f(flat_carry.dynamic, [x[0] for x in dynamic_data])
    new_dynamic, _ = ops.backend.scan(
        _scan_f, new_dynamic_0, [x[1:] for x in dynamic_data]
    )
    return flat_carry.unflatten(new_dynamic)


def map_reduce_over_dim(
    map_f: Callable[[TInputData], TMappedInputData],
    reduce_f: Callable[[TCarry, TMappedInputData], TCarry],
    data: TInputData,
    *,
    init: TCarry,
    dim: Dim,
    include_index_in_reduce: bool = False,
) -> TCarry:
    if include_index_in_reduce:
        f = lambda carry, x: reduce_f(carry, (x[0], map_f(x[1])))
    else:
        f = lambda carry, x: reduce_f(carry, map_f(x))
    return reduce_over_dim(
        f, data, init=init, dim=dim, include_index=include_index_in_reduce
    )


def vmap(f, in_axes):
    return backend.vmap(f, in_axes=in_axes)


if __name__ == "__main__":
    import doctest

    from spekk import ops

    ops.backend.set_backend("numpy")
    doctest.testmod()
