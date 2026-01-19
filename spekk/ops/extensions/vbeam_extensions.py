import functools
from typing import (
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from spekk import ops
from spekk.module.base import Module
from spekk.ops._backend import backend
from spekk.ops._types import Dim, Dims, undefined_dim
from spekk.ops._util import get_reduction_axes_and_resulting_dims
from spekk.ops.array_object import array

TFunc = TypeVar("TFunc", bound=Callable)
TCarry = TypeVar("TCarry")
TInputData = TypeVar("TInputData", bound=Module)
TMappedInputData = TypeVar("TMappedInputData")
TOutputData = TypeVar("TOutputData")
TReducedOutputData = TypeVar("TReducedOutputData")


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


def flatten(x: array, dim: Optional[Dim] = None) -> array:
    data = backend.flatten(x.data)
    if dim is None:
        dim = undefined_dim
    return array(data, dims=[dim])


def diag(v, k=0, dims: Dims = None):
    if v.ndim == 1 or v.ndim == 2:
        data = backend.diag(v.data, k=k)
        return ops.array(data, dims=dims)
    else:
        raise ValueError(f"diag only supported ndim 1 or 2, diag={v.ndim}")


def to_numpy(x: array):
    return backend.to_numpy(x.data)


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

    # Get the dim names and sizes of the dimensions that are not being merged.
    other_dim_names = [d for d in x.dims if d not in merged_dims]
    other_dim_sizes = [x.dim_sizes[d] for d in x.dims if d not in merged_dims]

    # Move the dimensions to the start and in the right order.
    x = ops.permute_dims(x, [*merged_dims, *other_dim_names])

    # Perform the actual reshape operation and return.
    new_dims = [new_dim_name, *other_dim_names]
    new_shape = [-1, *other_dim_sizes]
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


def _argf_over_dims(
    f,
    x: ops.array,
    *,
    axis: tuple[Dim] | list[Dim] | Dim | None = None,
) -> dict[Dim, int]:
    """Can be used like argmin or argmax, but allows applying over multiple dimensions
    at once and integrates better with named dimensions.

    f can be either argmin or argmax (or any other function that similarly returns an
    index).

    Example:
    >>> x = ops.array(np.random.randn(2, 3, 4), ["a", "b", "c"])
    >>> min_i = _argf_over_dims(ops.argmin, x, axis=["a", "b"])
    >>> assert all(x[min_i] == ops.min(x, axis=["a", "b"]))
    """

    if axis is None:
        axis = x.dims
    elif not isinstance(axis, (tuple, list)):
        axis = [axis]
    assert all(isinstance(dim, Dim) for dim in axis)

    dim, *dims = axis
    i = f(x, axis=dim)
    x = x[dim, i]
    result = {dim: i}
    for dim in dims:
        i = f(x, axis=dim)
        x = x[dim, i]
        result = {result_dim: indices[dim, i] for result_dim, indices in result.items()}
        result[dim] = i
    return result


argmin_over_dims = functools.partial(_argf_over_dims, ops.argmin)
argmax_over_dims = functools.partial(_argf_over_dims, ops.argmax)


@overload
def jit(
    f: Optional[TFunc] = None,
    /,
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
) -> TFunc: ...
@overload
def jit(
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
) -> Callable[[TFunc], TFunc]: ...
def jit(
    f: Optional[TFunc] = None,
    /,
    *,
    static_argnums: Sequence[int] = (),
    static_argnames: Sequence[str] = (),
) -> Union[TFunc, Callable[[TFunc], TFunc]]:
    """Just-in-time (JIT) compile `f` if the active backend supports it. Optionally
    mark arguments as static via `static_argnums` and `static_argnames`. Marking
    arguments as static means that the function is recompiled every time they change.
    Static arguments have to be hashable or be a nested list/tuple/dict of hashable
    arguments, or a `spekk.array`.

    For backends that doesn't support JIT (like numpy), this is a no-op."""

    if f is None:

        @functools.wraps(f)
        def wrapper(f: TFunc) -> TFunc:
            return jit(
                f, static_argnums=static_argnums, static_argnames=static_argnames
            )

        return wrapper
    return backend.jit(
        f,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
    )


def scan(fn, init, xs):
    return ops.backend.scan(fn, init, xs.data)


def reduce_over_dim(
    reduce_f: Callable[[TCarry, TInputData], TCarry],
    data: TInputData,
    *,
    init: TCarry,
    dim: Dim,
    include_index: bool = False,
    unroll: int | bool = 1,
) -> TCarry:
    from spekk.module import dim_sizes as get_dim_sizes
    from spekk.module import flatten

    dim_sizes = get_dim_sizes(data)
    if dim not in dim_sizes:
        raise ValueError(f"Dimension {dim} not found in the data.")

    # Flatten the initial carry state.
    flat_carry = flatten(init, flatten_spekk_arrays=True)

    # Flatten the object that we want to reduce over into a sequence of arrays that has
    # dim in its dimensions.
    flat_outer = flatten(data).filter_dynamic(
        lambda x: isinstance(x, ops.array) and (dim in x.dims)
    )
    # Ensure that the dimension being reduced over is at the first axis.
    dynamic = [ops.moveaxis(x, dim, 0) for x in flat_outer.dynamic]

    # Extract underlying data and dims
    dynamic_data = [x.data for x in dynamic]
    dynamic_dims = [x.dims for x in dynamic]

    def _scan_f(carry, dyn_data):
        nonlocal flat_carry
        # Reconstruct the carry from its flattened version.
        carry = flat_carry.treedef.unflatten(carry)

        # Remove index from dynamic data.
        if include_index:
            index = dyn_data[-1]
            dyn_data = dyn_data[:-1]

        # Rebuild the current dynamic arrays for this time step without the leading
        # dimension. It does not have the leading dimension because that is what we are
        # "iterating" over.
        current_dynamic = [
            ops.array(data, dims[1:]) for data, dims in zip(dyn_data, dynamic_dims)
        ]
        reduce_f_args = [carry, flat_outer.treedef.unflatten(current_dynamic)]
        # Add index as last argument to reduce_f.
        if include_index:
            reduce_f_args.append(index)

        # Unflatten the data, call reduce_f, and flatten the result
        flat_carry = flatten(
            reduce_f(*reduce_f_args),
            flatten_spekk_arrays=True,
        )
        return flat_carry.dynamic, None

    # Run the backend scan implementation on the dynamic backend data.
    if include_index:
        # Add indices to the end of dynamic data. NB! We NEED to extract it out again
        # before calling unflatten on the dynamic data.
        indices = backend.arange(dim_sizes[dim])
        dynamic_data.append(indices)

    new_dynamic_0, _ = _scan_f(flat_carry.dynamic, [x[0] for x in dynamic_data])
    new_dynamic, _ = ops.backend.scan(
        _scan_f,
        new_dynamic_0,
        [x[1:] for x in dynamic_data],
        unroll=unroll,
    )
    return flat_carry.treedef.unflatten(new_dynamic)


def map_reduce_over_dim(
    map_f: Callable[[TInputData], TMappedInputData],
    reduce_f: Callable[[TCarry, TMappedInputData], TCarry],
    data: TInputData,
    *,
    init: TCarry,
    dim: Dim,
    include_index_in_reduce: bool = False,
    unroll: int | bool = 1,
) -> TCarry:
    if include_index_in_reduce:
        f = lambda carry, part, index: reduce_f(carry, map_f(part), index)
    else:
        f = lambda carry, part: reduce_f(carry, map_f(part))
    return reduce_over_dim(
        f,
        data,
        init=init,
        dim=dim,
        include_index=include_index_in_reduce,
        unroll=unroll,
    )


def map_over_dim(
    map_f: Callable,
    data,
    *,
    dim: Dim | Sequence[Dim],
    include_index: bool = False,
    unroll: int | bool = 1,
):
    """
    Iterates over each element of `dim` in `data`, applies `map_f` to it, and returns
    a new data object of the results along the same dimension.
    """
    from spekk.module import TreeDef
    from spekk.module import dim_sizes as get_dim_sizes
    from spekk.module import flatten

    # Handle case where multiple dims are given. Then we map over each dimension
    # individually.
    if not isinstance(dim, Dim):
        if not isinstance(dim, Sequence):
            raise ValueError(
                f"dim must be either a Dim or a sequence of Dim. Got '{type(dim)}'."
            )
        dim, *remaining_dims = dim
        if remaining_dims:
            return ops.map_over_dim(
                lambda data_1: map_over_dim(map_f, data_1, dim=remaining_dims),
                data,
                dim=dim,
            )

    # Validate dimension
    # dim_sizes = data.dim_sizes
    dim_sizes = get_dim_sizes(data)
    if dim not in dim_sizes:
        raise ValueError(f"Dimension {dim} not found in the data.")

    # Optionally include index array
    if include_index:
        data = (ops.arange(dim_sizes[dim], dim=dim), data)

    # Flatten input data trees for dynamic arrays along `dim`
    flat_in = flatten(data).filter_dynamic(
        lambda x: isinstance(x, ops.array) and (dim in x.dims)
    )

    # Move `dim` to leading axis and extract raw data and dims
    moved = [ops.moveaxis(arr, dim, 0) for arr in flat_in.dynamic]
    in_data = tuple(arr.data for arr in moved)
    in_dims = tuple(arr.dims for arr in moved)

    # Flatten output tree structure by running map_f on a dummy first to capture shape
    # Prepare scanning
    flat_out_template: TreeDef = None
    out_dims = None

    def _scan_f(_, dyn_vals):  # dyn_vals is sequence of numpy arrays for this step
        nonlocal flat_out_template, out_dims
        # Reconstruct input arrays for this time-step
        current = [ops.array(val, dims[1:]) for val, dims in zip(dyn_vals, in_dims)]
        inp = flat_in.treedef.unflatten(current)
        # Flatten and cache template dims on first call
        flat_dynamic, flat_out_template = flatten(map_f(inp))
        flat_dynamic = tuple(
            ops.array(x) if not isinstance(x, ops.array) else x for x in flat_dynamic
        )
        out_dims = [[dim, *x.dims] for x in flat_dynamic]
        flat_dynamic = tuple(
            x.data if isinstance(x, ops.array) else None for x in flat_dynamic
        )
        # Return unchanged carry and this step's output dynamics
        return None, flat_dynamic

    # Initialize dummy carry and capture first step
    first_vals = [arr[0] for arr in in_data]
    carry0, first_out = _scan_f(None, first_vals)

    # Run scan over the rest
    rest_vals = [arr[1:] for arr in in_data]
    _, rest_out = ops.backend.scan(_scan_f, carry0, rest_vals, unroll=unroll)

    # Combine first and rest outputs for each dynamic output array
    all_out = []
    for first_arr, rest_arr in zip(first_out, rest_out):
        all_out.append(ops.backend.concat([first_arr[None, ...], rest_arr], axis=0))

    # Rewrap into ops.array with leading dimension restored
    wrapped = [ops.array(arr, dims=dims) for arr, dims in zip(all_out, out_dims)]

    # Reconstruct tree of outputs
    return flat_out_template.unflatten(wrapped)


def vmap(f, in_axes):
    return backend.vmap(f, in_axes=in_axes)


def grad(f=None, /, argnums: int | Sequence[int] = 0):
    """Create a function that evaluates the gradient of ``f``.

      f: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, standard Python containers or spekk
        Modules. Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It should return a scalar (which
        includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).

    Returns:
      A function with the same arguments as ``f``, that evaluates the gradient
      of ``f``. If ``argnums`` is an integer then the gradient has the same
      shape and type as the positional argument indicated by that integer. If
      argnums is a tuple of integers, the gradient is a tuple of values with the
      same shapes and types as the corresponding arguments.
    """
    from spekk.module.base import flatten

    # Allow the following syntax:
    #   @grad(argnums=1)
    #   def f(a, b): ...
    # which is shorthand for:
    #   @functools.partial(grad, argnums=1)
    #   def f(a, b): ...
    if f is None:
        return functools.partial(grad, argnums=argnums)

    # Ensure that argnums is a list because it simplifies the code further down. We
    # still have to keep information about whether the argnums was an integer or a
    # sequence to know whether we should return a single item or a tuple of items. If
    # argnums is a sequence of integers, the returned gradient is a tuple of values
    # with the same shapes and types as the corresponding arguments.
    is_single_argnum = isinstance(argnums, int)
    argnums = [argnums] if isinstance(argnums, int) else list(argnums)

    def outer(*args):
        # Extract only the arguments that are to be differentiated.
        grad_args = [arg for i, arg in enumerate(args) if i in argnums]
        # Flatten it down to values that are understood by the backend.
        flattened_grad_args = flatten(grad_args, flatten_spekk_arrays=True)

        # Define the function that accepts the flattened values and wrap it with grad.
        # We pass only the values that are to be differentiated to this function, so
        # argnums is set to be all arguments.
        @functools.partial(
            ops.backend.grad,
            argnums=range(len(flattened_grad_args.dynamic)),
        )
        def inner(*args_inner):
            # Unflatten the arguments used to calculate the gradient back to their
            # original types and structure.
            grad_args_inner = flattened_grad_args.treedef.unflatten(args_inner)

            # Get the full list of args by inserting the unflattened args at the right
            # positions.
            args_inner = list(args)
            for i, arg in zip(argnums, grad_args_inner):
                args_inner[i] = arg

            # Call the function that will be differentiated.
            result = f(*args_inner)

            # The result must be a scalar float value.
            if not isinstance(result, (float, array)):
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with type: {type(result)!r}."
                )
            if isinstance(result, array) and not result.ndim == 0:
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with shape: {result.shape!r}"
                )

            # Return the backend data of the result which must be a scalar value.
            if isinstance(result, array):
                result = result.data
            return result

        # Get the resulting gradient and unflatten to the original type and structure.
        # The call to unflatten is what allows us to take the gradient with regards to
        # an arbitrary structure of arguments; even Modules.
        result = inner(*flattened_grad_args.dynamic)
        result = flattened_grad_args.treedef.unflatten(result)

        # If argnums was a sequence of integers, the returned gradient is a tuple of
        # values with the same shapes and types as the corresponding arguments.
        # Otherwise, the result is just a single item and we return it.
        if is_single_argnum:
            assert len(result) == 1
            result = result[0]
        return result

    return outer


def value_and_grad(f=None, /, argnums: int | Sequence[int] = 0):
    """Create a function that evaluates both `f` and the gradient of ``f``.

      f: Function to be differentiated. Its arguments at positions specified by
        ``argnums`` should be arrays, scalars, standard Python containers or spekk
        Modules. Argument arrays in the positions specified by ``argnums`` must be of
        inexact (i.e., floating-point or complex) type. It should return a scalar (which
        includes arrays with shape ``()`` but not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).

    Returns:
      A function with the same arguments as ``f``, that evaluates both ``f``and the
      gradient of ``f``. If ``argnums`` is an integer then the gradient has the same
      shape and type as the positional argument indicated by that integer. If argnums
      is a tuple of integers, the gradient is a tuple of values with the same shapes
      and types as the corresponding arguments.
    """
    from spekk.module.base import flatten

    # Allow the following syntax:
    #   @value_and_grad(argnums=1)
    #   def f(a, b): ...
    # which is shorthand for:
    #   @functools.partial(value_and_grad, argnums=1)
    #   def f(a, b): ...
    if f is None:
        return functools.partial(grad, argnums=argnums)

    # Ensure that argnums is a list because it simplifies the code further down. We
    # still have to keep information about whether the argnums was an integer or a
    # sequence to know whether we should return a single item or a tuple of items. If
    # argnums is a sequence of integers, the returned gradient is a tuple of values
    # with the same shapes and types as the corresponding arguments.
    is_single_argnum = isinstance(argnums, int)
    argnums = [argnums] if isinstance(argnums, int) else list(argnums)

    def outer(*args):
        # Extract only the arguments that are to be differentiated.
        grad_args = [arg for i, arg in enumerate(args) if i in argnums]
        # Flatten it down to values that are understood by the backend.
        flattened_grad_args = flatten(grad_args, flatten_spekk_arrays=True)

        # Define the function that accepts the flattened values and wrap it with grad.
        # We pass only the values that are to be differentiated to this function, so
        # argnums is set to be all arguments.
        @functools.partial(
            ops.backend.value_and_grad,
            argnums=range(len(flattened_grad_args.dynamic)),
        )
        def inner(*args_inner):
            # Unflatten the arguments used to calculate the gradient back to their
            # original types and structure.
            grad_args_inner = flattened_grad_args.treedef.unflatten(args_inner)

            # Get the full list of args by inserting the unflattened args at the right
            # positions.
            args_inner = list(args)
            for i, arg in zip(argnums, grad_args_inner):
                args_inner[i] = arg

            # Call the function that will be differentiated.
            result = f(*args_inner)

            # The result must be a scalar float value.
            if not isinstance(result, (float, array)):
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with type: {type(result)!r}."
                )
            if isinstance(result, array) and not result.ndim == 0:
                raise ValueError(
                    "Gradient only defined for scalar-output functions. Got output "
                    f"with shape: {result.shape!r}"
                )

            # Return the backend data of the result which must be a scalar value.
            if isinstance(result, array):
                result = result.data
            return result

        # Get the resulting gradient and unflatten to the original type and structure.
        # The call to unflatten is what allows us to take the gradient with regards to
        # an arbitrary structure of arguments; even Modules.
        value, grad = inner(*flattened_grad_args.dynamic)
        assert value.ndim == 0  # We check for this in the inner function as well
        value = ops.array(value)
        grad = flattened_grad_args.treedef.unflatten(grad)

        # If argnums was a sequence of integers, the returned gradient is a tuple of
        # values with the same shapes and types as the corresponding arguments.
        # Otherwise, the result is just a single item and we return it.
        if is_single_argnum:
            assert len(grad) == 1
            grad = grad[0]
        return value, grad

    return outer


def wrap_backend_decorator(decorator):
    """Wraps a backend decorator so that it accepts spekk arrays and Modules."""
    from spekk.module.base import FlattenedTree, TreeDef, flatten

    def new_decorator(f=None, /, **decorator_kwargs):
        # Allow the following syntax:
        #   wrapped_decorator = wrap_backend_decorator(my_decorator)
        #   @wrapped_decorator(argnums=1)
        #   def f(a, b): ...
        # which is shorthand for:
        #   @functools.partial(wrapped_decorator, argnums=1)
        #   def f(a, b): ...
        if f is None:
            return functools.partial(new_decorator, **decorator_kwargs)

        def outer(*args, **kwargs):
            # Flatten the arguments down to values that are understood by the backend.
            flattened_input = flatten((args, kwargs), flatten_spekk_arrays=True)
            # flattened_output gets defined inside inner function.
            output_treedef: TreeDef = None  # type: ignore

            @functools.partial(decorator, **decorator_kwargs)
            def inner(inner_args):
                nonlocal output_treedef
                inner_args, inner_kwargs = flattened_input.treedef.unflatten(inner_args)
                result = f(*inner_args, **inner_kwargs)
                output_dynamic, output_treedef = flatten(
                    result, flatten_spekk_arrays=True
                )
                return output_dynamic

            inner_result = inner(flattened_input.dynamic)
            return output_treedef.unflatten(inner_result)

        return outer

    return new_decorator


checkpoint = wrap_backend_decorator(backend.checkpoint)


if __name__ == "__main__":
    import doctest

    from spekk import ops

    ops.backend.set_backend("numpy")
    doctest.testmod()
