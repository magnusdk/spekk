import functools
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple, Union

from spekk.ops._backend import backend
from spekk.ops._types import ArrayLike, Dim, Dims, _UndefinedDim, undefined_dim
from spekk.ops.array_object import array
from spekk.ops.data_types import _DType


def get_reduction_axes_and_resulting_dims(
    dim: Optional[Union[Dim, int, Tuple[Dim, ...], Tuple[int, ...]]],
    all_dims: Dims,
    keepdims: bool,
) -> Tuple[Optional[Union[int, Tuple[int, ...]]], Dims]:
    """Helper function for determining the correct axes and the resulting dimensions
    after performing a reduction operation (e.g.: sum, mean, std, etc). Given one or
    multiple dimensions to reduce over, return the corresponding axis/axes of the
    underlying array and the resulting dimensions.

    Examples
    Reducing over a single dimension:
    >>> get_reduction_axes_and_resulting_dims("dim1", ["dim1", "dim2"], keepdims=False)
    (0, ['dim2'])
    >>> get_reduction_axes_and_resulting_dims("dim2", ["dim1", "dim2"], keepdims=True)
    (1, ['dim1', 'dim2'])

    Reducing over multiple dimensions:
    >>> get_reduction_axes_and_resulting_dims(["dim1", "dim2"], ["dim1", "dim2"], keepdims=False)
    ((0, 1), [])
    >>> get_reduction_axes_and_resulting_dims(["dim1", "dim2"], ["dim1", "dim2"], keepdims=True)
    ((0, 1), ['dim1', 'dim2'])
    """
    # Allow sending explicit axis
    if isinstance(dim, int):
        axis = dim
        dims = list(all_dims)
        del dims[axis]

    # Reducing over all axes (when dim is None)
    elif dim is None:
        # Return all axes, and empty dimension list
        axis = tuple(range(len(all_dims)))
        dims = []

    # Reducing over a single axis
    elif isinstance(dim, Dim):
        new_dims = list(all_dims)
        new_dims.remove(dim)
        # Return the axis for the dimension and all dimensions minus the reduced dimension
        axis = all_dims.index(dim)
        dims = new_dims

    # Reducing over multiple axes
    elif isinstance(dim, Sequence):
        axes = []
        for dim1 in dim:
            # Allow sending explicit axis
            axis = all_dims.index(dim1) if isinstance(dim1, Dim) else dim1
            axes.append(axis)
        # Return the reduced-over axes and all dimensions minus the ones reduced over
        axis = tuple(canonicalize_axis(len(all_dims), axis) for axis in axes)
        dims = [dim1 for i, dim1 in enumerate(all_dims) if i not in axis]

    else:
        raise ValueError(
            "dim must be either None, a single dimension or axis, or a "
            "sequence of dimensions or axes."
        )

    # Handle keepdims
    if keepdims:
        dims = all_dims
    return axis, dims


def prepare_slicing_along_dim(
    x: array, i: array, dim: Dim
) -> Tuple[array, Tuple[slice, array], Dims]:
    from spekk import ops

    x, i = ensure_array(x), ensure_array(i)

    # Ensure dim is the first axis. This makes it easier to keep track of dimensions.
    if x.dims.index(dim) != 0:
        x = ops.moveaxis(x, dim, 0)

    common_dims = set(x.dims) & set(i.dims) - {dim}
    slices = [slice(None)] * x.ndim
    slices[x.dims.index(dim)] = i.data
    for d in common_dims:
        dim_idx = x.dims.index(d)
        dim_size = x.shape[dim_idx]
        broadcastable_shape = [1] * i.ndim
        broadcastable_shape[i.dims.index(d)] = dim_size
        slices[dim_idx] = backend.reshape(backend.arange(dim_size), broadcastable_shape)

    resulting_dims = i.dims + [d for d in x.dims if d not in i.dims and d != dim]
    return x, tuple(slices), resulting_dims


def ensure_array(x: ArrayLike, dtype: _DType = None) -> array:
    if not isinstance(x, array):
        dtype = dtype._to_backend_dtype() if dtype is not None else None
        x = backend.asarray(x, dtype=dtype)
        x = array(x, [undefined_dim] * x.ndim)
    return x


def canonicalize_axis(n: int, i: int) -> int:
    if i < 0:
        i += n
    return i


def get_broadcast_array_fn(*arrays: array) -> Callable[[array], array]:
    from spekk import ops

    # TODO: How to handle UndefinedDim?
    if any(isinstance(dim, _UndefinedDim) for x in arrays for dim in x._dims):
        # If all dimensions are undefined, fall back to regular broadcasting.
        if all(
            (isinstance(dim, _UndefinedDim) or (len(x._dims) == 0))
            for x in arrays
            for dim in x._dims
        ):
            return [
                ops.array(x)
                for x in ops.backend.broadcast_arrays(*[x.data for x in arrays])
            ]

        raise NotImplementedError()

    # Get the output dimension list of each array after broadcasting. Ordering of output dimensions are determined by the ordering of input arrays and their dimensions.
    output_dims = defaultdict(set)
    for arr in arrays:
        for size, dim in zip(arr.shape, arr._dims):
            if dim not in output_dims:
                output_dims[dim].add(size)

    output_dims = {dim: sizes.pop() for dim, sizes in output_dims.items()}

    def broadcast_array(arr: ops.array) -> ops.array:
        # Find the dimensions that need to be added to the array in order to be broadcastable with all other arrays. missing_dims is a dictionary from the name of the dimension to the size of that dimension.
        missing_dims = {
            dim: size for dim, size in output_dims.items() if dim not in arr._dims
        }

        # Add the new dimensions to the array's data, making it broadcastable.
        new_shape_broadcastable = arr._data.shape + (1,) * len(missing_dims)
        new_data = ops.backend.reshape(arr._data, new_shape_broadcastable)

        # Add the correct sizes to the corresponding dimensions
        new_shape = (*arr._data.shape, *missing_dims.values())
        new_data = ops.backend.broadcast_to(new_data, new_shape)
        # Also add them to the dimensions list
        new_dims = [*arr._dims, *missing_dims.keys()]

        # Put the dimensions in the correct order (same as all other arrays)
        arr = ops.permute_dims(ops.array(new_data, new_dims), list(output_dims.keys()))
        return arr

    return broadcast_array


def ensure_broadcastable(*arrays: array) -> Tuple[List[Dim], List[array]]:
    from spekk import ops

    # Check if the arrays have any undefined dimensions. If all the dimensions are
    # undefined, then we just return the arrays as-is (and pray that they are actually
    # broadcastable) and give the broadcasting responsibility to the underlying
    # backend. If some, but not all dimensions are undefined, raise a ValueError.
    all_dims_are_undefined_dims = True
    any_dims_are_undefined_dims = False
    for arr in arrays:
        if isinstance(arr, array):
            for dim in arr.dims:
                if isinstance(dim, _UndefinedDim):
                    any_dims_are_undefined_dims = True
                else:
                    all_dims_are_undefined_dims = False
    if any_dims_are_undefined_dims:
        if not all_dims_are_undefined_dims:
            raise ValueError()
        n_dim = max(arr.ndim for arr in arrays if isinstance(arr, ops.array))
        return [undefined_dim] * n_dim, arrays

    # Get the output dimension list of each array after broadcasting. Ordering of
    # output dimensions are determined by the ordering of input arrays and their
    # dimensions.
    output_dims = []
    for arr in arrays:
        if isinstance(arr, array):
            for dim in arr.dims:
                if dim not in output_dims:
                    output_dims.append(dim)

    # Expand dimensions if needed (using reshape) in the correct order.
    resulting_arrays = []
    for arr in arrays:
        if isinstance(arr, array):
            if arr.dims != output_dims:
                arr_dims_in_order = [dim for dim in output_dims if dim in arr.dims]
                arr = ops.permute_dims(arr, arr_dims_in_order)
            if len(arr.dims) != len(output_dims):
                # Add the new dimensions to the array's data, making it broadcastable.
                broadcastable_shape = [
                    arr.dim_sizes[dim] if dim in arr.dims else 1 for dim in output_dims
                ]
                arr = ops.reshape(arr, broadcastable_shape, output_dims)
        resulting_arrays.append(arr)

    return output_dims, resulting_arrays


def ensure_broadcastable_with(x: array, dims: Dims) -> array:
    from spekk import ops

    x_dims = get_dims(x)
    assert all(dim in dims for dim in x_dims)
    if x_dims != dims:
        x_dims_in_order = [dim for dim in dims if dim in x_dims]
        x = ops.permute_dims(x, x_dims_in_order)
    if len(x_dims) != len(dims):
        # Add the new dimensions to the array's data, making it broadcastable.
        broadcastable_shape = [x.dim_sizes[dim] if dim in x_dims else 1 for dim in dims]
        x = ops.reshape(x, broadcastable_shape, dims)
    return x


def ensure_backend_compatible_data(*data) -> list:
    new_data = []
    for item in data:
        if isinstance(item, array):
            item = item.data
        new_data.append(item)
    return new_data


def get_dims(obj) -> Dims:
    if isinstance(obj, array):
        return obj.dims
    elif hasattr(obj, "ndim"):
        return [undefined_dim] * obj.ndim
    elif isinstance(obj, (list, tuple)):
        raise NotImplementedError()
    return []


def cacheable(f: Callable) -> Callable:
    return f

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        from spekk.module.base import _MODULE_METHODS_CACHE

        for v in [*args, *kwargs.values()]:
            try:
                hash(v)
            except Exception:
                return f(*args, **kwargs)

        if _MODULE_METHODS_CACHE is not None:
            if f not in _MODULE_METHODS_CACHE:
                _MODULE_METHODS_CACHE[f] = functools.lru_cache(
                    maxsize=None, typed=True
                )(f)
            return _MODULE_METHODS_CACHE[f](*args, **kwargs)
        return f(*args, **kwargs)

    return wrapped


if __name__ == "__main__":
    import doctest

    doctest.testmod()
