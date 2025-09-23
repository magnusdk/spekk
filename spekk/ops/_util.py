import numbers
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from spekk import ops
from spekk.module.base import TContainer, TLeaf
from spekk.ops._backend import backend
from spekk.ops._types import ArrayLike, Dim, Dims, _UndefinedDim, undefined_dim
from spekk.ops.array_object import array
from spekk.ops.data_types import DType


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


def ensure_array(x: ArrayLike, dtype: DType = None) -> array:
    # if not isinstance(x, array):
    # dtype = _DType._to_backend_dtype(dtype) if dtype is not None else None
    # if not backend._is_backend_array(x) or (x.dtype != dtype and dtype is not None) :
    #     x = backend.asarray(x, dtype=dtype)
    # x = array(x, dtype=dtype)
    return array(x, dtype=dtype)


def canonicalize_axis(n: int, i: int) -> int:
    if i < 0:
        i += n
    return i


def temporarily_make_undefined_dims_defined[T: TLeaf | TContainer](
    x: T,
) -> tuple[T, set[Dim]]:
    from spekk import traverse, util

    temporary_dims = set()

    def _impl(x: ops.array) -> ops.array:
        if isinstance(x, ops.array):
            new_dims = []
            for dim in x.dims:
                if isinstance(dim, _UndefinedDim):
                    new_dim = util.random_dim_name()
                    new_dims.append(new_dim)
                    temporary_dims.add(new_dim)
                else:
                    new_dims.append(dim)
            x = ops.array(x.data, new_dims)
        return x

    x = traverse(x, map_leaf=_impl)
    return x, temporary_dims


def make_temporary_dims_undefined[T: TLeaf | TContainer](
    x: T,
    temporary_dims: set[Dim],
) -> T:
    from spekk import traverse

    def _impl(x: ops.array) -> ops.array:
        if isinstance(x, ops.array):
            new_dims = [
                _UndefinedDim() if dim in temporary_dims else dim for dim in x.dims
            ]
            x = ops.array(x.data, new_dims)
        return x

    x = traverse(x, map_leaf=_impl)
    return x


def get_broadcast_array_fn(
    *arrays: array, except_dims: Dims = ()
) -> Callable[[array], array]:
    """Returns a function that broadcasts arrays to compatible shapes.

    For dimensions in except_dims, each array keeps its original size. This is useful
    for operations like concat where the concat axis can have different sizes.
    """
    from spekk import ops

    n_arrays_with_undefined_dims = sum(
        any(isinstance(dim, _UndefinedDim) for dim in x._dims) for x in arrays
    )
    if n_arrays_with_undefined_dims == len(arrays):
        output_shape = np.broadcast_shapes(*[x.shape for x in arrays])

        def broadcast_array(arr: ops.array) -> ops.array:
            return ops.broadcast_to(arr, output_shape)

        return broadcast_array

    # Get the output dimension list of each array after broadcasting. Ordering of
    # output dimensions are determined by the ordering of input arrays and their
    # dimensions.
    combined_output_dims = defaultdict(set)
    for arr in arrays:
        for size, dim in zip(arr.shape, arr._dims):
            if dim not in combined_output_dims:
                combined_output_dims[dim].add(size)

    combined_output_dims = {
        dim: sizes.pop() for dim, sizes in combined_output_dims.items()
    }

    def broadcast_array(arr: ops.array) -> ops.array:
        # Ensure that the dimensions have the correct order before broadcasting.
        arr_dims_in_order = [dim for dim in combined_output_dims if dim in arr.dims]
        if arr.dims != arr_dims_in_order:
            arr = ops.permute_dims(arr, arr_dims_in_order)

        # Add singleton dimensions that will be broadcasted to the common shape.
        new_shape = []
        for dim in combined_output_dims:
            if dim in arr.dims:
                new_shape.append(arr.dim_sizes[dim])
            else:
                new_shape.append(1)  # Singleton dimension
        new_shape = tuple(new_shape)
        if arr.shape != new_shape:
            arr = ops.reshape(
                arr, shape=new_shape, dims=list(combined_output_dims.keys())
            )

        # Get the final common broadcasted shape.
        broadcasted_shape = []
        for dim, size in combined_output_dims.items():
            # Keep original size for except_dims (e.g., concat axis)
            if dim in except_dims:
                size = arr.dim_sizes[dim]
            broadcasted_shape.append(size)
        broadcasted_shape = tuple(broadcasted_shape)

        # Perform the actual broadcasting.
        if arr.shape != broadcasted_shape:
            arr = ops.broadcast_to(
                arr,
                shape=broadcasted_shape,
                dims=list(combined_output_dims.keys()),
            )

        return arr

    return broadcast_array


def ensure_broadcastable(
    *arrays: array, ensure_same_ndim: bool = False
) -> Tuple[List[Dim], List[array]]:
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
        ndim = max(arr.ndim for arr in arrays if isinstance(arr, ops.array))
        return [undefined_dim] * ndim, arrays

    # Get the output dimension list of each array after broadcasting. Ordering of
    # output dimensions are determined by the ordering of input arrays and their
    # dimensions.
    output_dims = []
    for arr in arrays:
        if isinstance(arr, array):
            for dim in arr.dims:
                if dim not in output_dims:
                    output_dims.append(dim)

    # Ensure that they are broadcastable :)
    broadcastable_arrays = [
        ensure_broadcastable_with(x, output_dims, ensure_same_ndim=ensure_same_ndim)
        for x in arrays
    ]
    return output_dims, broadcastable_arrays


def ensure_broadcastable_with(
    x: array, dims: Dims, *, ensure_same_ndim: bool = False
) -> array:
    from spekk import ops

    if isinstance(x, array):
        x_dims = get_dims(x)
        # Permute arr's dims if they are not in the same order as output_dims
        arr_dims_in_order = [dim for dim in dims if dim in x_dims]
        if x_dims != arr_dims_in_order:
            x = ops.permute_dims(x, arr_dims_in_order)

        # Make arr broadcastable with output_dims if it isn't already. It is
        # broadcastable if its dimensions equal the last dimensions of output_dims.
        start_index = len(dims) - len(arr_dims_in_order)
        if (ensure_same_ndim and len(dims) != len(x_dims)) or arr_dims_in_order != dims[
            start_index:
        ]:
            # Add the new dimensions to the array's data, making it broadcastable.
            broadcastable_shape = [
                x.dim_sizes[dim] if dim in x_dims else 1 for dim in dims
            ]
            x = ops.reshape(x, broadcastable_shape, dims)
    return x


def is_number(variable):
    return isinstance(variable, numbers.Number)


def ensure_backend_compatible_data(*data) -> list:
    new_data = []
    for item in data:
        if isinstance(item, array):
            item = item.data

        if ops.backend.backend_name == "torch" and is_number(
            item
        ):  # NOTE: Move logic to backends in future or disallow python ints/floats/etc
            # item = backend.asarray(item)
            item = ops.array(item).data
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
