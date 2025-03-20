from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Tuple, Union

from spekk.ops._backend import backend
from spekk.ops._types import ArrayLike, Dim, Dims, _UndefinedDim, undefined_dim
from spekk.ops.array_object import array
from spekk.ops.data_types import _DType
from spekk import ops
import numbers

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


def ensure_array(x: ArrayLike, dtype: _DType = None) -> array:
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
        arr_dims_in_order = [dim for dim in output_dims if dim in arr.dims]
        if arr.dims != arr_dims_in_order:
            arr = ops.permute_dims(arr, arr_dims_in_order)

        new_shape = []
        for dim in output_dims:
            if dim in arr.dims:
                new_shape.append(arr.dim_sizes[dim])
            else:
                new_shape.append(1)
        new_shape = tuple(new_shape)
        if arr.shape != new_shape:
            arr = ops.reshape(arr, shape=new_shape, dims=list(output_dims.keys()))

        broadcasted_shape = tuple(output_dims.values())
        if arr.shape != broadcasted_shape:
            arr = ops.broadcast_to(
                arr, shape=broadcasted_shape, dims=list(output_dims.keys())
            )
        return arr

        # Find the dimensions that need to be added to the array in order to be broadcastable with all other arrays. missing_dims is a dictionary from the name of the dimension to the size of that dimension.
        missing_dims = {
            dim: size for dim, size in output_dims.items() if dim not in arr._dims
        }

        # Add the new dimensions to the array's data, making it broadcastable.
        new_shape_broadcastable = arr._data.shape + (1,) * len(missing_dims)
        new_data = arr._data
        if arr.shape != new_shape_broadcastable:
            new_data = ops.backend.reshape(new_data, new_shape_broadcastable)

        # Add the correct sizes to the corresponding dimensions
        new_shape = (*arr._data.shape, *missing_dims.values())
        if new_data.shape != new_shape:
            new_data = ops.backend.broadcast_to(new_data, new_shape)
        # Also add them to the dimensions list
        new_dims = [*arr._dims, *missing_dims.keys()]

        # Put the dimensions in the correct order (same as all other arrays)
        if new_dims != output_dims:
            arr = ops.permute_dims(
                ops.array(new_data, new_dims), list(output_dims.keys())
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

        if ops.backend.backend_name=="torch" and is_number(item): # NOTE: Move logic to backends in future or disallow python ints/floats/etc
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
