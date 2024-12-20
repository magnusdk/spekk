from typing import Optional, Sequence, Tuple, Union

from spekk.ops._backend import backend
from spekk.ops._types import ArrayLike, Dim, Dims, UndefinedDim
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


def ensure_array(x: ArrayLike, dtype: _DType = None) -> array:
    if not isinstance(x, array):
        dtype = dtype._to_backend_dtype() if dtype is not None else None
        x = backend.asarray(x, dtype=dtype)
        x = array(x, [UndefinedDim()] * x.ndim)
    return x


def canonicalize_axis(n: int, i: int) -> int:
    if i < 0:
        i += n
    return i


if __name__ == "__main__":
    import doctest

    doctest.testmod()
