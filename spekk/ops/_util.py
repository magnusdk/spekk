from typing import Optional, Sequence, Tuple, Union

from spekk.ops._backend import backend
from spekk.ops._types import ArrayLike, Dim, Dims, UndefinedDim
from spekk.ops.array_object import array


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
        # Ensure that dim is a dimension (str), not an axis (int) going forward
        dim = all_dims[dim]

    # Reducing over all axes (when dim is None)
    if dim is None:
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
        new_dims = list(all_dims)
        for dim1 in dim:
            # Allow sending explicit axis
            if isinstance(dim1, Dim):
                axis = all_dims.index(dim1)
            else:
                axis = dim1

            axes.append(axis)
            new_dims.remove(dim1)
        # Return the reduced-over axes and all dimensions minus the ones reduced over
        axis = tuple(axes)
        dims = new_dims

    else:
        raise ValueError(
            "dim must be either None, a single dimension or axis, or a "
            "sequence of dimensions or axes."
        )

    # Handle keepdims
    if keepdims:
        dims = all_dims
    return axis, dims


def ensure_array(x: ArrayLike) -> array:
    if not isinstance(x, array):
        x = backend.asarray(x)
        x = array(x, [UndefinedDim()] * x.ndim)
    return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
