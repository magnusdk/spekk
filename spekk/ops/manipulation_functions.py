__all__ = [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "repeat",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]

from typing import Sequence

from spekk.ops._backend import backend
from spekk.ops._types import (
    ArrayLike,
    Dim,
    undefined_dim,
)
from spekk.ops._util import (
    canonicalize_axis,
    ensure_broadcastable,
    ensure_broadcastable_with,
    get_broadcast_array_fn,
)
from spekk.ops.array_object import array


def broadcast_arrays(*arrays: array) -> list[array]:
    """
    Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays: array
        the arrays to broadcast.

    Returns
    -------
    out: list[array]
        a list of broadcasted arrays. Each array has the same shape and dtype as its corresponding input array.
    """
    arrays = [array(x) for x in arrays]
    broadcast_array = get_broadcast_array_fn(*arrays)
    return [broadcast_array(arr) for arr in arrays]


def broadcast_to(
    x: array, /, shape: tuple[int, ...], dims: Sequence[str] | None = None
) -> array:
    """
    Broadcasts an array to a specified shape.

    Parameters
    ----------
    x: array
        array to broadcast.
    shape: tuple[int, ...]
        array shape. Must be compatible with ``x``. If the array is incompatible with the specified shape, an exception is raised.
    dims: Sequence[str] | None
        the new list of dimension names. Must have the same length as ``shape``.

    Returns
    -------
    out: array
        an array having a specified shape. Has the same data type as ``x``.

    Raises
    ------
    ValueError
        If ``shape`` and ``dims`` do not have the same length.
    """
    x = array(x)
    if dims is None:
        dims = [undefined_dim] * len(shape)
    elif len(dims) != len(shape):
        raise ValueError(
            "The number of dimensions must equal the number of axes when broadcasting."
        )
    else:
        x = ensure_broadcastable_with(x, dims)
    data = backend.broadcast_to(x.data, shape)
    return array(data, dims)


def concat(
    arrays: tuple[array, ...] | list[array],
    /,
    *,
    axis: int | str = 0,
) -> array:
    """
    Joins a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays: tuple[array, ...] | list[array]
        input arrays to join.
    axis: int | str
        dimension along which the arrays will be joined.

    Returns
    -------
    out: array
        an output array containing the concatenated values. The output dtype is determined by type promotion across the input arrays.
    """
    # TODO: What to do with 'UndefinedDims's arrays? Then we shouldn't broadcast here.
    # broadcasted_dims, arrays = ensure_broadcastable(*arrays, ensure_same_ndim=True)

    broadcast_array = get_broadcast_array_fn(*arrays, except_dims=[axis])
    arrays = [broadcast_array(arr) for arr in arrays]
    broadcasted_dims = arrays[0].dims
    if isinstance(axis, Dim):
        axis = broadcasted_dims.index(axis)
    data = backend.concat([arr._data for arr in arrays], axis=axis)
    return array(data, broadcasted_dims)


def expand_dims(x: array, /, *, axis: int | str = 0) -> array:
    """
    Expands the shape of an array by inserting a new axis (dimension) of size one at the
    position specified by ``axis``.

    If ``axis`` is a string, it is used as the name of the new dimension and the
    dimension is inserted at position 0. If ``axis`` is an integer, the new dimension is
    unnamed.

    Parameters
    ----------
    x: array
        input array.
    axis: int | str
        axis position (zero-based) or dimension name. If a negative integer is
        provided, the position is counted from the end. Valid integer values are in the
        closed interval ``[-N-1, N]`` where ``N`` is the number of dimensions of ``x``.

    Returns
    -------
    out: array
        an expanded output array having the same data type as ``x``.

    Raises
    ------
    IndexError
        If provided an invalid ``axis`` position.
    ValueError
        If the new dimension already exists.
    """
    x = array(x)

    if isinstance(axis, Dim):
        dim = axis
        axis = 0
    else:
        dim = undefined_dim
    data = backend.expand_dims(x._data, axis=axis)
    dims = list(x._dims)
    dims.insert(axis, dim)
    return array(data, dims)


def flip(x: array, /, *, axis: int | str | tuple[int | str, ...] | None = None) -> array:
    """
    Reverses the order of elements in an array along the given axis. The shape of the
    array is preserved.

    Parameters
    ----------
    x: array
        input array.
    axis: int | str | tuple[int | str, ...] | None
        axis (or axes) along which to flip. If ``None``, all axes are flipped. If more
        than one axis is provided, only those axes are flipped. Default: ``None``.

    Returns
    -------
    out: array
        an output array having the same data type and shape as ``x`` and whose elements,
        relative to ``x``, are reordered.
    """
    x = array(x)
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    elif isinstance(axis, tuple):
        axis = tuple(
            x._dims.index(dim1) if isinstance(dim1, Dim) else dim1 for dim1 in axis
        )
    return array(backend.flip(x._data, axis=axis), x._dims)


def moveaxis(
    x: array,
    source: int | str | tuple[int | str, ...] | list[int | str],
    destination: int | str | tuple[int | str, ...] | list[int | str],
    /,
) -> array:
    """
    Moves array axes (dimensions) to new positions, while leaving other axes in their
    original positions.

    Parameters
    ----------
    x: array
        input array.
    source: int | str | tuple[int | str, ...] | list[int | str]
        axes to move. Provided axes must be unique. If ``x`` has ``N`` dimensions, a
        valid integer axis resides on the half-open interval ``[-N, N)``.
    destination: int | str | tuple[int | str, ...] | list[int | str]
        desired positions for each respective ``source`` axis. Provided values must be
        unique. If ``x`` has ``N`` dimensions, a valid integer axis resides on the
        half-open interval ``[-N, N)``.

    Returns
    -------
    out: array
        an array containing reordered axes. The returned array has the same data type
        as ``x``.
    """
    x = array(x)

    # Ensure a tuple of sources and a tuple of destinations
    if not isinstance(source, (tuple, list)):
        source = [source]
    if not isinstance(destination, (tuple, list)):
        destination = [destination]

    # Convert all to integers (axes)
    source = tuple(
        x._dims.index(d) if isinstance(d, Dim) else canonicalize_axis(len(x.dims), d)
        for d in source
    )
    destination = tuple(
        x._dims.index(d) if isinstance(d, Dim) else canonicalize_axis(len(x.dims), d)
        for d in destination
    )

    data = backend.moveaxis(x._data, source, destination)
    dims = [dim for i, dim in enumerate(x.dims) if i not in source]
    for src, dest in zip(source, destination):
        dims.insert(dest, x.dims[src])
    return array(data, dims)


def permute_dims(x: array, /, axes: tuple[int | str, ...] | list[int | str]) -> array:
    """
    Permutes the axes (dimensions) of an array ``x``.

    Parameters
    ----------
    x: array
        input array.
    axes: tuple[int | str, ...] | list[int | str]
        a permutation of axes specifying the desired order. Accepts dimension names (strings) or positional indices (integers).

    Returns
    -------
    out: array
        an array containing the axes permutation. The returned array has the same data
        type as ``x``.
    """
    x = array(x)
    if all(isinstance(axis, Dim) for axis in axes):
        dims = axes
        axes = [x._dims.index(dim) for dim in axes]
    else:
        dims = [x._dims[axis] for axis in axes]
    data = backend.permute_dims(x._data, axes)
    return array(data, dims)


def repeat(
    x: array,
    repeats: int,
    /,
    *,
    axis: int | str | None = None,
) -> array:
    """
    Repeats each element of an array a specified number of times.

    Parameters
    ----------
    x: array
        input array containing elements to repeat.
    repeats: int
        the number of repetitions for each element.
    axis: int | str | None
        the axis (dimension) along which to repeat elements. If ``axis`` is ``None``,
        the input array is flattened in row-major (C-style) order before repeating, and
        the result is a one-dimensional array. A string value is interpreted as a
        dimension name. Default: ``None``.

    Returns
    -------
    out: array
        an output array containing repeated elements. The returned array has the same
        data type as ``x``. If ``axis`` is ``None``, the returned array is
        one-dimensional; otherwise, it has the same shape as ``x`` except along the
        repeated axis.
    """
    x = array(x)
    dims = [undefined_dim] if axis is None else x.dims
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    data = backend.repeat(x._data, repeats=repeats, axis=axis)
    return array(data, dims)


def reshape(
    x: array,
    /,
    shape: tuple[int, ...] | list[int],
    dims: Sequence[str] | None = None,
    *,
    copy: bool | None = None,
) -> array:
    """
    Reshapes an array without changing its data.

    Parameters
    ----------
    x: array
        input array to reshape.
    shape: tuple[int, ...] | list[int]
        a new shape compatible with the original shape. One shape dimension is allowed
        to be ``-1``. When a shape dimension is ``-1``, the corresponding output array
        shape dimension is inferred from the length of the array and the remaining
        dimensions.
    dims: Sequence[str] | None
        the new list of dimension names. Must have the same length as ``shape``.
        Default: ``None``.
    copy: bool | None
        whether or not to copy the input array. If ``True``, the function always
        copies. If ``False``, the function never copies. If ``None``, the function
        avoids copying if possible, and may copy otherwise. Default: ``None``.

    Returns
    -------
    out: array
        an output array having the same data type and elements as ``x``.

    Raises
    ------
    ValueError
        If ``copy=False`` and a copy would be necessary. Also raised if ``shape`` and
        ``dims`` do not have the same length.
    """
    x = array(x)
    if dims is not None and len(dims) != len(shape):
        raise ValueError(
            "The number of dimensions must equal the number of axes when reshaping."
        )
    if dims is None:
        dims = [undefined_dim] * len(shape)
    return array(backend.reshape(x._data, tuple(shape), copy=copy), dims)


def roll(
    x: array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | str | tuple[int | str, ...] | None = None,
) -> array:
    """
    Rolls array elements along a specified axis. Elements that roll beyond the last
    position are re-introduced at the first position, and vice versa.

    Parameters
    ----------
    x: array
        input array.
    shift: int | tuple[int, ...]
        number of places by which the elements are shifted. If ``shift`` is a tuple,
        then ``axis`` must be a tuple of the same size, and each axis is shifted by the
        corresponding element in ``shift``. If ``shift`` is an ``int`` and ``axis`` is a
        tuple, the same ``shift`` is used for all specified axes. A positive shift moves
        elements toward larger indices; a negative shift moves elements toward smaller
        indices.
    axis: int | str | tuple[int | str, ...] | None
        axis (or axes) along which to shift elements. A string value is interpreted as a
        dimension name. If ``axis`` is ``None``, the array is flattened, shifted, and
        then restored to its original shape. Default: ``None``.

    Returns
    -------
    out: array
        an output array having the same data type as ``x`` and whose elements, relative
        to ``x``, are shifted.
    """
    x = array(x)
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    elif isinstance(axis, tuple):
        axis = tuple(
            x._dims.index(dim1) if isinstance(dim1, Dim) else dim1 for dim1 in axis
        )
    return array(backend.roll(x._data, shift=shift, axis=axis), x._dims)


def squeeze(
    x: ArrayLike,
    /,
    axis: int | str | Sequence[int] | Sequence[str],
) -> array:
    """
    Removes singleton dimensions (axes) from ``x``.

    Parameters
    ----------
    x: ArrayLike
        input array.
    axis: int | str | Sequence[int] | Sequence[str]
        axis (or axes) to squeeze. A string value is interpreted as a dimension name.

    Returns
    -------
    out: array
        an output array having the same data type and elements as ``x``.

    Raises
    ------
    ValueError
        If a specified axis has a size greater than one (i.e., it is not a
        singleton dimension).
    """
    x = array(x)

    # Handle changes to dimensions and get actual axis (integer)
    dims = list(x._dims)
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
        del dims[axis]
    elif isinstance(axis, int):
        del dims[axis]
    elif isinstance(axis, Sequence):
        axis = tuple(
            x._dims.index(i) if isinstance(i, Dim) else canonicalize_axis(x.ndim, i)
            for i in axis
        )
        # Remove the squeezed dimensions
        dims = [dim for i, dim in enumerate(dims) if i not in axis]

    data = backend.squeeze(x._data, axis)
    return array(data, dims)


def stack(
    arrays: tuple[bool | int | float | complex | array, ...]
    | list[bool | int | float | complex | array],
    /,
    *,
    axis: int | str = 0,
) -> array:
    """
    Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays: tuple[bool | int | float | complex | array, ...] | list[bool | int | float | complex | array]
        input arrays to join. Each array must have the same shape.
    axis: int | str
        axis along which the arrays will be joined. Providing an ``axis`` specifies the
        index of the new axis in the dimensions of the result. For example, if ``axis``
        is ``0``, the new axis will be the first dimension and the output array will have
        shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the second
        dimension and the output array will have shape ``(A, N, B, C)``; and, if
        ``axis`` is ``-1``, the new axis will be the last dimension and the output array
        will have shape ``(A, B, C, N)``. When ``axis`` is a string, it is used as the
        name of the new dimension and the new axis is inserted at position 0. A valid
        integer ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank
        (number of dimensions) of the input arrays. Default: ``0``.

    Returns
    -------
    out: array
        an output array having rank ``N+1``, where ``N`` is the rank (number of
        dimensions) of the input arrays. If the input arrays have different data types,
        type promotion applies. If the input arrays have the same data type, the output
        array has the same data type as the input arrays.
    """
    arrays = broadcast_arrays(*arrays)

    if isinstance(axis, Dim):
        dim = axis
        axis = 0
    else:
        dim = undefined_dim
    data = backend.stack([arr._data for arr in arrays], axis=axis)
    broadcasted_dims = list(arrays[0].dims)
    if axis < 0:
        # Handle negative index for list.insert. We have to add an additional 1 to the
        # axis, otherwise -1 refers to the second-to-last position when it should be
        # the last position.
        axis += len(broadcasted_dims) + 1
    broadcasted_dims.insert(axis, dim)
    return array(data, broadcasted_dims)


def tile(x: array, repetitions: tuple[int, ...], /) -> array:
    """
    Constructs an array by tiling an input array.

    Parameters
    ----------
    x: array
        input array.
    repetitions: tuple[int, ...]
        number of repetitions along each axis (dimension).

        Let ``N = len(x.shape)`` and ``M = len(repetitions)``.

        If ``N > M``, ones are prepended until all axes (dimensions) are specified (e.g.,
        if ``x`` has shape ``(8,6,4,2)`` and ``repetitions`` is ``(3,3)``, then
        ``repetitions`` is treated as ``(1,1,3,3)``).

        If ``N < M``, singleton axes (dimensions) are prepended to ``x`` until ``x`` has
        as many axes as ``repetitions`` specifies (e.g., if ``x`` has shape ``(4,2)`` and
        ``repetitions`` is ``(3,3,3,3)``, then ``x`` is treated as if it has shape
        ``(1,1,4,2)``).

    Returns
    -------
    out: array
        a tiled output array. The returned array has the same data type as ``x`` and a
        rank (number of dimensions) equal to ``max(N, M)``. The size of each axis ``i``
        equals ``S[i]*r[i]``, where ``S`` is the shape after prepending singleton
        dimensions and ``r`` is the repetitions after prepending ones.
    """
    x = array(x)
    data = backend.tile(x.data, repetitions)
    dims = [undefined_dim] * (len(repetitions) - x.ndim) + x.dims
    return array(data, dims)


def unstack(x: array, /, *, axis: int | str = 0) -> tuple[array, ...]:
    """
    Splits an array into a sequence of arrays along the given axis.

    Parameters
    ----------
    x: array
        input array.
    axis: int | str
        axis along which the array will be split. A string value is interpreted as a
        dimension name. A valid integer ``axis`` must be on the interval ``[-N, N)``,
        where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis``
        outside of the required interval, the function raises an exception.
        Default: ``0``.

    Returns
    -------
    out: tuple[array, ...]
        tuple of slices along the given dimension. All the arrays have the same shape.
    """
    x = array(x)
    if isinstance(axis, Dim):
        dim = axis
        axis = x._dims.index(axis)
    else:
        dim = x._dims[axis]
    unstacked_data = backend.unstack(x._data, axis=axis)
    dims = list(x._dims)
    dims.remove(dim)
    return tuple(array(data, dims) for data in unstacked_data)
