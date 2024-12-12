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

from collections import defaultdict

from spekk.ops._backend import backend
from spekk.ops._types import (
    ArrayLike,
    Dim,
    Dims,
    List,
    Optional,
    Tuple,
    UndefinedDim,
    Union,
)
from spekk.ops._util import ensure_array
from spekk.ops.array_object import array


def broadcast_arrays(*arrays: array) -> List[array]:
    """
    Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays: array
        an arbitrary number of to-be broadcasted arrays.

    Returns
    -------
    out: List[array]
        a list of broadcasted arrays. Each array must have the same shape. Each array must have the same dtype as its corresponding input array.
    """
    arrays = [ensure_array(x) for x in arrays]
    # TODO: How to handle UndefinedDim?
    if any(isinstance(dim, UndefinedDim) for x in arrays for dim in x._dims):
        raise NotImplementedError()

    # Get the output dimension list of each array after broadcasting. Ordering of output dimensions are determined by the ordering of input arrays and their dimensions.
    output_dims = defaultdict(set)
    for arr in arrays:
        for size, dim in zip(arr.shape, arr._dims):
            if dim not in output_dims:
                output_dims[dim].add(size)

    output_dims = {dim: sizes.pop() for dim, sizes in output_dims.items()}

    # Expand dimensions if needed (using reshape) and place dimensions in the correct order using permute_dims.
    resulting_arrays = []
    for arr in arrays:
        # Find the dimensions that need to be added to the array in order to be broadcastable with all other arrays. missing_dims is a dictionary from the name of the dimension to the size of that dimension.
        missing_dims = {
            dim: size for dim, size in output_dims.items() if dim not in arr._dims
        }

        # Add the new dimensions to the array's data, making it broadcastable.
        new_shape_broadcastable = arr._data.shape + (1,) * len(missing_dims)
        new_data = backend.reshape(arr._data, new_shape_broadcastable)

        # Add the correct sizes to the corresponding dimensions
        new_shape = (*arr._data.shape, *missing_dims.values())
        new_data = backend.broadcast_to(new_data, new_shape)
        # Also add them to the dimensions list
        new_dims = [*arr._dims, *missing_dims.keys()]

        # Put the dimensions in the correct order (same as all other arrays)
        arr = permute_dims(array(new_data, new_dims), list(output_dims.keys()))
        resulting_arrays.append(arr)

    return resulting_arrays


def broadcast_to(
    x: array, /, shape: Tuple[int, ...], dims: Optional[Dims] = None
) -> array:
    """
    Broadcasts an array to a specified shape.

    Parameters
    ----------
    x: array
        array to broadcast.
    shape: Tuple[int, ...]
        array shape. Must be compatible with ``x`` (see :ref:`broadcasting`). If the array is incompatible with the specified shape, the function should raise an exception.
    dims: Sequence[Hashable]
        the new list of dimension names. It must have the same length as shape.

    Returns
    -------
    out: array
        an array having a specified shape. Must have the same data type as ``x``.

    Raises
    ------
    ValueError
        If shape and dims do not contain the same number of arguments, a "ValueError" is raised.
    """
    x = ensure_array(x)
    if dims is not None and len(dims) != len(shape):
        raise ValueError(
            "The number of dimensions must equal the number of axes when broadcasting."
        )
    return array(backend.broadcast_to(x, shape), dims)


def concat(
    arrays: Union[Tuple[array, ...], List[array]],
    /,
    *,
    axis: Dim,
) -> array:
    """
    Joins a sequence of arrays along an existing axis.

    Parameters
    ----------
    arrays: Union[Tuple[array, ...], List[array]]
        input arrays to join.
    dim: Dim
        dimension along which the arrays will be joined.

    Returns
    -------
    out: array
        an output array containing the concatenated values. If the input arrays have different data types, normal :ref:`type-promotion` must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.

        .. note::
           This specification leaves type promotion between data type families (i.e., ``intxx`` and ``floatxx``) unspecified.
    """
    arrays = broadcast_arrays(*arrays)
    if isinstance(axis, Dim):
        axis = arrays[0]._dims.index(axis)
    elif isinstance(axis, tuple):
        axis = tuple(arrays[0]._dims.index(dim1) for dim1 in axis)
    dims = arrays[0]._dims
    data = backend.concat([arr._data for arr in arrays], axis=dims.index(axis))
    return array(data, dims)


def expand_dims(x: array, /, *, axis: Union[Dim, int] = 0) -> array:
    """
    Expands the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``.

    Parameters
    ----------
    x: array
        input array.
    dim: Dim
        the name of the new dimension being added.
    axis: int
        axis position (zero-based). If ``x`` has rank (i.e, number of dimensions) ``N``, a valid ``axis`` must reside on the closed-interval ``[-N-1, N]``. If provided a negative ``axis``, the axis position at which to insert a singleton dimension must be computed as ``N + axis + 1``. Hence, if provided ``-1``, the resolved axis position must be ``N`` (i.e., a singleton dimension must be appended to the input array ``x``). If provided ``-N-1``, the resolved axis position must be ``0`` (i.e., a singleton dimension must be prepended to the input array ``x``).

    Returns
    -------
    out: array
        an expanded output array having the same data type as ``x``.

    Raises
    ------
    IndexError
        If provided an invalid ``axis`` position, an ``IndexError`` should be raised.
    ValueError
        If the new dimension already exists.
    """
    x = ensure_array(x)

    if isinstance(axis, Dim):
        dim = axis
        axis = 0
    else:
        dim = UndefinedDim()
    data = backend.expand_dims(x._data, axis)
    dims = list(x._dims)
    dims.insert(axis, dim)
    return array(data, dims)


def flip(x: array, /, *, axis: Optional[Union[Dim, Tuple[Dim, ...]]] = None) -> array:
    """
    Reverses the order of elements in an array along the given axis. The shape of the array must be preserved.

    Parameters
    ----------
    x: array
        input array.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis (or axes) along which to flip. If ``axis`` is ``None``, the function must flip all input array axes. If ``axis`` is negative, the function must count from the last dimension. If provided more than one axis, the function must flip only the specified axes. Default: ``None``.

    Returns
    -------
    out: array
        an output array having the same data type and shape as ``x`` and whose elements, relative to ``x``, are reordered.
    """
    x = ensure_array(x)
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    elif isinstance(axis, tuple):
        axis = tuple(x._dims.index(dim1) for dim1 in axis)
    return array(backend.flip(x._data, axis=axis), x._dims)


def moveaxis(
    x: array,
    source: Union[Dim, Tuple[Dim, ...]],
    destination: Union[Dim, Tuple[Dim, ...]],
    /,
) -> array:
    """
    Moves array axes (dimensions) to new positions, while leaving other axes in their original positions.

    Parameters
    ----------
    x: array
        input array.
    source: Union[int, Tuple[int, ...]]
        Axes to move. Provided axes must be unique. If ``x`` has rank (i.e, number of dimensions) ``N``, a valid axis must reside on the half-open interval ``[-N, N)``.
    destination: Union[int, Tuple[int, ...]]
        indices defining the desired positions for each respective ``source`` axis index. Provided indices must be unique. If ``x`` has rank (i.e, number of dimensions) ``N``, a valid axis must reside on the half-open interval ``[-N, N)``.

    Returns
    -------
    out: array
        an array containing reordered axes. The returned array must have the same data type as ``x``.

    Notes
    -----

    .. versionadded:: 2023.12
    """
    x = ensure_array(x)

    # Ensure a tuple of sources and a tuple of destinations
    if not isinstance(source, tuple):
        source = (source,)
    if not isinstance(destination, tuple):
        destination = (destination,)

    # Convert all to integers (axes)
    source = tuple(x._dims.index(d) if isinstance(d, Dim) else d for d in source)
    destination = tuple(
        x._dims.index(d) if isinstance(d, Dim) else d for d in destination
    )

    data = backend.moveaxis(x._data, source, destination)
    dims = list(x._dims)
    for src, dest in zip(source, destination):
        dims.remove(x._dims[src])
        dims.insert(dest, x._dims[src])
    return array(data, dims)


def permute_dims(x: array, /, axes: Tuple[Dim, ...]) -> array:
    """
    Permutes the axes (dimensions) of an array ``x``.

    Parameters
    ----------
    x: array
        input array.
    axes: Tuple[int, ...]
        tuple containing a permutation of ``(0, 1, ..., N-1)`` where ``N`` is the number of axes (dimensions) of ``x``.

    Returns
    -------
    out: array
        an array containing the axes permutation. The returned array must have the same data type as ``x``.
    """
    x = ensure_array(x)
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
    axis: Optional[Union[Dim, int]] = None,
) -> array:
    """
    Repeats each element of an array a specified number of times on a per-element basis.

    .. admonition:: Data-dependent output shape
        :class: important

        When ``repeats`` is an array, the shape of the output array for this function depends on the data values in the ``repeats`` array; hence, array libraries which build computation graphs (e.g., JAX, Dask, etc.) may find this function difficult to implement without knowing the values in ``repeats``. Accordingly, such libraries may choose to omit support for ``repeats`` arrays; however, conforming implementations must support providing a literal ``int``. See :ref:`data-dependent-output-shapes` section for more details.

    Parameters
    ----------
    x: array
        input array containing elements to repeat.
    repeats: Union[int, array]
        the number of repetitions for each element.

        If ``axis`` is ``None``, let ``N = prod(x.shape)`` and

        -   if ``repeats`` is an array, ``repeats`` must be broadcast compatible with the shape ``(N,)`` (i.e., be a one-dimensional array having shape ``(1,)`` or ``(N,)``).
        -   if ``repeats`` is an integer, ``repeats`` must be broadcasted to the shape `(N,)`.

        If ``axis`` is not ``None``, let ``M = x.shape[axis]`` and

        -   if ``repeats`` is an array, ``repeats`` must be broadcast compatible with the shape ``(M,)`` (i.e., be a one-dimensional array having shape ``(1,)`` or ``(M,)``).
        -   if ``repeats`` is an integer, ``repeats`` must be broadcasted to the shape ``(M,)``.

        If ``repeats`` is an array, the array must have an integer data type.

        .. note::
           For specification-conforming array libraries supporting hardware acceleration, providing an array for ``repeats`` may cause device synchronization due to an unknown output shape. For those array libraries where synchronization concerns are applicable, conforming array libraries are advised to include a warning in their documentation regarding potential performance degradation when ``repeats`` is an array.

    axis: Optional[int]
        the axis (dimension) along which to repeat elements. If ``axis`` is `None`, the function must flatten the input array ``x`` and then repeat elements of the flattened input array and return the result as a one-dimensional output array. A flattened input array must be flattened in row-major, C-style order. Default: ``None``.

    Returns
    -------
    out: array
        an output array containing repeated elements. The returned array must have the same data type as ``x``. If ``axis`` is ``None``, the returned array must be a one-dimensional array; otherwise, the returned array must have the same shape as ``x``, except for the axis (dimension) along which elements were repeated.

    Notes
    -----

    .. versionadded:: 2023.12
    """
    x = ensure_array(x)
    # How do we implement repeat with named dimensions?
    raise NotImplementedError()
    if not isinstance(repeats, int):
        raise NotImplementedError()
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
    data = backend.repeat(x._data, repeats=repeats, axis=axis)
    return array(data, x._dims)


def reshape(
    x: array,
    /,
    shape: Tuple[int, ...],
    dims: Optional[Dims] = None,
    *,
    copy: Optional[bool] = None,
) -> array:
    """
    Reshapes an array without changing its data.

    Parameters
    ----------
    x: array
        input array to reshape.
    shape: Tuple[int, ...]
        a new shape compatible with the original shape. One shape dimension is allowed to be ``-1``. When a shape dimension is ``-1``, the corresponding output array shape dimension must be inferred from the length of the array and the remaining dimensions.
    dims: Sequence[Hashable]
        the new list of dimension names. It must have the same length as shape.
    copy: Optional[bool]
        whether or not to copy the input array. If ``True``, the function must always copy. If ``False``, the function must never copy. If ``None``, the function must avoid copying, if possible, and may copy otherwise. Default: ``None``.

    Returns
    -------
    out: array
        an output array having the same data type and elements as ``x``.

    Raises
    ------
    ValueError
        If ``copy=False`` and a copy would be necessary, a ``ValueError``
        should be raised. Also, if shape and dims do not contain the same number of
        arguments, a "ValueError" is raised.
    """
    x = ensure_array(x)
    if len(dims) != len(shape):
        raise ValueError(
            "The number of dimensions must equal the number of axes when reshaping."
        )
    if dims is None:
        dims = [UndefinedDim()] * len(shape)
    return array(backend.reshape(x._data, shape, copy=copy), dims)


def roll(
    x: array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[Dim, Tuple[Dim, ...]]] = None,
) -> array:
    """
    Rolls array elements along a specified axis. Array elements that roll beyond the last position are re-introduced at the first position. Array elements that roll beyond the first position are re-introduced at the last position.

    Parameters
    ----------
    x: array
        input array.
    shift: Union[int, Tuple[int, ...]]
        number of places by which the elements are shifted. If ``shift`` is a tuple, then ``axis`` must be a tuple of the same size, and each of the given axes must be shifted by the corresponding element in ``shift``. If ``shift`` is an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for all specified axes. If a shift is positive, then array elements must be shifted positively (toward larger indices) along the dimension of ``axis``. If a shift is negative, then array elements must be shifted negatively (toward smaller indices) along the dimension of ``axis``.
    axis: Optional[Union[int, Tuple[int, ...]]]
        axis (or axes) along which elements to shift. If ``axis`` is ``None``, the array must be flattened, shifted, and then restored to its original shape. Default: ``None``.

    Returns
    -------
    out: array
        an output array having the same data type as ``x`` and whose elements, relative to ``x``, are shifted.
    """
    x = ensure_array(x)
    axis = (
        x._dims.index(axis)
        if isinstance(axis, Dim)
        else tuple(x._dims.index(dim1) for dim1 in axis)
    )
    return array(backend.roll(x._data, shift=shift, axis=axis), x._dims)


def squeeze(
    x: ArrayLike,
    /,
    axis: Union[Dim, Tuple[Dim, ...], int, Tuple[int, ...]],
) -> array:
    """
    Removes singleton dimensions (axes) from ``x``.

    Parameters
    ----------
    x: array
        input array.
    axis: Union[int, Tuple[int, ...]]
        axis (or axes) to squeeze.

    Returns
    -------
    out: array
        an output array having the same data type and elements as ``x``.

    Raises
    ------
    ValueError
        If a specified axis has a size greater than one (i.e., it is not a
        singleton dimension), a ``ValueError`` should be raised.
    """
    x = ensure_array(x)

    # Handle changes to dimensions and get actual axis (integer)
    dims = list(x._dims)
    if isinstance(axis, Dim):
        axis = x._dims.index(axis)
        del dims[axis]
    elif isinstance(axis, int):
        del dims[axis]
    elif isinstance(axis, tuple):
        axis = [x._dims.index(i) if isinstance(i, Dim) else i for i in axis]
        for i in axis:
            del dims[i]

    data = backend.squeeze(x._data, axis)
    return array(data, dims)


def stack(
    arrays: Union[Tuple[array, ...], List[array]],
    /,
    *,
    axis: Union[int, Dim] = 0,
) -> array:
    """
    Joins a sequence of arrays along a new axis.

    Parameters
    ----------
    arrays: Union[Tuple[array, ...], List[array]]
        input arrays to join. Each array must have the same shape.
    axis: int
        axis along which the arrays will be joined. Providing an ``axis`` specifies the index of the new axis in the dimensions of the result. For example, if ``axis`` is ``0``, the new axis will be the first dimension and the output array will have shape ``(N, A, B, C)``; if ``axis`` is ``1``, the new axis will be the second dimension and the output array will have shape ``(A, N, B, C)``; and, if ``axis`` is ``-1``, the new axis will be the last dimension and the output array will have shape ``(A, B, C, N)``. A valid ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis`` outside of the required interval, the function must raise an exception. Default: ``0``.

    Returns
    -------
    out: array
        an output array having rank ``N+1``, where ``N`` is the rank (number of dimensions) of ``x``. If the input arrays have different data types, normal :ref:`type-promotion` must apply. If the input arrays have the same data type, the output array must have the same data type as the input arrays.

        .. note::
           This specification leaves type promotion between data type families (i.e., ``intxx`` and ``floatxx``) unspecified.
    """
    arrays = broadcast_arrays(*arrays)

    if isinstance(axis, Dim):
        dim = axis
        axis = 0
    else:
        dim = UndefinedDim()
    data = backend.stack([arr._data for arr in arrays], axis=axis)
    dims = list(arrays[0]._dims)
    if axis < 0:
        # Handle negative index for list.insert. We have to add an additional 1 to the
        # axis, otherwise -1 refers to the second-to-last position when it should be
        # the last position.
        axis += len(dims) + 1
    dims.insert(axis, dim)
    return array(data, dims)


def tile(x: array, repetitions: Tuple[int, ...], /) -> array:
    """
    Constructs an array by tiling an input array.

    Parameters
    ----------
    x: array
        input array.
    repetitions: Tuple[int, ...]
        number of repetitions along each axis (dimension).

        Let ``N = len(x.shape)`` and ``M = len(repetitions)``.

        If ``N > M``, the function must prepend ones until all axes (dimensions) are specified (e.g., if ``x`` has shape ``(8,6,4,2)`` and ``repetitions`` is the tuple ``(3,3)``, then ``repetitions`` must be treated as ``(1,1,3,3)``).

        If ``N < M``, the function must prepend singleton axes (dimensions) to ``x`` until ``x`` has as many axes (dimensions) as ``repetitions`` specifies (e.g., if ``x`` has shape ``(4,2)`` and ``repetitions`` is the tuple ``(3,3,3,3)``, then ``x`` must be treated as if it has shape ``(1,1,4,2)``).

    Returns
    -------
    out: array
        a tiled output array. The returned array must have the same data type as ``x`` and must have a rank (i.e., number of dimensions) equal to ``max(N, M)``. If ``S`` is the shape of the tiled array after prepending singleton dimensions (if necessary) and ``r`` is the tuple of repetitions after prepending ones (if necessary), then the number of elements along each axis (dimension) must satisfy ``S[i]*r[i]``, where ``i`` refers to the ``i`` th axis (dimension).

    Notes
    -----

    .. versionadded:: 2023.12
    """
    x = ensure_array(x)
    # How do we implement tile with named dimensions?
    raise NotImplementedError()


def unstack(x: array, /, *, axis: Union[int, Dim]) -> Tuple[array, ...]:
    """
    Splits an array into a sequence of arrays along the given axis.

    Parameters
    ----------
    x: array
        input array.
    axis: int
        axis along which the array will be split. A valid ``axis`` must be on the interval ``[-N, N)``, where ``N`` is the rank (number of dimensions) of ``x``. If provided an ``axis`` outside of the required interval, the function must raise an exception. Default: ``0``.

    Returns
    -------
    out: Tuple[array, ...]
        tuple of slices along the given dimension. All the arrays have the same shape.

    Notes
    -----

    .. versionadded:: 2023.12
    """
    x = ensure_array(x)
    if isinstance(axis, Dim):
        dim = axis
        axis = x._dims.index(axis)
    else:
        dim = x._dims[axis]
    unstacked_data = backend.unstack(x._data, axis)
    dims = list(x._dims)
    dims.remove(dim)
    return tuple(array(data, dims) for data in unstacked_data)
