import collections
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    from spekk import Dim, Dims, ops


def _expand_indexing_objects(n_axes: int, indexing_objects: Sequence) -> tuple:
    """Ensure that the number of indexing objects equal the number of axes in the
    indexed array by expanding the indexing_objects.

    If the indexing objects contain an Ellipse object (usually written as `...`), then
    it is replaced by the appropriate number of `slice(None)` (usually written as `:`).
    Else, if the number of indexing objects is less than the number of axes, trailing
    `slice(None)` are appended.

    >>> from spekk import ops
    >>> x = ops.zeros((3,4,5,6))
    >>> indexing_objects = (0, 1)
    >>> _expand_indexing_objects(x.ndim, indexing_objects)
    (0, 1, slice(None, None, None), slice(None, None, None))

    >>> indexing_objects = (0, ..., 1)
    >>> _expand_indexing_objects(x.ndim, indexing_objects)
    (0, slice(None, None, None), slice(None, None, None), 1)
    """
    n_ellipsis = sum(idx is Ellipsis for idx in indexing_objects)
    if n_ellipsis > 1:
        # We can only have one ... when indexing.
        raise IndexError("An index can only have a single ellipsis (...).")
    elif n_ellipsis == 1:
        # Replace ... by slices, i.e.: x[0, ..., 1] becomes x[0, :, :, 1] if x.ndim==4.
        ellipsis_index = indexing_objects.index(Ellipsis)
        num_explicit = len(indexing_objects) - 1
        n_missing = n_axes - num_explicit
        indexing_objects = (
            indexing_objects[:ellipsis_index]
            + (slice(None),) * n_missing
            + indexing_objects[ellipsis_index + 1 :]
        )
    else:
        # Add trailing slices, i.e.: x[:, 0] becomes x[:, 0, :, :], if x.ndim==4.
        if len(indexing_objects) < n_axes:
            indexing_objects = indexing_objects + (slice(None),) * (
                n_axes - len(indexing_objects)
            )
    return indexing_objects


def _validate_and_handle_indexing_with_undefined_dim_arrays(
    dims: "Dims",
    indexing_objects: Sequence,
) -> tuple:
    """Replace undefined dimensions with an unambiguous dimension name, if possible. If
    not possible, an IndexError is raised.

    This allows indexing a specific axis by a 1-D array with an undefined dimension
    name. Then we assume that the indexing array has the same dimension name as the
    indexed axis. See example below:

    >>> from spekk import ops
    >>> dims = ["a", "b"]
    >>> indexing_objects = (0, ops.arange(2))
    >>> _validate_and_handle_indexing_with_undefined_dim_arrays(dims, indexing_objects)
    (0, array(shape=(2,), dims=['b'], dtype=_DType('int64'), data=[0 1]))

    Notice that the output array now has dims=['b']
    """
    from spekk import ops
    from spekk.ops._types import _UndefinedDim

    new_indexing_objects = []
    for dim, indexing_object in zip(dims, indexing_objects):
        if isinstance(indexing_object, ops.array) and any(
            isinstance(d, _UndefinedDim) for d in indexing_object.dims
        ):
            if indexing_object.ndim == 1:
                indexing_object = ops.array(indexing_object.data, [dim])
            else:
                raise IndexError(
                    "Indexing by an array with more than one undefined dim is not "
                    "supported because the resulting dimensions are ambiguous. Got "
                    f"dimensions {indexing_object.dims}."
                )
        new_indexing_objects.append(indexing_object)
    return tuple(new_indexing_objects)


def _parse_indexing_objects_by_dim(data_dims: "Dims", indexing_objects: tuple) -> tuple:
    """Parse indexing objects when indexing by dimension names. The output is a tuple
    with an indexing object for each axis (i.e.: no longer indexed by dimension name).

    >>> from spekk import ops
    >>> dims = ["a", "b", "c"]
    >>> indexing_objects = ("c", 0, "b", slice(5))
    >>> _parse_indexing_objects_by_dim(dims, indexing_objects)
    (slice(None, None, None), slice(None, 5, None), 0)
    """
    from spekk import Dim, ops

    if len(indexing_objects) % 2 != 0:
        raise IndexError(
            "You must provide an even number of indexing objects when indexing by "
            "dimension names."
        )

    # Assume indexing by dimension names
    dims = indexing_objects[::2]
    indexing_objects = indexing_objects[1::2]

    if not all(isinstance(dim, Dim) for dim in dims):
        raise IndexError(
            "Every other element of slices, starting from the first, must be a "
            "dimension name."
        )
    if not all(isinstance(d, (ops.array, slice, int)) for d in indexing_objects[1::2]):
        raise IndexError(
            "Every other element of slices, starting from the second, must be an "
            "index-like object."
        )

    _idx_dict = dict(zip(dims, indexing_objects))
    indexing_objects = tuple(_idx_dict.get(dim, slice(None)) for dim in data_dims)
    return indexing_objects


def _parse_indexing_objects(data_dims: "Dims", indexing_objects: tuple) -> tuple:
    from spekk import Dim, ops

    if len(indexing_objects) == 0:
        raise IndexError("You must index at least one axis.")
    if any(isinstance(dim, Dim) for dim in indexing_objects):
        indexing_objects = _parse_indexing_objects_by_dim(data_dims, indexing_objects)

    # Handle Ellipsis (aka ...) and the case where not all axes are explicitly given,
    # e.g.: arr with shape (2,3) that were indexed like arr[0] (contra indexed as
    # arr[0, :]).
    indexing_objects = _expand_indexing_objects(len(data_dims), indexing_objects)

    # Handle indexing with arrays with an undefined dimension. This means that we can
    # perform indexing like this:
    # x = ops.zeros((3, 4, 5), ["a", "b", "c"])
    # i = ops.array([0, 2])
    # x["c", i].dim_sizes == {"a": 3, "b": 4, "c": 2}
    indexing_objects = _validate_and_handle_indexing_with_undefined_dim_arrays(
        data_dims, indexing_objects
    )

    # Handle invalid indexing scenarios
    # Scenario 1.
    indexing_objects_dict = dict(zip(data_dims, indexing_objects))
    for dim_outer, i in zip(data_dims, indexing_objects):
        if isinstance(i, ops.array):
            for dim_inner in i.dim_sizes:
                if (
                    dim_inner in indexing_objects_dict
                    and dim_inner != dim_outer
                    and (
                        not isinstance(indexing_objects_dict[dim_inner], slice)
                        or indexing_objects_dict[dim_inner] != slice(None)
                    )
                ):
                    raise IndexError("TODO FOO")

    # Scenario 2.
    indexing_objects_sizes = collections.defaultdict(set)
    for i in indexing_objects:
        if isinstance(i, ops.array):
            for dim, size in i.dim_sizes.items():
                indexing_objects_sizes[dim].add(size)
    for dim, size in indexing_objects_sizes.items():
        if len(size) > 1:
            raise IndexError(
                f"Got inconsistent sizes for dimension '{dim}'. Sizes: "
                f"{indexing_objects_sizes[dim]}."
            )

    return indexing_objects


def _get_new_dims_basic_indexing(
    dims: List["Dim"],
    indexing_objects: tuple,
) -> List["Dim"]:
    if not indexing_objects:
        return dims

    current_dim, *remaining_dims = dims
    current_indexing_object, *remaining_indexing_objects = indexing_objects
    if isinstance(current_indexing_object, int):
        return _get_new_dims_basic_indexing(remaining_dims, remaining_indexing_objects)
    elif current_indexing_object is None:
        raise NotImplementedError(
            "Creating new axes using None is not supported in spekk. Use "
            "ops.expand_dims instead."
        )
    else:
        return [
            current_dim,
            *_get_new_dims_basic_indexing(remaining_dims, remaining_indexing_objects),
        ]


def _get_advanced_indexing_output_dims(
    dims: "Dims", indexing_objects: Sequence
) -> "Dims":
    from spekk import ops

    # Identify positions where advanced indexing (i.e. an ops.array) is used.
    advanced_positions = [
        i for i, idx in enumerate(indexing_objects) if isinstance(idx, (ops.array, int))
    ]

    # If there are advanced indices and they are non-contiguous,
    # then we are in case 1.
    if advanced_positions and (
        max(advanced_positions) - min(advanced_positions) + 1 != len(advanced_positions)
    ):
        # Case 1: non-contiguous advanced indices.
        # Collect all dimensions from advanced indices (in order)
        advanced_dims = []
        # And collect all dimensions that come from basic indexing *and are not dropped*
        basic_dims = []
        for i, idx in enumerate(indexing_objects):
            if isinstance(idx, ops.array):
                # Here we assume that each advanced index carries its own dims (which might be more than one)
                advanced_dims += idx.dims
            elif isinstance(idx, int):
                continue
            else:
                basic_dims.append(dims[i])
        output_dims = advanced_dims + basic_dims

    else:
        # Case 2: either no advanced indices or they are contiguous.
        # In this case, we “insert” the dimensions as they appear.
        output_dims = []
        for i, idx in enumerate(indexing_objects):
            if isinstance(idx, ops.array):
                output_dims.extend(idx.dims)
            elif isinstance(idx, int):
                continue
            else:
                output_dims.append(dims[i])

    # Remove duplicates
    unique_output_dims = []
    seen = set()
    for dim in output_dims:
        if dim not in seen:
            seen.add(dim)
            unique_output_dims.append(dim)
    return unique_output_dims


def getitem(x: "ops.array", indexing_objects: tuple) -> "ops.array":
    from spekk import ops
    from spekk.ops._util import ensure_broadcastable, ensure_broadcastable_with

    indexing_objects = _parse_indexing_objects(x.dims, indexing_objects)
    is_basic_slicing = all(
        isinstance(i, (int, slice)) or i is None or i is Ellipsis
        for i in indexing_objects
    )

    if is_basic_slicing:
        new_dims = _get_new_dims_basic_indexing(x.dims, indexing_objects)
        return ops.array(x.data.__getitem__(indexing_objects), new_dims)
    else:
        indexing_objects_dims, indexing_objects = ensure_broadcastable(
            *indexing_objects
        )

        _indexing_objects_tmp = []
        for (dim, dim_size), i in zip(x.dim_sizes.items(), indexing_objects):
            if isinstance(i, ops.array):
                _indexing_objects_tmp.append(i)
            else:
                if isinstance(i, slice) and dim in indexing_objects_dims:
                    i = ops.arange(
                        i.start if i.start is not None else 0,
                        i.stop if i.stop is not None else dim_size,
                        i.step if i.step is not None else 1,
                        dim=dim,
                    )
                    i = ensure_broadcastable_with(i, indexing_objects_dims)
                _indexing_objects_tmp.append(i)
        indexing_objects = _indexing_objects_tmp

        output_dims = _get_advanced_indexing_output_dims(x.dims, indexing_objects)
        indexing_objects = tuple(
            i.data if isinstance(i, ops.array) else i for i in indexing_objects
        )
        return ops.array(x.data.__getitem__(indexing_objects), output_dims)


# TODO: BELOW THIS LINE IS OLD CODE WHICH DOESN'T WORK
######################################################


def _parse_and_validate_slices(
    slices: tuple, data_dims: "ops.array"
) -> Tuple[Dict["Dim", Union["ops.array", slice, int]], "Dims"]:
    from spekk import Dim, ops
    from spekk.ops._util import ensure_broadcastable

    if not slices:
        raise ValueError("Must supply at least one (Dim, ops.array) pair.")
    if len(slices) % 2 != 0:
        raise ValueError(f"slices must be an even-length tuple, got {slices}.")
    if not all(isinstance(d, Dim) for d in slices[::2]):
        raise ValueError(
            "Every other element of slices, starting from the first, must be a Dim, "
            f"got {slices}."
        )
    if not all(isinstance(d, (ops.array, slice, int)) for d in slices[1::2]):
        raise ValueError(
            "Every other element of slices, starting from the second, must be an "
            f"index-like object, got {slices}."
        )

    dims = slices[::2]
    for dim in dims:
        if dim not in data_dims:
            raise ValueError(
                f"Attempting to index at dim {dim} which does not exist in x with dims "
                f"{data_dims}."
            )
    slices_dims, values = ensure_broadcastable(*slices[1::2])
    slices_dict = dict(zip(dims, values))
    for dim, i in slices_dict.items():
        if isinstance(i, slice):
            slices_dims.append(dim)

    for dim in slices_dims:
        if (
            dim in slices_dict
            and isinstance(slices_dict[dim], ops.array)
            and dim not in slices_dict[dim].dims
        ):
            raise ValueError("TODO: Help me figure out a good error message")

    return slices_dict, slices_dims


def _ensure_correct_mixed_indexing_output_dims(
    indices_per_axis: Dict["Dim", Union["ops.array", slice, int]],
    broadcasted_dims: "Dims",
) -> "Dims":
    """Return the output dimensions after indexing an array when mixing basic and
    advanced Numpy indexing.

    This is a bit finnicky and we're digging into details of NumPy array indexing. When
    we perform indexing that is a mix between arrays and "basic" indexing objects,
    figuring out the output shape is not trivial. If an axis uses slice indexing and
    there is a previous axis that does not use slice indexing, the slice dimensions are
    placed at the back of the output shape. Else they are placed in the front of the
    output shape. 🤷‍♂️

    NumPy docs: https://numpy.org/doc/2.2/user/basics.indexing.html#combining-advanced-and-basic-indexing
    """
    extra_dims = []
    encountered_non_slice = False
    place_extra_dims_in_back = False
    for dim, i in indices_per_axis.items():
        if isinstance(i, slice):
            if encountered_non_slice:
                # If we see a slice after we have seen a non-slice indexing object, the
                # extra dimensions — those with slice(None) indexing objects — should
                # be placed in the back (according to the Numpy documentaion.)
                place_extra_dims_in_back = True
            if i == slice(None) or dim not in broadcasted_dims:
                extra_dims.append(dim)
        else:
            encountered_non_slice = True

    if place_extra_dims_in_back:
        return broadcasted_dims + extra_dims
    else:
        return extra_dims + broadcasted_dims


def _finalize_indices(
    slices: Dict["Dim", Union["ops.array", slice, int]],
    output_dims: "Dims",
    x: "ops.array",
) -> Tuple[Dict["Dim", Union["ops.array", slice, int]], "Dims"]:
    from spekk import ops
    from spekk.ops._util import ensure_broadcastable_with

    # Get the index objects (in the correct order according to x.dims).
    _none = object()
    indices = {dim: _none for dim in x.dims}
    for dim, i in slices.items():
        indices[dim] = i
    for dim in output_dims:
        # Dimensions that are part of the indices and part of x needs to have a
        # broadcastable arange of values for the advanced indexing to be correct.
        if dim in x.dims and dim not in slices:
            indices[dim] = ensure_broadcastable_with(
                ops.arange(x.dim_sizes(dim), dim=dim), output_dims
            )
    indices = {dim: (slice(None) if i is _none else i) for dim, i in indices.items()}
    output_dims = _ensure_correct_mixed_indexing_output_dims(indices, output_dims)
    return indices, output_dims


def setitem(x: "ops.array", slices: tuple, value: "ops.array") -> "ops.array":
    from spekk import ops
    from spekk.ops._util import (
        ensure_broadcastable,
        ensure_broadcastable_with,
        get_broadcast_array_fn,
    )

    slices, slices_dims = _parse_and_validate_slices(slices, x.dims)

    # Give x any missing dimensions that are present in value and the slices.
    broadcast_array = get_broadcast_array_fn(
        x, value, *[i for i in slices.values() if isinstance(i, ops.array)]
    )
    x = broadcast_array(x)

    # Make value and slices broadcastable with each other.
    value = ensure_broadcastable_with(value, x.dims)
    output_dims, (value, *slice_values) = ensure_broadcastable(value, *slices.values())
    slices = {dim: i for dim, i in zip(slices.keys(), slice_values)}

    indices, output_dims = _finalize_indices(slices, output_dims, x)

    indices = tuple(i.data if isinstance(i, ops.array) else i for i in indices.values())
    data = ops.backend._setitem_impl(x.data, indices, value.data)
    return ops.array(data, output_dims)


class ArrayIndexUpdateHelper:
    def __init__(self, x: "ops.array"):
        self.x = x

    def __getitem__(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        return _ArrayIndexUpdateRef(self.x, slices)


class _ArrayIndexUpdateRef:
    def __init__(self, x: "ops.array", slices: tuple):
        self.x = x
        self.slices = slices

    def get(self) -> "ops.array":
        return getitem(self.x, self.slices)

    def set(self, value: "ops.array") -> "ops.array":
        raise NotImplementedError()

    def update(
        self, f: Callable[["ops.array"], "ops.array"], *args, **kwargs
    ) -> "ops.array":
        value = f(self.get(), *args, **kwargs)
        return self.set(value)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
