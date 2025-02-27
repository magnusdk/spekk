import collections
from typing import TYPE_CHECKING, Callable, List, Sequence

if TYPE_CHECKING:
    from spekk import Dim, Dims, ops

import itertools


def _index_of_first_ellipsis(objs) -> int:
    for i, obj in enumerate(objs):
        if obj is Ellipsis:
            return i
    raise IndexError()


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
        ellipsis_index = _index_of_first_ellipsis(indexing_objects)
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
    dims_not_in_data = {dim for dim in dims if dim not in data_dims}
    if len(dims_not_in_data) != 0:
        raise IndexError(
            f"Indexing dimensions {dims_not_in_data} does not exist in the data with "
            f"dimensions {data_dims}."
        )
    if len(set(dims)) != len(dims):
        raise IndexError(f"Got duplicate indexed dimensions: {dims}")

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
            for dim_inner in i.dim_sizes.keys():
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
                output_dims += idx.dims
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
    from spekk.ops._util import (
        ensure_backend_compatible_data,
        ensure_broadcastable,
        ensure_broadcastable_with,
    )

    indexing_objects = _parse_indexing_objects(x.dims, indexing_objects)

    # TODO: Clean up this hack
    # Short-circuit to NumPy broadcasting if:
    # - Any dimensions are undefined
    # - Only one indexing object is given and it has dtype=bool
    tmp_all_indexing_dims = set()
    n_arrays = 0
    n_arrays_with_dtype_bool = 0
    bool_array = None
    for indexing_object in indexing_objects:
        if isinstance(indexing_object, ops.array):
            tmp_all_indexing_dims.update(indexing_object.dims)
            n_arrays += 1
            if indexing_object.dtype == "bool":
                n_arrays_with_dtype_bool += 1
                bool_array = indexing_object
    if n_arrays == n_arrays_with_dtype_bool == 1:
        data = x.data[bool_array.data]
        return ops.array(data, [ops.undefined_dim])
    elif any(ops.is_undefined_dim(dim) for dim in tmp_all_indexing_dims):
        data = x.data.__getitem__(
            tuple(
                indexing_object.data
                if isinstance(indexing_object, ops.array)
                else indexing_object
                for indexing_object in indexing_objects
            )
        )
        return ops.array(data)

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
                    start, stop, step = i.indices(dim_size)
                    i = ops.arange(start, stop, step, dim=dim)
                    i = ensure_broadcastable_with(i, indexing_objects_dims)
                _indexing_objects_tmp.append(i)
        indexing_objects = _indexing_objects_tmp

        output_dims = _get_advanced_indexing_output_dims(x.dims, indexing_objects)
        indexing_objects = tuple(
            i.data if isinstance(i, ops.array) else i for i in indexing_objects
        )
        return ops.array(x.data.__getitem__(indexing_objects), output_dims)


def setitem(x: "ops.array", indexing_objects: tuple, value: "ops.array") -> "ops.array":
    from spekk import ops
    from spekk.ops._util import (
        ensure_backend_compatible_data,
        ensure_broadcastable,
        ensure_broadcastable_with,
    )

    # TODO: Clean up this hack
    # Short-circuit to NumPy broadcasting if:
    # - Any dimensions are undefined
    # - Only one indexing object is given and it has dtype=bool
    tmp_all_indexing_dims = set()
    n_arrays = 0
    n_arrays_with_dtype_bool = 0
    bool_array = None
    for indexing_object in indexing_objects:
        if isinstance(indexing_object, ops.array):
            tmp_all_indexing_dims.update(indexing_object.dims)
            n_arrays += 1
            if indexing_object.dtype == "bool":
                n_arrays_with_dtype_bool += 1
                bool_array = indexing_object
    if n_arrays == n_arrays_with_dtype_bool == 1:
        data = ops.backend._setitem_impl(
            x.data,
            (bool_array.data,),
            *ensure_backend_compatible_data(value),
        )
        return ops.array(data, x.dims)
    elif any(ops.is_undefined_dim(dim) for dim in tmp_all_indexing_dims):
        indexing_objects = _parse_indexing_objects(x.dims, indexing_objects)
        data = ops.backend._setitem_impl(
            x.data,
            tuple(
                indexing_object.data
                if isinstance(indexing_object, ops.array)
                else indexing_object
                for indexing_object in indexing_objects
            ),
            *ensure_backend_compatible_data(value),
        )
        return ops.array(data)

    # Calculate the union of dimensions and sizes of x, the indexing objects, and the
    # value. We order dimensions such that x's dimensions come first.
    dim_sizes = x.dim_sizes
    indexing_sizes = {}
    for indexing_object in [*indexing_objects, value]:
        if isinstance(indexing_object, ops.array):
            for dim, size in indexing_object.dim_sizes.items():
                if dim not in dim_sizes:
                    dim_sizes[dim] = size
                if dim not in indexing_sizes:
                    indexing_sizes[dim] = size

    # Ensure that x has all the dimensions. Note that this means that x may change
    # shape (become bigger!) after setitem, which differs from NumPy semantics.
    if x.dim_sizes != dim_sizes:
        x = ops.broadcast_to(
            x,
            shape=tuple(dim_sizes.values()),
            dims=tuple(dim_sizes.keys()),
        )

    indexing_objects = _parse_indexing_objects(x.dims, indexing_objects)

    indexing_objects = list(indexing_objects)
    value_dim_sizes = value.dim_sizes if isinstance(value, ops.array) else {}
    for dim, size in value_dim_sizes.items():
        i = x.dims.index(dim)
        indexing_object = indexing_objects[i]
        if isinstance(indexing_object, ops.array):
            if dim not in indexing_object.dims:
                indexing_objects[i] = ops.broadcast_to(
                    indexing_object,
                    shape=indexing_object.shape + (size,),
                    dims=indexing_object.dims + [dim],
                )
        if isinstance(indexing_object, slice):
            start, stop, step = indexing_object.indices(size)
            indexing_objects[i] = ops.arange(start, stop, step, dim=dim)
            indexing_objects[i] = ensure_broadcastable_with(
                indexing_objects[i], indexing_sizes
            )
    _, indexing_objects = ensure_broadcastable(*indexing_objects)

    is_basic_slicing = all(
        isinstance(i, (int, slice)) or i is None or i is Ellipsis
        for i in indexing_objects
    )
    output_dims = (
        _get_new_dims_basic_indexing(x.dims, indexing_objects)
        if is_basic_slicing
        else _get_advanced_indexing_output_dims(x.dims, indexing_objects)
    )
    value = ensure_broadcastable_with(value, output_dims)

    x, value, *indexing_objects = ensure_backend_compatible_data(
        x, value, *indexing_objects
    )
    data = ops.backend._setitem_impl(x, tuple(indexing_objects), value)
    return ops.array(data, list(dim_sizes.keys()))


class ArrayIndexUpdateHelper:
    def __init__(self, x: "ops.array"):
        self.x = x

    def __getitem__(self, slices):
        if isinstance(slices, dict):
            # Convert dict to the tuple syntax of interleaved dim and indexing objects.
            slices = tuple(itertools.chain(*slices.items()))
        elif not isinstance(slices, tuple):
            slices = (slices,)
        return _ArrayIndexUpdateRef(self.x, slices)


class _ArrayIndexUpdateRef:
    def __init__(self, x: "ops.array", slices: tuple):
        self.x = x
        self.slices = slices

    def get(self) -> "ops.array":
        return getitem(self.x, self.slices)

    def set(self, value: "ops.array") -> "ops.array":
        return setitem(self.x, self.slices, value)

    def update(
        self, f: Callable[["ops.array"], "ops.array"], *args, **kwargs
    ) -> "ops.array":
        value = f(self.get(), *args, **kwargs)
        return self.set(value)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
