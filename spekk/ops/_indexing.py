import abc
from collections import Counter
from dataclasses import dataclass, field, replace
from types import EllipsisType
from typing import Any, Callable, Optional, Sequence

from spekk import Dim, ops, util
from spekk.ops._types import _UndefinedDim
from spekk.ops._util import (
    ensure_backend_compatible_data,
    ensure_broadcastable_with,
    get_broadcast_array_fn,
)


# Marker class and singleton for representing axes that hasn't been explicitly indexed
# by the user, but may have been expanded while processing the indexing object.
class _NotExplicitlyIndexed: ...


_NOT_EXPLICITLY_INDEXED = _NotExplicitlyIndexed()


@dataclass
class IndexingBehavior:
    """Object for configuring the behavior of an indexing operation.

    For example, indexing on a Module object is different from indexing just an array.
    When indexing on a Module, we're really processing it as a tree of arrays and
    indexing each array individually. Not all arrays may have the dimension(s) being
    indexed, and in spekk this means that we skip indexing that dimension and only
    index the dimensions that exist on the array."""

    # only_index_dims is used to filter the dimensions indexed. This is only useful
    # when we are indexing all the leaves of a pytree of arrays and not all the arrays
    # have the indexed dimensions. Then the rule is to only index the arrays that have
    # the explicitly indexed dimensions.
    only_index_dims: Sequence[Dim] | None = None

    def apply_only_keep_dims_behavior[T: tuple](
        self, wrapped_indexing_object: "MaybeIndexingByDimsNotInArr[T]"
    ) -> "MaybeIndexingByDimsNotInArr[T]":
        if self.only_index_dims is not None:
            indexing_object = wrapped_indexing_object.indexing_object
            new_key = [
                index if dim in self.only_index_dims else _NOT_EXPLICITLY_INDEXED
                for dim, index in zip(indexing_object.arr.dims, indexing_object.key)
            ]
            indexing_object = replace(indexing_object, key=new_key)
            new_indexed_dims_not_in_arr = {
                dim: index
                for dim, index in wrapped_indexing_object.indexed_dims_not_in_arr.items()
                if dim in self.only_index_dims
            }
            wrapped_indexing_object = replace(
                wrapped_indexing_object,
                indexing_object=indexing_object,
                indexed_dims_not_in_arr=new_indexed_dims_not_in_arr,
            )
        return wrapped_indexing_object


@dataclass
class IndexingObject[T]:
    arr: ops.array
    key: T
    value: Optional[ops.array] = None
    new_positional_axes: list[int] = field(default_factory=list)
    new_named_dimensions: list[Dim] = field(default_factory=list)
    temporary_dims: set[Dim] = field(default_factory=set)
    indexing_behavior: IndexingBehavior = field(default_factory=IndexingBehavior)


@dataclass
class MaybeIndexingByDimsNotInArr[T]:
    indexing_object: IndexingObject[T]
    indexed_dims_not_in_arr: dict[Dim, T] = field(default_factory=dict)


def _validate_dim_index_pairs(dims: Sequence[Any], key: Sequence[Any]):
    for dim, num in Counter(dims).items():
        if num > 1:
            raise IndexError(
                f"Dimension {dim} was referenced more than once. When indexing by "
                "dimension-index pairs, all referenced dimensions must be unique."
            )
        if len(dims) != len(key):
            raise IndexError(
                f"When indexing by dimension-index pairs, the number of dimensions "
                f"must equal the number of indices. Got {len(dims)} dimensions and "
                f"{len(key)} indices."
            )
        if not all(isinstance(dim, Dim) for dim in dims):
            raise IndexError(
                "When indexing by dimension-index pairs, every other item must be a "
                "Dim object. Expected pattern: (Dim, index, Dim, index, ...). "
                "Non-Dim objects found at even positions: "
                f"{[obj for i, obj in enumerate(dims) if not isinstance(obj, Dim)]}."
            )


def _extract_out_new_positional_dimensions(
    key: tuple[int | slice | None | tuple[int, ...] | list[int] | ops.array, ...],
) -> tuple[
    tuple[int | slice | tuple[int, ...] | list[int] | ops.array, ...],
    list[int],
]:
    new_positional_axes = []
    new_key = []
    for axis, index in enumerate(key):
        if index is None:
            new_positional_axes.append(axis)
        else:
            new_key.append(index)
    return tuple(new_key), new_positional_axes


def ensure_tuple_key_representation(
    indexing_object: IndexingObject[Any],
) -> MaybeIndexingByDimsNotInArr[
    tuple[
        int
        | slice
        | _NotExplicitlyIndexed
        | EllipsisType
        | tuple[int, ...]
        | list[int]
        | ops.array,
        ...,
    ]
]:
    # Initialize set containing new dimensions created with explicit name. These
    # could for example be created via x[{"new_dim": None}] or x["new_dim", None].
    new_named_dimensions = list()
    # Initialize dict to keep track of dimensions that were explicitly indexed but is not part of the array. This usually ends in an IndexError but we don't handle that here.
    indexed_dims_not_in_arr = {}

    def _handle_indexing_by_dict(key: dict):
        key_tuple = tuple(
            key.get(dim, _NOT_EXPLICITLY_INDEXED) for dim in indexing_object.arr.dims
        )
        # Keep track of the names of any new dimensions.
        for dim, index in key.items():
            if index is None:
                new_named_dimensions.append(dim)
            elif dim not in indexing_object.arr.dims:
                indexed_dims_not_in_arr[dim] = index
        return key_tuple

    if isinstance(indexing_object.key, dict):
        key_tuple = _handle_indexing_by_dict(indexing_object.key)

    # If the raw_indexing_object is not a tuple it means that the user indexed the
    # array without any commas in the brackets, i.e.: like this: x[0] (and not like
    # this: x[0,]).
    # NOTE: Wrapping the indexing object in a tuple here is not a bug. Indexing
    # using a tuple is inherently ambiguous in Python.
    elif not isinstance(indexing_object.key, tuple):
        key_tuple = (indexing_object.key,)

    # Handle indexing by pairs of dim and index
    elif any(isinstance(index, Dim) for index in indexing_object.key):
        indexed_dims = indexing_object.key[::2]
        indexing_objects_for_dims = indexing_object.key[1::2]
        _validate_dim_index_pairs(indexed_dims, indexing_objects_for_dims)
        raw_indexing_object_dict = dict(zip(indexed_dims, indexing_objects_for_dims))
        key_tuple = _handle_indexing_by_dict(raw_indexing_object_dict)

    # Otherwise return the tuple as-is. We know that it must be a tuple here since
    # we check for the opposite in the if-statement further up.
    else:
        key_tuple = indexing_object.key

    key_tuple, new_positional_axes = _extract_out_new_positional_dimensions(key_tuple)
    indexing_object = replace(
        indexing_object,
        key=key_tuple,
        new_positional_axes=new_positional_axes,
        new_named_dimensions=new_named_dimensions,
    )
    return MaybeIndexingByDimsNotInArr(indexing_object, indexed_dims_not_in_arr)


def ensure_has_all_indexed_dims[T: tuple](
    wrapped_indexing_object: "MaybeIndexingByDimsNotInArr[T]",
) -> IndexingObject[T]:
    if wrapped_indexing_object.indexed_dims_not_in_arr:
        raise IndexError(
            "Indexing dimensions that does not exist in the array: "
            f"{set(wrapped_indexing_object.indexed_dims_not_in_arr.keys())}."
        )
    return wrapped_indexing_object.indexing_object


def validate_index_types(
    indexing_object: IndexingObject[
        tuple[
            int
            | slice
            | _NotExplicitlyIndexed
            | EllipsisType
            | tuple[int, ...]
            | list[int]
            | ops.array,
            ...,
        ]
    ],
):
    for index in indexing_object.key:
        if isinstance(
            index,
            (int, slice, _NotExplicitlyIndexed, tuple, EllipsisType, list, ops.array),
        ):
            continue
        raise IndexError(f"Unsupported index type: {type(index)}.")


def convert_tuple_and_list_indices_to_arrays(
    indexing_object: IndexingObject[
        tuple[
            int
            | slice
            | _NotExplicitlyIndexed
            | EllipsisType
            | tuple[int, ...]
            | list[int]
            | ops.array,
            ...,
        ]
    ],
) -> IndexingObject[
    tuple[int | slice | _NotExplicitlyIndexed | EllipsisType | ops.array, ...]
]:
    new_key: list[int | slice | EllipsisType | ops.array | _NotExplicitlyIndexed] = []
    for index in indexing_object.key:
        if isinstance(index, (list, tuple)):
            if not all(isinstance(x, (int, bool)) for x in index):
                index_element_types = {type(x) for x in index}
                raise IndexError(
                    "Only 1D lists or tuples of ints or bools  are valid as indexing "
                    f"objects, but got {index_element_types}. If you need a more "
                    "advanced indexing object, convert it to an array with integer or "
                    "bool dtype first."
                )
            index = ops.asarray(index)
        new_key.append(index)

    return replace(indexing_object, key=tuple(new_key))  # type: ignore


def broadcast_arr_with_value(
    indexing_object: IndexingObject[
        tuple[int | slice | _NotExplicitlyIndexed | EllipsisType | ops.array, ...]
    ],
) -> IndexingObject[
    tuple[int | slice | _NotExplicitlyIndexed | EllipsisType | ops.array, ...]
]:
    # Ensure the array is broadcastable by the value before we do anything else.
    # NOTE: spekk may change the number of dimensions of arrays when calling setitem,
    # which is never the case in NumPy.
    if isinstance(indexing_object.value, ops.array):
        broadcasted_dim_sizes = indexing_object.arr.dim_sizes.copy()
        for dim, size in indexing_object.value.dim_sizes.items():
            if dim not in broadcasted_dim_sizes:
                broadcasted_dim_sizes[dim] = size

        # Broadcast arr
        broadcasted_arr = ops.broadcast_to(
            indexing_object.arr,
            shape=tuple(broadcasted_dim_sizes.values()),
            dims=list(broadcasted_dim_sizes.keys()),
        )

        # Handle weird case where the user creates a new axis by indexing with None in
        # the context of setitem. This has no effect, but if the value also has this
        # "new" dimension then we should remove it from
        # indexing_object.new_named_dimensions. It is added to the array during the
        # broadcasting step above and it will propagate to the key via the
        # broadcast_key function, so it is no longer a new dimension.
        new_named_dimensions = [
            dim
            for dim in indexing_object.new_named_dimensions
            if dim not in indexing_object.value.dims
        ]
        indexing_object = replace(
            indexing_object,
            arr=broadcasted_arr,
            new_named_dimensions=new_named_dimensions,
        )
    return indexing_object


def expand_indexing_object(
    indexing_object: IndexingObject[
        tuple[int | slice | _NotExplicitlyIndexed | EllipsisType | ops.array, ...]
    ],
) -> IndexingObject[tuple[int | slice | _NotExplicitlyIndexed | ops.array, ...]]:
    # Count non-ellipsis indices
    non_ellipsis_count = sum(1 for idx in indexing_object.key if idx is not Ellipsis)
    ellipsis_count = sum(1 for idx in indexing_object.key if idx is Ellipsis)

    expanded_key = indexing_object.key
    if ellipsis_count > 1:
        raise IndexError("You may only index by a single ellipsis ('...')")
    elif ellipsis_count == 1:
        # Arrays in the tuple stops us from simply writing expanded_key.index(Ellipsis).
        # tuple.index calls __bool__ on the objects which gives the infamous
        # "ValueError: The truth value of an array with more than one element is
        # ambiguous. Use a.any() or a.all()" when called on higher dimensional arrays.
        ellipsis_index = next(i for i, x in enumerate(expanded_key) if x is Ellipsis)
        # Calculate how many dimensions the ellipsis should expand to
        ellipsis_dims = indexing_object.arr.ndim - non_ellipsis_count
        expanded_key = (
            expanded_key[:ellipsis_index]
            + ((_NOT_EXPLICITLY_INDEXED,) * ellipsis_dims)  # Inject expanded indices
            + expanded_key[ellipsis_index + 1 :]
        )

    expanded_key += (_NOT_EXPLICITLY_INDEXED,) * (
        indexing_object.arr.ndim - len(expanded_key)
    )
    return replace(indexing_object, key=expanded_key)  # type: ignore


def infer_undefined_dimensions_where_possible(
    indexing_object: IndexingObject[
        tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]
    ],
) -> IndexingObject[tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]]:
    new_key: list[int | slice | ops.array | _NotExplicitlyIndexed] = []
    for indexed_dim, index in zip(indexing_object.arr.dims, indexing_object.key):
        if isinstance(index, ops.array):
            # Handle case where we index by an array with a single undefined dimension.
            # We infer the dimension name based on the corresponding dimension in x.
            # spekk handles indexing by a 1D array like indexing by a slice.
            if index.ndim == 1 and isinstance(index.dims[0], _UndefinedDim):
                index = index.rename_dim(0, indexed_dim)
        new_key.append(index)
    return replace(indexing_object, key=tuple(new_key))


def all_dims_are_undefined(
    indexing_object: IndexingObject[
        tuple[int | slice | _NotExplicitlyIndexed | EllipsisType | ops.array, ...]
    ],
) -> bool:
    all_arr_dims_are_undefined = all(
        isinstance(dim, _UndefinedDim) for dim in indexing_object.arr.dims
    )
    all_indexing_dims_are_undefined = all(
        isinstance(dim, _UndefinedDim)
        for index in indexing_object.key
        if isinstance(index, ops.array)
        for dim in index.dims
    )
    return all_arr_dims_are_undefined and all_indexing_dims_are_undefined


def replace_not_explicitly_indexed_with_slice[T](
    indexing_object: IndexingObject[tuple[T | _NotExplicitlyIndexed, ...]],
) -> IndexingObject[tuple[T, ...]]:
    key = tuple(
        slice(None) if index is _NOT_EXPLICITLY_INDEXED else index
        for index in indexing_object.key
    )
    return replace(indexing_object, key=key)  # type: ignore


def has_ambiguous_undefined_dimensions(
    indexing_object: IndexingObject[
        tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]
    ],
) -> bool:
    implicitly_indexed_dims = set()
    for index in indexing_object.key:
        if isinstance(index, ops.array):
            # Handle special case where we have a boolean array mask with undefined
            # dimensions. In the context of indexing using named dimensions this is
            # ambiguous. Why? When indexing using a boolean array the underlying axes
            # of the referenced dimensions are flattened, but when a dimension is
            # undefined we can't know which axis to flatten.
            if index.dtype == "bool":
                if any(isinstance(dim, _UndefinedDim) for dim in index.dims):
                    return True

            # Else, add the dimensions to the set.
            implicitly_indexed_dims.update(index.dims)

    undefined_dims_in_implicitly_indexed_dims = [
        isinstance(dim, _UndefinedDim) for dim in implicitly_indexed_dims
    ]

    if any(isinstance(dim, _UndefinedDim) for dim in indexing_object.arr.dims):
        # It's OK that the indexed array has undefined dimensions as long as no
        # undefined dimensions exist in the indexing object and that all dimensions
        # in the indexing object exist in the indexed array.
        if any(undefined_dims_in_implicitly_indexed_dims) or any(
            dim not in indexing_object.arr.dims for dim in implicitly_indexed_dims
        ):
            return True  # There are ambiguous undefined dimensions ❌
    else:  # elif any(undefined_dims_in_indexing_object)
        # It's also OK that the indexing object has undefined dimensions as long as the
        # indexed array has no undefined dimensions.
        pass

    return False  # There are no ambiguous undefined dimensions ✅


def validate_and_process_boolean_indices(
    indexing_object: IndexingObject[
        tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]
    ],
) -> IndexingObject[tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]]:
    # Convert to dict representation because it is easier to work with in this context.
    key_dict = dict(zip(indexing_object.arr.dims, indexing_object.key))
    temporary_dims = indexing_object.temporary_dims.copy()

    # Create a copy of key_dict because we will update it in the loop.
    _key_dict_copy = key_dict.copy()
    for indexed_dim, index in _key_dict_copy.items():
        if isinstance(index, ops.array) and index.dtype == "bool":
            # Check that we don't try to apply a mask over a dimension that does not
            # exist.
            if any(dim not in indexing_object.arr.dims for dim in index.dims):
                raise IndexError(
                    "Indexing one or more dimensions that does not exist in the array: "
                    f"'{index.dims}' using a boolean array. When indexing by a boolean "
                    "array, all dimensions must be defined and exist in the indexed "
                    "array."
                )

            # Remove the boolean mask index from the indexed object dict. Further
            # down (after the call to ops.nonzero), we add it back at the correct
            # dimensions in the form of integer arrays.
            del key_dict[indexed_dim]

            # Handle case where a boolean mask references a dimension in the array
            # that is already being explicitly indexed. This is not allowed and an
            # error will be raised.
            conflicting_dims = {
                dim
                for dim in index.dims
                if dim in key_dict and key_dict[dim] is not _NOT_EXPLICITLY_INDEXED
            }
            if conflicting_dims:
                # When boolean indexing with multiple dimensions, those dimensions get
                # flattened into a single result dimension. If any of those dimensions
                # are also being sliced/indexed separately in the same operation, we'd
                # have conflicting operations: one trying to slice the dimension and
                # another trying to flatten it. The user should perform slicing first
                # as a separate indexing operation.
                raise IndexError(
                    f"Boolean mask for '{indexed_dim}' flattens dimensions "
                    f"{list(index.dims)} but these dimensions are also explicitly "
                    f"indexed: {conflicting_dims}. You should index these dimensions "
                    "before applying the mask."
                )

            # Check for size-mismatches. If the size of a dimension in the boolean
            # mask does not equal the corresponding shape in the array, and error
            # is raised.
            mismatched_size_dims = []
            for indexed_dim in index.dims:
                if indexed_dim in indexing_object.arr.dims:
                    if (
                        index.dim_sizes[indexed_dim]
                        != indexing_object.arr.dim_sizes[indexed_dim]
                    ):
                        mismatched_size_dims.append(indexed_dim)
            if mismatched_size_dims:
                raise IndexError(
                    "Mismatched dimension sizes for dimensions "
                    f"{mismatched_size_dims}. Array dimension sizes: "
                    f"{indexing_object.arr.dim_sizes}. Boolean indexing array "
                    f"dimension sizes: {index.dim_sizes}."
                )

            # Convert boolean masks to integer arrays manually (using ops.nonzero; see
            # further down). This is normally automatically done by the backend, but it
            # is simpler to do it ourselves in order to keep track of named dimensions.
            # The integer arrays will index the values of the dimensions in the boolean
            # mask and return a single new flattened dimension.
            if index.ndim == 1:
                # If the boolean array index has only one dimension, keep that
                # dimension in the output. spekk handles indexing by a 1D array like
                # indexing by a slice.
                new_dim_name = index.dims[0]
            else:
                # However, if there are more than one dimension in the boolean array,
                # then it becomes ambiguous what the resulting dimension should be
                # (remember that indexing by a boolean array always returns a flattened
                # result). The resulting dimension should be an undefined dimension,
                # hence we mark it as temporary which is converted to an undefined
                # dimension later.
                new_dim_name = util.random_dim_name(text="bool_index")
                temporary_dims.add(new_dim_name)

            expanded_boolean_index = dict(
                zip(index.dims, ops.nonzero(index, dim=new_dim_name))
            )
            key_dict.update(expanded_boolean_index)

    # Convert the dict-representation back to a sequence. NOTE: we have to get the
    # indices out by iterating over the arr's dims to get them in the correct order.
    # Order is not preserved otherwise because we may delete and re-add elements in the
    # above loop.
    new_key = tuple(key_dict.get(dim, slice(None)) for dim in indexing_object.arr.dims)

    return replace(
        indexing_object,
        key=new_key,
        temporary_dims=temporary_dims,
    )


def make_undefined_dims_temporary(
    indexing_object: IndexingObject[
        tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]
    ],
) -> IndexingObject[tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]]:
    new_key = []
    new_temporary_dims = indexing_object.temporary_dims.copy()
    for index in indexing_object.key:
        if isinstance(index, ops.array):
            new_dims = []
            for dim in index.dims:
                if isinstance(dim, _UndefinedDim):
                    new_dim = util.random_dim_name()
                    new_dims.append(new_dim)
                    new_temporary_dims.add(new_dim)
                else:
                    new_dims.append(dim)
            index = ops.array(index.data, new_dims)
        new_key.append(index)
    return replace(
        indexing_object,
        key=tuple(new_key),
        temporary_dims=new_temporary_dims,
    )


def broadcast_key(
    indexing_object: IndexingObject[
        tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]
    ],
) -> IndexingObject[tuple[int | slice | ops.array | _NotExplicitlyIndexed, ...]]:
    # Broadcast the arrays in the key.
    implicitly_referenced_dims = set()
    for index in indexing_object.key:
        if isinstance(index, ops.array):
            implicitly_referenced_dims.update(index.dims)

    new_key = []
    for dim, size, index in zip(
        indexing_object.arr.dims,
        indexing_object.arr.shape,
        indexing_object.key,
    ):
        # ...and dim is also being implicitly indexed through a dimension of
        # another array
        if dim in implicitly_referenced_dims:
            # ...then slices must be converted to arrays so that they can be
            # broadcasted
            if index is _NOT_EXPLICITLY_INDEXED:
                index = ops.arange(size, dim=dim)  # Treat as slice(None)
            elif isinstance(index, slice):
                start, stop, step = index.indices(size)
                index = ops.arange(start, stop, step, dim=dim)
            # ...and the resulting dimension sizes must match, but this is handled
            # inside the broadcasting function further down.
        new_key.append(index)

    broadcast_array_fn = get_broadcast_array_fn(
        *[index for index in new_key if isinstance(index, ops.array)]
    )
    new_key = tuple(
        broadcast_array_fn(index) if isinstance(index, ops.array) else index
        for index in new_key
    )

    return replace(indexing_object, key=new_key)


@dataclass
class ProcessedIndexer(abc.ABC):
    """A processed indexer where the only left to do is call either getitem or setitem
    (with no arguments)."""

    indexing_object: IndexingObject[tuple[int | slice | EllipsisType | ops.array, ...]]

    @abc.abstractmethod
    def getitem(self) -> ops.array: ...
    @abc.abstractmethod
    def setitem(self) -> ops.array: ...


class PositionalAxisIndexer(ProcessedIndexer):
    def getitem(self) -> ops.array:
        new_data = self.indexing_object.arr.data[self._get_key()]
        return ops.array(new_data)

    def setitem(self) -> ops.array:
        new_data = ops.backend._setitem_impl(
            self.indexing_object.arr.data,
            self._get_key(),
            *ensure_backend_compatible_data(self.indexing_object.value),
        )
        return ops.array(new_data)

    def _get_key(self) -> tuple[object, ...]:
        key: list[int | slice | EllipsisType | None | ops.array] = list(
            self.indexing_object.key
        )

        for axis in self.indexing_object.new_positional_axes:
            key.insert(axis, None)
        return tuple(
            index.data if isinstance(index, ops.array) else index for index in key
        )


class NamedDimensionsIndexer(ProcessedIndexer):
    def getitem(self) -> ops.array:
        resulting_backend_data = self.indexing_object.arr.data.__getitem__(
            self._get_key()
        )
        output_dims = self.dims_after_getitem()
        return ops.array(resulting_backend_data, output_dims)

    def setitem(self) -> ops.array:
        value = ensure_broadcastable_with(
            self.indexing_object.value,
            self.dims_after_getitem(),
        )
        resulting_backend_data = ops.backend._setitem_impl(
            self.indexing_object.arr.data,
            self._get_key(),
            *ensure_backend_compatible_data(value),
        )
        resulting_dims = [
            _UndefinedDim() if dim in self.indexing_object.temporary_dims else dim
            for dim in self.indexing_object.arr.dims
        ]
        return ops.array(resulting_backend_data, resulting_dims)

    def _get_key(self) -> tuple[Any, ...]:
        new_axes = (None,) * (
            len(self.indexing_object.new_positional_axes)
            + len(self.indexing_object.new_named_dimensions)
        )
        indexed_axes = tuple(
            index.data if isinstance(index, ops.array) else index
            for index in self.indexing_object.key
        )
        # New axes are always created before everything else when indexing with named
        # dimensions in spekk.
        return new_axes + indexed_axes

    def dims_after_getitem(self) -> list[Dim]:
        # NumPy advanced indexing rules: arrays/ints are advanced indices, slices are basic.
        # If advanced indices are contiguous, their dimensions go in place.
        # If non-contiguous, advanced dimensions come first.

        # The logic is complex so we implement it as a state-machine. Here are the 4
        # possible states:
        NOT_SEEN_ADVANCED_INDEX = "not_seen_advanced_index"
        SEEN_ADVANCED_INDEX = "seen_advanced_index"
        SEEN_ADVANCED_INDEX_FOLLOWED_BY_SLICE = "seen_advanced_index_followed_by_slice"
        ADVANCED_INDICES_ARE_NOT_CONTIGUOUS = "advanced_indices_are_not_contiguous"
        state = NOT_SEEN_ADVANCED_INDEX

        dims_before_first_advanced_index = []
        advanced_indices = []
        dims_after_first_advanced_index = []
        # Iterate through dimensions in their original order in the array
        for dim, index in zip(self.indexing_object.arr.dims, self.indexing_object.key):
            # We are in the initial state (NOT_SEEN_ADVANCED_INDEX) until we encounter an
            # advanced index (array or int).
            if state == NOT_SEEN_ADVANCED_INDEX:
                if isinstance(index, (ops.array, int)):
                    advanced_indices.append(index)
                    state = SEEN_ADVANCED_INDEX
                elif isinstance(index, slice):
                    dims_before_first_advanced_index.append(dim)

            # We are in the second state after seeing the first advanced index until we see
            # a slice index. We know add slice-indexed dimensions to the
            # dimensions_after_advanced list instead.
            elif state == SEEN_ADVANCED_INDEX:
                if isinstance(index, (ops.array, int)):
                    advanced_indices.append(index)
                elif isinstance(index, slice):
                    dims_after_first_advanced_index.append(dim)
                    state = SEEN_ADVANCED_INDEX_FOLLOWED_BY_SLICE

            # If we see another advanced index now, it means that they are not all next to
            # each other, which affects the final output dimensions. See where we create
            # output_dims below.
            elif state == SEEN_ADVANCED_INDEX_FOLLOWED_BY_SLICE:
                if isinstance(index, (ops.array, int)):
                    advanced_indices.append(index)
                    state = ADVANCED_INDICES_ARE_NOT_CONTIGUOUS
                elif isinstance(index, slice):
                    dims_after_first_advanced_index.append(dim)

            # This is the terminal state, so we just add the indices and continue.
            elif state == ADVANCED_INDICES_ARE_NOT_CONTIGUOUS:
                if isinstance(index, (ops.array, int)):
                    advanced_indices.append(index)
                elif isinstance(index, slice):
                    dims_after_first_advanced_index.append(dim)

        # Let's simplify our implementation a bit and convert all new axis (all the
        # places where the user indexed by None) to new named dimensions (with
        # temporary names which we convert to _UndefinedDim further down).
        # NamedDimensionsIndexer puts all new axes in front no matter where in the
        # indexing object they appeared.
        new_named_dimensions = self.indexing_object.new_named_dimensions.copy()
        temporary_dims = self.indexing_object.temporary_dims.copy()
        for _ in self.indexing_object.new_positional_axes:
            dim_name = util.random_dim_name()
            new_named_dimensions.append(dim_name)
            temporary_dims.add(dim_name)

        # Get the final output dimensions
        advanced_indices_dims = []
        for arr in advanced_indices:
            if isinstance(arr, ops.array):
                for dim in arr.dims:
                    if dim not in advanced_indices_dims:
                        advanced_indices_dims.append(dim)

        if state == ADVANCED_INDICES_ARE_NOT_CONTIGUOUS:
            dims = [
                *advanced_indices_dims,
                *new_named_dimensions,
                *dims_before_first_advanced_index,
                *dims_after_first_advanced_index,
            ]
        else:
            dims = [
                *new_named_dimensions,
                *dims_before_first_advanced_index,
                *advanced_indices_dims,
                *dims_after_first_advanced_index,
            ]

        # Finally, replace temporary dims with undefined dims and return.
        dims = [_UndefinedDim() if dim in temporary_dims else dim for dim in dims]
        return dims


def get_processed_indexer(
    arr: ops.array,
    indexing_object: object,
    value: Optional[ops.array] = None,
    *,
    indexing_behavior: IndexingBehavior = IndexingBehavior(),
) -> ProcessedIndexer:
    indexing_object = IndexingObject(
        arr,
        indexing_object,
        value,
        indexing_behavior=indexing_behavior,
    )

    indexing_object = ensure_tuple_key_representation(indexing_object)
    indexing_object = indexing_behavior.apply_only_keep_dims_behavior(indexing_object)
    indexing_object = ensure_has_all_indexed_dims(indexing_object)

    validate_index_types(indexing_object)
    indexing_object = convert_tuple_and_list_indices_to_arrays(indexing_object)
    if all_dims_are_undefined(indexing_object):
        indexing_object = replace_not_explicitly_indexed_with_slice(indexing_object)
        return PositionalAxisIndexer(indexing_object)

    indexing_object = broadcast_arr_with_value(indexing_object)
    indexing_object = expand_indexing_object(indexing_object)
    indexing_object = infer_undefined_dimensions_where_possible(indexing_object)
    if has_ambiguous_undefined_dimensions(indexing_object):
        raise IndexError(
            "Undefined dimensions causes ambiguity when indexing. Either make sure "
            "that all dimension names are defined or fallback to NumPy-style "
            "indexing by removing all named dimensions from the array and indices."
        )
    indexing_object = validate_and_process_boolean_indices(indexing_object)
    indexing_object = make_undefined_dims_temporary(indexing_object)
    indexing_object = broadcast_key(indexing_object)
    indexing_object = replace_not_explicitly_indexed_with_slice(indexing_object)
    return NamedDimensionsIndexer(indexing_object)


def getitem(
    arr: ops.array,
    indexing_object: object,
    indexing_behavior: IndexingBehavior | None = None,
) -> ops.array:
    indexer = get_processed_indexer(
        arr,
        indexing_object,
        indexing_behavior=indexing_behavior or IndexingBehavior(),
    )
    return indexer.getitem()


def setitem(
    arr: ops.array,
    indexing_object: Any,
    value: ops.array,
    indexing_behavior: IndexingBehavior | None = None,
) -> ops.array:
    indexer = get_processed_indexer(
        arr,
        indexing_object,
        value,
        indexing_behavior=indexing_behavior or IndexingBehavior(),
    )
    return indexer.setitem()


class ArrayIndexUpdateHelper:
    def __init__(self, x: ops.array):
        self.x = x
        self.indexing_behavior = IndexingBehavior()

    def __getitem__(self, indexing_object: object):
        return _ArrayIndexUpdateRef(self.x, indexing_object, self.indexing_behavior)


class _ArrayIndexUpdateRef:
    def __init__(
        self,
        x: ops.array,
        indexing_object: object,
        indexing_behavior: IndexingBehavior,
    ):
        self.x = x
        self.indexing_object = indexing_object
        self.indexing_behavior = indexing_behavior

    def get(self) -> ops.array:
        return getitem(self.x, self.indexing_object, self.indexing_behavior)

    def set(self, value: ops.array) -> ops.array:
        return setitem(self.x, self.indexing_object, value, self.indexing_behavior)

    def update(self, f: Callable[[ops.array], ops.array], *args, **kwargs) -> ops.array:
        value = f(self.get(), *args, **kwargs)
        return self.set(value)
