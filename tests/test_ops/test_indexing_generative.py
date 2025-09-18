import itertools
import math
from dataclasses import dataclass

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from spekk import ops
from spekk.ops._types import _UndefinedDim

ops.backend.set_backend("numpy")

dim_sizes = {dim: size for dim, size in zip("abcdef", itertools.count(start=2))}
dim_gen = st.sampled_from(list(dim_sizes.keys()))
arr_dims_gen = st.lists(dim_gen, min_size=1, unique=True)


@st.composite
def shuffled_dict_gen(draw, original_dict: dict) -> dict:
    keys = draw(st.permutations(list(original_dict.keys())))
    return {key: original_dict[key] for key in keys}


@st.composite
def indexing_types_dims_gen(draw, arr_dims: list[str]):
    indexing_types = {
        "basic_indexing": [],
        "bool_simple_mask_indexing": [],
        "bool_multi_dimensional_mask_indexing": [],
        "integer_array_indexing": [],
    }
    referenced_dims = draw(st.lists(st.sampled_from(arr_dims), min_size=1, unique=True))
    for dim in referenced_dims:
        indexing_type = draw(st.sampled_from(list(indexing_types.keys())))
        indexing_types[indexing_type].append(dim)

    dims_not_in_arr = [dim for dim in dim_sizes.keys() if dim not in arr_dims]
    new_dims = (
        draw(st.lists(st.sampled_from(dims_not_in_arr), unique=True))
        if len(dims_not_in_arr) != 0
        else []
    )
    indexing_types["new_dims"] = new_dims
    indexing_types["all"] = new_dims + referenced_dims
    return indexing_types


@st.composite
def indexing_object_gen(draw, arr_dims: list[str]):
    indexing_types_dims = draw(indexing_types_dims_gen(arr_dims))

    # We keep track of what the resulting dimension sizes will be in updated_dim_sizes.
    # We use it to ensure that all indexing objects are effectively broadcastable
    # (including cases where we combine slices and array indices).
    resulting_dim_sizes = {}
    undefined_dim_sizes = []

    # The final indexing object that will be returned.
    indexing_object = {}

    # TODO: Explain
    sub_referenced_dims = set()
    removed_dims = set()
    new_non_broadcasted_dims = set()

    # Add all new dimension indexing objects.
    for dim in indexing_types_dims["new_dims"]:
        indexing_object[dim] = None
        resulting_dim_sizes[dim] = 1
        new_non_broadcasted_dims.add(dim)

    # Add the basic indexing objects. These can not affect other dimensions that the
    # one that they're referencing, which is why we add them before adding more
    # advanced indices.
    for dim in indexing_types_dims["basic_indexing"]:
        index_type = draw(
            st.one_of(
                # Select first element only (removes dimension)
                st.just("single_element"),
                # Slice that selects all elements
                st.just("slice_all"),
                # Slice that selects a subset of elements (all except first)
                st.just("slice_subset"),
            )
        )
        if index_type == "single_element":
            indexing_object[dim] = 0
            removed_dims.add(dim)
            # resulting_dim_sizes[dim] = 0  # This dimension is removed
        elif index_type == "slice_all":
            indexing_object[dim] = slice(None)  # Select all
            resulting_dim_sizes[dim] = dim_sizes[dim]
        elif index_type == "slice_subset":
            n = draw(st.integers(1, dim_sizes[dim] - 1))
            indexing_object[dim] = slice(n)  # Select all except first
            resulting_dim_sizes[dim] = n  # Size is decreased

    # Simple boolean masks can also only affect the referenced dimension.
    for dim in indexing_types_dims["bool_simple_mask_indexing"]:
        mask = ops.array(
            [
                draw(st.booleans())
                for _ in range(resulting_dim_sizes.get(dim, dim_sizes[dim]))
            ],
            dims=[dim],
        )
        indexing_object[dim] = mask
        resulting_dim_sizes[dim] = int(ops.sum(mask))

    for dim in indexing_types_dims["bool_multi_dimensional_mask_indexing"]:
        possible_dims = [
            dim
            for dim in arr_dims
            if dim not in indexing_types_dims["all"] and dim not in sub_referenced_dims
        ]
        if len(possible_dims) < 2:
            continue
        dims = draw(st.lists(st.sampled_from(possible_dims), min_size=2, unique=True))
        shape = [dim_sizes[sub_dim] for sub_dim in dims]
        sub_referenced_dims.update(dims)
        total_size = math.prod(shape)
        mask = ops.reshape(
            ops.array(
                draw(st.lists(st.booleans(), min_size=total_size, max_size=total_size))
            ),
            tuple(shape),
            dims,
        )
        undefined_dim_sizes.append(int(ops.sum(mask)))
        indexing_object[dim] = mask
        for sub_dim in dims:
            removed_dims.add(sub_dim)
            resulting_dim_sizes[sub_dim] = 0

    for dim in indexing_types_dims["integer_array_indexing"]:
        # We want to create array indexing objects. These arrays may have multiple
        # dimensions. The size of the dimensions must match the resulting size of
        # applying all other indexing objects, which we keep track of in
        # updated_dim_sizes. If updated_dim_sizes[dim] is 0, then the dimension has
        # been removed (f.ex. from indexing using a single integer) and can not be
        # referenced again.
        possible_dims = [
            dim
            for dim in dim_sizes.keys()
            if dim not in removed_dims and dim not in indexing_types_dims["new_dims"]
        ]
        if len(possible_dims) == 0:
            continue

        # Generate random dimensions and corresponding sizes.
        dims = draw(st.lists(st.sampled_from(possible_dims), min_size=1, unique=True))
        shape = []
        for sub_dim in dims:
            sub_size = resulting_dim_sizes.get(
                sub_dim,
                draw(st.integers(min_value=1, max_value=8))
                if sub_dim == dim
                else dim_sizes[sub_dim],
            )
            shape.append(sub_size)
            resulting_dim_sizes[sub_dim] = sub_size
        # Indexing with an integer array where the referenced dimension is not part of
        # the array dimensions, that dimension is removed. In practice, it is like
        # selecting a single element from the referenced dimension for each array
        # dimension.
        if dim not in dims and dim not in resulting_dim_sizes:
            removed_dims.add(dim)

        indexing_object[dim] = ops.zeros(tuple(shape), dtype="int32", dims=dims)

    # Shuffle the dict because ordering shouldn't matter.
    indexing_object = draw(shuffled_dict_gen(indexing_object))

    resulting_dim_sizes = {
        **{dim: dim_sizes[dim] for dim in arr_dims},
        **resulting_dim_sizes,
    }
    for dim in removed_dims:
        del resulting_dim_sizes[dim]
    return (
        indexing_object,
        resulting_dim_sizes,
        undefined_dim_sizes,
        new_non_broadcasted_dims,
        removed_dims,
    )


@dataclass
class GetitemScenario:
    arr: ops.array
    indexing_object: dict[str, None | list[str] | slice | int | ops.array]
    defined_dim_sizes: dict[str, int]
    undefined_dim_sizes: list[int]
    new_non_broadcasted_dims: set[int]
    removed_dims: set[int]

    @staticmethod
    @st.composite
    def gen(draw):
        arr_dims = draw(
            st.lists(st.sampled_from(list(dim_sizes.keys())), min_size=1, unique=True)
        )
        (
            indexing_object,
            defined_dim_sizes,
            undefined_dim_sizes,
            new_non_broadcasted_dims,
            removed_dims,
        ) = draw(indexing_object_gen(arr_dims))
        arr = ops.zeros(tuple((dim_sizes[dim] for dim in arr_dims)), dims=arr_dims)
        return GetitemScenario(
            arr,
            indexing_object,
            defined_dim_sizes,
            undefined_dim_sizes,
            new_non_broadcasted_dims,
            removed_dims,
        )


@dataclass
class SetitemScenario:
    arr: ops.array
    indexing_object: dict[str, None | list[str] | slice | int | ops.array]
    dim_sizes: dict[str, int]
    value: ops.array

    @st.composite
    @staticmethod
    def gen(draw):
        scenario: GetitemScenario = draw(GetitemScenario.gen())

        possible_dims = [
            *scenario.defined_dim_sizes.keys(),
            *[dim for dim in dim_sizes.keys() if dim not in scenario.removed_dims],
        ]
        if possible_dims:
            value_dims = draw(st.lists(st.sampled_from(possible_dims), unique=True))
        else:
            value_dims = []
        value_sizes = [
            scenario.defined_dim_sizes.get(dim, dim_sizes[dim]) for dim in value_dims
        ]
        value = ops.ones(tuple(value_sizes), dims=value_dims)

        # Add value dim_sizes first, THEN arr dim_sizes so that they are overwritten
        # correctly.
        defined_dim_sizes = {**value.dim_sizes, **scenario.arr.dim_sizes}

        return SetitemScenario(
            scenario.arr,
            scenario.indexing_object,
            defined_dim_sizes,
            value,
        )


@given(GetitemScenario.gen())
def test_getitem(scenario: GetitemScenario):
    result = scenario.arr[scenario.indexing_object]

    assert len(scenario.defined_dim_sizes) + len(scenario.undefined_dim_sizes) == len(
        result.dim_sizes
    )
    undefined_dim_sizes = set()
    for dim, size in scenario.defined_dim_sizes.items():
        if isinstance(dim, _UndefinedDim):
            undefined_dim_sizes.add(size)
        assert size == result.dim_sizes[dim]
    assert undefined_dim_sizes == undefined_dim_sizes


@given(SetitemScenario.gen())
def test_setitem(scenario: SetitemScenario):
    arr = ops.array(np.array(scenario.arr.data), scenario.arr.dims)
    arr[scenario.indexing_object] = scenario.value
    assert scenario.dim_sizes == arr.dim_sizes
