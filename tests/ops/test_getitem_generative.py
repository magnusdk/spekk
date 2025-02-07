import collections
import dataclasses
from typing import Dict, Union

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from spekk import ops
from spekk.ops._types import Dim


@dataclasses.dataclass
class Setup:
    data_dim_sizes: Dict[Dim, int]
    # If an indexing object is a dict, then it represents the dim_sizes of an array.
    indexing_objects: Dict[Dim, Union[Dict[Dim, int], int, slice]]

    def perform_indexing(self):
        x = ops.zeros(
            shape=tuple(self.data_dim_sizes.values()),
            dims=list(self.data_dim_sizes.keys()),
        )
        indexing_objects = []
        for dim, i in self.indexing_objects.items():
            indexing_objects.append(dim)
            if isinstance(i, dict):
                i = ops.zeros(tuple(i.values()), dims=list(i.keys()), dtype="int32")
            indexing_objects.append(i)

        return x[tuple(indexing_objects)]

    @property
    def expected_output_dim_sizes(self):
        output_dim_sizes = dict(self.data_dim_sizes)
        for dim, i in self.indexing_objects.items():
            if isinstance(i, dict):
                assert all(
                    _size == self.data_dim_sizes[_dim]
                    for _dim, _size in i.items()
                    if _dim in self.data_dim_sizes and _dim != dim
                ), (
                    "Sanity check that we have not generated an invalid indexing setup."
                    " If this fails, something in the generator is incorrect. Indexing "
                    "with an array that has a dimension in the indexed array (except "
                    "the indexed dimension) means that the dimension should have the "
                    "same size as in the indexed array. 😵‍💫"
                )
                if dim in output_dim_sizes:
                    del output_dim_sizes[dim]
                output_dim_sizes.update(i)
            elif isinstance(i, int):
                del output_dim_sizes[dim]
            elif isinstance(i, slice):
                output_dim_sizes[dim] = len(range(self.data_dim_sizes[dim])[i])
        return output_dim_sizes


dim_name_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=2
)

dim_size_strategy = st.integers(min_value=2, max_value=8)

data_dim_sizes_strategy = st.dictionaries(
    keys=dim_name_strategy,
    values=dim_size_strategy,
    min_size=1,
    max_size=5,
)


def array_indexing_object_strategy(data_dim_sizes: Dict[str, int]):
    dims_in_data = st.sampled_from(list(data_dim_sizes.items()))
    new_dims = st.tuples(
        dim_name_strategy.filter(lambda s: s not in data_dim_sizes),
        dim_size_strategy,
    )
    dict_entries = st.one_of(dims_in_data, new_dims)
    return st.lists(dict_entries, min_size=1, max_size=5).map(dict)


@st.composite
def setup_strategy(draw):
    data_dim_sizes = draw(data_dim_sizes_strategy)
    indexing_objects_strategy = st.dictionaries(
        st.sampled_from(list(data_dim_sizes.keys())),
        st.one_of(
            array_indexing_object_strategy(data_dim_sizes),
            st.just(0),
            st.slices(1),
        ),
        min_size=1,
        max_size=5,
    )

    indexing_objects = draw(indexing_objects_strategy)
    return Setup(data_dim_sizes, indexing_objects)


def invalid_case_1(setup: Setup):
    for dim_outer, i in setup.indexing_objects.items():
        if isinstance(i, dict):
            for dim_inner in i.keys():
                if (
                    dim_inner in setup.indexing_objects
                    and dim_inner != dim_outer
                    and (
                        not isinstance(setup.indexing_objects[dim_inner], slice)
                        or setup.indexing_objects[dim_inner] != slice(None)
                    )
                ):
                    return True
    return False


def invalid_case_2(setup: Setup):
    indexing_objects_sizes = collections.defaultdict(set)
    for i in setup.indexing_objects.values():
        if isinstance(i, dict):
            for dim, size in i.items():
                indexing_objects_sizes[dim].add(size)
    return any(len(size) > 1 for size in indexing_objects_sizes.values())


@given(setup_strategy())
def test_getitem_generative(setup: Setup):
    if invalid_case_1(setup):
        with pytest.raises(IndexError):
            setup.perform_indexing()
    elif invalid_case_2(setup):
        with pytest.raises(IndexError):
            setup.perform_indexing()
    else:
        result = setup.perform_indexing()
        assert result.dim_sizes == setup.expected_output_dim_sizes
