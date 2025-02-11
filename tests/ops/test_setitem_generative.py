import dataclasses
from typing import Dict, Union

import hypothesis.strategies as st
from hypothesis import given, settings

from spekk import ops
from spekk.ops._types import Dim


@dataclasses.dataclass
class Setup:
    data_dim_sizes: Dict[Dim, int]
    # If an indexing object is a dict, then it represents the dim_sizes of an array.
    indexing_objects: Dict[Dim, Union[Dict[Dim, int], int, slice]]
    value: Dict[Dim, int]

    def perform_indexing_update(self):
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

        if isinstance(self.value, dict):
            value = ops.ones(
                shape=tuple(self.value.values()),
                dims=list(self.value.keys()),
            )
        elif isinstance(self.value, int):
            value = self.value

        x[tuple(indexing_objects)] = value
        return x

    @property
    def expected_output_dim_sizes(self):
        output_dim_sizes = dict(self.value) if isinstance(self.value, dict) else {}
        output_dim_sizes.update(self.data_dim_sizes)
        indexing_dim_sizes = {}
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
                indexing_dim_sizes.update(i)
        for dim, size in indexing_dim_sizes.items():
            if dim not in output_dim_sizes:
                output_dim_sizes[dim] = size
        return output_dim_sizes


dim_name_strategy = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=2
)

dim_size_strategy = st.integers(min_value=2, max_value=6)

data_dim_sizes_strategy = st.dictionaries(
    keys=dim_name_strategy,
    values=dim_size_strategy,
    min_size=1,
    max_size=6,
)


def array_indexing_object_strategy(data_dim_sizes: Dict[str, int]):
    dims_in_data = st.sampled_from(list(data_dim_sizes.items()))
    new_dims = st.tuples(
        dim_name_strategy.filter(lambda s: s not in data_dim_sizes),
        dim_size_strategy,
    )
    dict_entries = st.one_of(dims_in_data, new_dims)
    return st.lists(dict_entries, min_size=1, max_size=4).map(dict)


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
        max_size=6,
    )
    value_strategy = st.one_of(data_dim_sizes_strategy, st.just(0))

    indexing_objects = draw(indexing_objects_strategy)
    value_dim_sizes = draw(value_strategy)
    return Setup(data_dim_sizes, indexing_objects, value_dim_sizes)


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


def invalid_case_2(setup: Setup) -> bool:
    value_dim_sizes = setup.value if isinstance(setup.value, dict) else {}
    for indexed_dim, indexing_object in setup.indexing_objects.items():
        if isinstance(indexing_object, dict):
            for dim, size in indexing_object.items():
                if value_dim_sizes.get(dim, None) != size:
                    return True
        elif isinstance(indexing_object, int):
            if value_dim_sizes.get(indexed_dim, None) != 1:
                return True
        elif isinstance(indexing_object, slice):
            new_size = len(range(setup.data_dim_sizes[indexed_dim])[indexing_object])
            if value_dim_sizes.get(indexed_dim, None) != new_size:
                return True
    return False


def invalid_case_3(setup: Setup) -> bool:
    value_dim_sizes = setup.value if isinstance(setup.value, dict) else {}
    for dim, size in value_dim_sizes.items():
        for indexing_object in setup.indexing_objects.values():
            if (
                isinstance(indexing_object, dict)
                and dim in indexing_object
                and indexing_object[dim] != size
            ):
                return True
        else:
            if dim in setup.data_dim_sizes and setup.data_dim_sizes[dim] != size:
                return True
    return False


@settings(max_examples=2000)
@given(setup_strategy())
def test_setitem_generative(setup: Setup):
    if invalid_case_1(setup):
        # TODO: Assert raised IndexError
        return
    elif invalid_case_2(setup):
        # TODO: Assert raised IndexError
        return
    elif invalid_case_3(setup):
        # TODO: Assert raised IndexError
        return
    else:
        result = setup.perform_indexing_update()
        assert result.dim_sizes == setup.expected_output_dim_sizes
