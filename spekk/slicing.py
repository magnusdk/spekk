from typing import Any, Sequence, TypeAlias, Union

import numpy as np

from spekk import Spec, traverse_leaves

IndicesT: TypeAlias = Union[int, slice, Sequence[int], None]


def _getitem_along_axis(arr: np.ndarray, axis: int, indices: IndicesT):
    if axis is not None:
        slices = tuple([slice(None)] * axis + [indices])
        arr = arr.__getitem__(slices)
    return arr


def slice_dims(
    spec: Spec,
    data: Any,
    slice_definitions: Sequence[Union[str, IndicesT]],
):
    sliced_data = data
    for dim, indices in zip(slice_definitions[::2], slice_definitions[1::2]):
        sliced_data = traverse_leaves(
            lambda x, axis: _getitem_along_axis(x, axis, indices),
            data,
            spec.indices_for(dim, conform=data),
            is_leaf=lambda _, maybe_index: isinstance(maybe_index, int),
        )

    sliced_spec = spec
    for dim, indices in zip(slice_definitions[::2], slice_definitions[1::2]):
        if isinstance(indices, int):
            sliced_spec = sliced_spec.remove_dimension(dim)

    return sliced_spec, sliced_data


def iterate_over(spec: Spec, data: Any, dim: str):
    size = spec.size(data, dim)
    for i in range(size):
        _, sliced_data = slice_dims(spec, data, (dim, i))
        yield sliced_data


def test_slice():
    spec = Spec({"a": {"b": ["dim0", "dim1"]}})
    data = {"a": {"b": np.ones((2, 3))}, "c": np.ones((4,))}
    sliced_spec, sliced_data = slice_dims(spec, data, ("dim1", 0))
    print(sliced_spec, sliced_data)


if __name__ == "__main__":
    test_slice()
