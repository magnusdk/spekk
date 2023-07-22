"Slice data using a :class:`~spekk.spec.Spec`."

from typing import Protocol, Sequence, Union

from spekk.spec import Spec
from spekk.trees import Tree, leaves, update

IndicesT = Union[int, slice, Sequence[int], None]


class Sliceable(Protocol):
    def __getitem__(self, indices: IndicesT):
        ...


_all_slice = slice(None, None, None)


def slice_array_1(arr: Sliceable, axis: int, indices: IndicesT):
    """Select the indices along the given axis of an array.

    >>> import numpy as np
    >>> arr = np.array([[1,2,3], [4,5,6]])
    >>> slice_array_1(arr, 0, 0)
    array([1, 2, 3])
    >>> slice_array_1(arr, 1, [0, 2])
    array([[1, 3],
           [4, 6]])
    >>> slice_array_1(arr, None, [0, 2])
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    if axis is not None:
        slices = tuple([_all_slice for _ in range(axis)] + [indices])
        arr = arr.__getitem__(slices)
    return arr


def slice_data(
    data: Tree, spec: Spec, slice_definitions: Sequence[Union[str, IndicesT]]
):
    """Slice the data given a spec and a dict (kwargs) of slice definitions. The keys
    of ``slice_definitions`` are the names of the dimensions to slice, and the values 
    are the indices to slice along that dimension. Returns the sliced data and the
    (possibly modified) spec.

    The data may have an arbitrary tree-like shape (nested ``dict``, ``list``, 
    ``tuple``, etc.), but the leaves must support numpy indexing. The returned sliced 
    data has the same shape as the input data. The spec argument describes the shape of 
    the data.

    If the indices for a dimension in ``slice_definitions`` is just an integer, said
    dimension is removed from the returned spec.

    >>> import numpy as np
    >>> data = {"foo": ({"bar": np.ones((4,5))},)}
    >>> spec = Spec({"foo": ({"bar": ["a", "b"]},)})

    Getting the first item in the ``"a"`` dimension removes that dimension from the 
    spec:

    >>> slice_data(data, spec, ("a", 0))
    {'foo': ({'bar': array([1., 1., 1., 1., 1.])},)}

    We can slice multiple dimensions at once:

    >>> slice_data(data, spec, ("a", 0, "b", slice(0, 3)))
    {'foo': ({'bar': array([1., 1., 1.])},)}

    Here, both the ``"a"`` and ``"b"`` dimensions are removed from the spec:

    >>> slice_data(data, spec, ("a", 0, "b", 0))
    {'foo': ({'bar': 1.0},)}
    """
    is_axis = lambda x: isinstance(x, int) or x is None
    for dimension, indices in zip(slice_definitions[::2], slice_definitions[1::2]):
        for leaf in leaves(spec.index_for(dimension), is_axis):
            data = update(
                data, lambda a: slice_array_1(a, leaf.value, indices), leaf.path
            )
        if isinstance(indices, int):
            spec = spec.remove_dimension(dimension)
    return data


def slice_spec(spec: Spec, slice_definitions: Sequence[Union[str, IndicesT]]):
    for dimension, indices in zip(slice_definitions[::2], slice_definitions[1::2]):
        if isinstance(indices, int):
            spec = spec.remove_dimension(dimension)
    return spec


if __name__ == "__main__":
    import doctest

    doctest.testmod()
