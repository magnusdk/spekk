from typing import Protocol, Sequence, Union

from spekk2.spec import Spec
from spekk2.trees import Tree, update
from spekk2.trees.common import leaves

IndicesT = Union[int, slice, Sequence[int], None]


class Sliceable(Protocol):
    def __getitem__(self, indices: IndicesT):
        ...


_all_slice = slice(None, None, None)


def slice_array(arr: Sliceable, axis: int, indices: IndicesT):
    """Select the indices along the given axis of an array.

    >>> import numpy as np
    >>> arr = np.array([[1,2,3], [4,5,6]])
    >>> slice_array(arr, 0, 0)
    array([1, 2, 3])
    >>> slice_array(arr, 1, [0, 2])
    array([[1, 3],
           [4, 6]])
    >>> slice_array(arr, None, [0, 2])
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    if axis is not None:
        slices = tuple([_all_slice for _ in range(axis)] + [indices])
        arr = arr.__getitem__(slices)
    return arr


def slice_dimensions(data: Tree, spec: Spec, **slice_definitions: IndicesT):
    """Slice the data given a spec and a dict (kwargs) of slice definitions. The keys
    of slice_definitions are the names of the dimensions to slice, and the values are
    the indices to slice along that dimension. Returns the sliced data and the
    (possibly modified) spec.

    The data may have an arbitrary tree-like shape (nested dicts, lists, tuples, etc.),
    but the leaves must support numpy indexing. The returned sliced data has the same
    shape as the input data. The spec argument describes the shape of the data.

    If a indices for a dimension in slice_definitions is just an integer, said
    dimension is removed from the returned spec.

    >>> import numpy as np
    >>> data = {"foo": ({"bar": np.ones((4,5))},)}
    >>> spec = Spec({"foo": ({"bar": ["a", "b"]},)})

    Getting the first item in the "a" dimension removes that dimension from the spec:
    >>> slice_dimensions(data, spec, a=0)
    ({'foo': ({'bar': array([1., 1., 1., 1., 1.])},)}, Spec({foo: ("{bar: ['b']}",)}))

    We can slice multiple dimensions at once:
    >>> slice_dimensions(data, spec, a=0, b=slice(0, 3))
    ({'foo': ({'bar': array([1., 1., 1.])},)}, Spec({foo: ("{bar: ['b']}",)}))

    Here, both the "a" and "b" dimensions are removed from the spec:
    >>> slice_dimensions(data, spec, a=0, b=0)
    ({'foo': ({'bar': 1.0},)}, Spec({foo: ('{bar: []}',)}))
    """
    is_axis = lambda x: isinstance(x, int) or x is None
    for dimension, indices in slice_definitions.items():
        for axis, path in leaves(spec.index_for(dimension), is_axis):
            data = update(data, lambda a: slice_array(a, axis, indices), path)
        if isinstance(indices, int):
            spec = spec.remove_dimension(dimension)
    return data, spec


if __name__ == "__main__":
    import doctest

    doctest.testmod()
