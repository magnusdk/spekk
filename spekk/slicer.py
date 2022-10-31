from dataclasses import dataclass
from typing import Any, Optional, TypeVar, Union

from spekk import Shape, Spec

# Keep all elements for the axis. Same as writing just : when slicing, e.g. arr[:].
_all_slice = slice(None, None, None)
T = TypeVar("T")


def slice_data(
    data: Union[Any, dict],
    shape_or_spec: Union[Shape, Spec],
    dim_slices: tuple,
):
    if isinstance(shape_or_spec, Shape):
        shape = shape_or_spec
        new_data = data
        for dim, indices in zip(dim_slices[::2], dim_slices[1::2]):
            assert isinstance(indices, slice), "You may only use slices"
            axis = shape.index(dim)
            if axis is not None:
                slices = tuple([_all_slice for _ in range(axis)] + [indices])
                new_data = new_data.__getitem__(slices)
        return new_data

    elif isinstance(shape_or_spec, Spec):
        spec = shape_or_spec
        new_data = data.copy()
        for dim, indices in zip(dim_slices[::2], dim_slices[1::2]):
            assert isinstance(indices, slice), "You may only use slices"
            indices_map = spec.indices_for(dim, index_by_name=True)
            if not indices_map:
                raise ValueError(
                    f"The dimension '{dim}' was not found in the spec. Is there a typo?"
                )
            for name, axis in indices_map.items():
                slices = tuple([_all_slice for _ in range(axis)] + [indices])
                new_data[name] = new_data[name].__getitem__(slices)
        return new_data, spec

    else:
        raise ValueError(
            f"Second argument must either be a Shape of a Spec. Received object with type {type(shape_or_spec)}."
        )


@dataclass
class Slicer:
    """A class that can "slice" data according to a spec.

    If the spec specifies that the data has a dimension, e.g. "transmits", then you can
    select a slice of the data across that dimension using a Slicer.

    >>> import numpy as np
    >>> spec = Spec({"signal": ["frames", "transmits"]})
    >>> slicer = Slicer(spec, ("transmits", slice(1,2))) # Take the second transmit only
    >>> slicer(np.array([[1,2,3], [4,5,6]]), "signal")
    array([[2],
           [5]])
    """

    spec: Spec
    slice_definitions: tuple

    def __post_init__(self):
        if len(self.slice_definitions) % 2 != 0:
            raise ValueError(
                """The slice definitions must be a concatenation of (dim, slice) pairs.\
 For example: ("frames", slice(0, 10), "transmits", slice(25, 35, 2))"""
            )
        for dim in self.slice_definitions[::2]:
            if not self.spec.has_dim(dim):
                raise ValueError(
                    f"The dimension '{dim}' was not found in the spec. Is there a typo?"
                )
        for indices in self.slice_definitions[1::2]:
            if not isinstance(indices, slice):
                raise ValueError(
                    f"The indices must be slices, not {type(indices).__name__}."
                )

    def __call__(self, data: T, name: Optional[str] = None) -> T:
        if name is None:
            return slice_data(data, self.spec, self.slice_definitions)
        else:
            return slice_data(data, self.spec[name], self.slice_definitions)

    def __add__(self, other) -> "Slicer":
        if isinstance(other, tuple):
            return Slicer(self.spec, self.slice_definitions + other)
        else:
            return NotImplemented
