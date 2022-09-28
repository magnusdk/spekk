from dataclasses import dataclass
from typing import Any, Callable, Tuple

from spekk.spec import Spec

# Keep all elements for the axis. Same as writing just : when slicing, e.g. arr[:, 0].
_all_slice = slice(None, None, None)


def slice_dims(data: dict, spec: Spec, dim_slices) -> Tuple[dict, Spec]:
    new_data = data.copy()
    for dim, indices in zip(dim_slices[::2], dim_slices[1::2]):
        assert isinstance(indices, slice), "You may (currently) only use slices"
        indices_map = spec.indices_for(dim, index_by_name=True)
        if not indices_map:
            raise ValueError(
                f"The dimension '{dim}' was not found in the spec. Is there a typo?"
            )
        for name, axis in indices_map.items():
            slices = tuple([_all_slice for _ in range(axis)] + [indices])
            new_data[name] = new_data[name].__getitem__(slices)
    return new_data, spec


@dataclass
class Slicer:
    obj: Any
    lift: Callable
    unlift: Callable

    def __getitem__(self, dim_slices):
        data, spec = self.lift(self.obj)
        new_data, new_spec = slice_dims(data, spec, dim_slices)
        return self.unlift(new_data, new_spec)
