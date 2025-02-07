from typing import TYPE_CHECKING

import spekk.ops as ops
from spekk.ops._types import Dim, Dims

if TYPE_CHECKING:
    from spekk.module.trees import TreeLike


class DimSlicer:
    def __init__(self, data: "TreeLike", dim: Dim):
        self.data = data
        self.dim = dim

    def __getitem__(self, key):
        from spekk.module.trees import flatten

        if isinstance(self.data, ops.array):
            s = (slice(None),) * self.data.dims.index(self.dim) + (key, Ellipsis)
            return self.data.__getitem__(s)

        # Flatten data such that all dynamic data are ops.array instances with the
        # given dimension.
        flattened = flatten(
            self.data,
            is_static=lambda x: not isinstance(x, ops.array) or self.dim not in x.dims,
            is_tree_like=lambda x: not isinstance(x, ops.array),
        )
        # Iterate through and slice the ops.array instances.
        sliced_arrays = []
        for x in flattened.dynamic:
            s = (slice(None),) * x.dims.index(self.dim) + (key, Ellipsis)
            sliced_arrays.append(x.__getitem__(s))
        # Get back the original data object
        return flattened.treedef.unflatten(sliced_arrays)


class SlicingMixin:
    def slice_dim(self, dim: Dim):
        return DimSlicer(self, dim)

    @property
    def dim_size(self):
        from spekk.module.trees import flatten

        flattened = flatten(
            self,
            is_static=lambda x: not isinstance(x, ops.array),
            is_tree_like=lambda x: not isinstance(x, ops.array),
        )
        dim_sizes = {}
        for arr in flattened.dynamic:
            for d, s in zip(arr.dims, arr.shape):
                dim_sizes[d] = s
        return dim_sizes

    @property
    def dims(self) -> Dims:
        from spekk.module.trees import flatten

        flattened = flatten(
            self,
            is_static=lambda x: not isinstance(x, ops.array),
            is_tree_like=lambda x: not isinstance(x, ops.array),
        )
        dims = set()
        for x in flattened.dynamic:
            dims.update(x.dims)
        return list(dims)
