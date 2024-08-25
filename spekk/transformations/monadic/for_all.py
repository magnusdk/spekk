from typing import Callable, Optional

from spekk.spec import Spec
from spekk.transformations.monadic.base import Transformation
from spekk.transformations.specced_vmap import T_vmap, specced_vmap
from spekk.transformations.util import compose


class ForAll(Transformation):
    vmap_impl: Optional[T_vmap] = None

    def __init__(self, dim: str, *additional_dims: str):
        self.dims = [dim, *additional_dims]

    def transform_input_spec(self, spec: Spec):
        return spec.remove_dimension(*self.dims)

    def transform_output_spec(self, spec: Spec):
        for dim in self.dims:
            spec = spec.add_dimension(dim)
        return spec

    def transform_function(self, f: Callable, input_spec: Spec, returned_spec: Spec):
        # removed_dimensions keeps track of what dimensions have been removed by
        # "previous" steps of ForAll
        removed_dimensions = set(self.dims)
        for dim in self.dims:
            removed_dimensions.remove(dim)
            f = specced_vmap(
                f, input_spec.remove_dimension(*removed_dimensions), dim, self.vmap_impl
            )
        return f

