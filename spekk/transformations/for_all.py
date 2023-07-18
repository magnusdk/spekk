from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from spekk import Spec, util
from spekk.transformations import Transformation, common

T_in_axes = Sequence[Optional[int]]
T_vmap = Callable[[callable, T_in_axes], callable]


@dataclass
class ForAll(Transformation):
    dimension: str
    vmap_impl: Optional[T_vmap] = None

    def transform_function(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        if not input_spec.has_dimension(self.dimension):
            raise ValueError(f"Spec does not contain the dimension {self.dimension}.")
        return specced_vmap(
            to_be_transformed, input_spec, self.dimension, self.vmap_impl
        )

    def transform_input_spec(self, spec: Spec) -> Spec:
        # The returned function works on 1 element of the dimension at a time, so it has
        # one less dimension.
        return spec.remove_dimension(self.dimension)

    def transform_output_spec(self, spec: Spec) -> Spec:
        # We re-add the dimension after the function has been applied to each element.
        return spec.add_dimension(self.dimension)

    def __repr__(self) -> str:
        return f'ForAll("{self.dimension}")'


def specced_vmap(
    f: callable,
    spec: Spec,
    dimension: str,
    vmap_impl: Optional[T_vmap] = None,
):
    """Similar to vmap, but flattens/decomposes the kwargs to a list that is supported
    by vmap. It also ensures that each input to the vmapped function (vmap(f, ...))
    does not have any field that does not have the given dimension.
    """
    if vmap_impl is None:
        vmap_impl = python_vmap

    def wrapped(*_unsupported_positional_args, **kwargs):
        if _unsupported_positional_args:
            raise ValueError(
                "Positional arguments are not supported in specced_vmap. Use keyword arguments instead."
            )

        flattened_args, in_axes, unflatten = util.flatten(kwargs, spec, dimension)

        def f_with_unflattening_args(*args):
            original_kwargs = unflatten(args)
            return f(**original_kwargs)

        vmapped_f = vmap_impl(f_with_unflattening_args, in_axes)
        return vmapped_f(*flattened_args)

    return wrapped


def python_vmap(f, in_axes):
    def wrapped(*args):
        sizes = [common.shape(arg)[a] for arg, a in zip(args, in_axes) if a is not None]
        size = sizes[0]
        results = []
        for i in range(size):
            args_i = [
                common.getitem_along_axis(arg, a, i) if a is not None else arg
                for arg, a in zip(args, in_axes)
            ]
            results.append(f(*args_i))
        return results

    return wrapped
