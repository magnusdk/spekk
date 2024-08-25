from typing import TYPE_CHECKING

from spekk.transformations.monadic.base import TransformedFunction
from spekk.trees import Tree

if TYPE_CHECKING:
    from spekk.spec import Spec


def compose(x, *fns) -> TransformedFunction:
    for wrap in fns:
        x = wrap(x)
    return x


def flatten_and_split_dynamic_and_static(data, spec: "Spec", dim: str):
    indices = spec.indices_for(dim, conform=data)
    flattened_data, original_unflatten = Tree.flatten(data)
    in_axes, _ = Tree.flatten(indices)

    dynamic_argnums = []
    dynamic_args = []
    static_argnums = []
    static_args = []
    dynamic_in_axes = []
    for i, (x, axis) in enumerate(zip(flattened_data, in_axes)):
        if axis is None:
            static_argnums.append(i)
            static_args.append(x)
        else:
            dynamic_argnums.append(i)
            dynamic_args.append(x)
            dynamic_in_axes.append(axis)

    def unflatten_with_static(vmapped_args, static_args):
        args_original_order = [None] * (len(vmapped_args) + len(static_args))
        for i, arg in zip(dynamic_argnums, vmapped_args):
            args_original_order[i] = arg
        for i, arg in zip(static_argnums, static_args):
            args_original_order[i] = arg
        return original_unflatten(args_original_order)

    return dynamic_args, static_args, tuple(dynamic_in_axes), unflatten_with_static
