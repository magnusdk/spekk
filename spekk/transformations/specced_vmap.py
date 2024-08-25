import functools
from typing import TYPE_CHECKING, Callable, Optional, Sequence, TypeVar, Union

from spekk.transformations.util import flatten_and_split_dynamic_and_static
from spekk.trees import Tree

if TYPE_CHECKING:
    from spekk.spec import Spec

F = TypeVar("F", bound=Callable)
T_in_axes = Sequence[Union[int, None]]
T_vmap = Callable[[F, T_in_axes], F]


def vmap_for_loop_impl(f: F, in_axes: T_in_axes) -> F:
    import numpy as np

    def wrapped(*args):
        if len(args) != len(in_axes):
            raise ValueError(
                f"The number of args ({len(args)}) does not match the number of "
                f"in_axes ({len(in_axes)})."
            )
        if len(args) == 0:
            return f(*args)

        def get_args_at_index(i):
            return [
                arg[(slice(None),) * axis + (i,)] if axis is not None else arg
                for arg, axis in zip(args, in_axes)
            ]

        sizes = {
            arg.shape[axis] for arg, axis in zip(args, in_axes) if axis is not None
        }
        if len(sizes) > 1:
            raise ValueError(
                "Inconsistent sizes between the arguments for the specified in_axes "
                f"axes. Sizes: {sizes}"
            )

        size = list(sizes)[0]
        all_results = [f(*get_args_at_index(i)) for i in range(size)]
        result1 = all_results[0]
        if Tree.is_tree_like(result1):
            return Tree.merge(*all_results, by=lambda xs: np.stack(xs, axis=0))
        else:
            return np.stack(all_results, axis=0)

    return wrapped


def specced_vmap(
    f: F,
    spec: "Spec",
    dim: str,
    base_vmap_impl: Optional[T_vmap] = None,
) -> F:
    """specced_vmap is like vmap, but takes a Spec and the name of a dimension that will be vectorized over instead of in_axes.

    specced_vmap also destructures both the input and output so that the underlying vmap implementation only sees tuples of arrays that are vectorized over. The mapping function f is agnostic to the destructuring and sees the object normally.

    Currently only kwargs are supported as arguments to f because I can't figure out the best way to represent both positional and keyword arguments in a Spec."""
    if dim not in spec.dimensions:
        raise ValueError(
            f"Could not find dimension '{dim}' "
            f"in spec with dimensions: {spec.dimensions}."
        )

    # Use the slow Python for-loop implementation of `vmap` if not provided.
    if base_vmap_impl is None:
        base_vmap_impl = vmap_for_loop_impl

    @functools.wraps(f)
    def wrapped_outer(*args, **kwargs):
        if args:
            raise ValueError("Positional args not supported. Use kwargs instead.")

        # Flatten the arguments into dynamic and static args. Only dynamic_args are
        # actually vmapped over, everything else is treated as "static". The vmap
        # implementation won't see static args.
        dynamic_args, static_args, in_axes, unflatten = (
            flatten_and_split_dynamic_and_static(kwargs, spec, dim)
        )

        # These are set inside `wrapped_inner` using the `nonlocal` statement.
        flattened_data_inner, unflatten_inner = None, None

        # wrapped_inner unflattens the args back to its original structure before
        # calling f on them. Note that these args have now been vmapped over and will
        # have one fewer dimension.
        def wrapped_inner(*args):
            # Unflatten args, alongside static args, back to the original structure.
            kwargs_inner = unflatten(args, static_args)
            # Call f on the vmapped, unflattened args.
            result_inner = f(**kwargs_inner)

            # Use nonlocal variables so that we can unflatten the final result.
            nonlocal flattened_data_inner, unflatten_inner
            # We flatten the result again so that the output of the vmapped function is
            # a tuple. NOTE: it is not guaranteed to be a tuple of only arrays/tensors
            # which may confuse some vmap implementations. A larger discussion is
            # whether we should include the concept of dynamic and static values in a
            # spekk trees. Inspiration for this is Equinox's filter_vmap.
            flattened_data_inner, unflatten_inner = Tree.flatten(result_inner)
            return flattened_data_inner

        # vmap wrapped_inner and call it
        vmapped_wrapped_inner = base_vmap_impl(wrapped_inner, in_axes)
        result_outer = vmapped_wrapped_inner(*dynamic_args)
        # Unflatten it (a final time!) to have the same structure as the output of f.
        return unflatten_inner(result_outer)

    # Return wrapped_outer, which is essentially vmap(f).
    return wrapped_outer
