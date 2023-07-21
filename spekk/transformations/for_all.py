""":class:`ForAll` transforms a function that works on scalar inputs such that it works 
on arrays instead (vectorization), and can be used with :func:`jax.vmap`."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from spekk import Spec, trees, util
from spekk.transformations import Transformation, common

T_in_axes = Sequence[Optional[int]]
T_vmap = Callable[[callable, T_in_axes], callable]


@dataclass
class ForAll(Transformation):
    """Vectorize/"make looped" a function such that it works on arrays instead of
    scalars.
    """

    dimension: str  #: The dimension to vectorize/loop over.
    vmap_impl: Optional[
        T_vmap
    ] = None  #: The ``vmap`` implementation to use. Defaults to a simple Python implementation, but can also (for example) be set to :func:`jax.vmap`.

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
        return spec.update_leaves(lambda dimensions: [self.dimension, *dimensions])

    def __repr__(self) -> str:
        return f'ForAll("{self.dimension}")'


def specced_vmap(
    f: callable,
    spec: Spec,
    dimension: str,
    vmap_impl: Optional[T_vmap] = None,
):
    """Similar to ``vmap``, but flattens/decomposes the ``kwargs`` to a list that is
    supported by ``vmap``.
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
    """A simple Python implementation of JAX's :func:`jax.vmap` based on for-loops."""

    def wrapped(*args):
        sizes = [util.shape(arg)[a] for arg, a in zip(args, in_axes) if a is not None]
        size = sizes[0]
        if not all(s == size for s in sizes):
            raise ValueError(
                f"Cannot apply python_vmap to arguments with different sizes over the \
in_axes: {sizes=}, {in_axes=}"
            )

        # The result for each item in the dimension.
        all_results = [
            f(*common.get_args_for_index(args, in_axes, i)) for i in range(size)
        ]
        result0 = all_results[0]

        # Combine the results such that the returned object has the same shape as each
        # individual result.
        combined_result = result0
        for leaf in trees.leaves(
            result0, lambda x: isinstance(x, list) or not trees.has_treedef(x)
        ):
            values = [trees.get(_result, leaf.path) for _result in all_results]
            combined_result = trees.set(combined_result, values, leaf.path)
        return combined_result

    return wrapped
