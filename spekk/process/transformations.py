from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence, Union

from spekk.process.axis import Axis, concretize_axes
from spekk.spec import Spec
from spekk.trees import has_treedef, leaves


@dataclass
class Kernel(ABC):
    f: Callable
    required_spec: Spec
    output_spec: Spec

    def __repr__(self) -> str:
        return f"Kernel({self.f})"


class Transformation(ABC):
    """A Transformation object acts as a higher-order function that takes a function and
    a spec and returns a new transformed function. It also knows what happens to the
    spec when the transformation is applied."""

    @abstractmethod
    def __call__(self, f: Callable, input_spec: Spec, output_spec: Spec) -> Callable:
        ...

    @abstractmethod
    def preprocess_spec(self, spec: Spec) -> Spec:
        ...

    @abstractmethod
    def postprocess_spec(self, spec: Spec) -> Spec:
        ...


@dataclass
class ForAll(Transformation):
    """A transformation that vectorizes a function over a dimension such that it can
    take multiple values at once.

    vmap is a higher-order function that is used to vectorize another function. An
    example of this is jax.vmap from the JAX library, and you should refer to their
    documentation for more details.
    """

    dimension: str
    vmap: Callable[[Callable, Sequence[Union[int, None]]], Callable]

    def __call__(self, f: Callable, input_spec: Spec, output_spec: Spec) -> Callable:
        """Vectorize the function over the dimension."""
        in_axes = input_spec.index_for(self.dimension)
        if isinstance(in_axes, dict):
            in_axes = list(in_axes.values())
        return self.vmap(f, in_axes)

    def preprocess_spec(self, spec: Spec) -> Spec:
        """Remove the dimension from the spec that is passed to the transformed
        function. Vectorization ensures that the function only has to know about one
        value at a time, so we remove the dimension.

        See Transformation.preprocess_spec for more information."""
        return spec.remove_dimension(self.dimension)

    def postprocess_spec(self, spec: Spec) -> Spec:
        """We add the dimension back to the spec as the result contains the dimension
        that was vectorized over.

        See Transformation.postprocess_spec for more information."""
        return spec.add_dimension(self.dimension)

    def __repr__(self) -> str:
        return f'ForAll("{self.dimension}", {self.vmap.__qualname__})'


class Apply(Transformation):
    """A transformation that applies f to the result of the transformed function.

    It also knows how to update the spec given the args and kwargs (if all axes are
    defined using the Axis class).

    >>> import numpy as np
    >>> x = np.ones((3, 4))
    >>> f = lambda x: x + 1
    >>> input_spec = Spec({"x": ("a", "b")})
    >>> output_spec = Spec(("a", "b"))
    >>> transform_apply = Apply(np.sum, Axis("a"))
    >>> transformed_f = transform_apply(f, input_spec, output_spec)
    >>> transformed_f(x)
    array([6., 6., 6., 6.])
    >>> transform_apply.postprocess_spec(output_spec)
    Spec(('b',))
    """

    def __init__(self, f: Callable, *args, **kwargs):
        self.f, self.args, self.kwargs = f, args, kwargs

    def __call__(self, fn_to_wrap, input_spec: Spec, output_spec: Spec) -> Callable:
        """Return a function that calls fn_to_wrap and applies self.f to the result."""

        def inner(*inner_args) -> Callable:
            result = fn_to_wrap(*inner_args)
            args, kwargs = concretize_axes(output_spec, self.args, self.kwargs)
            return self.f(result, *args, **kwargs)

        return inner

    def preprocess_spec(self, spec: Spec) -> Spec:
        return spec  # Apply doesn't change the input.

    def postprocess_spec(self, spec: Spec) -> Spec:
        """Update the spec according to the Axis objects in self.args and self.kwargs.

        See Axis docstring for more details."""

        state = spec
        tree = (self.args, self.kwargs)
        for leaf in leaves(tree, lambda x: isinstance(x, Axis) or not has_treedef(x)):
            if isinstance(leaf.value, Axis):
                state = state.update_leaves(leaf.value.new_dimensions)
        return state

    def __repr__(self) -> str:
        args_str = ", ".join([str(arg) for arg in self.args])
        kwargs_str = ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        repr_str = f"Apply({str(self.f.__qualname__)}"
        if self.args:
            repr_str += f", {args_str}"
        if self.kwargs:
            repr_str += f", {kwargs_str}"
        return repr_str + ")"


if __name__ == "__main__":
    import doctest

    doctest.testmod()
