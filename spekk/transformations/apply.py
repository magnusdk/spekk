":class:`Apply` applies function ``f`` to the output of the wrapped function."

from typing import Callable

import spekk.transformations.common as common
from spekk import Spec, trees
from spekk.transformations.axis import Axis, concretize_axes
from spekk.transformations.base import Transformation


class Apply(Transformation):
    """Transform a function such that ``f`` is applied to the output of it.

    Attributes:
        f: The function to apply to the result of the wrapped function.
        args: Optional extra positional arguments to pass to ``f``.
        kwargs: Optional extra keyword arguments to pass to ``f``.
    """

    def __init__(self, f: Callable, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def transform_function(
        self, to_be_transformed: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        def with_applied_f(*args, **kwargs):
            result = to_be_transformed(*args, **kwargs)
            args, kwargs = concretize_axes(output_spec, self.args, self.kwargs)
            return self.f(result, *args, **kwargs)

        return with_applied_f

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        tree = (self.args, self.kwargs)
        for leaf in trees.leaves(
            tree, lambda x: isinstance(x, Axis) or not trees.has_treedef(x)
        ):
            if isinstance(leaf.value, Axis):
                spec = spec.update_leaves(leaf.value.new_dimensions)
        extra_output_spec_transform = getattr(self, "extra_output_spec_transform", None)
        if extra_output_spec_transform:
            spec = extra_output_spec_transform(spec)
        return spec

    def with_extra_output_spec_transform(self, t: Callable[[Spec], Spec]):
        copy = Apply(self.f, *self.args, **self.kwargs)
        copy.extra_output_spec_transform = t
        return copy

    def __repr__(self) -> str:
        args_str = ", ".join([str(arg) for arg in self.args])
        kwargs_str = ", ".join([f"{k}={str(v)}" for k, v in self.kwargs.items()])
        repr_str = f"Apply({common.get_fn_name(self.f)}"
        if self.args:
            repr_str += f", {args_str}"
        if self.kwargs:
            repr_str += f", {kwargs_str}"
        # Make sure the repr string is not too long
        if len(repr_str) > 140:
            repr_str = repr_str[: (140 - len("… <truncated>"))] + "… <truncated>"
        return repr_str + ")"
