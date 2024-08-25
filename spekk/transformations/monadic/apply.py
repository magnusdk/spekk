from functools import reduce
from typing import Callable

from spekk import common
from spekk.spec import Spec
from spekk.transformations.monadic.axis import Axis, concretize_axes
from spekk.transformations.monadic.base import Transformation
from spekk.trees import traverse_iter


class Apply(Transformation):
    """Transform a function f such that the given function is applied to the result.
    See Axis for how to reference dimensions from a Spec.

    Example:
    >>> f = lambda x, y: x+y
    >>> transformed_f = Apply(lambda x: x+3)(f)
    >>> assert f(1,2) == 6, "1+2+3 == 6"
    """

    def __init__(self, function: Callable, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def transform_input_spec(self, spec: Spec):
        return spec

    def transform_output_spec(self, spec: Spec):
        for x in traverse_iter((self.args, self.kwargs)):
            if isinstance(x, Axis):
                spec = x.update_spec(spec)
        return spec

    def transform_function(self, f: Callable, input_spec: Spec, returned_spec: Spec):
        extra_args, extra_kwargs = concretize_axes(
            returned_spec, (self.args, self.kwargs)
        )

        def with_applied(*args, **kwargs):
            inner_result = f(*args, **kwargs)
            return self.function(inner_result, *extra_args, **extra_kwargs)

        return with_applied

    def __repr__(self) -> str:
        args_str = reduce(lambda carry, x: carry + f", {x}", self.args, "")
        kwargs_str = reduce(
            lambda carry, x: carry + f", {x[0]}={str(x[1])}", self.kwargs.items(), ""
        )
        return f"Apply({common.get_fn_name(self.function)}{args_str}{kwargs_str})"
