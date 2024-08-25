import functools
import operator
from typing import Any, Callable, Dict, Optional, Sequence, Type

from vbeam.fastmath import numpy as np

from spekk.spec import Spec
from spekk.transformations.monadic.axis import Axis, concretize_axes
from spekk.transformations.monadic.base import Transformation
from spekk.transformations.specced_map_reduce import (
    T_reduce,
    T_reduce_f,
    T_reduction_result,
    specced_map_reduce,
)
from spekk.trees import traverse_iter


class Reduce(Transformation):
    reduce_impl: Optional[T_reduce] = None

    def __init__(
        self,
        dim: str,
        reduce_fn: T_reduce_f,
        initial_value: Optional[T_reduction_result] = None,
        enumerate: bool = False,
        extra_args: Sequence = (),
        extra_kwargs: Dict[str, Any] = {},
    ):
        self.dim = dim
        self.reduce_fn = reduce_fn
        self.initial_value = initial_value
        self.enumerate = enumerate
        self.extra_args = extra_args
        self.extra_kwargs = extra_kwargs

    def transform_input_spec(self, spec: Spec):
        return spec.remove_dimension(self.dim)

    def transform_output_spec(self, spec: Spec):
        for x in traverse_iter((self.extra_args, self.extra_kwargs)):
            if isinstance(x, Axis):
                spec = x.update_spec(spec)
        return spec

    def transform_function(self, f: Callable, input_spec: Spec, returned_spec: Spec):
        extra_args, extra_kwargs = concretize_axes(
            returned_spec, (self.extra_args, self.extra_kwargs)
        )

        @functools.wraps(self.reduce_fn)
        def reduce_fn_with_extra_args(carry, x):
            return self.reduce_fn(carry, x, *extra_args, **extra_kwargs)

        return specced_map_reduce(
            f,
            reduce_fn_with_extra_args,
            input_spec,
            self.dim,
            self.initial_value,
            self.enumerate,
            np.reduce,
        )

    @classmethod
    def Sum(cls: Type["Reduce"], dim: str, initial_value: float = 0.0) -> "Reduce":
        """Transformation that iteratively adds the results of the wrapped function for
        each item in the given dimension."""
        return cls(dim, operator.add, initial_value)

    @classmethod
    def Product(cls: Type["Reduce"], dim: str, initial_value: float = 1.0) -> "Reduce":
        """Transformation that iteratively multiplies the results of the wrapped
        function for each item in the given dimension."""
        return cls(dim, operator.mul, initial_value)
