from typing import Callable

from spekk.spec import Spec
from spekk.transformations.monadic.base import Transformation


class Wrap(Transformation):
    def __init__(self, f: Callable, *args, **kwargs):
        self.f = f
        self.args = args
        self.kwargs = kwargs

    def transform_input_spec(self, spec: Spec) -> Spec:
        return spec

    def transform_output_spec(self, spec: Spec) -> Spec:
        return spec

    def transform_function(
        self, to_be_wrapped: callable, input_spec: Spec, output_spec: Spec
    ) -> callable:
        return self.f(to_be_wrapped, *self.args, **self.kwargs)
