from dataclasses import dataclass
from functools import reduce
from typing import Any, Sequence, Union

from spekk.spec import Spec, is_spec_dimensions
from spekk.trees import traverse, traverse_leaves


def replace(xs: list, a: Any, b: list):
    return reduce(lambda carry, x: carry + b if x == a else carry + [x], xs, [])


@dataclass
class Axis:
    dim: str
    keep: bool = False
    becomes: Sequence[str] = ()

    def update_spec(self, spec: Spec) -> Spec:
        if self.becomes:
            return traverse_leaves(
                lambda x: Spec(replace(x, self.dim, self.becomes)),
                spec,
                is_leaf=is_spec_dimensions,
            )
        if self.keep:
            return spec
        return spec.remove_dimension(self.dim)


class AxisConcretizationError(ValueError):
    def __init__(self, axis: Axis, spec: Spec):
        super().__init__(f'Could not find dimension "{axis.dim}" in the spec {spec}.')


def concretize_axes(spec: Spec, data: Any):
    spec_dimensions = spec.dimensions

    def f(x):
        if isinstance(x, Axis):
            if x.dim not in spec_dimensions:
                raise AxisConcretizationError(x, spec)
            return spec.indices_for(x.dim)
        return x

    return traverse(f, data)


def test_axis_update_spec_keep():
    spec = Spec({"a": ["d1", "d2"]})
    axis = Axis("d2", keep=True)
    assert axis.update_spec(spec) == spec


def test_axis_update_spec_becomes():
    spec = Spec({"a": ["d1", "d2"]})
    axis = Axis("d2", becomes=["d3", "d4"])
    assert axis.update_spec(spec) == Spec({"a": ["d1", "d3", "d4"]})


test_axis_update_spec_keep()
test_axis_update_spec_becomes()
