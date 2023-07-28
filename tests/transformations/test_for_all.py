import numpy as np

from spekk import Spec
from spekk.transformations import ForAll, compose


def test_multiple_for_alls():
    kernel = lambda x: x + 1
    data = {"x": np.ones((2, 3, 4))}
    spec = Spec({"x": ["a", "b", "c"]})

    tf1 = compose(kernel, ForAll("a"), ForAll("b"), ForAll("c")).build(spec)
    tf2 = compose(kernel, ForAll(["a", "b", "c"])).build(spec)

    assert tf1(**data) == tf2(**data)
    assert tf1.input_spec == tf2.input_spec
    assert tf1.passed_spec == tf2.passed_spec
    assert tf1.returned_spec == tf2.returned_spec
    assert tf1.output_spec == tf2.output_spec
