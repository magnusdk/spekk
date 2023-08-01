import jax
import numpy as np

from spekk import Spec, util
from spekk.transformations import ForAll, compose


def test_multiple_for_alls():
    f = lambda x: x + 1
    data = {"x": np.ones((2, 3, 4, 5))}
    spec = Spec({"x": ["2", "3", "4", "5"]})

    tf1 = compose(f, ForAll("3"), ForAll("5"), ForAll("2"), ForAll("4")).build(spec)
    tf2 = compose(f, ForAll("4", "2", "5", "3")).build(spec)  # Reversed order

    assert tf1(**data) == tf2(**data)
    assert tf1.input_spec == tf2.input_spec
    assert tf1.output_spec == tf2.output_spec
