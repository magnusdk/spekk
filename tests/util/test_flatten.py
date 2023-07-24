from random import Random

import hypothesis as ht
import hypothesis.strategies as st
import numpy as np
from test_helpers.generators.spec import specs
from test_helpers.mock_data import generate_mock_data

from spekk import Spec, trees, util


def test_flattening_dict():
    foo_x = np.ones((2, 3)) * 4
    foo_y = np.ones((2,)) * 2
    bar = np.ones((3,)) * 3
    baz_z = np.ones((4,))
    obj = {
        "foo": {
            "x": foo_x,
            "y": [foo_y, foo_y],
        },
        "bar": bar,
        "baz": {"z": baz_z},
    }
    spec = Spec(
        {
            "foo": {
                "x": ["b", "a"],
                "y": [["a"], ["a"]],
            }
        }
    )

    # "x" and "y" should be split apart since "x" contains dimension "b"
    flattened, in_axes, unflatten = util.flatten(obj, spec, "b")
    np.testing.assert_equal(flattened, [foo_x, [foo_y, foo_y], bar, {"z": baz_z}])
    assert in_axes == [0, None, None, None]
    np.testing.assert_equal(obj, unflatten(flattened))

    # "x" and the list items under "y" should be split apart since they all contain
    # dimension "a"
    flattened, in_axes, unflatten = util.flatten(obj, spec, "a")
    np.testing.assert_equal(flattened, [foo_x, foo_y, foo_y, bar, {"z": baz_z}])
    assert in_axes == [1, 0, 0, None, None]
    np.testing.assert_equal(obj, unflatten(flattened))


@ht.given(specs())
def test_flatten_unflatten_data(spec: Spec):
    """Test that for any spec with corresponding valid data, flattening and 
    unflattening the data for any dimension results in the original data."""
    data = generate_mock_data(spec)
    for dimension in spec.dimensions:
        flattened, _, unflatten = util.flatten(data, spec, dimension)
        assert trees.are_equal(
            unflatten(flattened),
            data,
            lambda x: isinstance(x, (np.ndarray, float, int)),
            np.array_equal,
        )

@ht.given(specs())
def test_flattened_in_axes(spec: Spec):
    """Test that for any spec with corresponding valid data, the `in_axes` calculated 
    from flattening the data have at least 1 non-None value."""
    data = generate_mock_data(spec)
    for dimension in spec.dimensions:
        _, in_axes, _ = util.flatten(data, spec, dimension)
        assert not all(x is None for x in in_axes)
