import numpy as np
import pytest

from spekk.trees import traverse
from spekk.spec import Spec, is_spec_dimensions


def test_spec_unwrapping():
    "Creating a new Spec will unwrap all nested Spec objects."
    spec = traverse(
        lambda x: x,
        Spec({"a": {"b": Spec(["dim1", "dim2"])}, "c": ["dim1", "dim2"]}),
        should_stop=is_spec_dimensions,
    )
    assert spec.data == {"a": {"b": ["dim1", "dim2"]}, "c": ["dim1", "dim2"]}


def test_adding_dimension():
    spec = Spec({"a": {"b": ["dim1", "dim2"]}, "c": ["dim1", "dim2"]})
    assert spec.add_dimension("dim3") == Spec(
        {"a": {"b": ["dim3", "dim1", "dim2"]}, "c": ["dim3", "dim1", "dim2"]}
    )
    assert spec.add_dimension("dim3", 1) == Spec(
        {"a": {"b": ["dim1", "dim3", "dim2"]}, "c": ["dim1", "dim3", "dim2"]}
    )
    assert spec.add_dimension("dim3", -1) == Spec(
        {"a": {"b": ["dim1", "dim2", "dim3"]}, "c": ["dim1", "dim2", "dim3"]}
    )

    # Test adding a dimension using lenses (I.e.: spec.at(...))
    assert spec.at("c").add_dimension("dim3") == Spec(
        {"a": {"b": ["dim1", "dim2"]}, "c": ["dim3", "dim1", "dim2"]}
    )


def test_removing_dimension():
    spec = Spec({"a": {"b": ["dim1", "dim2"]}, "c": ["dim1", "dim3"]})
    assert spec.remove_dimension("dim1") == Spec({"a": {"b": ["dim2"]}, "c": ["dim3"]})

    # Removing dimensions also automatically prunes empty branches
    assert spec.remove_dimension("dim1", "dim2") == Spec({"c": ["dim3"]})

    # Test removing a dimension using lenses
    assert spec.at("c").remove_dimension("dim1") == Spec(
        {"a": {"b": ["dim1", "dim2"]}, "c": ["dim3"]}
    )
    assert spec.at("c").remove_dimension("dim1", "dim3") == Spec(
        {"a": {"b": ["dim1", "dim2"]}}
    )


def test_is_spec_dimensions():
    # A list of strings are considered dimensions
    assert is_spec_dimensions(["dim1", "dim2"])
    # So is an empty list
    assert is_spec_dimensions([])
    # ...and None means that we don't care about the dimensions
    assert is_spec_dimensions(None)

    # Dimensions can be wrapped with Spec
    assert is_spec_dimensions(Spec(["dim1", "dim2"]))
    assert is_spec_dimensions(Spec([]))

    # Dimensions can only be a list of strings, not f.ex. integers
    assert not is_spec_dimensions([1, 2, 3])
    assert not is_spec_dimensions(Spec([1, 2, 3]))

    # A tree datastructure, f.ex. a dict or list (of non-strings), is not dimensions
    assert not is_spec_dimensions(Spec({"a": ["dim1", "dim2"]}))
    assert not is_spec_dimensions(Spec([["dim1", "dim2"]]))


def test_conform():
    spec = Spec({"b": ["dim1", "dim2"]})
    data = {
        "a": [np.array([[1], [2], [3]])],
        "b": {
            "c": np.array([4, 6]),
            "d": [6, 7],
        },
    }
    assert spec.conform(data) == Spec(
        {
            "a": [None],
            "b": {
                "c": ["dim1", "dim2"],
                "d": [["dim1", "dim2"], ["dim1", "dim2"]],
            },
        }
    )


def test_indices_for():
    spec = Spec(
        {
            "a": [["dim1", "dim2"]],
            "b": ["dim1"],
        }
    )
    data = {
        "a": [np.array([[1], [2]])],
        "b": {"c": np.array([3, 4])},
    }
    assert spec.indices_for("dim1") == {"a": [0], "b": 0}
    assert spec.indices_for("dim2") == {"a": [1], "b": None}
    assert spec.indices_for("dim1") == {"a": [0], "b": 0}
    assert spec.indices_for("dim2", conform=data) == {"a": [1], "b": {"c": None}}


def test_dimensions():
    spec = Spec({})
    assert spec.dimensions == set()
    spec = Spec({"b": ["dim1", "dim2"]})
    assert spec.dimensions == {"dim1", "dim2"}
    spec = Spec({"a": {"b": ["dim1", "dim2"]}, "c": ["dim1", "dim3"]})
    assert spec.dimensions == {"dim1", "dim2", "dim3"}


def test_size():
    spec = Spec(
        {
            "a": [["dim1", "dim2"]],
            "b": {"c": ["dim1"]},
        }
    )
    data = {
        "a": [np.array([[1], [2]])],
        "b": {"c": np.array([3, 4])},
    }
    assert spec.size(data, "dim1") == 2
    assert spec.size(data, "dim2") == 1
    assert spec.size(data) == {"dim1": 2, "dim2": 1}


if __name__ == "__main__":
    pytest.main([__file__])
