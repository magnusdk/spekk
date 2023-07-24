import numpy as np

from spekk import Spec

spec = Spec({"foo": ["a", "b"], "bar": ["b"]})
deeper_spec = Spec({"foo": {"baz": ["a", "b"], "quaz": ["c"]}, "bar": ["b"]})
sequence_spec = Spec([["a", "b"], ["b"]])


def test_getitem():
    # Getting the value or subtree at the given path returns a new Spec object.
    assert spec.get(["foo"]) == Spec(["a", "b"])
    assert spec.get(["bar"]) == Spec(["b"])

    assert spec["foo"] == Spec(["a", "b"])
    assert spec["bar"] == Spec(["b"])

    assert spec["foo"].tree == ["a", "b"]
    assert spec["bar"].tree == ["b"]


def test_is_leaf():
    assert not spec.is_leaf()
    assert spec["foo"].is_leaf()
    assert spec["bar"].is_leaf()


def test_remove_dimension():
    assert spec.remove_dimension("a") == Spec({"foo": ["b"], "bar": ["b"]})
    assert spec.remove_dimension("b") == Spec({"foo": ["a"], "bar": []})


def test_index_for():
    assert spec.index_for("a") == {"foo": 0, "bar": None}
    assert spec.index_for("b") == {"foo": 1, "bar": 0}


def test_get_dimensions():
    assert spec.dimensions == {"a", "b"}
    assert spec.remove_dimension("b").dimensions == {"a"}


def test_has_dimension():
    assert spec.has_dimension("a")
    assert spec.has_dimension("b")
    assert not spec.has_dimension("c")


def test_add_dimension():
    assert spec.add_dimension("c", ["foo"], 0) == Spec(
        {"foo": ["c", "a", "b"], "bar": ["b"]}
    )
    assert spec.add_dimension("c", ["foo"], 1) == Spec(
        {"foo": ["a", "c", "b"], "bar": ["b"]}
    )
    assert spec.add_dimension("c", ["foo"], 2) == Spec(
        {"foo": ["a", "b", "c"], "bar": ["b"]}
    )
    assert spec.add_dimension("c", ["bar"], 0) == Spec(
        {"foo": ["a", "b"], "bar": ["c", "b"]}
    )
    assert spec.add_dimension("c", ["bar"], 1) == Spec(
        {"foo": ["a", "b"], "bar": ["b", "c"]}
    )


def test_replace():
    # Removing a path by setting it to None
    assert spec.replace({"foo": None}) == Spec({"bar": ["b"]})
    # Replacing a path with a subtree
    assert spec.replace({"foo": {"baz": []}}) == Spec(
        {"foo": {"baz": []}, "bar": ["b"]}
    )

    # Removing a deeper path with a subtree potentially removes the whole subtree
    assert deeper_spec.replace({"foo": {"baz": None}}) == Spec(
        {"foo": {"quaz": ["c"]}, "bar": ["b"]}
    )
    assert deeper_spec.replace({"foo": {"baz": None, "quaz": None}}) == Spec(
        {"bar": ["b"]}
    )

    # A leaf is treated as a leaf, even when the original tree is a list
    assert sequence_spec.replace([{"foo": ["a", "b"]}]) == Spec(
        [{"foo": ["a", "b"]}, ["b"]]
    )
    assert sequence_spec.replace(["a", "b"]) == Spec(["a", "b"])


def test_update_leaves():
    assert spec.update_leaves(lambda dims: dims + ["c"]) == Spec(
        {"foo": ["a", "b", "c"], "bar": ["b", "c"]}
    )


def test_size():
    assert spec.size({"foo": np.ones([2, 3]), "bar": np.ones([3])}) == {"a": 2, "b": 3}
