from dataclasses import dataclass
import numpy as np
import pytest

from spekk import Module, field, ops, replace, replace_at, traverse, update_at

# TODO: Remove me
ops.backend.set_backend("numpy")


def test_creating_class():
    class A(Module):
        a: float
        b: float

    obj = A(1.0, 20.0)
    assert obj.a == 1.0
    assert obj.b == 20.0


def test_creating_subclass():
    class A(Module):
        a: int
        b: int

    class B(A):
        c: int

    obj = B(1, 2, 3)
    assert obj.a == 1
    assert obj.b == 2
    assert obj.c == 3


def test_creating_class_with_custom_init():
    class A(Module):
        a: float
        b: float

        def __init__(self, a: float):
            self.a = a
            self.b = 20.0

    obj = A(1.0)
    assert obj.a == 1.0
    assert obj.b == 20.0


def test_creating_class_with_post_init():
    class A(Module):
        a: float
        b: float

        def __post_init__(self):
            self.b *= 2

    obj = A(1.0, 10.0)
    assert obj.a == 1.0
    assert obj.b == 20.0


def test_module_dim_sizes():
    class A(Module):
        a: ops.array
        b: ops.array

    class B(Module):
        c: A
        d: ops.array

    obj = B(
        A(
            ops.ones((2, 3, 4), dims=["d2", "d3", "d4"]),
            ops.ones((5, 4), dims=["d5", "d4"]),
        ),
        ops.ones((6, 2), dims=["d6", "d2"]),
    )
    assert obj.dim_sizes == {"d2": 2, "d3": 3, "d4": 4, "d5": 5, "d6": 6}

    # Inconsistent sizes for a dimension means a set is returned for that dimension.
    obj = B(
        A(
            ops.ones((2,), dims=["d"]),
            ops.ones((5,), dims=["d"]),
        ),
        ops.ones((4,), dims=["d"]),
    )
    assert obj.dim_sizes == {"d": {2, 5, 4}}

    # Undefined dimensions are put into its own dict.
    obj = B(
        A(
            ops.ones((2,), dims=["d2"]),
            ops.ones((5,)),
        ),
        ops.ones((2,), dims=["d2"]),
    )
    assert obj.dim_sizes["d2"] == 2
    # It is a bit difficult to make assertions about undefined dimensions (with the
    # current implementation).
    assert len(obj.dim_sizes) == 2


def test_module_at():
    class A(Module):
        a: ops.array
        b: ops.array

    class B(Module):
        c: A
        d: ops.array

    obj = B(
        A(
            ops.ones((2, 3, 4), dims=["d2", "d3", "d4"]),
            ops.ones((5, 4), dims=["d5", "d4"]),
        ),
        ops.ones((6, 2), dims=["d6", "d2"]),
    )
    sliced_obj = obj.at["d4", ::2, "d5", 0].get()
    assert sliced_obj.dim_sizes == {"d2": 2, "d3": 3, "d4": 2, "d6": 6}

    sliced_obj = obj.at["d4", ::2, "d5", 0].set(0)
    assert sliced_obj.dim_sizes == {"d2": 2, "d3": 3, "d4": 4, "d5": 5, "d6": 6}


###############


def test_replace():
    class A(Module):
        a: int
        b: int

    obj = A(1, 2)
    new_obj = replace(obj, a=42)
    assert new_obj.a == 42
    assert obj.a == 1
    assert new_obj.b == 2


# Tests for replace_at
def test_replace_at():
    class A(Module):
        x: int
        y: int

    class B(Module):
        a: A
        b: int

    obj = B(A(10, 20), 30)
    # Replace 'x' inside the nested A instance with 99.
    new_obj = replace_at(obj, ["a", "x"], 99)
    assert new_obj.a.x == 99
    assert obj.a.x == 10
    assert new_obj.a.y == 20
    assert new_obj.b == 30


def test_replace_at_empty_path():
    class A(Module):
        a: int

    a_obj = A(10)
    # An empty path should return the new value directly.
    new_value = "new"
    new_obj = replace_at(a_obj, [], new_value)
    assert new_obj == new_value


# Tests for update_at
def test_update_at():
    class A(Module):
        x: int
        y: int

    class B(Module):
        a: A
        b: int

    obj = B(A(10, 20), 30)

    def add_five(x):
        return x + 5

    # Update the 'x' field of the nested A instance.
    new_obj = update_at(obj, ["a", "x"], add_five)
    assert new_obj.a.x == 15
    assert obj.a.x == 10
    assert new_obj.a.y == 20
    assert new_obj.b == 30


def test_update_at_empty_path():
    class A(Module):
        a: int

    obj = A(10)

    def double(obj):
        # Here f receives the whole Module instance.
        return replace(obj, a=obj.a * 2)

    # With an empty path, update_at applies f to the entire object.
    new_obj = update_at(obj, [], double)
    assert new_obj.a == 20
    assert obj.a == 10


# TODO LOOK THROUGH REMAINING
# Tests for traverse
def test_traverse_map_leaf():
    class A(Module):
        a: int
        b: list

    obj = A(3, [1, 2, 3])

    def multiply(x):
        if isinstance(x, int):
            return x * 10
        return x

    new_obj = traverse(obj, map_leaf=multiply)
    assert new_obj.a == 30
    assert new_obj.b == [10, 20, 30]


def test_traverse_on_container():
    # Test traversing over a list of dictionaries.
    data = [{"a": 1}, {"a": 2}]

    def add_one(x):
        if isinstance(x, int):
            return x + 5
        return x

    new_data = traverse(data, map_leaf=add_one)
    assert new_data == [{"a": 6}, {"a": 7}]


def test_traverse_map_static_field():
    # Test that static fields are mapped using map_static_field.
    class A(Module):
        a: int
        b: int = field(static=True)

    obj = A(5, 100)

    def static_mapper(x):
        return x + 1

    new_obj = traverse(obj, map_static_field=static_mapper)
    # Field 'a' is not static so it remains unchanged; 'b' is static and is modified.
    assert new_obj.a == 5
    assert new_obj.b == 101


if __name__ == "__main__":
    test_module_at()
