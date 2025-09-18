"""Test suite for the new setitem implementation in _indexing.py"""

from spekk import ops
from spekk.ops._indexing import setitem


def test_basic_setitem_positional():
    """Test basic setitem with named dimension indexing (converted from positional test)"""
    x = ops.zeros((3, 4), dims=["a", "b"])
    value = ops.ones((4,), dims=["b"])

    # Set first row using named dimension
    result = setitem(x, {"a": 0}, value)

    # Check that the result has the same dimensions as x
    assert result.dims == ["a", "b"]
    assert result.shape == (3, 4)

    # Check that the first row was set to ones
    first_row = result["a", 0]
    assert first_row.shape == value.shape
    assert ops.all(first_row == value)
    # Check that other rows are still zeros
    second_row = result["a", 1]
    assert ops.all(second_row == 0)


def test_basic_setitem_named_dimensions():
    """Test basic setitem with named dimension indexing"""
    x = ops.zeros((3, 4), dims=["a", "b"])
    value = ops.ones((4,), dims=["b"])

    # Set first row using named dimension
    result = setitem(x, {"a": 0}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (3, 4)
    assert ops.all(result["a", 0] == value)


def test_setitem_with_new_dimensions():
    """Test setitem where value has new dimensions that get added to x"""
    x = ops.zeros((3, 4), dims=["a", "b"])
    # Value has a new dimension "c"
    value = ops.ones((4, 2), dims=["b", "c"])

    # This should expand x to include dimension "c"
    result = setitem(x, {"a": 0}, value)

    # Result should now have 3 dimensions: a, b, c
    expected_dims = ["a", "b", "c"]
    assert set(result.dims) == set(expected_dims)
    assert result.shape == (3, 4, 2)  # x expanded to include c=2


def test_setitem_with_array_indexing():
    """Test setitem with array indexing"""
    x = ops.zeros((5, 4), dims=["a", "b"])
    # Index specific elements of dimension "a"
    indices = ops.array([0, 2, 4], dims=["idx"])
    value = ops.ones((3, 4), dims=["idx", "b"])

    result = setitem(x, {"a": indices}, value)

    # Result should have original dimensions plus the indexing dimension
    expected_dims = ["a", "b", "idx"]
    assert set(result.dims) == set(expected_dims)


def test_setitem_boolean_indexing():
    """Test setitem with boolean indexing"""
    x = ops.array([1, 2, 3, 4, 5], dims=["a"])
    mask = ops.array([True, False, True, False, False], dims=["a"])
    value = ops.array([10, 30], dims=["a"])  # Will be flattened

    result = setitem(x, (mask,), value)

    # Should have same dimensions as original
    assert result.dims == x.dims
    # Elements at positions 0 and 2 should be updated
    expected = ops.array([10, 2, 30, 4, 5], dims=["a"])
    assert ops.all(result == expected)


def test_setitem_slice_indexing():
    """Test setitem with slice indexing"""
    x = ops.zeros((5, 4), dims=["a", "b"])
    value = ops.ones((3, 4), dims=["a", "b"])

    result = setitem(x, {"a": slice(1, 4)}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (5, 4)
    # Check that rows 1-3 were set to ones
    sliced_result = result["a", slice(1, 4)]
    assert ops.all(sliced_result == value)
