"""
Tests for named dimension indexing edge cases that might fail.
These tests are meant to identify potential bugs in the indexing implementation.
"""

import pytest
from spekk import ops
from spekk.ops._indexing import setitem

ops.backend.set_backend("numpy")
import numpy as np


def create_test_array(shape, dims):
    """Create a test array with given shape and dimension names."""
    data = np.arange(np.prod(shape)).reshape(shape)
    return ops.array(data, dims)


# Skip: this syntax is not allowed
# Test 1: Mixed tuple and dictionary syntax in complex scenarios
# def test_mixed_syntax_complex():
#    """Test complex mixed syntax that might confuse the parser."""
#    x = create_test_array((2, 3, 4), ["a", "b", "c"])
#    indices = ops.array([0, 1], ["new_dim"])
#    # This might fail if the parser can't handle mixed syntax correctly
#    result = x[{"a": 0}, "b", indices]
#    assert result.dim_sizes == {"new_dim": 2, "c": 4}


# Skip: indexing by float not allowed (also raised IndexError, but we shouldn't test this)
# Test 2: Indexing with float arrays (should fail)
# def test_float_array_indexing():
#    """Test that float arrays as indices raise errors."""
#    x = create_test_array((3, 4), ["a", "b"])
#    float_indices = ops.array([0.5, 1.5], ["new_dim"])
#    with pytest.raises(TypeError):
#        x["a", float_indices]


# Test 3: Boolean indexing with wrong dimension sizes
def test_boolean_wrong_size():
    """Test boolean indexing with mask that has wrong size."""
    x = create_test_array((3, 4), ["a", "b"])
    wrong_mask = ops.array([True, False], ["a"])  # Size 2, but "a" has size 3
    with pytest.raises(IndexError):
        x["a", wrong_mask]


# Test 4: Setitem with mismatched value shapes
def test_setitem_wrong_shape():
    """Test setitem with value that has wrong shape."""
    x = create_test_array((3, 4), ["a", "b"])
    wrong_value = ops.zeros((2, 3), dims=["a", "b"])  # Wrong shape
    with pytest.raises(ValueError):
        setitem(x, {"a": 0}, wrong_value)


# Skip: this is 100% intended behavior. Dimensions are broadcasted automatically by name, even in setitem, meaning that new dimensions can be added during in-place operations.
# Test 5: Setitem with completely different dimension names
# def test_setitem_wrong_dims():
#    """Test setitem with value that has completely different dimension names."""
#    x = create_test_array((3, 4), ["a", "b"])
#    wrong_dims_value = ops.reshape(
#        ops.ones((4,)), (4,), ["c"]
#    )  # Different dimension name
#    with pytest.raises(ValueError):
#        setitem(x, {"a": 0}, wrong_dims_value)


# Test 6: Setitem with boolean indexing edge cases
def test_setitem_boolean_edge_cases():
    """Test setitem with boolean indexing edge cases."""
    x = create_test_array((4, 3), ["a", "b"])
    mask = ops.array([True, False, True, False], ["a"])
    # Value has wrong number of elements for the mask
    wrong_value = ops.reshape(ops.ones((3, 3)), (3, 3), ["a", "b"])  # Should be (2, 3)
    with pytest.raises(ValueError):
        setitem(x, {"a": mask}, wrong_value)


# Skip: this is not valid syntax. If a dimension is referenced in tuple syntaxing, there must be an even number of elements in the tuple because it will be converted to a dictionary from dimension name to indexing object.
# Test 7: Newaxis (None) in named indexing
# def test_newaxis_in_named_indexing():
#    """Test newaxis (None) in named dimension indexing."""
#    x = create_test_array((2, 3), ["a", "b"])
#    # This might fail if newaxis handling is not implemented correctly
#    result = x["a", 0, None]
#    assert result.shape == (1, 3)


# Test 8: Setitem with scalar to multi-dimensional selection
def test_setitem_scalar_to_multidim():
    """Test setitem with scalar value to multi-dimensional selection."""
    x = create_test_array((3, 4), ["a", "b"])
    # This might fail if scalar broadcasting is not handled correctly
    result = setitem(x, {"a": 0}, 999)
    assert result.shape == (3, 4)


# Test 9: Circular reference in dimension names during indexing
def test_circular_dim_reference():
    """Test circular reference in dimension names during complex indexing."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    # Index "a" with array having dimension "b", and "b" with array having dimension "a"
    indices_a = ops.array([0, 1, 2, 1], ["b"])
    indices_b = ops.array([0, 1, 2], ["a"])
    # This might fail with circular dimension reference
    result = x["a", indices_a, "b", indices_b]
    assert result.dim_sizes == {"a": 3, "b": 4, "c": 5}


# Skip: this is also totally valid. Dimensions are broadcasted by name, new dimensions (and corresponding size) is added to the existing array and indices.
# Test 10: Multiple setitem operations with conflicting dimensions
# def test_multiple_setitem_conflicts():
#    """Test multiple setitem operations with conflicting dimensions."""
#    x = create_test_array((3, 4), ["a", "b"])
#    value = ops.reshape(ops.ones((2, 4)), (2, 4), ["c", "b"])  # Different dim name
#    # This should fail due to dimension name conflict
#    with pytest.raises(ValueError):
#        setitem(x, {"a": slice(0, 2)}, value)


# Test 11: Advanced indexing with dimension size mismatches
def test_advanced_dimension_mismatch():
    """Test advanced indexing with dimension size mismatches."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    # Create indices that cause dimension size conflicts
    indices_a = ops.array([0, 1], ["shared"])
    indices_b = ops.array([0, 1, 2], ["shared"])  # Same dim name, different size
    # This should fail due to dimension size mismatch
    with pytest.raises(IndexError):
        x["a", indices_a, "b", indices_b]


# Test 12: Boolean mask with wrong dimension names
def test_boolean_mask_wrong_dims():
    """Test boolean mask with wrong dimension names."""
    x = create_test_array((3, 4), ["a", "b"])
    # Create boolean mask with wrong dimension names
    wrong_mask = ops.array([True, False, True], ["wrong_dim"])
    # This should fail due to wrong dimension names
    with pytest.raises(IndexError):
        x["a", wrong_mask]


# Skip: we don't test for stuff like this. It is trivial and handled by the underlying backend arrays.
# Test 13: Indexing with complex number indices
# def test_complex_number_indices():
#    """Test indexing with complex number indices."""
#    x = create_test_array((3, 4), ["a", "b"])
#    # This should fail with complex number indices
#    with pytest.raises((TypeError, ValueError)):
#        x["a", 1 + 2j]

# Skip: same for this; trivially invalid.
# Test 14: Setitem with None value
# def test_setitem_none_value():
#    """Test setitem with None value."""
#    x = create_test_array((3, 4), ["a", "b"])
#    # This might fail with None value handling
#    with pytest.raises((TypeError, ValueError)):
#        setitem(x, {"a": 0}, None)


# Actually, this should not raise an error. A tuple is converted to an array with the same dimensions as the one being referenced.
# Test 15: Indexing with tuple as index
def test_tuple_as_index():
    """Test indexing with tuple as index."""
    x = create_test_array((3, 4), ["a", "b"])
    result = x["a", (0, 1)]
    assert result.dim_sizes == {"a": 2, "b": 4}


# Skip: we don't need to test this. Our test should revolve around the named dimensions and implicit broadcasting. This type of tests (indexing by large negative number) is handled by underlying backend array implementation.
# Test 16: Extremely large negative indices
# def test_extremely_large_negative_indices():
#    """Test extremely large negative indices."""
#    x = create_test_array((3, 4), ["a", "b"])
#    # This should fail with extremely large negative index
#    with pytest.raises(IndexError):
#        x["a", -1000]


# Skip: this test is testing indexing with integer array. This is tested elsewhere.
# Test 17: Boolean indexing with non-boolean array
# def test_non_boolean_array_as_mask():
#    """Test boolean indexing with non-boolean array."""
#    x = create_test_array((3, 4), ["a", "b"])
#    non_bool_mask = ops.array([0, 1, 0], ["a"])  # Integer array, not boolean
#    # This should work as integer indexing, not boolean
#    result = x["a", non_bool_mask]
#    assert result.dim_sizes == {"a": 3, "b": 4}


# Test 18: Indexing with list instead of array
def test_list_indexing():
    """Test indexing with Python list instead of array."""
    x = create_test_array((3, 4), ["a", "b"])
    # This might fail with list indexing
    result = x["a", [0, 1]]
    assert result.shape == (2, 4)


# Skip: this is totally valid. Arrays are implicitly broadcasted by dimension name, even during in-place operations like setitem. Dimensions in value not present in array are added to array with the corresponding size.
# Test 19: Setitem with mismatched broadcast dimensions
# def test_setitem_broadcast_mismatch():
#    """Test setitem with mismatched broadcast dimensions."""
#    x = create_test_array((3, 4), ["a", "b"])
#    # Create value with incompatible broadcast dimensions
#    value = ops.reshape(ops.ones((2, 2)), (2, 2), ["c", "d"])
#    # This should fail due to broadcast mismatch
#    with pytest.raises(ValueError):
#        setitem(x, {"a": slice(0, 2), "b": slice(0, 2)}, value)


# Test 20: Complex mixed indexing with errors
def test_complex_mixed_indexing_errors():
    """Test complex mixed indexing that should cause errors."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    # Mix different types of problematic indices
    indices = ops.array([0, 1, 2, 3], ["b"])  # Size 4, but b has size 4 (this is ok)
    # But then try to index b with slice that creates size mismatch
    with pytest.raises(IndexError):
        x["a", indices, "b", slice(0, 2)]  # This creates size mismatch


if __name__ == "__main__":
    pytest.main([__file__])
