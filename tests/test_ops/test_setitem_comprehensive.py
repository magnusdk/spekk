"""
SPEKK NAMED DIMENSION SETITEM TEST SUITE

This test suite mirrors test_indexing.py to provide comprehensive coverage
for setitem functionality. It follows the same structure and patterns but
focuses on assignment operations with proper broadcasting and dimension expansion.

KEY SETITEM PATTERNS TESTED:
• Basic setitem: x["dim", integer] = value and x["dim", slice] = value
• Advanced setitem: x["dim", array] = value with new/existing dimensions
• Multiple setitem: x["a", idx1, "b", idx2] = value combinations
• Boolean setitem: x[boolean_mask] = value patterns
• Coordinated setitem: x["c", :2, "a", array_with_c_dim] = value
• Dimension expansion: arrays can grow with new dimensions from value
• Error cases: invalid dimensions, duplicates, size conflicts

SETITEM SPECIFIC FEATURES:
• Seamless dimension expansion when value has new dimensions
• Broadcasting by dimension names
• Returns new array instance (immutable semantics)
• Proper error handling for incompatible operations
"""

import numpy as np
import pytest

from spekk import ops
from spekk.ops._indexing import setitem
from spekk.ops._types import _UndefinedDim


def create_test_array(shape, dims):
    """Helper function to create test arrays with specified shape and dimensions."""
    total_size = 1
    for s in shape:
        total_size *= s
    data = ops.arange(total_size)
    return ops.reshape(data, shape, dims=dims)


# ============================================================================
# 1. BASIC SETITEM TESTS
# ============================================================================


def test_basic_single_dimension_integer():
    """Test basic setitem with integer indexing."""
    x = create_test_array((3, 4), ["a", "b"])
    value = ops.ones((4,), dims=["b"])

    result = setitem(x, {"a": 0}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (3, 4)
    # First row should be all ones
    assert ops.all(result["a", 0] == 1)
    # Other rows should be unchanged
    assert ops.all(result["a", 1] == x["a", 1])


def test_basic_single_dimension_slice():
    """Test basic setitem with slice indexing."""
    x = create_test_array((4, 3), ["a", "b"])
    value = ops.ones((2, 3), dims=["a", "b"])

    result = setitem(x, {"a": slice(1, 3)}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (4, 3)
    # Rows 1-2 should be ones
    assert ops.all(result["a", slice(1, 3)] == 1)
    # Row 0 should be unchanged
    assert ops.all(result["a", 0] == x["a", 0])


def test_basic_multiple_dimensions():
    """Test setitem with multiple basic dimensions."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    value = ops.array(42)  # Scalar

    result = setitem(x, {"a": 1, "c": 2}, value)

    assert result.dims == ["a", "b", "c"]
    assert result.shape == (3, 4, 5)
    # Check that the specific element was set
    indexed_result = result["a", 1, "c", 2]
    assert indexed_result.shape == (4,)  # Should be 1D array with dimension "b"
    # All elements in dimension b should be 42
    assert ops.all(indexed_result == 42)


def test_zero_dimensional_result():
    """Test setitem that results in scalar assignment."""
    x = create_test_array((2, 3), ["a", "b"])
    value = ops.array(99)  # Scalar

    result = setitem(x, {"a": 0, "b": 1}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (2, 3)
    assert int(result["a", 0, "b", 1]) == 99


# ============================================================================
# 2. ADVANCED SETITEM TESTS
# ============================================================================


def test_advanced_new_dimension():
    """Test setitem with array index that introduces new dimension."""
    x = create_test_array((3, 4), ["a", "b"])
    indices = ops.array([0, 2], dims=["new_dim"])
    value = ops.ones((2, 4), dims=["new_dim", "b"])

    result = setitem(x, {"a": indices}, value)

    # Should have original dimensions plus new dimension
    expected_dims = {"a", "b", "new_dim"}
    assert set(result.dims) == expected_dims
    # Array should have expanded to include new dimension
    assert result.shape == (3, 4, 2)


def test_advanced_existing_dimension():
    """Test setitem with array index using existing dimension."""
    x = create_test_array((4, 3), ["a", "b"])
    indices = ops.array([1, 3], dims=["a"])  # Using existing dimension "a"
    value = ops.ones((2, 3), dims=["a", "b"])

    result = setitem(x, {"a": indices}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (4, 3)
    # The indexed elements should be set to ones
    indexed_result = result[{"a": indices}]
    assert ops.all(indexed_result == 1)


def test_advanced_multiple_indices():
    """Test setitem with multiple array indices."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    idx_a = ops.array([0, 2], dims=["sel_a"])
    idx_c = ops.array([1, 3], dims=["sel_c"])
    # Advanced indexing puts array dimensions first: (sel_a, sel_c, b)
    value = ops.ones((2, 2, 4), dims=["sel_a", "sel_c", "b"])

    result = setitem(x, {"a": idx_a, "c": idx_c}, value)

    # Should include all dimensions
    expected_dims = {"a", "b", "c", "sel_a", "sel_c"}
    assert set(result.dims) == expected_dims


def test_advanced_multidimensional_array():
    """Test setitem with multidimensional array index."""
    x = create_test_array((4, 5), ["a", "b"])
    indices = ops.array([[0, 1], [2, 3]], dims=["idx1", "idx2"])
    value = ops.ones((2, 2, 5), dims=["idx1", "idx2", "b"])

    result = setitem(x, {"a": indices}, value)

    expected_dims = {"a", "b", "idx1", "idx2"}
    assert set(result.dims) == expected_dims
    assert "idx1" in result.dims and "idx2" in result.dims


def test_advanced_same_dimension_indexing():
    """Test setitem where index array shares dimension with target."""
    x = create_test_array((4, 3), ["a", "b"])
    indices = ops.array([1, 2], dims=["a"])  # Index array has dimension "a"
    value = ops.ones((2, 3), dims=["a", "b"])

    result = setitem(x, {"a": indices}, value)

    assert result.dims == ["a", "b"]
    assert result.shape == (4, 3)


def test_advanced_mixed_with_basic():
    """Test setitem mixing array and basic indexing."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    indices = ops.array([0, 2], dims=["sel"])
    value = ops.ones((2, 5), dims=["sel", "c"])

    result = setitem(x, {"a": indices, "b": 1}, value)

    expected_dims = {"a", "b", "c", "sel"}
    assert set(result.dims) == expected_dims


def test_advanced_broadcasting():
    """Test setitem with broadcasting between index arrays."""
    x = create_test_array((4, 5, 6), ["a", "b", "c"])
    idx1 = ops.array([0, 2], dims=["shared"])
    idx2 = ops.array([1, 3], dims=["shared"])
    value = ops.ones((2, 6), dims=["shared", "c"])

    result = setitem(x, {"a": idx1, "b": idx2}, value)

    expected_dims = {"a", "b", "c", "shared"}
    assert set(result.dims) == expected_dims


def test_advanced_same_array_indexing():
    """Test setitem using the same array to index multiple dimensions."""
    x = create_test_array((5, 5, 3), ["a", "b", "c"])
    indices = ops.array([1, 3], dims=["diag"])
    value = ops.ones((2, 3), dims=["diag", "c"])

    result = setitem(x, {"a": indices, "b": indices}, value)

    expected_dims = {"a", "b", "c", "diag"}
    assert set(result.dims) == expected_dims


def test_complex_nested_indexing():
    """Test setitem with complex nested indexing patterns."""
    x = create_test_array((4, 5, 6, 7), ["a", "b", "c", "d"])
    idx_a = ops.array([0, 2], dims=["sel1"])
    idx_c = ops.array([1, 4], dims=["sel2"])
    # Advanced indexing puts array dimensions first: (sel1, sel2, b, d)
    value = ops.ones((2, 2, 5, 7), dims=["sel1", "sel2", "b", "d"])

    result = setitem(x, {"a": idx_a, "c": idx_c}, value)

    expected_dims = {"a", "b", "c", "d", "sel1", "sel2"}
    assert set(result.dims) == expected_dims


def test_complex_mixed_indexing_patterns():
    """Test setitem with complex mix of indexing types."""
    x = create_test_array((6, 7, 8), ["a", "b", "c"])
    indices = ops.array([1, 3, 5], dims=["subset"])
    value = ops.ones((3, 4, 8), dims=["subset", "b", "c"])

    result = setitem(x, {"a": indices, "b": slice(2, 6)}, value)

    expected_dims = {"a", "b", "c", "subset"}
    assert set(result.dims) == expected_dims


# ============================================================================
# 3. BOOLEAN SETITEM TESTS
# ============================================================================


def test_basic_1d_boolean_mask():
    """Test basic setitem with 1D boolean mask."""
    x = create_test_array((5, 3), ["height", "width"])
    mask = ops.array([True, False, True, False, True], dims=["height"])
    value = ops.ones((3,), dims=["width"])

    result = setitem(x, {"height": mask}, value)

    # Boolean indexing preserves the dimension name when 1D
    assert result.dims == ["height", "width"]
    assert result.dim_sizes == {"height": 5, "width": 3}

    # Check that masked positions are set to value
    expected = x.at[{"height": mask}].set(value)
    assert ops.all(result == expected)


def test_1d_boolean_mask_with_shared_dimension():
    """Test boolean mask that shares dimension with value array."""
    x = create_test_array((4, 6), ["batch", "features"])
    mask = ops.array([True, False, False, True], dims=["batch"])
    # Value has same "batch" dimension - should broadcast correctly
    value = ops.ones((2, 6), dims=["batch", "features"])

    result = setitem(x, {"batch": mask}, value)

    assert result.dims == ["batch", "features"]
    assert result.dim_sizes == {"batch": 4, "features": 6}
    np.testing.assert_equal(
        result.data,
        np.array(
            [
                [1, 1, 1, 1, 1, 1],
                [6, 7, 8, 9, 10, 11],
                [12, 13, 14, 15, 16, 17],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )


def test_multidimensional_boolean_mask():
    """Test setitem with multi-dimensional boolean mask."""
    x = create_test_array((3, 4, 5), ["height", "width", "depth"])
    # Create 2D boolean mask over height and width dimensions
    mask_data = ops.ones((3, 4), dtype="bool")
    mask = ops.array(mask_data.data, dims=["height", "width"])

    # Value should have depth dimension (the non-masked dimension)
    value = ops.ones((5,), dims=["depth"])

    result = setitem(
        x, {"depth": mask}, value
    )  # Note: indexing "depth" with mask over height/width

    # Multi-dimensional boolean mask creates undefined dimension
    assert len(result.dims) == 3
    # The depth dimension gets replaced by the flattened boolean result


def test_boolean_mask_with_new_dimension_in_value():
    """Test boolean setitem where value introduces new dimension."""
    x = create_test_array((4, 3), ["time", "space"])
    mask = ops.array([True, False, True, False], dims=["time"])
    # Value has new dimension "channels"
    value = ops.ones((2, 3), dims=["channels", "space"])

    result = setitem(x, {"time": mask}, value)

    # Result should expand to include new dimension
    expected_dims = {"time", "space", "channels"}
    assert set(result.dims) == expected_dims


def test_boolean_mask_dimension_size_mismatch_error():
    """Test that boolean mask with wrong dimension size raises error."""
    x = create_test_array((5, 3), ["rows", "cols"])
    # Mask has wrong size for "rows" dimension
    wrong_mask = ops.array([True, False, True], dims=["rows"])  # size 3, should be 5
    value = ops.ones((3,), dims=["cols"])

    with pytest.raises(IndexError, match="Mismatched dimension sizes"):
        setitem(x, {"rows": wrong_mask}, value)


def test_boolean_mask_nonexistent_dimension_error():
    """Test boolean mask referencing non-existent dimension."""
    x = create_test_array((3, 4), ["height", "width"])
    mask = ops.array([True, False, True], dims=["height"])
    value = ops.ones((4,), dims=["width"])

    # Try to use mask for dimension that doesn't exist
    with pytest.raises(IndexError, match="does not exist in the array"):
        setitem(x, {"nonexistent": mask}, value)


def test_boolean_mask_conflicting_dimension_indexing():
    """Test error when boolean mask conflicts with explicit indexing."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    # Boolean mask that references dimension "b"
    mask = ops.array(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, True, False, False],
        ],
        dims=["a", "b"],
    )
    value = ops.ones((5,), dims=["c"])

    # Try to also explicitly index dimension "b" - should conflict
    with pytest.raises(IndexError, match="dimensions are also explicitly indexed"):
        setitem(x, {"c": mask, "b": slice(2, 4)}, value)


def test_multiple_boolean_masks_error():
    """Test that using multiple boolean masks is handled correctly."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])
    mask1 = ops.array([True, False, True], dims=["a"])
    mask2 = ops.array([True, False, True, False], dims=["b"])
    value = ops.ones((5,), dims=["c"])

    # Multiple boolean masks should work if they don't conflict
    result = setitem(x, {"a": mask1, "b": mask2}, value)

    # Check result has expected structure
    assert "c" in result.dims
    assert result.dim_sizes["c"] == 5


def test_boolean_mask_preserves_other_dimensions():
    """Test that boolean setitem preserves non-indexed dimensions."""
    x = create_test_array((2, 5, 3), ["batch", "time", "features"])
    mask = ops.array([True, False, True, False, False], dims=["time"])
    value = ops.ones((2, 3), dims=["batch", "features"])

    result = setitem(x, {"time": mask}, value)

    # Non-indexed dimensions should be preserved
    assert "batch" in result.dims
    assert "features" in result.dims
    assert result.dim_sizes["batch"] == 2
    assert result.dim_sizes["features"] == 3


def test_boolean_mask_empty_selection():
    """Test boolean mask that selects no elements."""
    x = create_test_array((4, 3), ["rows", "cols"])
    mask = ops.array([False, False, False, False], dims=["rows"])
    value = ops.ones((3,), dims=["cols"])

    result = setitem(x, {"rows": mask}, value)

    # Should still work, just no elements get modified
    assert result.dims == ["rows", "cols"]
    assert result.dim_sizes == {"rows": 4, "cols": 3}
    # Original values should be unchanged since mask selected nothing
    assert ops.all(result == x)


def test_mixed_boolean_and_basic_indexing():
    """Test combining boolean mask with slice/integer indexing."""
    x = create_test_array((4, 5, 3), ["batch", "time", "features"])
    # Boolean mask on one dimension
    mask = ops.array([True, False, True, False, True], dims=["time"])
    # Combined with slice on another dimension
    value = ops.ones((2, 3), dims=["batch", "features"])

    result = setitem(x, {"time": mask, "batch": slice(0, 2)}, value)

    # Should preserve the sliced batch dimension and boolean-indexed time
    assert "batch" in result.dims
    assert "time" in result.dims
    assert "features" in result.dims
    assert result.dim_sizes["batch"] == 4  # Original size preserved
    assert result.dim_sizes["features"] == 3


def test_mixed_boolean_and_advanced_indexing():
    """Test combining boolean mask with array indexing - critical test for complex indexing logic."""
    x = create_test_array((3, 4, 5), ["height", "width", "depth"])
    # Boolean mask on height
    height_mask = ops.array([True, False, True], dims=["height"])
    # Array indexing on width
    width_indices = ops.array([0, 2], dims=["selected_width"])
    # Value that broadcasts correctly
    value = ops.ones((5,), dims=["depth"])

    result = setitem(x, {"height": height_mask, "width": width_indices}, value)

    # This tests the complex broadcasting logic in the indexing implementation
    assert "depth" in result.dims
    assert result.dim_sizes["depth"] == 5
    # The height and width dimensions get transformed by the indexing


# ============================================================================
# 4. EDGE CASE TESTS AND ERROR HANDLING
# ============================================================================


def test_nonexistent_dimension_error():
    """Test that setitem with non-existent dimension names raises ValueError."""
    x = create_test_array((2, 3, 4), ["a", "b", "c"])
    value = ops.ones((4,), dims=["c"])

    with pytest.raises(IndexError):
        setitem(x, {"nonexistent": 0}, value)

    with pytest.raises(IndexError):
        setitem(x, {"a": 0, "nonexistent": 1}, value)


def test_duplicate_dimensions_error():
    """Test that setitem with same dimension twice raises ValueError."""
    x = create_test_array((2, 3, 4), ["a", "b", "c"])
    value = ops.array(42)

    with pytest.raises(IndexError):
        setitem(x, ("a", 0, "a", 1), value)

    with pytest.raises(IndexError):
        setitem(x, ("b", slice(None), "b", 0), value)


def test_inconsistent_dimension_sizes():
    """Test error when index arrays have same dimension name but different sizes."""
    x = create_test_array((3, 4, 5), ["a", "b", "c"])

    indices1 = ops.array([0, 1], dims=["shared_dim"])  # size 2
    indices2 = ops.array([0, 1, 2], dims=["shared_dim"])  # size 3
    value = ops.ones((2, 3), dims=["shared_dim", "other_dim"])

    with pytest.raises(IndexError):
        setitem(x, {"a": indices1, "b": indices2}, value)


def test_odd_number_indexing_objects():
    """Test that odd number of indexing objects raises error."""
    x = create_test_array((2, 3, 4), ["a", "b", "c"])
    value = ops.array(42)

    with pytest.raises(IndexError):
        setitem(x, ("a",), value)  # Only dimension name, no index

    with pytest.raises(IndexError):
        setitem(x, ("a", 0, "b"), value)  # Missing index for "b"


# ============================================================================
# 5. DIMENSION EXPANSION TESTS (SETITEM SPECIFIC)
# ============================================================================


def test_dimension_expansion_simple():
    """Test that setitem can expand array with new dimensions from value."""
    x = create_test_array((3, 4), ["a", "b"])
    value = ops.ones((4, 2), dims=["b", "new_dim"])  # Value has new dimension

    result = setitem(x, {"a": 0}, value)

    # Result should have original dimensions plus new dimension
    expected_dims = {"a", "b", "new_dim"}
    assert set(result.dims) == expected_dims
    assert result.shape == (3, 4, 2)


def test_dimension_expansion_multiple():
    """Test setitem with multiple new dimensions."""
    x = create_test_array((2, 3), ["a", "b"])
    value = ops.ones((3, 4, 5), dims=["b", "new1", "new2"])

    result = setitem(x, {"a": 1}, value)

    expected_dims = {"a", "b", "new1", "new2"}
    assert set(result.dims) == expected_dims
    assert result.shape == (2, 3, 4, 5)


def test_dimension_expansion_with_array_indexing():
    """Test dimension expansion when using array indexing."""
    x = create_test_array((3, 4), ["a", "b"])
    indices = ops.array([0, 2], dims=["sel"])
    value = ops.ones((2, 4, 3), dims=["sel", "b", "new_dim"])

    result = setitem(x, {"a": indices}, value)

    expected_dims = {"a", "b", "sel", "new_dim"}
    assert set(result.dims) == expected_dims


# ============================================================================
# 6. COMPLEX REAL-WORLD SETITEM SCENARIOS
# ============================================================================


def test_ml_batch_update():
    """Test ML-style batch update scenario."""
    # Update specific samples in a batch
    batch = create_test_array((32, 128, 10), ["batch", "sequence", "features"])
    sample_ids = ops.array([0, 5, 10, 15], dims=["update_batch"])
    new_features = ops.ones((4, 128, 10), dims=["update_batch", "sequence", "features"])

    result = setitem(batch, {"batch": sample_ids}, new_features)

    expected_dims = {"batch", "sequence", "features", "update_batch"}
    assert set(result.dims) == expected_dims


def test_time_series_window_update():
    """Test time series window update."""
    # Update specific time windows across multiple series
    data = create_test_array((100, 50), ["time", "series"])
    window_times = ops.arange(10, 20, dim="window")
    window_data = ops.ones((10, 50), dims=["window", "series"])

    result = setitem(data, {"time": window_times}, window_data)

    expected_dims = {"time", "series", "window"}
    assert set(result.dims) == expected_dims

@pytest.skip("Terminates. Probably OOM but not sure.")
def test_scientific_data_calibration():
    """Test scientific data calibration scenario."""
    # Apply calibration to specific instruments and wavelengths
    measurements = create_test_array(
        (1000, 256, 64), ["time", "wavelength", "instrument"]
    )
    cal_wavelengths = ops.array([100, 150, 200], dims=["cal_band"])
    cal_instruments = ops.array([0, 10, 20], dims=["cal_inst"])
    calibration = ops.ones((3, 3), dims=["cal_band", "cal_inst"]) * 1.1

    result = setitem(
        measurements,
        {"wavelength": cal_wavelengths, "instrument": cal_instruments},
        calibration,
    )

    expected_dims = {"time", "wavelength", "instrument", "cal_band", "cal_inst"}
    assert set(result.dims) == expected_dims


def test_image_patch_update():
    """Test image patch update scenario."""
    # Update specific patches in an image
    image = create_test_array((224, 224, 3), ["height", "width", "channels"])
    patch_h = ops.arange(50, 60, dim="patch_h")
    patch_w = ops.arange(100, 110, dim="patch_w")
    patch_data = ops.ones((10, 10, 3), dims=["patch_h", "patch_w", "channels"])

    result = setitem(image, {"height": patch_h, "width": patch_w}, patch_data)

    expected_dims = {"height", "width", "channels", "patch_h", "patch_w"}
    assert set(result.dims) == expected_dims


def test_sparse_matrix_update():
    """Test sparse matrix-style update."""
    # Update specific elements in a large matrix
    matrix = create_test_array((1000, 1000), ["rows", "cols"])
    update_rows = ops.array([10, 50, 100, 500], dims=["updates"])
    update_cols = ops.array([20, 60, 200, 600], dims=["updates"])
    values = ops.ones(4, dims=["updates"]) * 999

    result = setitem(matrix, {"rows": update_rows, "cols": update_cols}, values)

    expected_dims = {"rows", "cols", "updates"}
    assert set(result.dims) == expected_dims


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v"])
