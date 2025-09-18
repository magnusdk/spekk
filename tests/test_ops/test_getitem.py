from spekk import ops


def test_basic_axis_based_indexing():
    x = ops.zeros((4, 5, 6, 7), dims=["d4", "d5", "d6", "d7"])

    assert x[0].dim_sizes == {"d5": 5, "d6": 6, "d7": 7}, (
        'Indexing with an integer on "d4" removes it.'
    )
    assert x[1, 2].dim_sizes == {"d6": 6, "d7": 7}, (
        'Indexing with integers on "d4" and "d5" removes them.'
    )
    assert x[:, 0].dim_sizes == {"d4": 4, "d6": 6, "d7": 7}, (
        'Indexing with a slice on "d4" and an integer on "d5" removes "d5".'
    )
    assert x[:].dim_sizes == {"d4": 4, "d5": 5, "d6": 6, "d7": 7}, (
        "Full slice (:) preserves all dimensions."
    )
    assert x[:, 3, 1:4].dim_sizes == {"d4": 4, "d6": 3, "d7": 7}, (
        'Slice on "d4", integer on "d5", slice on "d6": "d5" is removed and "d6" becomes size 3.'
    )
    assert x[..., 0].dim_sizes == {"d4": 4, "d5": 5, "d6": 6}, (
        'Ellipsis with an integer on "d7" removes it.'
    )
    assert x[-1, -2, :, :].dim_sizes == {"d6": 6, "d7": 7}, (
        'Negative indices on "d4" and "d5" remove them.'
    )
    assert x[::2, :, ::-1, :].dim_sizes == {"d4": 2, "d5": 5, "d6": 6, "d7": 7}, (
        'Slicing with steps adjusts "d4" to size 2.'
    )
    assert x[1:3, 0, ..., 2:5].dim_sizes == {"d4": 2, "d6": 6, "d7": 3}, (
        'Mixed slicing: integer on "d5" removes it; "d4" becomes size 2 and "d7" becomes size 3.'
    )


def test_basic_dimension_name_based_indexing():
    x = ops.zeros((4, 5, 6, 7), dims=["d4", "d5", "d6", "d7"])

    assert x["d4", 0].dim_sizes == {"d5": 5, "d6": 6, "d7": 7}, (
        'Indexing with an integer on "d4" removes it.'
    )
    assert x["d7", 1, "d5", 2].dim_sizes == {"d4": 4, "d6": 6}, (
        'Indexing with integers on "d7" and "d5" removes them.'
    )
    assert x["d5", :, "d7", 0].dim_sizes == {"d4": 4, "d5": 5, "d6": 6}, (
        'Indexing with a slice on "d5" and an integer on "d4" removes "d5".'
    )
    assert x["d4", :, "d6", 3, "d7", 1:4].dim_sizes == {"d4": 4, "d5": 5, "d7": 3}, (
        'Slice on "d4", integer on "d6", slice on "d7": "d6" is removed and "d7" becomes size 3.'
    )


def test_advanced_axis_based_indexing():
    x = ops.zeros((4), dims=["d4"])
    i1 = ops.zeros((8, 9), dtype="int32", dims=["d8", "d9"])
    assert x[i1].dim_sizes == {"d8": 8, "d9": 9}, (
        'Indexing a dimension with an array with dimensions "d8" and "d9" outputs an array with dimensions "d8" and "d9".'
    )

    # Testing mixed advanced and basic indexing
    x = ops.zeros((4, 5, 6), dims=["d4", "d5", "d6"])
    i1 = ops.zeros((8, 9), dtype="int32", dims=["d8", "d9"])
    assert x[i1, :, :].dim_sizes == {"d5": 5, "d6": 6, "d8": 8, "d9": 9}
    assert x[:, i1, :].dim_sizes == {"d4": 4, "d6": 6, "d8": 8, "d9": 9}
    assert x[:, :, i1].dim_sizes == {"d4": 4, "d5": 5, "d8": 8, "d9": 9}

    # Testing more complex mixed advanced and basic indexing by iterating over all
    # possible ways of indexing two axes in an array with ndim=4.
    x = ops.zeros((4, 5, 6, 7), dims=["d4", "d5", "d6", "d7"])
    i1 = ops.zeros((8, 9), dtype="int32", dims=["d8", "d9"])
    i2 = ops.zeros((2,), dtype="int32", dims=["d2"])
    for axis_1 in range(x.ndim - 1):
        for axis_2 in range(axis_1 + 1, x.ndim):
            indexing_objects = [slice(None)] * x.ndim
            indexing_objects[axis_1] = i1
            indexing_objects[axis_2] = i2
            expected_result = {**x.dim_sizes, **i1.dim_sizes, **i2.dim_sizes}
            del expected_result[x.dims[axis_1]]
            del expected_result[x.dims[axis_2]]
            assert x.__getitem__(tuple(indexing_objects)).dim_sizes == expected_result

    # Mixing indexing by arrays, ints and slices.
    i1 = ops.zeros((8, 9), dtype="int32", dims=["d8", "d9"])
    assert x[i1, 2, :, slice(1, 4)].dim_sizes == {"d6": 6, "d7": 3, "d8": 8, "d9": 9}
    assert x[0, 2, i1, :].dim_sizes == {"d7": 7, "d8": 8, "d9": 9}
    assert x[i1, slice(1, 4), 2, :].dim_sizes == {"d5": 3, "d7": 7, "d8": 8, "d9": 9}

    # Indexing with array that has some of the same dimensions as the indexed array.
    x = ops.zeros((4, 5, 6), dims=["d4", "d5", "d6"])
    i1 = ops.zeros((8, 5), dtype="int32", dims=["d8", "d5"])
    assert x[i1].dim_sizes == {"d5": 5, "d6": 6, "d8": 8}
    assert x[i1].ndim == 3


def test_indexing_with_undefined_dims():
    x = ops.zeros((4, 5, 6, 7))
    assert x[0].shape == (5, 6, 7)
    assert x[1, 2].shape == (6, 7)
    assert x[:, 0].shape == (4, 6, 7)
    assert x[:].shape == (4, 5, 6, 7)
    assert x[:, 3, 1:4].shape == (4, 3, 7)
    assert x[..., 0].shape == (4, 5, 6)
    assert x[-1, -2, :, :].shape == (6, 7)
    assert x[::2, :, ::-1, :].shape == (2, 5, 6, 7)
    assert x[1:3, 0, ..., 2:5].shape == (2, 6, 3)

    # Indexing by an array with undefined dims is OK if it is one-dimensional
    x = ops.zeros((4, 5, 6, 7), dims=["d4", "d5", "d6", "d7"])
    i1 = ops.zeros((2,), dtype="int32")
    assert x[i1].dim_sizes == {"d4": 2, "d5": 5, "d6": 6, "d7": 7}
    assert x[:, i1].dim_sizes == {"d4": 4, "d5": 2, "d6": 6, "d7": 7}
    assert x[:, :, i1].dim_sizes == {"d4": 4, "d5": 5, "d6": 2, "d7": 7}
    assert x[..., i1].dim_sizes == {"d4": 4, "d5": 5, "d6": 6, "d7": 2}


if __name__ == "__main__":
    test_basic_axis_based_indexing()
    test_basic_dimension_name_based_indexing()
    test_advanced_axis_based_indexing()
    test_indexing_with_undefined_dims()
