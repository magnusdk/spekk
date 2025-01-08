from spekk import ops
from spekk.ops._types import UndefinedDim

ops.backend.set_backend("numpy")

arr = ops.reshape(
    ops.arange(2 * 3 * 4 * 5),
    (2, 3, 4, 5),
    dims=["dim2", "dim3", "dim4", "dim5"],
)


def test_index_by_int():
    result = arr[0, ...]
    assert result.shape == (3, 4, 5)
    assert result.dims == ["dim3", "dim4", "dim5"]

    result = arr[0, 0, ...]
    assert result.shape == (4, 5)
    assert result.dims == ["dim4", "dim5"]

    result = arr[0, 0, 0, ...]
    assert result.shape == (5,)
    assert result.dims == ["dim5"]

    result = arr[0, 0, 0, 0]
    assert result.shape == ()
    assert result.dims == []


def test_index_by_int_and_slice():
    result = arr[:, 0, ...]
    assert result.shape == (2, 4, 5)
    assert result.dims == ["dim2", "dim4", "dim5"]

    result = arr[0, :, 0:1, ...]
    assert result.shape == (3, 1, 5)
    assert result.dims == ["dim3", "dim4", "dim5"]


def test_index_by_none():
    result = arr[None]
    assert result.shape == (1, 2, 3, 4, 5)
    assert isinstance(result.dims[0], UndefinedDim)
    assert result.dims[1:] == ["dim2", "dim3", "dim4", "dim5"]

    result = arr[:, 0, :, None, ...]
    assert result.shape == (2, 4, 1, 5)
    assert isinstance(result.dims[2], UndefinedDim)
    assert result.dims[:2] + result.dims[3:] == ["dim2", "dim4", "dim5"]


def test_index_by_array():
    i = ops.array([0, 1], dims=["foo"])
    result = arr[:, i, ...]
    assert result.shape == (2, 2, 4, 5)
    assert result.dims == ["dim2", "dim3", "dim4", "dim5"]

    result = arr[..., i]
    assert result.shape == (2, 3, 4, 2)
    assert result.dims == ["dim2", "dim3", "dim4", "dim5"]


def test_set_item():
    a = ops.reshape(ops.arange(24), (3, 4, 2), dims=["azimuths", "depths", "rx"])
    im = ops.zeros((6, 4), dims=["azimuths", "depths"])
    im[ops.arange(3), ...] += a
    assert im.shape == (6, 4, 2)
    assert im.dims == ["azimuths", "depths", "rx"]
    assert ops.all(im[:3] == a)
    assert ops.all(im[3:] == 0)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
