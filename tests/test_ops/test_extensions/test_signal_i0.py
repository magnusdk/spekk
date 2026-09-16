"""Bessel primitive comparisons against SciPy."""

import numpy as np
import pytest
from scipy.special import i0 as scipy_i0

from spekk import ops


@pytest.fixture(autouse=True)
def numpy_backend():
    with ops.backend.temporary_backend("numpy"):
        yield


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_i0_values_and_dimensions(dtype):
    values = np.linspace(-80, 80, 126, dtype=dtype).reshape(3, 42)
    result = ops.i0(ops.array(values, dims=["batch", "time"]))
    assert result.dims == ["batch", "time"]
    assert ops.to_numpy(result).dtype == dtype
    np.testing.assert_allclose(ops.to_numpy(result), scipy_i0(values), rtol=3e-6 if dtype == np.float32 else 1e-14)


@pytest.mark.parametrize("value", [0, 1, -2, 10.0, 100.0, 700.0])
def test_i0_scalar(value):
    result = ops.signal.i0(value)
    assert result.shape == ()
    assert float(result) == pytest.approx(scipy_i0(value), rel=1e-14)


def test_i0_empty_and_complex():
    assert ops.i0(ops.array([], dims=["time"])).shape == (0,)
    with pytest.raises(ValueError):
        ops.i0(1j)


@pytest.mark.parametrize("backend_name", ["numpy", "jax", "torch"])
def test_i0_backend_values(backend_name):
    pytest.importorskip(backend_name)
    with ops.backend.temporary_backend(backend_name):
        values = np.linspace(-40, 40, 81, dtype=np.float32)
        actual = ops.i0(ops.array(values, dims=["time"]))
        assert actual.dims == ["time"]
        np.testing.assert_allclose(ops.to_numpy(actual), scipy_i0(values), rtol=3e-6)


def test_i0_jax_jit():
    jax = pytest.importorskip("jax")
    with jax.enable_x64(True), ops.backend.temporary_backend("jax"):
        values = np.array([-100, -8, -1, 0, 1, 8, 100], dtype=np.float64)
        actual = ops.jit(ops.i0)(ops.array(values, dims=["time"]))
        np.testing.assert_allclose(ops.to_numpy(actual), scipy_i0(values), rtol=1e-13)