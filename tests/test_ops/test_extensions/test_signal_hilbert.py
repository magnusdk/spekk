"""Tests for ``spekk.ops.signal.hilbert`` against ``scipy.signal.hilbert``."""

import numpy as np
import pytest
from scipy.signal import hilbert as scipy_hilbert

import spekk.ops as ops

# Use the numpy backend so float64 is preserved, allowing a tight comparison
# against scipy (which computes in float64). This mirrors the convention in
# other spekk test modules (e.g. test_matmul.py, test_indexing.py).
ops.backend.set_backend("numpy")

RTOL = 1e-9
ATOL = 1e-10


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 7, 8, 16, 31, 64, 100])
def test_hilbert_1d_matches_scipy(seed: int, length: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(length)
    result = ops.signal.hilbert(ops.array(x, ["time"]), axis="time")
    expected = scipy_hilbert(x, axis=-1)
    np.testing.assert_allclose(ops.to_numpy(result), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("size", [8, 9, 16, 17])
@pytest.mark.parametrize("N", [1, 4, 7, 8, 9, 15, 16, 17, 32, 33])
def test_hilbert_with_N_padding_and_trimming(size: int, N: int):
    rng = np.random.default_rng(size * 100 + N)
    x = rng.standard_normal(size)
    result = ops.signal.hilbert(ops.array(x, ["time"]), N, axis="time")
    expected = scipy_hilbert(x, N, axis=-1)
    np.testing.assert_allclose(ops.to_numpy(result), expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shape", [(3, 16), (4, 17), (1, 8), (5, 1)])
def test_hilbert_2d_along_last_axis_name_int_neg(shape):
    rng = np.random.default_rng(sum(shape))
    x = rng.standard_normal(shape)
    xa = ops.array(x, ["channel", "time"])
    expected = scipy_hilbert(x, axis=1)
    for result in (
        ops.signal.hilbert(xa, axis="time"),
        ops.signal.hilbert(xa, axis=1),
        ops.signal.hilbert(xa, axis=-1),
    ):
        np.testing.assert_allclose(
            ops.to_numpy(result), expected, rtol=RTOL, atol=ATOL
        )


@pytest.mark.parametrize("shape", [(16, 3), (17, 4), (8, 1)])
def test_hilbert_2d_along_first_axis(shape):
    rng = np.random.default_rng(sum(shape) + 1)
    x = rng.standard_normal(shape)
    xa = ops.array(x, ["time", "channel"])
    result = ops.signal.hilbert(xa, axis="time")
    expected = scipy_hilbert(x, axis=0)
    np.testing.assert_allclose(ops.to_numpy(result), expected, rtol=RTOL, atol=ATOL)


def test_hilbert_3d_along_named_axis():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((3, 16, 5))
    xa = ops.array(x, ["a", "time", "b"])
    result = ops.signal.hilbert(xa, axis="time")
    expected = scipy_hilbert(x, axis=1)
    np.testing.assert_allclose(ops.to_numpy(result), expected, rtol=RTOL, atol=ATOL)


def test_hilbert_real_part_recovers_input():
    t = ops.linspace(0, 1, 200, dim="time")
    x = ops.cos(2 * ops.pi * 5 * t) + 0.5 * ops.sin(2 * ops.pi * 12 * t)
    analytic = ops.signal.hilbert(x, axis="time")
    np.testing.assert_allclose(
        ops.to_numpy(ops.real(analytic)), ops.to_numpy(x), rtol=RTOL, atol=ATOL
    )


def test_hilbert_envelope_of_am_signal():
    # The envelope of an amplitude-modulated signal should track the modulation.
    t = np.linspace(0, 1, 1024, endpoint=False)
    envelope = 1.0 + 0.5 * np.cos(2 * np.pi * 3 * t)
    carrier = np.cos(2 * np.pi * 120 * t)
    x = envelope * carrier

    analytic = ops.signal.hilbert(ops.array(x, ["time"]), axis="time")
    recovered = ops.to_numpy(ops.abs(analytic))
    # Ignore the edges where the Hilbert transform has boundary effects.
    np.testing.assert_allclose(recovered[50:-50], envelope[50:-50], rtol=0, atol=2e-2)


def test_hilbert_complex_input_raises():
    x = ops.array(np.ones(8, dtype=np.complex128), ["time"])
    with pytest.raises(ValueError):
        ops.signal.hilbert(x, axis="time")


@pytest.mark.parametrize("bad_N", [0, -1, -8])
def test_hilbert_nonpositive_N_raises(bad_N: int):
    x = ops.array(np.ones(8), ["time"])
    with pytest.raises(ValueError):
        ops.signal.hilbert(x, bad_N, axis="time")


def test_hilbert_jit_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(256)
    xa = ops.array(x, ["time"])

    jitted = ops.jit(lambda a: ops.signal.hilbert(a, axis="time"))
    result = jitted(xa)
    expected = scipy_hilbert(x, axis=-1)
    np.testing.assert_allclose(ops.to_numpy(result), expected, rtol=1e-7, atol=1e-8)
