from typing import Literal

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

import spekk.ops as ops


@given(
    x=arrays(
        dtype=np.complex64,
        shape=array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=st.complex_numbers(
            min_magnitude=0,
            max_magnitude=10,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    filter=arrays(
        dtype=np.float32,
        shape=array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=100),
        elements=st.floats(
            min_value=-10,
            max_value=10,
            allow_nan=False,
            allow_infinity=False,
        ),
    ),
    mode=...,
)
def test_fftconvolve(
    x: np.ndarray,
    filter: np.ndarray,
    mode: Literal["full", "same", "valid"],
):
    from scipy.signal import fftconvolve

    x = ops.array(x, ["time"])
    filter = ops.array(filter, ["filter_coefficients"])
    mode = "same"
    assume(filter.size < x.size)

    result = ops.fftconvolve(
        x, filter, mode=mode, axis="time", filter_axis="filter_coefficients"
    )
    expected = fftconvolve(x, filter, mode)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-4)
