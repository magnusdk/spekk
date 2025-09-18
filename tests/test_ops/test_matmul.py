import hypothesis.extra.numpy as hnp
import numpy as np
import pytest
from hypothesis import assume, given

from spekk import ops

ops.backend.set_backend("numpy")


@given(
    a_shape=hnp.array_shapes(min_dims=1, max_dims=5, min_side=2, max_side=6),
    b_shape=hnp.array_shapes(min_dims=1, max_dims=5, min_side=2, max_side=6),
)
def test_matmul(a_shape, b_shape):
    # Assume unique dimensions
    assume(len(set(a_shape)) == len(a_shape))
    assume(len(set(b_shape)) == len(b_shape))

    a = np.zeros(a_shape)
    b = np.zeros(b_shape)
    spekk_a = ops.array(a, [f"d{s}" for s in a.shape])
    spekk_b = ops.array(b, [f"d{s}" for s in b.shape])

    try:
        np_result = np.matmul(a, b)
        # Assume unique dimensions after matmul as well
        assume(len(set(np_result.shape)) == len(np_result.shape))
    except Exception:
        with pytest.raises(Exception):
            spekk_result = ops.matmul(spekk_a, spekk_b)
    else:
        spekk_result = ops.matmul(spekk_a, spekk_b)
        np.testing.assert_allclose(np_result, spekk_result)
        assert spekk_result.dims == [f"d{s}" for s in spekk_result.shape]
