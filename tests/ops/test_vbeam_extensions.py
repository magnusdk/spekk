import random
from typing import Literal

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays
from tqdm import tqdm

import spekk.ops as ops


# TODO: Make me into a hypothesis test
def foo():
    with ops.backend.temporary_backend("numpy"):
        if __name__ == "__main__":
            for _ in tqdm(range(1000)):
                dims = {name: size + 1 for name, size in zip("abcdefghijk", range(10))}

                # Construct a random data array x that will be indexed
                x_dims = random.sample(
                    list(dims.keys()), random.randint(1, len(dims) - 1)
                )
                x = ops.array(np.random.randn(*[dims[d] for d in x_dims]), x_dims)

                # Get a random dimension that will be indexed.
                x_indexing_dim = random.choice(x_dims)

                # Construct a random indices array i. It contains a dimension i_indexing_dim
                # which is used to index x. The size of i_indexing_dim may be smaller than
                # x_sample_dim.
                i_dims = random.sample(
                    list(dims.keys()), random.randint(1, len(dims) - 1)
                )
                if x_indexing_dim in i_dims:
                    i_dims.remove(x_indexing_dim)
                # Convert the x_indexing_dim to a potentially smaller i_indexing_dim.
                i_indexing_dim = x_indexing_dim + "_i"
                dims[i_indexing_dim] = random.randint(1, dims[x_indexing_dim])
                i_dims.append(i_indexing_dim)
                random.shuffle(i_dims)

                i = ops.array(
                    np.random.randint(
                        0, dims[x_indexing_dim], tuple(dims[d] for d in i_dims)
                    ),
                    i_dims,
                )

                # Construct a dictionary of the expected dimension sizes.
                expected_output_dim_sizes = x.dim_size() | i.dim_size()
                # The resulting array will have the indexing dimension of the indices in place
                # of the indexing dimension of the data.
                del expected_output_dim_sizes[x_indexing_dim]

                try:
                    result = ops.take_along_dim(x, i, x_indexing_dim)
                    assert expected_output_dim_sizes == result.dim_size()
                except Exception:
                    print(f"{x.dims=}")
                    print(f"{i.dims=}")
                    print(f"{x_indexing_dim=}")
                    print(f"{i_indexing_dim=}")
                    print(f"{expected_output_dim_sizes=}")
                    try:
                        print(f"{result.dim_size()=}")
                    except Exception:
                        pass
                    raise


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


# if __name__ == "__main__": foo()
